use std::collections::HashSet;

use crate::backend::{
    driver::{apply_patterns_and_fold_greedily, GreedyConfig},
    optimizer::OptimizeContext,
    pattern::{ElementwiseBinaryOpView, OpRewritePattern, PatternSet},
    rewriter::ProgramRewriter,
    spec::{
        BroadcastToSpec, Dimension, ElementwiseBinaryOp, Function, Operand, Operation,
        PortableBackend, ReshapeDim, ReshapeSpec, Shape, TensorSpec, ValueId, ValueType,
    },
};

use super::static_values::static_values;
use super::{FunctionPass, FunctionPassResult};

/// Moves a static scale of a `dot_general` result into the static right operand.
///
/// `dot_general(x, w) * broadcast(s)` equals `dot_general(x, w * broadcast(s'))` when `s` varies
/// only along result axes that come from `w`, with `s'` laid out along those axes of `w`. Because
/// `w` and `s` depend only on parameters and literals, [`super::ParamOnlyFoldToParamPass`] then
/// folds the scaled operand into one parameter, so the scale costs nothing per run. A convolution
/// absorbs the BatchNorm scale after it this way.
#[derive(Default)]
pub struct DotScaleFoldPass {
    config: GreedyConfig,
}

impl DotScaleFoldPass {
    const NAME: &'static str = "dot-scale-fold";
}

impl<B: PortableBackend + 'static> FunctionPass<B> for DotScaleFoldPass {
    fn name(&self) -> &'static str {
        Self::NAME
    }

    fn run(&self, function: &mut Function, cx: &mut OptimizeContext<B>) -> FunctionPassResult {
        // Without the parameter fold, the scaled weight would be recomputed on every run.
        if cx.services().params.is_none() {
            return FunctionPassResult::default();
        }
        let statics = static_values(function, cx);
        let mut patterns = PatternSet::new();
        patterns.insert_view::<ElementwiseBinaryOpView, _>(ScaleIntoDotOperand { statics });
        let stats = apply_patterns_and_fold_greedily(function, &patterns.freeze(), &self.config);
        FunctionPassResult {
            changed: stats.applied > 0 || stats.dce_removed > 0,
            iterations: stats.iterations,
            rewrites_applied: stats.applied,
            erased_insts: stats.dce_removed,
        }
    }
}

struct ScaleIntoDotOperand {
    statics: HashSet<ValueId>,
}

impl OpRewritePattern<ElementwiseBinaryOpView> for ScaleIntoDotOperand {
    fn match_and_rewrite(
        &self,
        view: ElementwiseBinaryOpView,
        rewriter: &mut ProgramRewriter,
    ) -> bool {
        if view.op != ElementwiseBinaryOp::Mul {
            return false;
        }
        let [Operand::Value(lhs), Operand::Value(rhs)] = view.operands.as_slice() else {
            return false;
        };
        let (lhs, rhs) = (*lhs, *rhs);
        [(lhs, rhs), (rhs, lhs)]
            .into_iter()
            .any(|(product, scale)| self.fold(&view, product, scale, rewriter))
    }
}

impl ScaleIntoDotOperand {
    fn fold(
        &self,
        view: &ElementwiseBinaryOpView,
        product: ValueId,
        scale: ValueId,
        rewriter: &mut ProgramRewriter,
    ) -> bool {
        let Some(dot) = rewriter.inst_of(product) else {
            return false;
        };
        let Operation::DotGeneral(spec) = rewriter.op(dot).clone() else {
            return false;
        };
        let [Operand::Value(x), Operand::Value(w)] = *rewriter.operands(dot) else {
            return false;
        };
        let Some(broadcast) = rewriter.inst_of(scale) else {
            return false;
        };
        let (Operation::BroadcastTo(_), [Operand::Value(s)]) =
            (rewriter.op(broadcast), rewriter.operands(broadcast))
        else {
            return false;
        };
        let s = *s;
        // Scaling `w` pays off only when the scaled `w` folds away. Other users of the product need
        // it unscaled.
        if !self.statics.contains(&w)
            || !self.statics.contains(&s)
            || rewriter.users_of(product).len() != 1
        {
            return false;
        }
        // A cast `w` comes from a narrower stored weight. Scaling it would store a widened copy.
        if rewriter
            .inst_of(w)
            .is_some_and(|inst| matches!(rewriter.op(inst), Operation::Cast(_)))
        {
            return false;
        }
        let (Some(w_spec), Some(s_spec), Some(x_spec), ValueType::Tensor(out_spec)) = (
            tensor_spec(rewriter, w),
            tensor_spec(rewriter, s),
            tensor_spec(rewriter, x),
            &view.result_type,
        ) else {
            return false;
        };
        if w_spec.dtype != out_spec.dtype || s_spec.dtype != out_spec.dtype {
            return false;
        }
        let (Some(w_dims), Some(s_dims), Some(out_dims)) = (
            w_spec.shape.static_dims(),
            s_spec.shape.static_dims(),
            out_spec.shape.static_dims(),
        ) else {
            return false;
        };

        // The result axes are the batch axes, the free axes of `x`, then the free axes of `w`.
        let w_free: Vec<usize> = (0..w_dims.len())
            .filter(|axis| !spec.batch_rhs.contains(axis) && !spec.contract_rhs.contains(axis))
            .collect();
        let Some(first_w_axis) = out_dims.len().checked_sub(w_free.len()) else {
            return false;
        };
        if s_dims.len() > out_dims.len()
            || x_spec.shape.rank().checked_sub(spec.contract_lhs.len()) != Some(first_w_axis)
        {
            return false;
        }
        // `broadcast_to` aligns `s` with the trailing result axes.
        let offset = out_dims.len() - s_dims.len();
        let mut scale_dims = vec![1; w_dims.len()];
        for (axis, &extent) in s_dims.iter().enumerate() {
            if extent == 1 {
                continue;
            }
            let Some(free) = (offset + axis).checked_sub(first_w_axis) else {
                return false;
            };
            scale_dims[w_free[free]] = extent;
        }

        let tensor = |dims: &[usize]| {
            ValueType::Tensor(TensorSpec::new(
                out_spec.dtype,
                Shape::new(
                    dims.iter()
                        .map(|&d| Dimension::Static(d))
                        .collect::<Vec<_>>(),
                ),
            ))
        };
        let reshape = Operation::Reshape(ReshapeSpec {
            new_shape: scale_dims
                .iter()
                .map(|&d| ReshapeDim::Explicit(Dimension::Static(d)))
                .collect(),
        });
        let broadcast = Operation::BroadcastTo(BroadcastToSpec {
            result_shape: w_spec.shape.clone(),
        });
        let at = view.root;
        let inserted = (|| {
            let (_, aligned) = rewriter.insert_before(
                at,
                reshape,
                vec![Operand::Value(s)],
                tensor(&scale_dims),
            )?;
            let (_, expanded) = rewriter.insert_before(
                at,
                broadcast,
                vec![Operand::Value(aligned)],
                tensor(&w_dims),
            )?;
            let (_, scaled_w) = rewriter.insert_before(
                at,
                Operation::ElementwiseBinary(ElementwiseBinaryOp::Mul),
                vec![Operand::Value(w), Operand::Value(expanded)],
                tensor(&w_dims),
            )?;
            rewriter.insert_before(
                at,
                Operation::DotGeneral(spec),
                vec![Operand::Value(x), Operand::Value(scaled_w)],
                view.result_type.clone(),
            )
        })();
        let Ok((_, scaled)) = inserted else {
            return false;
        };
        rewriter.replace_all_uses(view.result, scaled);
        for result in &mut rewriter.func.result_ids {
            if *result == view.result {
                *result = scaled;
            }
        }
        rewriter
            .erase_inst(view.root)
            .expect("replaced multiply has no users");
        rewriter
            .erase_inst(dot)
            .expect("the multiply was the only user of the dot");
        true
    }
}

fn tensor_spec(rewriter: &ProgramRewriter, value: ValueId) -> Option<TensorSpec> {
    match rewriter.type_of(value)? {
        ValueType::Tensor(spec) => Some(spec.clone()),
        _ => None,
    }
}
