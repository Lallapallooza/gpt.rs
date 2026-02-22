use crate::backend::{
    driver::{apply_patterns_and_fold_greedily, GreedyConfig},
    optimizer::OptimizeContext,
    pattern::{OpRewritePattern, PatternSet, ReshapeOpView},
    rewriter::ProgramRewriter,
    spec::{
        Dimension, Function, Operand, Operation, PortableBackend, ReshapeDim, ReshapeSpec,
        ValueType,
    },
};

use super::{FunctionPass, FunctionPassResult};

/// Remove reshapes that do not change the logical shape.
pub struct EliminateIdentityReshape;

impl OpRewritePattern<ReshapeOpView> for EliminateIdentityReshape {
    fn match_and_rewrite(&self, view: ReshapeOpView, rewriter: &mut ProgramRewriter) -> bool {
        let [Operand::Value(source)] = view.operands.as_slice() else {
            return false;
        };
        let Some(ValueType::Tensor(input_ty)) = rewriter.type_of(*source).cloned() else {
            return false;
        };
        let ValueType::Tensor(result_ty) = view.result_type.clone() else {
            return false;
        };

        if input_ty.shape != result_ty.shape || input_ty.dtype != result_ty.dtype {
            return false;
        }

        rewriter.replace_all_uses(view.result, *source);
        for result_id in &mut rewriter.func.result_ids {
            if *result_id == view.result {
                *result_id = *source;
            }
        }
        rewriter
            .erase_inst(view.root)
            .expect("reshape canonicalization erase should succeed");
        true
    }
}

/// Fold reshape chains into a single reshape from the original source to the final shape.
pub struct CollapseReshapeChain;

impl OpRewritePattern<ReshapeOpView> for CollapseReshapeChain {
    fn match_and_rewrite(&self, view: ReshapeOpView, rewriter: &mut ProgramRewriter) -> bool {
        let [Operand::Value(inner_value)] = view.operands.as_slice() else {
            return false;
        };
        let Some(inner_inst) = rewriter.inst_of(*inner_value) else {
            return false;
        };
        let Operation::Reshape(_inner_spec) = rewriter.op(inner_inst).clone() else {
            return false;
        };

        let [Operand::Value(base_value)] = rewriter.operands(inner_inst) else {
            return false;
        };

        let Some(ValueType::Tensor(base_ty)) = rewriter.type_of(*base_value).cloned() else {
            return false;
        };
        let ValueType::Tensor(result_ty) = view.result_type.clone() else {
            return false;
        };

        // Only fold when all shapes are static and element counts match to avoid dynamic surprises.
        let Some(_base_dims) = base_ty.shape.static_dims() else {
            return false;
        };
        let Some(result_dims) = result_ty.shape.static_dims() else {
            return false;
        };
        let Some(base_elems) = base_ty.element_count() else {
            return false;
        };
        let Some(result_elems) = result_ty.element_count() else {
            return false;
        };
        if base_elems != result_elems {
            return false;
        }

        let new_spec = ReshapeSpec {
            new_shape: result_dims
                .iter()
                .copied()
                .map(|d| ReshapeDim::Explicit(Dimension::Static(d)))
                .collect(),
        };
        let output_ty = ValueType::Tensor(result_ty.clone());
        let Ok((_, folded_value)) = rewriter.insert_before(
            view.root,
            Operation::Reshape(new_spec),
            vec![Operand::Value(*base_value)],
            output_ty,
        ) else {
            return false;
        };

        rewriter.replace_all_uses(view.result, folded_value);
        for result_id in &mut rewriter.func.result_ids {
            if *result_id == view.result {
                *result_id = folded_value;
            }
        }
        rewriter
            .erase_inst(view.root)
            .expect("reshape canonicalization erase should succeed");
        true
    }
}

/// Canonical reshape simplifications (identity removal, chain folding).
pub struct ReshapeCanonicalizationPass {
    config: GreedyConfig,
}

impl ReshapeCanonicalizationPass {
    const NAME: &'static str = "reshape-canonicalize";

    pub fn new(config: GreedyConfig) -> Self {
        Self { config }
    }
}

impl Default for ReshapeCanonicalizationPass {
    fn default() -> Self {
        Self::new(GreedyConfig::default())
    }
}

impl<B: PortableBackend + 'static> FunctionPass<B> for ReshapeCanonicalizationPass {
    fn name(&self) -> &'static str {
        Self::NAME
    }

    fn run(&self, function: &mut Function, _cx: &mut OptimizeContext<B>) -> FunctionPassResult {
        let mut patterns = PatternSet::new();
        patterns.insert_view::<ReshapeOpView, _>(EliminateIdentityReshape);
        patterns.insert_view::<ReshapeOpView, _>(CollapseReshapeChain);
        let frozen = patterns.freeze();
        let stats = apply_patterns_and_fold_greedily(function, &frozen, &self.config);
        FunctionPassResult {
            changed: stats.applied > 0 || stats.dce_removed > 0,
            iterations: stats.iterations,
            rewrites_applied: stats.applied,
            erased_insts: stats.dce_removed,
        }
    }
}
