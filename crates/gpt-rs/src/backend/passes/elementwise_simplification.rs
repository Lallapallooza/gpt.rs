use std::convert::TryInto;

use crate::backend::{
    driver::{apply_patterns_and_fold_greedily, GreedyConfig},
    optimizer::OptimizeContext,
    pattern::{ElementwiseBinaryOpView, OpRewritePattern, PatternSet},
    rewriter::ProgramRewriter,
    spec::{
        Dimension, ElementwiseBinaryOp, Function, Operand, Operation, PortableBackend,
        TensorLiteral,
    },
};

use super::{FunctionPass, FunctionPassResult};

fn scalar_f32_from_literal(lit: &TensorLiteral) -> Option<f32> {
    if lit.spec.dtype != crate::backend::spec::DType::F32 {
        return None;
    }
    let static_dims =
        lit.spec.shape.dims().iter().all(|d| {
            matches!(d, Dimension::Static(v) if *v == 1) || matches!(d, Dimension::Static(0))
        });
    let rank_zero = lit.spec.shape.dims().is_empty();
    if !rank_zero && !static_dims {
        return None;
    }
    let bytes: &[u8] = lit.bytes.as_ref();
    if bytes.len() != 4 {
        return None;
    }
    let arr: [u8; 4] = bytes.try_into().ok()?;
    Some(f32::from_le_bytes(arr))
}

fn scalar_f32_from_value(
    value: crate::backend::spec::ValueId,
    rewriter: &ProgramRewriter,
) -> Option<f32> {
    let inst = rewriter.inst_of(value)?;
    match rewriter.op(inst) {
        Operation::Constant(lit) => scalar_f32_from_literal(lit),
        Operation::BroadcastTo(_) => match rewriter.operands(inst) {
            [Operand::Value(src)] => scalar_f32_from_value(*src, rewriter),
            [Operand::Literal(lit)] => scalar_f32_from_literal(lit),
            _ => None,
        },
        _ => None,
    }
}

/// Removes algebraic no-ops in elementwise binary expressions (add zero, mul by one, div by one).
pub struct ElementwiseSimplificationPass {
    config: GreedyConfig,
}

impl ElementwiseSimplificationPass {
    const NAME: &'static str = "elementwise-simplify";

    pub fn new(config: GreedyConfig) -> Self {
        Self { config }
    }
}

impl Default for ElementwiseSimplificationPass {
    fn default() -> Self {
        Self::new(GreedyConfig::default())
    }
}

impl<B: PortableBackend + 'static> FunctionPass<B> for ElementwiseSimplificationPass {
    fn name(&self) -> &'static str {
        Self::NAME
    }

    fn run(&self, function: &mut Function, _cx: &mut OptimizeContext<B>) -> FunctionPassResult {
        let mut patterns = PatternSet::new();
        patterns.insert_view::<ElementwiseBinaryOpView, _>(SimplifyElementwiseBinary);
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

struct SimplifyElementwiseBinary;

impl OpRewritePattern<ElementwiseBinaryOpView> for SimplifyElementwiseBinary {
    fn match_and_rewrite(
        &self,
        view: ElementwiseBinaryOpView,
        rewriter: &mut ProgramRewriter,
    ) -> bool {
        let lhs_value = match view.operands.first() {
            Some(Operand::Value(v)) => Some(*v),
            _ => None,
        };
        let rhs_value = match view.operands.get(1) {
            Some(Operand::Value(v)) => Some(*v),
            _ => None,
        };

        let lhs_scalar = match view.operands.first() {
            Some(Operand::Literal(lit)) => scalar_f32_from_literal(lit),
            Some(Operand::Value(v)) => scalar_f32_from_value(*v, rewriter),
            _ => None,
        };
        let rhs_scalar = match view.operands.get(1) {
            Some(Operand::Literal(lit)) => scalar_f32_from_literal(lit),
            Some(Operand::Value(v)) => scalar_f32_from_value(*v, rewriter),
            _ => None,
        };

        match view.op {
            ElementwiseBinaryOp::Add => {
                if matches!(rhs_scalar, Some(v) if v == 0.0) {
                    if let Some(lhs) = lhs_value {
                        replace_with_value(view, lhs, rewriter);
                        return true;
                    }
                }
                if matches!(lhs_scalar, Some(v) if v == 0.0) {
                    if let Some(rhs) = rhs_value {
                        replace_with_value(view, rhs, rewriter);
                        return true;
                    }
                }
            }
            ElementwiseBinaryOp::Sub => {
                if matches!(rhs_scalar, Some(v) if v == 0.0) {
                    if let Some(lhs) = lhs_value {
                        replace_with_value(view, lhs, rewriter);
                        return true;
                    }
                }
            }
            ElementwiseBinaryOp::Mul => {
                if matches!(rhs_scalar, Some(v) if v == 1.0) {
                    if let Some(lhs) = lhs_value {
                        replace_with_value(view, lhs, rewriter);
                        return true;
                    }
                }
                if matches!(lhs_scalar, Some(v) if v == 1.0) {
                    if let Some(rhs) = rhs_value {
                        replace_with_value(view, rhs, rewriter);
                        return true;
                    }
                }
                if matches!(rhs_scalar, Some(v) if v == 0.0) {
                    if let Some(rhs) = rhs_value {
                        replace_with_value(view, rhs, rewriter);
                        return true;
                    }
                }
                if matches!(lhs_scalar, Some(v) if v == 0.0) {
                    if let Some(lhs) = lhs_value {
                        replace_with_value(view, lhs, rewriter);
                        return true;
                    }
                }
            }
            ElementwiseBinaryOp::Div => {
                if matches!(rhs_scalar, Some(v) if v == 1.0) {
                    if let Some(lhs) = lhs_value {
                        replace_with_value(view, lhs, rewriter);
                        return true;
                    }
                }
            }
            _ => {}
        }

        false
    }
}

fn replace_with_value(
    view: ElementwiseBinaryOpView,
    replacement: crate::backend::spec::ValueId,
    rewriter: &mut ProgramRewriter,
) {
    rewriter.replace_all_uses(view.result, replacement);
    for result_id in &mut rewriter.func.result_ids {
        if *result_id == view.result {
            *result_id = replacement;
        }
    }
    if let Some(inst) = rewriter.inst_of(view.result) {
        rewriter.erase_inst(inst);
    }
}
