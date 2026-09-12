use std::collections::BTreeMap;

use gpt_rs::backend::{
    driver::{apply_patterns_and_fold_greedily, GreedyConfig},
    optimizer::{FunctionPass, OptimizeContext, PassResult},
    pattern::{DotGeneralOpView, OpRewritePattern, PatternSet},
    rewriter::ProgramRewriter,
    spec::{
        CastSpec, CustomCallSpec, DType, DotGeneralSpec, Function, Operand, Operation, TensorSpec,
        ValueType,
    },
};

use crate::targets::TARGET_LINEAR_NT_F32_BF16;

use super::utils::tensor_spec_of;

/// Whether `spec` is the PyTorch linear layout `x [M, K] . w [N, K] -> [M, N]`.
fn is_linear_nt(spec: &DotGeneralSpec, x: &TensorSpec, w: &TensorSpec) -> bool {
    spec.batch_lhs.is_empty()
        && spec.batch_rhs.is_empty()
        && spec.contract_lhs.as_slice() == [1]
        && spec.contract_rhs.as_slice() == [1]
        && x.shape.rank() == 2
        && w.shape.rank() == 2
}

/// Rewrites `dot_general(x: f32, cast(w: bf16 -> f32))` in linear layout into
/// [`TARGET_LINEAR_NT_F32_BF16`], so no f32 copy of the weight is materialised. Other users of the
/// cast keep it.
struct FoldBf16WeightCast;

impl OpRewritePattern<DotGeneralOpView> for FoldBf16WeightCast {
    fn match_and_rewrite(&self, view: DotGeneralOpView, rewriter: &mut ProgramRewriter) -> bool {
        let [Operand::Value(x), Operand::Value(weight)] = view.operands.as_slice() else {
            return false;
        };
        let Some(cast) = rewriter.inst_of(*weight) else {
            return false;
        };
        let (Operation::Cast(CastSpec { dtype: DType::F32 }), [Operand::Value(source)]) =
            (rewriter.op(cast), rewriter.operands(cast))
        else {
            return false;
        };
        let source = *source;
        let (Some(x_spec), Some(w_spec), ValueType::Tensor(out_spec)) = (
            tensor_spec_of(rewriter, *x),
            tensor_spec_of(rewriter, source),
            &view.result_type,
        ) else {
            return false;
        };
        if x_spec.dtype != DType::F32
            || w_spec.dtype != DType::Bf16
            || out_spec.dtype != DType::F32
            || !is_linear_nt(&view.spec, &x_spec, &w_spec)
        {
            return false;
        }

        let op = Operation::CustomCall(CustomCallSpec {
            target: TARGET_LINEAR_NT_F32_BF16.to_string(),
            attrs: BTreeMap::new(),
        });
        let Ok((_, linear)) = rewriter.insert_before(
            view.root,
            op,
            vec![Operand::Value(*x), Operand::Value(source)],
            view.result_type.clone(),
        ) else {
            return false;
        };
        rewriter.replace_all_uses(view.result, linear);
        for result in &mut rewriter.func.result_ids {
            if *result == view.result {
                *result = linear;
            }
        }
        rewriter
            .erase_inst(view.root)
            .expect("replaced dot_general has no users");
        true
    }
}

#[derive(Default)]
pub struct CLinearBf16WeightPass {
    config: GreedyConfig,
}

impl CLinearBf16WeightPass {
    const NAME: &'static str = "c-linear-bf16-weight";
}

impl FunctionPass<crate::CBackend> for CLinearBf16WeightPass {
    fn name(&self) -> &'static str {
        Self::NAME
    }

    fn run(
        &self,
        function: &mut Function,
        _cx: &mut OptimizeContext<crate::CBackend>,
    ) -> PassResult {
        let mut patterns = PatternSet::new();
        patterns.insert_view::<DotGeneralOpView, _>(FoldBf16WeightCast);
        let stats = apply_patterns_and_fold_greedily(function, &patterns.freeze(), &self.config);
        PassResult {
            changed: stats.applied > 0 || stats.dce_removed > 0,
            iterations: stats.iterations,
            rewrites_applied: stats.applied,
            erased_insts: stats.dce_removed,
        }
    }
}
