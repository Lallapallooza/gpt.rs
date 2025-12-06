use crate::backend::{
    driver::{apply_patterns_and_fold_greedily, GreedyConfig},
    optimizer::OptimizeContext,
    pattern::{BroadcastOpView, OpRewritePattern, PatternSet},
    rewriter::ProgramRewriter,
    spec::{BroadcastToSpec, Dimension, Function, Operand, Operation, PortableBackend, ValueType},
};

use super::{FunctionPass, FunctionPassResult};

/// Removes `broadcast_to` when it is a no-op (identity).
pub struct EliminateIdentityBroadcast;

impl OpRewritePattern<BroadcastOpView> for EliminateIdentityBroadcast {
    fn match_and_rewrite(&self, view: BroadcastOpView, rewriter: &mut ProgramRewriter) -> bool {
        let [Operand::Value(source)] = view.operands.as_slice() else {
            return false;
        };

        let source_ty = match rewriter.type_of(*source) {
            Some(ValueType::Tensor(tensor)) => tensor.clone(),
            _ => return false,
        };
        let ValueType::Tensor(result_ty) = view.result_type.clone() else {
            return false;
        };

        if source_ty.shape != result_ty.shape {
            return false;
        }

        rewriter.replace_all_uses(view.result, *source);
        for result_id in &mut rewriter.func.result_ids {
            if *result_id == view.result {
                *result_id = *source;
            }
        }
        rewriter.erase_inst(view.root);
        true
    }
}

/// Folds nested broadcasts into a single broadcast from the original operand.
pub struct CollapseBroadcastChain;

impl OpRewritePattern<BroadcastOpView> for CollapseBroadcastChain {
    fn match_and_rewrite(&self, view: BroadcastOpView, rewriter: &mut ProgramRewriter) -> bool {
        let [Operand::Value(inner_value)] = view.operands.as_slice() else {
            return false;
        };

        let Some(inner_inst) = rewriter.inst_of(*inner_value) else {
            return false;
        };
        let Operation::BroadcastTo(_inner_spec) = rewriter.op(inner_inst).clone() else {
            return false;
        };

        // Base operand that feeds into the inner broadcast.
        let [Operand::Value(base_value)] = rewriter.operands(inner_inst) else {
            return false;
        };

        let base_ty = match rewriter.type_of(*base_value) {
            Some(ValueType::Tensor(tensor)) => tensor.clone(),
            _ => return false,
        };
        let ValueType::Tensor(outer_result_ty) = view.result_type.clone() else {
            return false;
        };

        if !is_broadcast_compatible(&base_ty.shape, &outer_result_ty.shape) {
            return false;
        }

        let new_spec = BroadcastToSpec {
            result_shape: view.spec.result_shape.clone(),
        };

        let operands = vec![Operand::Value(*base_value)];
        let output_ty = ValueType::Tensor(outer_result_ty.clone());

        let Ok((_, combined_value)) = rewriter.insert_before(
            view.root,
            Operation::BroadcastTo(new_spec),
            operands,
            output_ty,
        ) else {
            return false;
        };

        rewriter.replace_all_uses(view.result, combined_value);
        for result_id in &mut rewriter.func.result_ids {
            if *result_id == view.result {
                *result_id = combined_value;
            }
        }
        rewriter.erase_inst(view.root);
        true
    }
}

fn is_broadcast_compatible(
    from: &crate::backend::spec::Shape,
    to: &crate::backend::spec::Shape,
) -> bool {
    let from_dims = from.dims();
    let to_dims = to.dims();
    if from_dims.len() > to_dims.len() {
        return false;
    }
    let rank_diff = to_dims.len() - from_dims.len();
    for (idx, from_dim) in from_dims.iter().enumerate() {
        let to_dim = &to_dims[idx + rank_diff];
        let (Dimension::Static(from_size), Dimension::Static(to_size)) = (from_dim, to_dim) else {
            return false;
        };
        if *from_size != 1 && from_size != to_size {
            return false;
        }
    }
    true
}

/// Applies canonical broadcast simplifications across a PTIR function body.
pub struct BroadcastCanonicalizationPass {
    config: GreedyConfig,
}

impl BroadcastCanonicalizationPass {
    const NAME: &'static str = "broadcast-canonicalize";

    pub fn new(config: GreedyConfig) -> Self {
        Self { config }
    }
}

impl Default for BroadcastCanonicalizationPass {
    fn default() -> Self {
        Self::new(GreedyConfig::default())
    }
}

impl<B: PortableBackend + 'static> FunctionPass<B> for BroadcastCanonicalizationPass {
    fn name(&self) -> &'static str {
        Self::NAME
    }

    fn run(&self, function: &mut Function, _cx: &mut OptimizeContext<B>) -> FunctionPassResult {
        let mut patterns = PatternSet::new();
        patterns.insert_view::<BroadcastOpView, _>(EliminateIdentityBroadcast);
        patterns.insert_view::<BroadcastOpView, _>(CollapseBroadcastChain);
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
