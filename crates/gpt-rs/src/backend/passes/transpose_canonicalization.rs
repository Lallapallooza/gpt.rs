use crate::backend::{
    driver::{apply_patterns_and_fold_greedily, GreedyConfig},
    optimizer::OptimizeContext,
    pattern::{OpRewritePattern, PatternSet, TransposeOpView},
    rewriter::ProgramRewriter,
    spec::{Function, Operand, Operation, PortableBackend, TransposeSpec},
};

use super::{FunctionPass, FunctionPassResult};

fn is_identity_perm(perm: &[usize]) -> bool {
    perm.iter().copied().eq(0..perm.len())
}

fn compose_perms(first: &[usize], second: &[usize]) -> Option<Vec<usize>> {
    if first.len() != second.len() {
        return None;
    }
    let mut result = Vec::with_capacity(first.len());
    for &axis in second {
        if axis >= first.len() {
            return None;
        }
        result.push(first[axis]);
    }
    Some(result)
}

/// Remove transposes that keep axes in place.
pub struct EliminateIdentityTranspose;

impl OpRewritePattern<TransposeOpView> for EliminateIdentityTranspose {
    fn match_and_rewrite(&self, view: TransposeOpView, rewriter: &mut ProgramRewriter) -> bool {
        if !is_identity_perm(&view.spec.perm) {
            return false;
        }
        let [Operand::Value(src)] = view.operands.as_slice() else {
            return false;
        };
        rewriter.replace_all_uses(view.result, *src);
        for result_id in &mut rewriter.func.result_ids {
            if *result_id == view.result {
                *result_id = *src;
            }
        }
        rewriter.erase_inst(view.root);
        true
    }
}

/// Fold transpose(transpose(x, p1), p2) -> transpose(x, p1∘p2).
pub struct CollapseTransposeChain;

impl OpRewritePattern<TransposeOpView> for CollapseTransposeChain {
    fn match_and_rewrite(&self, view: TransposeOpView, rewriter: &mut ProgramRewriter) -> bool {
        let [Operand::Value(inner_value)] = view.operands.as_slice() else {
            return false;
        };
        let Some(inner_inst) = rewriter.inst_of(*inner_value) else {
            return false;
        };
        let Operation::Transpose(inner_spec) = rewriter.op(inner_inst).clone() else {
            return false;
        };

        // Compose permutations: first inner, then outer.
        let Some(composed) = compose_perms(&inner_spec.perm, &view.spec.perm) else {
            return false;
        };

        // If composition becomes identity, drop both transposes.
        if is_identity_perm(&composed) {
            let base_value = match rewriter.operands(inner_inst) {
                [Operand::Value(base_value)] => *base_value,
                _ => return false,
            };
            rewriter.replace_all_uses(view.result, base_value);
            for result_id in &mut rewriter.func.result_ids {
                if *result_id == view.result {
                    *result_id = base_value;
                }
            }
            rewriter.erase_inst(view.root);
            return true;
        }

        let output_ty = view.result_type.clone();
        let new_spec = TransposeSpec { perm: composed };
        let base_value = match rewriter.operands(inner_inst) {
            [Operand::Value(base_value)] => *base_value,
            _ => return false,
        };

        let Ok((_, new_value)) = rewriter.insert_before(
            view.root,
            Operation::Transpose(new_spec),
            vec![Operand::Value(base_value)],
            output_ty,
        ) else {
            return false;
        };

        rewriter.replace_all_uses(view.result, new_value);
        for result_id in &mut rewriter.func.result_ids {
            if *result_id == view.result {
                *result_id = new_value;
            }
        }
        rewriter.erase_inst(view.root);
        true
    }
}

/// Canonical transpose simplifications (identity removal, chain folding).
pub struct TransposeCanonicalizationPass {
    config: GreedyConfig,
}

impl TransposeCanonicalizationPass {
    const NAME: &'static str = "transpose-canonicalize";

    pub fn new(config: GreedyConfig) -> Self {
        Self { config }
    }
}

impl Default for TransposeCanonicalizationPass {
    fn default() -> Self {
        Self::new(GreedyConfig::default())
    }
}

impl<B: PortableBackend + 'static> FunctionPass<B> for TransposeCanonicalizationPass {
    fn name(&self) -> &'static str {
        Self::NAME
    }

    fn run(&self, function: &mut Function, _cx: &mut OptimizeContext<B>) -> FunctionPassResult {
        let mut patterns = PatternSet::new();
        patterns.insert_view::<TransposeOpView, _>(EliminateIdentityTranspose);
        patterns.insert_view::<TransposeOpView, _>(CollapseTransposeChain);
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
