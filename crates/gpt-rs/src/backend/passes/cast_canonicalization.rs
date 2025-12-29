use crate::backend::{
    driver::{apply_patterns_and_fold_greedily, GreedyConfig},
    optimizer::OptimizeContext,
    pattern::{CastOpView, OpRewritePattern, PatternSet},
    rewriter::ProgramRewriter,
    spec::{Function, Operand, PortableBackend},
};

use super::{FunctionPass, FunctionPassResult};

/// Pattern that removes casts when the operand already has the desired type.
pub struct EliminateRedundantCast;

impl OpRewritePattern<CastOpView> for EliminateRedundantCast {
    fn match_and_rewrite(&self, view: CastOpView, rewriter: &mut ProgramRewriter) -> bool {
        let [Operand::Value(source)] = view.operands.as_slice() else {
            return false;
        };
        let Some(input_ty) = rewriter.type_of(*source) else {
            return false;
        };
        if input_ty != &view.result_type {
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
            .expect("cast canonicalization erase should succeed");
        true
    }
}

/// Applies canonical cast simplifications across a PTIR function body.
pub struct CastCanonicalizationPass {
    config: GreedyConfig,
}

impl CastCanonicalizationPass {
    const NAME: &'static str = "cast-canonicalize";

    pub fn new(config: GreedyConfig) -> Self {
        Self { config }
    }
}

impl Default for CastCanonicalizationPass {
    fn default() -> Self {
        Self::new(GreedyConfig::default())
    }
}

impl<B: PortableBackend + 'static> FunctionPass<B> for CastCanonicalizationPass {
    fn name(&self) -> &'static str {
        Self::NAME
    }

    fn run(&self, function: &mut Function, _cx: &mut OptimizeContext<B>) -> FunctionPassResult {
        let mut patterns = PatternSet::new();
        patterns.insert_view::<CastOpView, _>(EliminateRedundantCast);
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
