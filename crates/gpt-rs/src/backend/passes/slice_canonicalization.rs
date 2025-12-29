use crate::backend::{
    driver::{apply_patterns_and_fold_greedily, GreedyConfig},
    optimizer::OptimizeContext,
    pattern::{OpRewritePattern, PatternSet, SliceOpView},
    rewriter::ProgramRewriter,
    spec::{Function, Operand, Operation, PortableBackend, SliceSpec},
};

use super::{FunctionPass, FunctionPassResult};

fn compose_slice_specs(inner: &SliceSpec, outer: &SliceSpec) -> Option<SliceSpec> {
    if inner.starts.len() != outer.starts.len() || inner.sizes.len() != outer.sizes.len() {
        return None;
    }

    let mut starts = Vec::with_capacity(inner.starts.len());
    let mut sizes = Vec::with_capacity(inner.sizes.len());

    for ((i_start, i_size), (o_start, o_size)) in inner
        .starts
        .iter()
        .zip(inner.sizes.iter())
        .zip(outer.starts.iter().zip(outer.sizes.iter()))
    {
        if *o_start > *i_size {
            return None;
        }
        let new_start = i_start.saturating_add(*o_start);
        if new_start > *i_start + i_size {
            return None;
        }
        let max_len = i_size.saturating_sub(*o_start);
        if *o_size > max_len {
            return None;
        }
        starts.push(new_start);
        sizes.push(*o_size);
    }

    Some(SliceSpec { starts, sizes })
}

/// Folds nested slices into a single slice when ranges allow.
pub struct SliceCanonicalizationPass {
    config: GreedyConfig,
}

impl SliceCanonicalizationPass {
    const NAME: &'static str = "slice-canonicalize";

    pub fn new(config: GreedyConfig) -> Self {
        Self { config }
    }
}

impl Default for SliceCanonicalizationPass {
    fn default() -> Self {
        Self::new(GreedyConfig::default())
    }
}

impl<B: PortableBackend + 'static> FunctionPass<B> for SliceCanonicalizationPass {
    fn name(&self) -> &'static str {
        Self::NAME
    }

    fn run(&self, function: &mut Function, _cx: &mut OptimizeContext<B>) -> FunctionPassResult {
        let mut patterns = PatternSet::new();
        patterns.insert_view::<SliceOpView, _>(CollapseSliceChain);
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

struct CollapseSliceChain;

impl OpRewritePattern<SliceOpView> for CollapseSliceChain {
    fn match_and_rewrite(&self, view: SliceOpView, rewriter: &mut ProgramRewriter) -> bool {
        let [Operand::Value(inner_value)] = view.operands.as_slice() else {
            return false;
        };
        let Some(inner_inst) = rewriter.inst_of(*inner_value) else {
            return false;
        };
        let Operation::Slice(inner_spec) = rewriter.op(inner_inst).clone() else {
            return false;
        };

        let Some(composed) = compose_slice_specs(&inner_spec, &view.spec) else {
            return false;
        };

        let [Operand::Value(base_value)] = rewriter.operands(inner_inst) else {
            return false;
        };

        let output_ty = view.result_type.clone();
        let Ok((_, new_value)) = rewriter.insert_before(
            view.root,
            Operation::Slice(composed),
            vec![Operand::Value(*base_value)],
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
        rewriter
            .erase_inst(view.root)
            .expect("slice canonicalization erase should succeed");
        true
    }
}
