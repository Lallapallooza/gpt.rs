use std::sync::Arc;

use crate::backend::fusion::{
    discover_candidates, materialize_hints, select_non_overlapping, HintCostModel, HintLegalizer,
    SelectedFusion,
};
use crate::backend::optimizer::{FunctionPass, OptimizeContext, PassResult};
use crate::backend::spec::{Function, HintKind, PortableBackend};

mod memo;

pub struct FusionHintPass<B: PortableBackend + 'static> {
    legalizer: Arc<dyn HintLegalizer<B>>,
    cost_model: Arc<dyn HintCostModel<B>>,
    min_score: i64,
}

impl<B: PortableBackend + 'static> FusionHintPass<B> {
    pub fn new(
        legalizer: Arc<dyn HintLegalizer<B>>,
        cost_model: Arc<dyn HintCostModel<B>>,
    ) -> Self {
        Self {
            legalizer,
            cost_model,
            min_score: 0,
        }
    }

    pub fn with_min_score(mut self, min_score: i64) -> Self {
        self.min_score = min_score;
        self
    }

    fn materialize_and_report(function: &mut Function, selected: &[SelectedFusion]) -> PassResult {
        for selected_hint in selected {
            emit_selected(selected_hint.candidate.kind);
        }
        let changed = materialize_hints(function, selected);
        PassResult {
            changed,
            iterations: 1,
            rewrites_applied: selected.len(),
            erased_insts: 0,
        }
    }

    fn select_fusions(
        &self,
        function: &Function,
        candidates: Vec<crate::backend::fusion::FusionCandidate>,
        cx: &OptimizeContext<B>,
    ) -> Vec<SelectedFusion> {
        let mut eligible = Vec::<SelectedFusion>::new();
        for candidate in candidates {
            emit_discovered(candidate.kind);
            let policy = match self.legalizer.can_fuse(function, &candidate, cx) {
                Ok(policy) => policy,
                Err(_reason) => {
                    crate::profiling::cache_event("fusion_hint_rejected_legalize");
                    continue;
                }
            };
            let score = self.cost_model.score(function, &candidate, cx);
            if score <= self.min_score {
                crate::profiling::cache_event("fusion_hint_rejected_cost");
                continue;
            }
            eligible.push(SelectedFusion {
                candidate,
                policy,
                score,
            });
        }

        let selection = select_non_overlapping(&eligible);
        for _ in 0..selection.rejected_overlap {
            crate::profiling::cache_event("fusion_hint_rejected_overlap");
        }
        selection
            .selected
            .into_iter()
            .map(|idx| eligible[idx].clone())
            .collect()
    }
}

impl<B: PortableBackend + 'static> FunctionPass<B> for FusionHintPass<B> {
    fn name(&self) -> &'static str {
        "fusion_hint_regions"
    }

    fn run(&self, function: &mut Function, cx: &mut OptimizeContext<B>) -> PassResult {
        let memo_key = memo::build_key(function, cx.backend().backend_name(), self.min_score);

        match memo::lookup(function, &memo_key) {
            memo::MemoLookup::Selected(selected) => {
                return Self::materialize_and_report(function, selected.as_slice());
            }
            memo::MemoLookup::Candidates(candidates) => {
                let selected = self.select_fusions(function, candidates, cx);
                memo::store_selected(&memo_key, function, selected.as_slice());
                return Self::materialize_and_report(function, selected.as_slice());
            }
            memo::MemoLookup::Miss => {}
        }

        let candidates = match discover_candidates(function) {
            Ok(candidates) => candidates,
            Err(_) => return PassResult::default(),
        };
        memo::store_candidates(memo_key.clone(), function, candidates.as_slice());

        let selected = self.select_fusions(function, candidates, cx);
        memo::store_selected(&memo_key, function, selected.as_slice());
        Self::materialize_and_report(function, selected.as_slice())
    }
}

fn emit_discovered(kind: HintKind) {
    match kind {
        HintKind::ElementwiseDag => {
            crate::profiling::cache_event("fusion_hint_discovered_elementwise_dag")
        }
        HintKind::DotEpilogue => {
            crate::profiling::cache_event("fusion_hint_discovered_dot_epilogue")
        }
        HintKind::ReductionChain => {
            crate::profiling::cache_event("fusion_hint_discovered_reduction_chain")
        }
    }
}

fn emit_selected(kind: HintKind) {
    match kind {
        HintKind::ElementwiseDag => {
            crate::profiling::cache_event("fusion_hint_selected_elementwise_dag")
        }
        HintKind::DotEpilogue => crate::profiling::cache_event("fusion_hint_selected_dot_epilogue"),
        HintKind::ReductionChain => {
            crate::profiling::cache_event("fusion_hint_selected_reduction_chain")
        }
    }
}
