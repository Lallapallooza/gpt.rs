use std::collections::HashMap;
use std::sync::Mutex;

use once_cell::sync::Lazy;
use serde::Serialize;

use crate::backend::fusion::{FusionCandidate, FusionDetail, SelectedFusion};
use crate::backend::hashing::fnv1a_hash;
use crate::backend::op_signature::operation_kind;
use crate::backend::spec::{Function, HintKind, HintPolicy, Operand, TensorSpec, ValueId};

const FUSION_MEMO_VERSION: u32 = 2;
const FUSION_MEMO_CAPACITY: usize = 512;

static FUSION_MEMO: Lazy<Mutex<HashMap<FusionMemoKey, FusionMemoEntry>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

#[derive(Clone, Hash, PartialEq, Eq)]
pub(super) struct FusionMemoKey {
    backend: String,
    graph_hash: u64,
    min_score: i64,
    version: u32,
}

#[derive(Clone, Default)]
struct FusionMemoEntry {
    candidate_templates: Vec<FusionCandidateTemplate>,
    selected_templates: Vec<SelectedFusionTemplate>,
    selected_cached: bool,
}

#[derive(Clone)]
struct FusionCandidateTemplate {
    id: u32,
    kind: HintKind,
    anchor_pos: usize,
    inst_positions: Vec<usize>,
    input_positions: Vec<usize>,
    export_positions: Vec<usize>,
    detail: FusionDetail,
}

#[derive(Clone)]
struct SelectedFusionTemplate {
    candidate: FusionCandidateTemplate,
    policy: HintPolicy,
    score: i64,
}

pub(super) enum MemoLookup {
    Selected(Vec<SelectedFusion>),
    Candidates(Vec<FusionCandidate>),
    Miss,
}

pub(super) fn build_key(function: &Function, backend: &str, min_score: i64) -> FusionMemoKey {
    FusionMemoKey {
        backend: backend.to_string(),
        graph_hash: function_graph_hash(function),
        min_score,
        version: FUSION_MEMO_VERSION,
    }
}

pub(super) fn lookup(function: &Function, key: &FusionMemoKey) -> MemoLookup {
    let entry = {
        let Ok(cache) = FUSION_MEMO.lock() else {
            crate::profiling::cache_event("fusion_hint_memo_poisoned");
            return MemoLookup::Miss;
        };
        cache.get(key).cloned()
    };

    let Some(entry) = entry else {
        crate::profiling::cache_event("fusion_hint_candidate_memo_miss");
        crate::profiling::cache_event("fusion_hint_selected_memo_miss");
        return MemoLookup::Miss;
    };

    if entry.selected_cached {
        match rehydrate_selected(function, entry.selected_templates.as_slice()) {
            Some(selected) => {
                crate::profiling::cache_event("fusion_hint_selected_memo_hit");
                return MemoLookup::Selected(selected);
            }
            None => {
                crate::profiling::cache_event("fusion_hint_selected_memo_rehydrate_miss");
            }
        }
    } else {
        crate::profiling::cache_event("fusion_hint_selected_memo_miss");
    }

    if !entry.candidate_templates.is_empty() {
        if let Some(candidates) =
            rehydrate_candidates(function, entry.candidate_templates.as_slice())
        {
            crate::profiling::cache_event("fusion_hint_candidate_memo_hit");
            return MemoLookup::Candidates(candidates);
        }
        crate::profiling::cache_event("fusion_hint_candidate_memo_rehydrate_miss");
    }

    MemoLookup::Miss
}

pub(super) fn store_candidates(
    key: FusionMemoKey,
    function: &Function,
    candidates: &[FusionCandidate],
) {
    let Some(candidate_templates) = template_candidates(function, candidates) else {
        crate::profiling::cache_event("fusion_hint_candidate_memo_template_miss");
        return;
    };

    with_cache_mut(|cache| {
        let entry = cache.entry(key).or_insert_with(FusionMemoEntry::default);
        entry.candidate_templates = candidate_templates;
        entry.selected_templates.clear();
        entry.selected_cached = false;
    });
}

pub(super) fn store_selected(
    key: &FusionMemoKey,
    function: &Function,
    selected: &[SelectedFusion],
) {
    let Some(selected_templates) = template_selected(function, selected) else {
        crate::profiling::cache_event("fusion_hint_selected_memo_template_miss");
        return;
    };

    with_cache_mut(|cache| {
        let entry = cache
            .entry(key.clone())
            .or_insert_with(FusionMemoEntry::default);
        entry.selected_templates = selected_templates;
        entry.selected_cached = true;
    });
}

fn with_cache_mut<F>(f: F)
where
    F: FnOnce(&mut HashMap<FusionMemoKey, FusionMemoEntry>),
{
    let Ok(mut cache) = FUSION_MEMO.lock() else {
        crate::profiling::cache_event("fusion_hint_memo_poisoned");
        return;
    };
    if cache.len() >= FUSION_MEMO_CAPACITY {
        cache.clear();
        crate::profiling::cache_event("fusion_hint_memo_clear");
    }
    f(&mut cache);
}

fn function_value_order(function: &Function) -> Vec<ValueId> {
    let mut values = Vec::with_capacity(function.parameter_ids.len() + function.body.len());
    values.extend(function.parameter_ids.iter().copied());
    values.extend(function.body.iter().map(|inst| inst.id));
    values
}

fn function_value_positions(function: &Function) -> HashMap<ValueId, usize> {
    function_value_order(function)
        .into_iter()
        .enumerate()
        .map(|(idx, value)| (value, idx))
        .collect()
}

fn template_candidate(
    positions: &HashMap<ValueId, usize>,
    candidate: &FusionCandidate,
) -> Option<FusionCandidateTemplate> {
    let anchor_pos = *positions.get(&candidate.anchor)?;
    let inst_positions = candidate
        .inst_values
        .iter()
        .map(|value| positions.get(value).copied())
        .collect::<Option<Vec<_>>>()?;
    let input_positions = candidate
        .inputs
        .iter()
        .map(|value| positions.get(value).copied())
        .collect::<Option<Vec<_>>>()?;
    let export_positions = candidate
        .exports
        .iter()
        .map(|value| positions.get(value).copied())
        .collect::<Option<Vec<_>>>()?;
    Some(FusionCandidateTemplate {
        id: candidate.id,
        kind: candidate.kind,
        anchor_pos,
        inst_positions,
        input_positions,
        export_positions,
        detail: candidate.detail.clone(),
    })
}

fn template_candidates(
    function: &Function,
    candidates: &[FusionCandidate],
) -> Option<Vec<FusionCandidateTemplate>> {
    let positions = function_value_positions(function);
    candidates
        .iter()
        .map(|candidate| template_candidate(&positions, candidate))
        .collect()
}

fn template_selected(
    function: &Function,
    selected: &[SelectedFusion],
) -> Option<Vec<SelectedFusionTemplate>> {
    let positions = function_value_positions(function);
    selected
        .iter()
        .map(|selected| {
            Some(SelectedFusionTemplate {
                candidate: template_candidate(&positions, &selected.candidate)?,
                policy: selected.policy,
                score: selected.score,
            })
        })
        .collect()
}

fn rehydrate_candidate(
    values: &[ValueId],
    template: &FusionCandidateTemplate,
) -> Option<FusionCandidate> {
    let value_at = |position: usize| -> Option<ValueId> { values.get(position).copied() };
    let anchor = value_at(template.anchor_pos)?;
    let inst_values = template
        .inst_positions
        .iter()
        .map(|position| value_at(*position))
        .collect::<Option<Vec<_>>>()?;
    let inputs = template
        .input_positions
        .iter()
        .map(|position| value_at(*position))
        .collect::<Option<Vec<_>>>()?;
    let exports = template
        .export_positions
        .iter()
        .map(|position| value_at(*position))
        .collect::<Option<Vec<_>>>()?;
    Some(FusionCandidate {
        id: template.id,
        kind: template.kind,
        anchor,
        inst_values,
        inputs,
        exports,
        detail: template.detail.clone(),
    })
}

fn rehydrate_candidates(
    function: &Function,
    templates: &[FusionCandidateTemplate],
) -> Option<Vec<FusionCandidate>> {
    let values = function_value_order(function);
    templates
        .iter()
        .map(|template| rehydrate_candidate(values.as_slice(), template))
        .collect()
}

fn rehydrate_selected(
    function: &Function,
    templates: &[SelectedFusionTemplate],
) -> Option<Vec<SelectedFusion>> {
    let values = function_value_order(function);
    templates
        .iter()
        .map(|template| {
            Some(SelectedFusion {
                candidate: rehydrate_candidate(values.as_slice(), &template.candidate)?,
                policy: template.policy,
                score: template.score,
            })
        })
        .collect()
}

#[derive(Serialize)]
struct FunctionGraphSignature {
    parameters: usize,
    body: Vec<InstructionGraphSignature>,
}

#[derive(Serialize)]
struct InstructionGraphSignature {
    op_kind: &'static str,
    operands: Vec<OperandGraphSignature>,
}

#[derive(Serialize)]
enum OperandGraphSignature {
    Value(u32),
    TupleElement { tuple: u32, index: usize },
    Literal(TensorSpec),
}

fn function_graph_hash(function: &Function) -> u64 {
    let mut mapping = HashMap::<ValueId, u32>::new();
    let mut next = 0u32;
    let mut canonical_value = |value: ValueId| {
        *mapping.entry(value).or_insert_with(|| {
            let out = next;
            next = next.saturating_add(1);
            out
        })
    };

    for parameter in &function.parameter_ids {
        canonical_value(*parameter);
    }

    let mut body = Vec::with_capacity(function.body.len());
    for inst in &function.body {
        let _ = canonical_value(inst.id);
        let operands = inst
            .operands
            .iter()
            .map(|operand| match operand {
                Operand::Value(value) => OperandGraphSignature::Value(canonical_value(*value)),
                Operand::TupleElement { tuple, index } => OperandGraphSignature::TupleElement {
                    tuple: canonical_value(*tuple),
                    index: *index,
                },
                Operand::Literal(literal) => OperandGraphSignature::Literal(literal.spec.clone()),
            })
            .collect();
        body.push(InstructionGraphSignature {
            op_kind: operation_kind(&inst.op),
            operands,
        });
    }

    let signature = FunctionGraphSignature {
        parameters: function.parameter_ids.len(),
        body,
    };
    match bincode::serialize(&signature) {
        Ok(bytes) => fnv1a_hash(&bytes),
        Err(_) => 0,
    }
}
