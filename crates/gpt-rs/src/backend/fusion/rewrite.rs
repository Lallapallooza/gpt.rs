use std::collections::HashMap;

use crate::backend::spec::{CustomCallAttr, Function, HintRegion, Instruction, ValueId};

use super::{SelectedFusion, FUSION_ATTR_GENERATED};

pub fn materialize_hints(function: &mut Function, selected: &[SelectedFusion]) -> bool {
    let inst_index = instruction_index(function);
    let mut generated = Vec::new();
    for fused in selected {
        let mut ordered = fused
            .candidate
            .inst_values
            .iter()
            .filter_map(|value| inst_index.get(value).cloned())
            .collect::<Vec<_>>();
        ordered.sort_by_key(|(pos, _)| *pos);
        ordered.dedup_by_key(|(pos, _)| *pos);
        if ordered.is_empty() {
            continue;
        }
        let body = ordered
            .into_iter()
            .map(|(_, inst)| inst)
            .collect::<Vec<Instruction>>();
        let mut attrs = fused.candidate.attrs();
        attrs.insert(
            FUSION_ATTR_GENERATED.to_string(),
            CustomCallAttr::Bool(true),
        );
        attrs.insert("fusion_score".to_string(), CustomCallAttr::I64(fused.score));
        generated.push(HintRegion {
            id: fused.candidate.id,
            kind: fused.candidate.kind,
            policy: fused.policy,
            inputs: fused.candidate.inputs.clone(),
            exports: fused.candidate.exports.clone(),
            body,
            attrs,
        });
    }

    generated.sort_by_key(|hint| hint.id);
    let mut next_hints = function
        .hints
        .iter()
        .filter(|hint| !is_generated_hint(hint))
        .cloned()
        .collect::<Vec<_>>();
    next_hints.extend(generated);
    if function.hints == next_hints {
        return false;
    }
    function.hints = next_hints;
    true
}

fn instruction_index(function: &Function) -> HashMap<ValueId, (usize, Instruction)> {
    let mut out = HashMap::with_capacity(function.body.len());
    for (idx, inst) in function.body.iter().enumerate() {
        out.insert(inst.id, (idx, inst.clone()));
    }
    out
}

fn is_generated_hint(hint: &HintRegion) -> bool {
    matches!(
        hint.attrs.get(FUSION_ATTR_GENERATED),
        Some(CustomCallAttr::Bool(true))
    )
}
