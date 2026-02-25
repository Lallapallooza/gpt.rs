use std::collections::HashSet;

use gpt_rs::backend::conversion::{ConversionError, ConversionResult};
use gpt_rs::backend::fusion::{
    FUSION_ATTR_KIND, FUSION_ATTR_VERSION, FUSION_KIND_DOT_EPILOGUE_V1,
    FUSION_KIND_ELEMENTWISE_DAG_V1,
};
use gpt_rs::backend::spec::{
    CustomCallAttr, CustomCallSpec, Function, HintKind, HintPolicy, Instruction, Operand,
    Operation, Program, ValueId,
};

use crate::targets::{TARGET_DOT_BIAS_FUSED_F32_V1, TARGET_ELEMENTWISE_FUSED_F32_V1};

pub fn lower_hint_regions_to_custom_calls(program: &Program) -> ConversionResult<Program> {
    let mut lowered = program.clone();
    for function in &mut lowered.functions {
        lower_function_hints(function)?;
        reorder_function_body_topologically(function)?;
    }
    Ok(lowered)
}

fn lower_function_hints(function: &mut Function) -> ConversionResult<()> {
    if function.hints.is_empty() {
        return Ok(());
    }

    let hints = function.hints.clone();
    for hint in hints {
        let Some(target) = custom_call_target_for_hint(&hint.kind, &hint.attrs) else {
            if hint.policy == HintPolicy::Required {
                return Err(ConversionError::new(format!(
                    "required hint id={} kind={:?} is not supported by triton lowering",
                    hint.id, hint.kind
                )));
            }
            continue;
        };

        if hint.exports.len() != 1 {
            if hint.policy == HintPolicy::Required {
                return Err(ConversionError::new(format!(
                    "required hint id={} must export exactly one value",
                    hint.id
                )));
            }
            continue;
        }
        let export = hint.exports[0];
        let Some((insert_pos, output)) = find_export_output(function, export) else {
            if hint.policy == HintPolicy::Required {
                return Err(ConversionError::new(format!(
                    "required hint id={} export value {} is missing in function body",
                    hint.id, export.0
                )));
            }
            continue;
        };

        let covered = hint.body.iter().map(|inst| inst.id).collect::<HashSet<_>>();
        if covered.is_empty() {
            if hint.policy == HintPolicy::Required {
                return Err(ConversionError::new(format!(
                    "required hint id={} has empty body",
                    hint.id
                )));
            }
            continue;
        }
        if !covered.contains(&export) {
            if hint.policy == HintPolicy::Required {
                return Err(ConversionError::new(format!(
                    "required hint id={} body does not contain export value {}",
                    hint.id, export.0
                )));
            }
            continue;
        }
        if !inputs_available_before(function, hint.inputs.as_slice(), insert_pos, &covered) {
            if hint.policy == HintPolicy::Required {
                return Err(ConversionError::new(format!(
                    "required hint id={} references inputs that are unavailable at insertion point",
                    hint.id
                )));
            }
            continue;
        }

        let mut removed_any = false;
        function.body.retain(|inst| {
            let keep = !covered.contains(&inst.id);
            if !keep {
                removed_any = true;
            }
            keep
        });
        if !removed_any {
            if hint.policy == HintPolicy::Required {
                return Err(ConversionError::new(format!(
                    "required hint id={} did not match any body instructions",
                    hint.id
                )));
            }
            continue;
        }

        let replacement = Instruction {
            id: export,
            op: Operation::CustomCall(CustomCallSpec {
                target: target.to_string(),
                attrs: hint.attrs.clone(),
            }),
            operands: hint
                .inputs
                .iter()
                .map(|value| Operand::Value(*value))
                .collect(),
            output,
        };
        if insert_pos > function.body.len() {
            function.body.push(replacement);
        } else {
            function.body.insert(insert_pos, replacement);
        }
    }

    Ok(())
}

fn custom_call_target_for_hint(
    kind: &HintKind,
    attrs: &std::collections::BTreeMap<String, CustomCallAttr>,
) -> Option<&'static str> {
    let version = custom_call_i64(attrs.get(FUSION_ATTR_VERSION))?;
    if version != 1 {
        return None;
    }
    let fusion_kind = custom_call_string(attrs.get(FUSION_ATTR_KIND))?;
    match kind {
        HintKind::ElementwiseDag if fusion_kind == FUSION_KIND_ELEMENTWISE_DAG_V1 => {
            Some(TARGET_ELEMENTWISE_FUSED_F32_V1)
        }
        HintKind::DotEpilogue if fusion_kind == FUSION_KIND_DOT_EPILOGUE_V1 => {
            Some(TARGET_DOT_BIAS_FUSED_F32_V1)
        }
        HintKind::ReductionChain => None,
        _ => None,
    }
}

fn find_export_output(
    function: &Function,
    export: ValueId,
) -> Option<(usize, gpt_rs::backend::spec::ValueType)> {
    for (idx, instruction) in function.body.iter().enumerate() {
        if instruction.id == export {
            return Some((idx, instruction.output.clone()));
        }
    }
    None
}

fn custom_call_i64(attr: Option<&CustomCallAttr>) -> Option<i64> {
    match attr {
        Some(CustomCallAttr::I64(value)) => Some(*value),
        _ => None,
    }
}

fn custom_call_string(attr: Option<&CustomCallAttr>) -> Option<&str> {
    match attr {
        Some(CustomCallAttr::String(value)) => Some(value.as_str()),
        _ => None,
    }
}

fn inputs_available_before(
    function: &Function,
    inputs: &[ValueId],
    insert_pos: usize,
    covered: &HashSet<ValueId>,
) -> bool {
    for input in inputs {
        if covered.contains(input) {
            return false;
        }
        if function.parameter_ids.contains(input) {
            continue;
        }
        let Some(def_pos) = function.body.iter().position(|inst| inst.id == *input) else {
            return false;
        };
        if def_pos >= insert_pos {
            return false;
        }
    }
    true
}

fn reorder_function_body_topologically(function: &mut Function) -> ConversionResult<()> {
    if function.body.len() < 2 {
        return Ok(());
    }

    let mut def_index = std::collections::HashMap::with_capacity(function.body.len());
    for (idx, instruction) in function.body.iter().enumerate() {
        def_index.insert(instruction.id, idx);
    }

    let mut indegree = vec![0usize; function.body.len()];
    let mut users = vec![Vec::<usize>::new(); function.body.len()];
    for (idx, instruction) in function.body.iter().enumerate() {
        let mut deps = 0usize;
        for operand in &instruction.operands {
            let maybe_dep = match operand {
                Operand::Value(value) => def_index.get(value).copied(),
                Operand::TupleElement { tuple, .. } => def_index.get(tuple).copied(),
                Operand::Literal(_) => None,
            };
            if let Some(dep_idx) = maybe_dep {
                if dep_idx != idx {
                    deps = deps.saturating_add(1);
                    users[dep_idx].push(idx);
                }
            }
        }
        indegree[idx] = deps;
    }

    let mut ready = std::collections::VecDeque::new();
    for (idx, deps) in indegree.iter().enumerate() {
        if *deps == 0 {
            ready.push_back(idx);
        }
    }

    let mut scheduled = Vec::with_capacity(function.body.len());
    while let Some(idx) = ready.pop_front() {
        scheduled.push(function.body[idx].clone());
        for user_idx in &users[idx] {
            let slot = indegree
                .get_mut(*user_idx)
                .expect("topological indegree index must be valid");
            if *slot > 0 {
                *slot -= 1;
                if *slot == 0 {
                    ready.push_back(*user_idx);
                }
            }
        }
    }

    if scheduled.len() != function.body.len() {
        return Err(ConversionError::new(format!(
            "failed to topologically reorder function '{}' (cycle or unresolved dependency)",
            function.name
        )));
    }

    function.body = scheduled;
    Ok(())
}
