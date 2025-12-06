use std::collections::{HashMap, HashSet};

use anyhow::{anyhow, bail, Result};

use crate::backend::optimizer::{FunctionPass, OptimizeContext, PassResult};
use crate::backend::spec::{
    Function, Operand, PortableBackend, Program, ProgramBuilder, TensorLiteral, ValueId, ValueType,
};
use crate::tensor::InputRole;

#[derive(Default)]
pub struct ParamOnlyFoldToParamPass;

impl ParamOnlyFoldToParamPass {
    const NAME: &'static str = "param-only-fold-to-param";
}

impl<B: PortableBackend + 'static> FunctionPass<B> for ParamOnlyFoldToParamPass {
    fn name(&self) -> &'static str {
        Self::NAME
    }

    fn run(&self, function: &mut Function, cx: &mut OptimizeContext<B>) -> PassResult {
        let Some(params) = cx.services().params else {
            return PassResult::default();
        };

        let mut total_changed = false;

        loop {
            let def_map = build_def_map(function);
            let type_map = build_type_map(function);
            let mut memo_static: HashMap<ValueId, bool> = HashMap::new();

            let mut dynamic_users: HashSet<ValueId> = HashSet::new();
            for inst in &function.body {
                let value_static =
                    is_static_value(function, cx, &def_map, &mut memo_static, inst.id);
                if value_static {
                    continue;
                }
                for operand in &inst.operands {
                    match operand {
                        Operand::Value(id) => {
                            dynamic_users.insert(*id);
                        }
                        Operand::TupleElement { tuple, .. } => {
                            dynamic_users.insert(*tuple);
                        }
                        Operand::Literal(_) => {}
                    }
                }
            }

            let mut targets: Vec<ValueId> = Vec::new();
            for inst in &function.body {
                if !is_static_value(function, cx, &def_map, &mut memo_static, inst.id) {
                    continue;
                }
                if dynamic_users.contains(&inst.id) || function.result_ids.contains(&inst.id) {
                    targets.push(inst.id);
                }
            }

            if targets.is_empty() {
                break;
            }

            let mut changed_this_round = false;

            for target in targets {
                if !uses_value(function, target) && !function.result_ids.contains(&target) {
                    continue;
                }

                let Ok(desc) = describe_static_subgraph(function, cx, &def_map, target) else {
                    continue;
                };

                if desc.leaf_param_stable_ids.len() != 1 {
                    continue;
                }

                let Some(output_ty) = type_map.get(&target).cloned() else {
                    continue;
                };

                let leaf_stable_id = desc.leaf_param_stable_ids[0];
                let dag_hash = dag_hash_value(function, cx, &def_map, &type_map, target);
                let derived_id = derived_param_id(cx.backend().backend_name(), dag_hash, &desc);
                let failure_key = failure_cache_key(cx.backend().backend_name(), dag_hash);
                if cx.is_failed_fold_key(failure_key) {
                    crate::profiling::cache_event("param_fold_failure_cache_hit");
                    continue;
                }

                if params.get(derived_id).is_some() {
                    crate::profiling::cache_event("param_resolver_derived_hit");
                } else {
                    crate::profiling::cache_event("param_resolver_derived_miss");

                    let leaf_handle = match params.get(leaf_stable_id) {
                        Some(handle) => {
                            crate::profiling::cache_event("param_resolver_leaf_hit");
                            handle
                        }
                        None => {
                            crate::profiling::cache_event("param_resolver_leaf_miss");
                            cx.record_failed_fold_key(failure_key);
                            continue;
                        }
                    };

                    let subgraph = match build_subgraph_program(
                        function,
                        target,
                        &type_map,
                        &desc.leaf_param_ids,
                        &desc.inst_set,
                    ) {
                        Ok(program) => program,
                        Err(_) => {
                            cx.record_failed_fold_key(failure_key);
                            continue;
                        }
                    };

                    let _suspend = crate::profiling::suspend();
                    let outputs = cx.backend().run_program(&subgraph, &[leaf_handle]);
                    let mut outputs = match outputs {
                        Ok(v) => v,
                        Err(_) => {
                            cx.record_failed_fold_key(failure_key);
                            continue;
                        }
                    };
                    if outputs.len() != 1 {
                        cx.record_failed_fold_key(failure_key);
                        continue;
                    }
                    let handle = outputs.pop().expect("length checked");
                    params.set(derived_id, handle);
                    crate::profiling::cache_event("param_resolver_derived_set");
                }

                let Ok(replacement) = cx.entry_mut().get_or_add_param(
                    function,
                    InputRole::Param,
                    Some(derived_id),
                    output_ty,
                ) else {
                    continue;
                };

                replace_uses(function, target, replacement);
                changed_this_round = true;
            }

            if !changed_this_round {
                break;
            }
            total_changed |= changed_this_round;
        }

        PassResult {
            changed: total_changed,
            iterations: 0,
            rewrites_applied: 0,
            erased_insts: 0,
        }
    }
}

struct StaticSubgraphDesc {
    leaf_param_ids: Vec<ValueId>,
    leaf_param_stable_ids: Vec<u64>,
    inst_set: HashSet<ValueId>,
    literal_hash: u64,
}

fn build_def_map(function: &Function) -> HashMap<ValueId, usize> {
    let mut def_map: HashMap<ValueId, usize> = HashMap::with_capacity(function.body.len());
    for (idx, inst) in function.body.iter().enumerate() {
        def_map.insert(inst.id, idx);
    }
    def_map
}

fn build_type_map(function: &Function) -> HashMap<ValueId, ValueType> {
    let mut types: HashMap<ValueId, ValueType> = HashMap::new();
    for (id, ty) in function
        .parameter_ids
        .iter()
        .zip(function.parameters.iter())
    {
        types.insert(*id, ty.clone());
    }
    for inst in &function.body {
        types.insert(inst.id, inst.output.clone());
    }
    types
}

fn is_static_value<B: PortableBackend + 'static>(
    function: &Function,
    cx: &OptimizeContext<'_, B>,
    def_map: &HashMap<ValueId, usize>,
    memo: &mut HashMap<ValueId, bool>,
    value: ValueId,
) -> bool {
    if let Some(flag) = memo.get(&value).copied() {
        return flag;
    }

    if function.parameter_ids.contains(&value) {
        let flag = cx.entry().role_of(value) == Some(InputRole::Param);
        memo.insert(value, flag);
        return flag;
    }

    let Some(&idx) = def_map.get(&value) else {
        memo.insert(value, false);
        return false;
    };

    let inst = &function.body[idx];
    let mut flag = true;
    for operand in &inst.operands {
        match operand {
            Operand::Literal(_) => {}
            Operand::Value(dep) => {
                if !is_static_value(function, cx, def_map, memo, *dep) {
                    flag = false;
                    break;
                }
            }
            Operand::TupleElement { .. } => {
                flag = false;
                break;
            }
        }
    }

    memo.insert(value, flag);
    flag
}

fn describe_static_subgraph<B: PortableBackend + 'static>(
    function: &Function,
    cx: &OptimizeContext<'_, B>,
    def_map: &HashMap<ValueId, usize>,
    target: ValueId,
) -> Result<StaticSubgraphDesc> {
    let mut inst_set: HashSet<ValueId> = HashSet::new();
    let mut leaf_set: HashSet<ValueId> = HashSet::new();
    let mut literal_hash = fnv1a_init();

    fn dfs<B: PortableBackend + 'static>(
        value: ValueId,
        function: &Function,
        cx: &OptimizeContext<'_, B>,
        def_map: &HashMap<ValueId, usize>,
        inst_set: &mut HashSet<ValueId>,
        leaf_set: &mut HashSet<ValueId>,
        literal_hash: &mut u64,
    ) -> Result<()> {
        if function.parameter_ids.contains(&value) {
            if cx.entry().role_of(value) == Some(InputRole::Param) {
                leaf_set.insert(value);
                return Ok(());
            }
            bail!("static fold reached non-param input {:?}", value);
        }

        let Some(&idx) = def_map.get(&value) else {
            bail!("static fold target depends on unknown value {:?}", value);
        };
        if !inst_set.insert(value) {
            return Ok(());
        }
        let inst = function
            .body
            .get(idx)
            .ok_or_else(|| anyhow!("missing defining instruction for {:?}", value))?;
        for operand in &inst.operands {
            match operand {
                Operand::Literal(lit) => {
                    *literal_hash = hash_tensor_literal(*literal_hash, lit);
                }
                Operand::Value(dep) => {
                    dfs::<B>(
                        *dep,
                        function,
                        cx,
                        def_map,
                        inst_set,
                        leaf_set,
                        literal_hash,
                    )?;
                }
                Operand::TupleElement { .. } => bail!("tuple operands are not supported"),
            }
        }
        Ok(())
    }

    dfs::<B>(
        target,
        function,
        cx,
        def_map,
        &mut inst_set,
        &mut leaf_set,
        &mut literal_hash,
    )?;

    let mut leaf_param_ids: Vec<ValueId> = leaf_set.into_iter().collect();
    leaf_param_ids.sort_by_key(|id| id.0);

    let mut leaf_param_stable_ids: Vec<u64> = Vec::with_capacity(leaf_param_ids.len());
    for id in &leaf_param_ids {
        let stable_id = cx
            .entry()
            .stable_id_of(*id)
            .ok_or_else(|| anyhow!("param input {:?} missing stable id", id))?;
        leaf_param_stable_ids.push(stable_id);
    }
    leaf_param_stable_ids.sort_unstable();
    leaf_param_stable_ids.dedup();

    Ok(StaticSubgraphDesc {
        leaf_param_ids,
        leaf_param_stable_ids,
        inst_set,
        literal_hash,
    })
}

fn build_subgraph_program(
    function: &Function,
    target: ValueId,
    type_map: &HashMap<ValueId, ValueType>,
    leaf_params: &[ValueId],
    inst_set: &HashSet<ValueId>,
) -> Result<Program> {
    let mut builder = ProgramBuilder::new();
    let mut mapping: HashMap<ValueId, ValueId> = HashMap::new();

    for leaf in leaf_params {
        let ty = type_map
            .get(leaf)
            .cloned()
            .ok_or_else(|| anyhow!("missing type for leaf param {:?}", leaf))?;
        let new_id = builder.add_parameter(ty);
        mapping.insert(*leaf, new_id);
    }

    for inst in &function.body {
        if !inst_set.contains(&inst.id) {
            continue;
        }

        let operands = inst
            .operands
            .iter()
            .map(|operand| match operand {
                Operand::Value(id) => mapping
                    .get(id)
                    .copied()
                    .map(Operand::Value)
                    .ok_or_else(|| anyhow!("missing mapping for operand {:?}", id)),
                Operand::Literal(lit) => Ok(Operand::Literal(lit.clone())),
                Operand::TupleElement { .. } => Err(anyhow!("tuple operands are not supported")),
            })
            .collect::<Result<Vec<_>>>()?;

        let output = inst.output.clone();
        let new_id = builder.emit_single(inst.op.clone(), operands, output);
        mapping.insert(inst.id, new_id);
    }

    let result_id = mapping
        .get(&target)
        .copied()
        .ok_or_else(|| anyhow!("missing mapping for fold target {:?}", target))?;
    let sub_fn = builder.finish("param_only_fold", vec![result_id]);
    Ok(Program::new("param_only_fold").with_functions(vec![sub_fn]))
}

fn replace_uses(function: &mut Function, from: ValueId, to: ValueId) {
    if from == to {
        return;
    }
    for inst in &mut function.body {
        for operand in &mut inst.operands {
            match operand {
                Operand::Value(id) if *id == from => {
                    *operand = Operand::Value(to);
                }
                Operand::TupleElement { tuple, .. } if *tuple == from => {
                    *tuple = to;
                }
                _ => {}
            }
        }
    }
    for id in &mut function.result_ids {
        if *id == from {
            *id = to;
        }
    }
}

fn uses_value(function: &Function, value: ValueId) -> bool {
    for inst in &function.body {
        for operand in &inst.operands {
            match operand {
                Operand::Value(id) if *id == value => return true,
                Operand::TupleElement { tuple, .. } if *tuple == value => return true,
                _ => {}
            }
        }
    }
    false
}

fn dag_hash_value<B: PortableBackend + 'static>(
    function: &Function,
    cx: &OptimizeContext<'_, B>,
    def_map: &HashMap<ValueId, usize>,
    type_map: &HashMap<ValueId, ValueType>,
    value: ValueId,
) -> u64 {
    let mut memo: HashMap<ValueId, u64> = HashMap::new();
    fn rec<B: PortableBackend + 'static>(
        function: &Function,
        cx: &OptimizeContext<'_, B>,
        def_map: &HashMap<ValueId, usize>,
        type_map: &HashMap<ValueId, ValueType>,
        value: ValueId,
        memo: &mut HashMap<ValueId, u64>,
    ) -> u64 {
        if let Some(&h) = memo.get(&value) {
            return h;
        }

        let mut hash = fnv1a_init();
        if function.parameter_ids.contains(&value) {
            hash = fnv1a_bytes(hash, b"param");
            let role = cx.entry().role_of(value);
            hash = fnv1a_u64(hash, role.map(|r| r as u64).unwrap_or(0));
            let stable_id = cx.entry().stable_id_of(value).unwrap_or(0);
            hash = fnv1a_u64(hash, stable_id);
            if let Some(ty) = type_map.get(&value) {
                if let Ok(bytes) = bincode::serialize(ty) {
                    hash = fnv1a_bytes(hash, &bytes);
                }
            }
            memo.insert(value, hash);
            return hash;
        }

        let Some(&idx) = def_map.get(&value) else {
            hash = fnv1a_bytes(hash, b"unknown");
            memo.insert(value, hash);
            return hash;
        };

        let inst = &function.body[idx];
        hash = fnv1a_bytes(hash, b"inst");
        if let Ok(bytes) = bincode::serialize(&inst.op) {
            hash = fnv1a_bytes(hash, &bytes);
        }
        if let Some(ty) = type_map.get(&inst.id) {
            if let Ok(bytes) = bincode::serialize(ty) {
                hash = fnv1a_bytes(hash, &bytes);
            }
        }

        for operand in &inst.operands {
            match operand {
                Operand::Value(dep) => {
                    hash = fnv1a_u64(hash, rec(function, cx, def_map, type_map, *dep, memo));
                }
                Operand::Literal(lit) => {
                    hash = hash_tensor_literal(hash, lit);
                }
                Operand::TupleElement { tuple, index } => {
                    hash = fnv1a_bytes(hash, b"tuple");
                    hash = fnv1a_u64(hash, rec(function, cx, def_map, type_map, *tuple, memo));
                    hash = fnv1a_u64(hash, *index as u64);
                }
            }
        }

        memo.insert(value, hash);
        hash
    }

    rec(function, cx, def_map, type_map, value, &mut memo)
}

fn derived_param_id(backend_name: &str, dag_hash: u64, desc: &StaticSubgraphDesc) -> u64 {
    let mut hash = fnv1a_init();
    hash = fnv1a_bytes(hash, b"ptir:derived_param");
    hash = fnv1a_bytes(hash, backend_name.as_bytes());
    hash = fnv1a_u64(hash, dag_hash);
    hash = fnv1a_u64(hash, desc.literal_hash);
    for leaf in &desc.leaf_param_stable_ids {
        hash = fnv1a_u64(hash, *leaf);
    }
    hash | (1u64 << 63)
}

fn failure_cache_key(backend_name: &str, dag_hash: u64) -> u64 {
    let mut hash = fnv1a_init();
    hash = fnv1a_bytes(hash, b"ptir:fold_fail");
    hash = fnv1a_bytes(hash, backend_name.as_bytes());
    fnv1a_u64(hash, dag_hash)
}

fn hash_tensor_literal(seed: u64, lit: &TensorLiteral) -> u64 {
    let mut hash = fnv1a_bytes(seed, b"lit");
    if let Ok(spec) = bincode::serialize(&lit.spec) {
        hash = fnv1a_bytes(hash, &spec);
    }
    hash = fnv1a_bytes(hash, lit.bytes.as_ref());
    hash
}

fn fnv1a_init() -> u64 {
    0xcbf29ce484222325
}

fn fnv1a_bytes(mut hash: u64, bytes: &[u8]) -> u64 {
    const PRIME: u64 = 0x100000001b3;
    for byte in bytes {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(PRIME);
    }
    hash
}

fn fnv1a_u64(hash: u64, value: u64) -> u64 {
    fnv1a_bytes(hash, &value.to_le_bytes())
}
