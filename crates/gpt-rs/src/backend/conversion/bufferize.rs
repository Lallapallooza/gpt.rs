use std::collections::HashMap;

use thiserror::Error;

use crate::backend::spec::{
    DType, Program, Region, RegionId, Shape, TensorSpec, ValueId, ValueType,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LiveRange {
    pub start: usize,
    pub end: usize,
}

impl LiveRange {
    pub fn new(start: usize, end: usize) -> Self {
        Self { start, end }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AliasKind {
    None,
    Identity,
    View,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BufferUsage(u8);

impl BufferUsage {
    const PARAMETER: u8 = 1;
    const RESULT: u8 = 2;
    const TEMPORARY: u8 = 4;

    pub fn empty() -> Self {
        BufferUsage(0)
    }

    pub fn parameter() -> Self {
        BufferUsage(Self::PARAMETER)
    }

    pub fn result() -> Self {
        BufferUsage(Self::RESULT)
    }

    pub fn temporary() -> Self {
        BufferUsage(Self::TEMPORARY)
    }

    pub fn contains_parameter(self) -> bool {
        (self.0 & Self::PARAMETER) != 0
    }

    pub fn contains_result(self) -> bool {
        (self.0 & Self::RESULT) != 0
    }

    pub fn contains_temporary(self) -> bool {
        (self.0 & Self::TEMPORARY) != 0
    }

    pub fn merge(mut self, other: BufferUsage) -> Self {
        self.0 |= other.0;
        self
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct BufferKey {
    pub value: ValueId,
    pub path: Vec<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BufferSpec {
    pub value: ValueId,
    pub path: Vec<usize>,
    pub dtype: DType,
    pub shape: Shape,
    pub byte_len: Option<usize>,
    pub usage: BufferUsage,
    pub alias_group: usize,
    pub alias_kind: AliasKind,
    pub alias_of: Option<BufferKey>,
    pub live_range: LiveRange,
    pub slot: Option<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BufferSlot {
    pub id: usize,
    pub dtype: DType,
    pub byte_len: Option<usize>,
    pub usage: BufferUsage,
}

#[derive(Debug, Clone, Default)]
pub struct FunctionBufferPlan {
    pub buffers: Vec<BufferSpec>,
    pub values: HashMap<BufferKey, usize>,
    pub slots: Vec<BufferSlot>,
}

impl FunctionBufferPlan {
    pub fn buffer_for(&self, value: ValueId) -> Option<&BufferSpec> {
        self.buffer_for_path(value, &[])
    }

    pub fn buffer_for_path(&self, value: ValueId, path: &[usize]) -> Option<&BufferSpec> {
        let key = BufferKey {
            value,
            path: path.to_vec(),
        };
        self.values
            .get(&key)
            .and_then(|index| self.buffers.get(*index))
    }

    pub fn buffer_for_tuple_element(&self, value: ValueId, index: usize) -> Option<&BufferSpec> {
        self.buffer_for_path(value, &[index])
    }

    pub fn buffers_for_value(&self, value: ValueId) -> Vec<&BufferSpec> {
        let mut matches: Vec<(&BufferSpec, Vec<usize>)> = self
            .values
            .iter()
            .filter_map(|(key, idx)| {
                if key.value == value {
                    self.buffers.get(*idx).map(|spec| (spec, key.path.clone()))
                } else {
                    None
                }
            })
            .collect();
        matches.sort_by(|a, b| a.1.cmp(&b.1));
        matches.into_iter().map(|(spec, _)| spec).collect()
    }

    pub fn slot_for_value(&self, value: ValueId) -> Option<&BufferSlot> {
        self.slot_for_path(value, &[])
    }

    pub fn slot_for_path(&self, value: ValueId, path: &[usize]) -> Option<&BufferSlot> {
        let key = BufferKey {
            value,
            path: path.to_vec(),
        };
        let buffer = self
            .values
            .get(&key)
            .and_then(|index| self.buffers.get(*index))?;
        let slot = buffer.slot?;
        self.slots.get(slot)
    }
}

#[derive(Debug, Clone, Default)]
pub struct BufferPlan {
    pub functions: HashMap<String, FunctionBufferPlan>,
    pub regions: HashMap<RegionId, FunctionBufferPlan>,
}

impl BufferPlan {
    pub fn function(&self, name: &str) -> Option<&FunctionBufferPlan> {
        self.functions.get(name)
    }

    pub fn region(&self, id: RegionId) -> Option<&FunctionBufferPlan> {
        self.regions.get(&id)
    }
}

#[derive(Debug, Clone, Default)]
pub struct BufferizeOptions {
    pub require_static_shapes: bool,
    pub require_known_dtypes: bool,
}

#[derive(Debug, Error)]
pub enum BufferizeError {
    #[error("dynamic shape not allowed for value {value:?} at path {path:?}")]
    DynamicShape { value: ValueId, path: Vec<usize> },
    #[error("dtype {dtype:?} has unknown storage size for value {value:?} at path {path:?}")]
    UnknownDType {
        value: ValueId,
        path: Vec<usize>,
        dtype: DType,
    },
    #[error("buffer byte length overflow for value {value:?} at path {path:?}")]
    ByteLenOverflow { value: ValueId, path: Vec<usize> },
    #[error("value {value:?} appears with incompatible types")]
    TypeMismatch { value: ValueId },
}

pub fn plan_buffers(program: &Program) -> Result<BufferPlan, BufferizeError> {
    plan_buffers_with(program, &BufferizeOptions::default())
}

pub fn plan_buffers_with(
    program: &Program,
    options: &BufferizeOptions,
) -> Result<BufferPlan, BufferizeError> {
    let mut plan = BufferPlan::default();

    for function in &program.functions {
        let func_plan = plan_function(function, options)?;
        plan.functions.insert(function.name.clone(), func_plan);
    }

    for region in &program.regions {
        let region_plan = plan_region(region, options)?;
        plan.regions.insert(region.id, region_plan);
    }

    Ok(plan)
}

fn plan_function(
    function: &crate::backend::spec::Function,
    options: &BufferizeOptions,
) -> Result<FunctionBufferPlan, BufferizeError> {
    let mut plan = FunctionBufferPlan::default();

    for (id, ty) in function
        .parameter_ids
        .iter()
        .zip(function.parameters.iter())
    {
        insert_value(&mut plan, *id, ty, BufferUsage::parameter(), options)?;
    }

    for instruction in &function.body {
        insert_value(
            &mut plan,
            instruction.id,
            &instruction.output,
            BufferUsage::temporary(),
            options,
        )?;
    }

    for (id, ty) in function.result_ids.iter().zip(function.results.iter()) {
        insert_value(&mut plan, *id, ty, BufferUsage::result(), options)?;
    }

    finalize_plan(
        &mut plan,
        &function.parameter_ids,
        &function.body,
        Some(&function.result_ids),
    );

    Ok(plan)
}

fn plan_region(
    region: &Region,
    options: &BufferizeOptions,
) -> Result<FunctionBufferPlan, BufferizeError> {
    let mut plan = FunctionBufferPlan::default();

    // Region parameter ids are not explicit; assume they are numbered in order.
    for (index, ty) in region.parameters.iter().enumerate() {
        let id = ValueId(index as u32);
        insert_value(&mut plan, id, ty, BufferUsage::parameter(), options)?;
    }

    for instruction in &region.body {
        insert_value(
            &mut plan,
            instruction.id,
            &instruction.output,
            BufferUsage::temporary(),
            options,
        )?;
    }

    let result_ids = region_result_ids(region);
    finalize_plan(&mut plan, &[], &region.body, result_ids.as_ref());

    Ok(plan)
}

fn region_result_ids(region: &Region) -> Option<Vec<ValueId>> {
    if region.results.is_empty() {
        return Some(Vec::new());
    }
    if region.body.len() < region.results.len() {
        return None;
    }
    let start = region.body.len() - region.results.len();
    Some(region.body[start..].iter().map(|inst| inst.id).collect())
}

fn finalize_plan(
    plan: &mut FunctionBufferPlan,
    parameter_ids: &[ValueId],
    instructions: &[crate::backend::spec::Instruction],
    result_ids: Option<&Vec<ValueId>>,
) {
    let live_ranges = compute_live_ranges(parameter_ids, instructions, result_ids);
    let alias_info = compute_alias_groups(plan, instructions);
    assign_live_ranges(plan, &live_ranges, &alias_info);
    assign_slots(plan, instructions, &alias_info);
}

fn insert_value(
    plan: &mut FunctionBufferPlan,
    value: ValueId,
    ty: &ValueType,
    usage: BufferUsage,
    options: &BufferizeOptions,
) -> Result<(), BufferizeError> {
    insert_value_recursive(plan, value, ty, usage, options, &mut Vec::new())
}

fn insert_value_recursive(
    plan: &mut FunctionBufferPlan,
    value: ValueId,
    ty: &ValueType,
    usage: BufferUsage,
    options: &BufferizeOptions,
    path: &mut Vec<usize>,
) -> Result<(), BufferizeError> {
    match ty {
        ValueType::Tensor(spec) => {
            let byte_len = compute_byte_len(spec, options, value, path)?;
            let key = BufferKey {
                value,
                path: path.clone(),
            };
            if let Some(index) = plan.values.get(&key).copied() {
                let existing = plan
                    .buffers
                    .get_mut(index)
                    .expect("buffer index must be valid");
                if existing.dtype != spec.dtype || existing.shape != spec.shape {
                    return Err(BufferizeError::TypeMismatch { value });
                }
                existing.usage = existing.usage.merge(usage);
            } else {
                let spec = BufferSpec {
                    value,
                    path: path.clone(),
                    dtype: spec.dtype,
                    shape: spec.shape.clone(),
                    byte_len,
                    usage,
                    alias_group: 0,
                    alias_kind: AliasKind::None,
                    alias_of: None,
                    live_range: LiveRange::new(0, 0),
                    slot: None,
                };
                let index = plan.buffers.len();
                plan.buffers.push(spec);
                plan.values.insert(key, index);
            }
            Ok(())
        }
        ValueType::Tuple(elements) => {
            for (index, element) in elements.iter().enumerate() {
                path.push(index);
                insert_value_recursive(plan, value, element, usage, options, path)?;
                path.pop();
            }
            Ok(())
        }
    }
}

fn compute_byte_len(
    spec: &TensorSpec,
    options: &BufferizeOptions,
    value: ValueId,
    path: &[usize],
) -> Result<Option<usize>, BufferizeError> {
    if options.require_static_shapes && spec.shape.static_dims().is_none() {
        return Err(BufferizeError::DynamicShape {
            value,
            path: path.to_vec(),
        });
    }
    if options.require_known_dtypes && spec.dtype.size_in_bytes().is_none() {
        return Err(BufferizeError::UnknownDType {
            value,
            path: path.to_vec(),
            dtype: spec.dtype,
        });
    }

    if spec.shape.static_dims().is_some() && spec.dtype.size_in_bytes().is_some() {
        match spec.byte_len() {
            Some(bytes) => Ok(Some(bytes)),
            None => Err(BufferizeError::ByteLenOverflow {
                value,
                path: path.to_vec(),
            }),
        }
    } else {
        Ok(None)
    }
}

fn compute_live_ranges(
    parameter_ids: &[ValueId],
    instructions: &[crate::backend::spec::Instruction],
    result_ids: Option<&Vec<ValueId>>,
) -> HashMap<ValueId, LiveRange> {
    let mut ranges = HashMap::new();
    for id in parameter_ids {
        ranges.insert(*id, LiveRange::new(0, 0));
    }
    for (idx, inst) in instructions.iter().enumerate() {
        let pos = idx + 1;
        ranges.insert(inst.id, LiveRange::new(pos, pos));
    }
    for (idx, inst) in instructions.iter().enumerate() {
        let pos = idx + 1;
        for operand in &inst.operands {
            match operand {
                crate::backend::spec::Operand::Value(id) => {
                    if let Some(range) = ranges.get_mut(id) {
                        range.end = range.end.max(pos);
                    }
                }
                crate::backend::spec::Operand::TupleElement { tuple, .. } => {
                    if let Some(range) = ranges.get_mut(tuple) {
                        range.end = range.end.max(pos);
                    }
                }
                crate::backend::spec::Operand::Literal(_) => {}
            }
        }
    }
    if let Some(result_ids) = result_ids {
        let end = instructions.len() + 1;
        for id in result_ids {
            if let Some(range) = ranges.get_mut(id) {
                range.end = range.end.max(end);
            }
        }
    }
    ranges
}

fn compute_alias_groups(
    plan: &FunctionBufferPlan,
    instructions: &[crate::backend::spec::Instruction],
) -> HashMap<BufferKey, (AliasKind, BufferKey)> {
    let mut alias_info = HashMap::new();
    for inst in instructions {
        let (alias_kind, input) = match inst.op {
            crate::backend::spec::Operation::Reshape(_) => {
                (AliasKind::Identity, operand_value(&inst.operands))
            }
            crate::backend::spec::Operation::StopGradient => {
                (AliasKind::Identity, operand_value(&inst.operands))
            }
            crate::backend::spec::Operation::Slice(_) => {
                (AliasKind::View, operand_value(&inst.operands))
            }
            crate::backend::spec::Operation::Transpose(_) => {
                (AliasKind::View, operand_value(&inst.operands))
            }
            _ => (AliasKind::None, None),
        };
        if alias_kind == AliasKind::None {
            continue;
        }
        let Some(input) = input else {
            continue;
        };
        let out_key = BufferKey {
            value: inst.id,
            path: Vec::new(),
        };
        let in_key = BufferKey {
            value: input,
            path: Vec::new(),
        };
        if plan.values.contains_key(&out_key) && plan.values.contains_key(&in_key) {
            alias_info.insert(out_key, (alias_kind, in_key));
        }
    }
    alias_info
}

fn assign_live_ranges(
    plan: &mut FunctionBufferPlan,
    live_ranges: &HashMap<ValueId, LiveRange>,
    alias_info: &HashMap<BufferKey, (AliasKind, BufferKey)>,
) {
    let mut group_roots: HashMap<BufferKey, BufferKey> = HashMap::new();
    for key in plan.values.keys() {
        group_roots.insert(key.clone(), key.clone());
    }

    for (out_key, (_, in_key)) in alias_info {
        let root_out = find_root(&group_roots, out_key);
        let root_in = find_root(&group_roots, in_key);
        if root_out != root_in {
            group_roots.insert(root_out, root_in);
        }
    }

    let mut group_ids: HashMap<BufferKey, usize> = HashMap::new();
    let mut next_group = 0usize;
    for key in plan.values.keys() {
        let root = find_root(&group_roots, key);
        let id = *group_ids.entry(root).or_insert_with(|| {
            let id = next_group;
            next_group += 1;
            id
        });
        if let Some(idx) = plan.values.get(key).copied() {
            if let Some(spec) = plan.buffers.get_mut(idx) {
                spec.alias_group = id;
                if let Some(range) = live_ranges.get(&spec.value) {
                    spec.live_range = *range;
                }
                if let Some((kind, alias_of)) = alias_info.get(key) {
                    spec.alias_kind = *kind;
                    spec.alias_of = Some(alias_of.clone());
                }
            }
        }
    }

    let mut identity_roots: HashMap<BufferKey, BufferKey> = HashMap::new();
    for key in plan.values.keys() {
        identity_roots.insert(key.clone(), key.clone());
    }
    for (out_key, (kind, in_key)) in alias_info {
        if *kind != AliasKind::Identity {
            continue;
        }
        let root_out = find_root(&identity_roots, out_key);
        let root_in = find_root(&identity_roots, in_key);
        if root_out != root_in {
            identity_roots.insert(root_out, root_in);
        }
    }

    let mut identity_ranges: HashMap<BufferKey, LiveRange> = HashMap::new();
    for key in plan.values.keys() {
        let root = find_root(&identity_roots, key);
        if let Some(idx) = plan.values.get(key).copied() {
            let range = plan.buffers[idx].live_range;
            identity_ranges
                .entry(root)
                .and_modify(|group_range| {
                    group_range.start = group_range.start.min(range.start);
                    group_range.end = group_range.end.max(range.end);
                })
                .or_insert(range);
        }
    }

    for key in plan.values.keys() {
        let root = find_root(&identity_roots, key);
        if let Some(idx) = plan.values.get(key).copied() {
            if let Some(range) = identity_ranges.get(&root).copied() {
                plan.buffers[idx].live_range = range;
            }
        }
    }
}

fn assign_slots(
    plan: &mut FunctionBufferPlan,
    instructions: &[crate::backend::spec::Instruction],
    alias_info: &HashMap<BufferKey, (AliasKind, BufferKey)>,
) {
    let const_ids: std::collections::HashSet<ValueId> = instructions
        .iter()
        .filter_map(|inst| match inst.op {
            crate::backend::spec::Operation::Constant(_) => Some(inst.id),
            _ => None,
        })
        .collect();

    let mut groups: HashMap<(DType, Option<usize>), Vec<usize>> = HashMap::new();
    for (idx, buffer) in plan.buffers.iter().enumerate() {
        if buffer.usage.contains_parameter() || buffer.usage.contains_result() {
            continue;
        }
        if const_ids.contains(&buffer.value) {
            continue;
        }
        if buffer.alias_kind == AliasKind::Identity {
            continue;
        }
        let key = (buffer.dtype, buffer.byte_len);
        groups.entry(key).or_default().push(idx);
    }

    let mut slots: Vec<BufferSlot> = Vec::new();
    let mut slot_end: Vec<usize> = Vec::new();

    for ((dtype, byte_len), indices) in groups {
        let mut sorted = indices;
        sorted.sort_by_key(|idx| plan.buffers[*idx].live_range.start);
        let mut local_slots: Vec<usize> = Vec::new();
        for idx in sorted {
            let range = plan.buffers[idx].live_range;
            let mut assigned = None;
            for &slot_id in &local_slots {
                if slot_end[slot_id] < range.start {
                    assigned = Some(slot_id);
                    break;
                }
            }
            let slot_id = match assigned {
                Some(id) => id,
                None => {
                    let id = slots.len();
                    slots.push(BufferSlot {
                        id,
                        dtype,
                        byte_len,
                        usage: BufferUsage::empty(),
                    });
                    slot_end.push(0);
                    local_slots.push(id);
                    id
                }
            };
            plan.buffers[idx].slot = Some(slot_id);
            slot_end[slot_id] = slot_end[slot_id].max(range.end);
        }
    }

    let mut alias_slots = vec![None; plan.buffers.len()];
    for (idx, alias_slot) in alias_slots.iter_mut().enumerate() {
        let buffer = &plan.buffers[idx];
        if buffer.alias_kind != AliasKind::Identity {
            continue;
        }
        let mut next = buffer.alias_of.clone();
        while let Some(alias_of) = next {
            if let Some(alias_idx) = plan.values.get(&alias_of).copied() {
                if let Some(slot) = plan.buffers[alias_idx].slot {
                    *alias_slot = Some(slot);
                    break;
                }
                next = plan.buffers[alias_idx].alias_of.clone();
            } else {
                break;
            }
        }
    }

    for (idx, buffer) in plan.buffers.iter_mut().enumerate() {
        if buffer.alias_kind == AliasKind::Identity {
            if let Some(slot) = alias_slots[idx] {
                buffer.slot = Some(slot);
            }
        }
        if let Some(slot) = buffer.slot {
            if let Some(slot_spec) = slots.get_mut(slot) {
                slot_spec.usage = slot_spec.usage.merge(buffer.usage);
            }
        }
        if alias_info
            .get(&BufferKey {
                value: buffer.value,
                path: buffer.path.clone(),
            })
            .is_none()
            && buffer.alias_kind == AliasKind::None
        {
            buffer.alias_of = None;
        }
    }

    plan.slots = slots;
}

fn operand_value(operands: &[crate::backend::spec::Operand]) -> Option<ValueId> {
    match operands.first() {
        Some(crate::backend::spec::Operand::Value(id)) => Some(*id),
        Some(crate::backend::spec::Operand::TupleElement { tuple, .. }) => Some(*tuple),
        _ => None,
    }
}

fn find_root(parents: &HashMap<BufferKey, BufferKey>, key: &BufferKey) -> BufferKey {
    let mut current = key.clone();
    loop {
        let next = parents
            .get(&current)
            .cloned()
            .unwrap_or_else(|| current.clone());
        if next == current {
            return current;
        }
        current = next;
    }
}
