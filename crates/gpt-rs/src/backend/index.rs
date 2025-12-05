use std::collections::{HashMap, HashSet};

use smallvec::SmallVec;
use thiserror::Error;

use crate::backend::spec::{Function, Instruction, Operand, ValueId, ValueType};

/// Stable identifier assigned to each instruction when indexing a function body.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub struct InstId(pub u32);

/// Total "definition" identifier for a value (MLIR-style `BlockArgument | OpResult`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DefId {
    /// Function parameter at the given index.
    Param { index: u32 },
    /// Instruction producing the value.
    Inst(InstId),
}

/// Captures structural indices for a single PTIR function body.
#[derive(Debug, Clone)]
pub struct FunctionIndices {
    inst_values: HashMap<InstId, ValueId>,
    value_types: HashMap<ValueId, ValueType>,
    params: Vec<ValueId>,
    param_index_of: HashMap<ValueId, u32>,
    pub pos_of: HashMap<InstId, usize>,
    pub value_to_inst: HashMap<ValueId, InstId>,
    pub users: HashMap<ValueId, SmallVec<[InstId; 4]>>,
    pub version: HashMap<InstId, u32>,
    pub next_value: u32,
    pub next_inst: u32,
}

impl FunctionIndices {
    /// Builds indices for the provided PTIR function and validates SSA invariants.
    pub fn build(function: &Function) -> Result<Self, FunctionIndexError> {
        let mut seen_values: HashSet<ValueId> = HashSet::new();
        let mut value_types: HashMap<ValueId, ValueType> = HashMap::new();
        let mut params: Vec<ValueId> = Vec::with_capacity(function.parameters.len());
        let mut param_index_of: HashMap<ValueId, u32> = HashMap::new();
        let mut pos_of = HashMap::new();
        let mut value_to_inst = HashMap::new();
        let mut inst_values = HashMap::new();
        let mut users: HashMap<ValueId, SmallVec<[InstId; 4]>> = HashMap::new();

        for (index, (param_id, param_ty)) in function
            .parameter_ids
            .iter()
            .zip(function.parameters.iter())
            .enumerate()
        {
            if !seen_values.insert(*param_id) {
                return Err(FunctionIndexError::DuplicateValue { value: *param_id });
            }
            params.push(*param_id);
            param_index_of.insert(*param_id, index as u32);
            value_types.insert(*param_id, param_ty.clone());
        }

        for (index, instruction) in function.body.iter().enumerate() {
            let inst_id = InstId(index as u32);

            for operand in &instruction.operands {
                let referenced = match operand {
                    Operand::Value(value) => *value,
                    Operand::TupleElement { tuple, .. } => *tuple,
                    Operand::Literal(_) => continue,
                };
                if !value_types.contains_key(&referenced) {
                    return Err(FunctionIndexError::MissingValueDefinition { value: referenced });
                }
                users.entry(referenced).or_default().push(inst_id);
            }

            if !seen_values.insert(instruction.id) {
                return Err(FunctionIndexError::DuplicateValue {
                    value: instruction.id,
                });
            }

            pos_of.insert(inst_id, index);
            value_to_inst.insert(instruction.id, inst_id);
            inst_values.insert(inst_id, instruction.id);
            value_types.insert(instruction.id, instruction.output.clone());
        }

        for result_id in &function.result_ids {
            if !value_types.contains_key(result_id) {
                return Err(FunctionIndexError::MissingValueDefinition { value: *result_id });
            }
        }

        let mut max_value = 0u32;
        for id in &function.parameter_ids {
            max_value = max_value.max(id.0);
        }
        for inst in &function.body {
            max_value = max_value.max(inst.id.0);
        }

        let version = pos_of.keys().map(|id| (*id, 0u32)).collect();
        Ok(FunctionIndices {
            inst_values,
            value_types,
            params,
            param_index_of,
            pos_of,
            value_to_inst,
            users,
            version,
            next_value: max_value + 1,
            next_inst: function.body.len() as u32,
        })
    }

    /// Returns the instruction position for the provided identifier.
    pub fn position(&self, inst: InstId) -> Option<usize> {
        self.pos_of.get(&inst).copied()
    }

    /// Returns the SSA value defined by the provided instruction.
    pub fn value_of(&self, inst: InstId) -> Option<ValueId> {
        self.inst_values.get(&inst).copied()
    }

    /// Returns the instruction producing the given value, if already defined.
    pub fn inst_of(&self, value: ValueId) -> Option<InstId> {
        self.value_to_inst.get(&value).copied()
    }

    /// Returns the total "definition" for the given value.
    pub fn def_of(&self, value: ValueId) -> Option<DefId> {
        if let Some(index) = self.param_index_of.get(&value).copied() {
            return Some(DefId::Param { index });
        }
        self.inst_of(value).map(DefId::Inst)
    }

    pub fn value_of_def(&self, def: DefId) -> Option<ValueId> {
        match def {
            DefId::Param { index } => self.params.get(index as usize).copied(),
            DefId::Inst(inst) => self.value_of(inst),
        }
    }

    /// Returns the type associated with a given SSA value.
    pub fn type_of(&self, value: ValueId) -> Option<&ValueType> {
        self.value_types.get(&value)
    }

    /// Returns the users recorded for a given SSA value.
    pub fn users_of(&self, value: ValueId) -> &[InstId] {
        self.users
            .get(&value)
            .map(|list| list.as_slice())
            .unwrap_or(&[])
    }

    pub fn contains(&self, inst: InstId) -> bool {
        self.pos_of.contains_key(&inst)
    }

    pub fn version(&self, inst: InstId) -> Option<u32> {
        self.version.get(&inst).copied()
    }

    pub fn ordered_inst_ids(&self) -> Vec<InstId> {
        let mut entries: Vec<_> = self
            .pos_of
            .iter()
            .map(|(inst, pos)| (*inst, *pos))
            .collect();
        entries.sort_by_key(|&(_, pos)| pos);
        entries.into_iter().map(|(inst, _)| inst).collect()
    }

    pub(crate) fn allocate_inst(&mut self) -> InstId {
        let inst = InstId(self.next_inst);
        self.next_inst += 1;
        inst
    }

    pub(crate) fn allocate_value(&mut self) -> ValueId {
        let value = ValueId(self.next_value);
        self.next_value += 1;
        value
    }

    pub(crate) fn insert_instruction(
        &mut self,
        inst_id: InstId,
        pos: usize,
        value_id: ValueId,
        output: ValueType,
        instruction: &Instruction,
    ) -> Result<(), FunctionIndexError> {
        if self.value_to_inst.contains_key(&value_id) {
            return Err(FunctionIndexError::DuplicateValue { value: value_id });
        }

        self.shift_positions_from(pos, 1);
        self.pos_of.insert(inst_id, pos);
        self.value_to_inst.insert(value_id, inst_id);
        self.inst_values.insert(inst_id, value_id);
        self.value_types.insert(value_id, output);
        self.version.insert(inst_id, 0);
        self.add_operand_users(inst_id, &instruction.operands)?;
        Ok(())
    }

    pub(crate) fn remove_instruction(&mut self, inst: InstId, instruction: &Instruction) {
        if let Some(pos) = self.pos_of.remove(&inst) {
            self.shift_positions_from(pos + 1, -1);
        }
        if let Some(value) = self.inst_values.remove(&inst) {
            self.value_to_inst.remove(&value);
            self.value_types.remove(&value);
            self.users.remove(&value);
        }
        self.version.remove(&inst);
        self.remove_operand_users(inst, &instruction.operands);
    }

    pub(crate) fn update_operand_use(
        &mut self,
        inst: InstId,
        from: ValueId,
        to: ValueId,
    ) -> Result<(), FunctionIndexError> {
        if !self.value_types.contains_key(&to) {
            return Err(FunctionIndexError::MissingValueDefinition { value: to });
        }

        if let Some(list) = self.users.get_mut(&from) {
            list.retain(|id| *id != inst);
            if list.is_empty() {
                self.users.remove(&from);
            }
        }
        self.users.entry(to).or_default().push(inst);
        Ok(())
    }

    pub(crate) fn clear_users(&mut self, value: ValueId) {
        self.users.remove(&value);
    }

    fn shift_positions_from(&mut self, start: usize, delta: isize) {
        if delta == 0 {
            return;
        }
        for (_, pos) in self.pos_of.iter_mut() {
            if *pos >= start {
                if delta.is_positive() {
                    *pos += delta.unsigned_abs();
                } else {
                    *pos -= delta.unsigned_abs();
                }
            }
        }
    }

    fn add_operand_users(
        &mut self,
        inst: InstId,
        operands: &[Operand],
    ) -> Result<(), FunctionIndexError> {
        for operand in operands {
            let referenced = match operand {
                Operand::Value(value) => *value,
                Operand::TupleElement { tuple, .. } => *tuple,
                Operand::Literal(_) => continue,
            };
            if !self.value_types.contains_key(&referenced) {
                return Err(FunctionIndexError::MissingValueDefinition { value: referenced });
            }
            self.users.entry(referenced).or_default().push(inst);
        }
        Ok(())
    }

    fn remove_operand_users(&mut self, inst: InstId, operands: &[Operand]) {
        for operand in operands {
            let referenced = match operand {
                Operand::Value(value) => *value,
                Operand::TupleElement { tuple, .. } => *tuple,
                Operand::Literal(_) => continue,
            };
            if let Some(list) = self.users.get_mut(&referenced) {
                list.retain(|id| *id != inst);
                if list.is_empty() {
                    self.users.remove(&referenced);
                }
            }
        }
    }
}

/// Errors surfaced when building SSA indices for a function.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum FunctionIndexError {
    #[error("duplicate value definition for %{value:?}")]
    DuplicateValue { value: ValueId },
    #[error("value %{value:?} is used but never defined")]
    MissingValueDefinition { value: ValueId },
}
