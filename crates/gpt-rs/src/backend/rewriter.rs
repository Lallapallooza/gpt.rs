use crate::backend::{
    index::{DefId, FunctionIndexError, FunctionIndices, InstId},
    spec::{Function, Instruction, Operand, Operation, TensorLiteral, ValueId, ValueType},
};

/// Mutable IR editor with stable instruction identifiers and SSA accounting.
pub struct ProgramRewriter<'a> {
    pub func: &'a mut Function,
    indices: FunctionIndices,
}

impl<'a> ProgramRewriter<'a> {
    /// Creates a rewriter for the provided PTIR function, indexing its body.
    pub fn new(func: &'a mut Function) -> Result<Self, FunctionIndexError> {
        let indices = FunctionIndices::build(func)?;
        Ok(Self { func, indices })
    }

    /// Returns the operation referenced by `inst`.
    pub fn op(&self, inst: InstId) -> &Operation {
        let pos = self
            .indices
            .position(inst)
            .expect("instruction id must be valid");
        &self.func.body[pos].op
    }

    /// Returns the operands for the given instruction.
    pub fn operands(&self, inst: InstId) -> &[Operand] {
        let pos = self
            .indices
            .position(inst)
            .expect("instruction id must be valid");
        &self.func.body[pos].operands
    }

    /// Returns the SSA value produced by the instruction.
    pub fn value_of(&self, inst: InstId) -> ValueId {
        self.indices
            .value_of(inst)
            .expect("instruction must have a value")
    }

    /// Returns the type recorded for the value.
    pub fn type_of(&self, value: ValueId) -> Option<&ValueType> {
        self.indices.type_of(value)
    }

    /// Returns the instruction defining the provided value.
    pub fn inst_of(&self, value: ValueId) -> Option<InstId> {
        self.indices.inst_of(value)
    }

    /// Returns the total "definition" for the provided value.
    pub fn def_of(&self, value: ValueId) -> Option<DefId> {
        self.indices.def_of(value)
    }

    pub fn value_of_def(&self, def: DefId) -> Option<ValueId> {
        self.indices.value_of_def(def)
    }

    pub fn type_of_def(&self, def: DefId) -> Option<&ValueType> {
        let value = self.value_of_def(def)?;
        self.type_of(value)
    }

    pub fn op_of_def(&self, def: DefId) -> Option<&Operation> {
        match def {
            DefId::Param { .. } => None,
            DefId::Inst(inst) => Some(self.op(inst)),
        }
    }

    pub fn operands_of_def(&self, def: DefId) -> &[Operand] {
        match def {
            DefId::Param { .. } => &[],
            DefId::Inst(inst) => self.operands(inst),
        }
    }

    /// Returns the recorded users for the value.
    pub fn users_of(&self, value: ValueId) -> &[InstId] {
        self.indices.users_of(value)
    }

    pub fn contains(&self, inst: InstId) -> bool {
        self.indices.contains(inst)
    }

    /// Returns the current version counter for an instruction.
    pub fn version(&self, inst: InstId) -> Option<u32> {
        self.indices.version(inst)
    }

    pub fn insts_in_order(&self) -> Vec<InstId> {
        self.indices.ordered_inst_ids()
    }

    /// Replaces all uses of `from` with `to`.
    pub fn replace_all_uses(&mut self, from: ValueId, to: ValueId) {
        if from == to {
            return;
        }
        let consumers = self.indices.users_of(from).to_vec();
        for inst in consumers {
            let pos = self
                .indices
                .position(inst)
                .expect("instruction id must be valid");
            let instruction = &mut self.func.body[pos];
            for operand in &mut instruction.operands {
                match operand {
                    Operand::Value(value) if *value == from => *value = to,
                    Operand::TupleElement { tuple, .. } if *tuple == from => *tuple = to,
                    _ => {}
                }
            }
            self.indices
                .update_operand_use(inst, from, to)
                .expect("operand replacement must succeed");
            self.bump_version(inst);
        }
        self.indices.clear_users(from);
    }

    /// Erases the instruction identified by `inst`.
    pub fn erase_inst(&mut self, inst: InstId) {
        let value = self
            .indices
            .value_of(inst)
            .expect("instruction must be in map");
        if !self.indices.users_of(value).is_empty() {
            panic!("attempting to erase instruction with live uses");
        }
        let pos = self
            .indices
            .position(inst)
            .expect("instruction id must be valid");
        let instruction = self.func.body.remove(pos);
        self.indices.remove_instruction(inst, &instruction);
    }

    /// Inserts a new instruction before `at`, returning its identifiers.
    pub fn insert_before(
        &mut self,
        at: InstId,
        op: Operation,
        operands: Vec<Operand>,
        output: ValueType,
    ) -> Result<(InstId, ValueId), FunctionIndexError> {
        let pos = self
            .indices
            .position(at)
            .expect("insertion point must exist");
        self.insert_at_pos(pos, op, operands, output)
    }

    /// Materialises a constant literal by inserting a `Constant` operation before `at`.
    pub fn materialize_constant(
        &mut self,
        at: InstId,
        literal: TensorLiteral,
        output: ValueType,
    ) -> Result<(InstId, ValueId), FunctionIndexError> {
        self.insert_before(at, Operation::Constant(literal), Vec::new(), output)
    }

    /// Verifies basic SSA invariants after mutations.
    pub fn verify(&self) -> bool {
        FunctionIndices::build(self.func).is_ok()
    }

    /// Bumps the version counter for an instruction.
    pub fn bump_version(&mut self, inst: InstId) {
        if let Some(ver) = self.indices.version.get_mut(&inst) {
            *ver = ver.wrapping_add(1);
        }
    }

    fn insert_at_pos(
        &mut self,
        pos: usize,
        op: Operation,
        operands: Vec<Operand>,
        output: ValueType,
    ) -> Result<(InstId, ValueId), FunctionIndexError> {
        let inst_id = self.indices.allocate_inst();
        let value_id = self.indices.allocate_value();
        self.func.body.insert(
            pos,
            Instruction {
                id: value_id,
                op,
                operands,
                output: output.clone(),
            },
        );
        self.indices.insert_instruction(
            inst_id,
            pos,
            value_id,
            output.clone(),
            &self.func.body[pos],
        )?;
        Ok((inst_id, value_id))
    }
}
