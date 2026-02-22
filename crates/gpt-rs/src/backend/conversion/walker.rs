use crate::backend::spec::{Function, HintRegion, Instruction, Program, Region};

pub trait ProgramVisitor {
    fn on_program(&mut self, _program: &Program) {}
    fn on_function(&mut self, _function: &Function) {}
    fn on_instruction(&mut self, _function: &Function, _index: usize, _inst: &Instruction) {}
    fn on_hint(&mut self, _function: &Function, _index: usize, _hint: &HintRegion) {}
    fn on_hint_instruction(
        &mut self,
        _function: &Function,
        _hint: &HintRegion,
        _index: usize,
        _inst: &Instruction,
    ) {
    }
    fn on_region(&mut self, _region: &Region) {}
    fn on_region_instruction(&mut self, _region: &Region, _index: usize, _inst: &Instruction) {}
}

pub fn walk_program<V: ProgramVisitor + ?Sized>(program: &Program, visitor: &mut V) {
    visitor.on_program(program);
    for function in &program.functions {
        walk_function(function, visitor);
    }
    for region in &program.regions {
        walk_region(region, visitor);
    }
}

fn walk_function<V: ProgramVisitor + ?Sized>(function: &Function, visitor: &mut V) {
    visitor.on_function(function);
    for (idx, inst) in function.body.iter().enumerate() {
        visitor.on_instruction(function, idx, inst);
    }
    for (hint_idx, hint) in function.hints.iter().enumerate() {
        visitor.on_hint(function, hint_idx, hint);
        for (idx, inst) in hint.body.iter().enumerate() {
            visitor.on_hint_instruction(function, hint, idx, inst);
        }
    }
}

fn walk_region<V: ProgramVisitor + ?Sized>(region: &Region, visitor: &mut V) {
    visitor.on_region(region);
    for (idx, inst) in region.body.iter().enumerate() {
        visitor.on_region_instruction(region, idx, inst);
    }
}
