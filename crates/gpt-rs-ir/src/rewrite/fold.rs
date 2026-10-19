//! Folding without computing on elements: the op's own folder, then the folds its traits imply.

use crate::ir::{FoldIn, Folded, Inst, Module, OpDyn, Type, Value};
use crate::rewrite::{fired, Fired};
use crate::traits::Operation;
use crate::Op;

/// What an op folds to.
#[derive(Debug, Clone, PartialEq)]
pub enum FoldResult {
    /// The op's results become these.
    Results(Vec<Folded>),
    /// The op stays, with these operands in place of its own: the same results in canonical form.
    Operands(Vec<Value>),
}

/// The operands of an op as its folder sees them.
fn fold_inputs<'m>(module: &'m Module, operands: &[Value]) -> Vec<FoldIn<'m>> {
    operands
        .iter()
        .map(|&value| FoldIn {
            value,
            ty: module.ty(value),
            constant: module.constant(value),
        })
        .collect()
}

/// The first fold one of the op's traits gives.
fn fold_by_traits(
    module: &Module,
    op: &Op,
    operands: &[Value],
    results: &[Type],
) -> Option<FoldResult> {
    let view = Operation {
        module,
        op,
        operands,
        results,
    };
    op.traits().iter().find_map(|op_trait| {
        let fold = op_trait.fold(&view)?;
        fired(Fired::TraitFold(op_trait.name()));
        Some(fold)
    })
}

/// What `op` applied to `operands` folds to, or `None`: its own folder's answer, or else its
/// traits'. When a trait puts the operands in canonical form, folding starts over with them.
pub fn fold_op(
    module: &Module,
    op: &Op,
    operands: &[Value],
    segments: Option<&[usize]>,
    results: &[Type],
) -> Option<FoldResult> {
    let mut current = operands.to_vec();
    loop {
        if let Some(folded) = op.dyn_fold(&fold_inputs(module, &current), segments, results) {
            return Some(FoldResult::Results(folded));
        }
        match fold_by_traits(module, op, &current, results) {
            Some(fold @ FoldResult::Results(_)) => return Some(fold),
            Some(FoldResult::Operands(next)) if next != current => current = next,
            _ => break,
        }
    }
    (current != operands).then_some(FoldResult::Operands(current))
}

/// What the instruction `inst` folds to, or `None`.
pub fn fold_inst(module: &Module, inst: Inst) -> Option<FoldResult> {
    let results: Vec<Type> = module.result_types(inst).into_iter().cloned().collect();
    fold_op(
        module,
        module.op(inst),
        module.operands(inst),
        module.operand_segments(inst),
        &results,
    )
}
