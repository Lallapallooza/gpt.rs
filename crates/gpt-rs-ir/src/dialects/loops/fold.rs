//! The folders of the `loop` ops, which `#[fold]` registers.

use crate::dialects::loops::ir::*;
use crate::ir::{fold, Fold, FoldCx, FoldOut, Folded, Folding};

/// An index whose expression is one of its operands is that operand.
#[fold]
impl Fold for Index {
    fn fold(&self, ins: IndexOperands<Folding>, _cx: &FoldCx) -> Option<IndexResults<FoldOut>> {
        let position = self.expr.as_iter()?;
        Some(IndexResults {
            output: Folded::Value(ins.ins[position].value),
        })
    }
}
