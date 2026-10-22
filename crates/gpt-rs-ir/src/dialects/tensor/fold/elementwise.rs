//! The folders of the elementwise ops.

use crate::dialects::tensor::ir::*;
use crate::ir::{fold, Fold, FoldCx, FoldOut, Folded, Folding};

/// `select(pred, on_true, on_false)` is `on_true` when `pred` is all true or both branches are
/// the same value, and `on_false` when `pred` is all false.
#[fold]
impl Fold for Select {
    fn fold(&self, ins: SelectOperands<Folding>, _cx: &FoldCx) -> Option<SelectResults<FoldOut>> {
        let picked = if ins.pred.is_splat(true) || ins.on_true.value == ins.on_false.value {
            ins.on_true
        } else if ins.pred.is_splat(false) {
            ins.on_false
        } else {
            return None;
        };
        Some(SelectResults {
            output: picked.into(),
        })
    }
}

/// A cast to the input's own type is the input.
#[fold]
impl Fold for Cast {
    fn fold(&self, ins: CastOperands<Folding>, cx: &FoldCx) -> Option<CastResults<FoldOut>> {
        (ins.input.ty == cx.result_type(0)).then_some(CastResults {
            output: Folded::Value(ins.input.value),
        })
    }
}
