//! The folders of the indexing ops.

use crate::dialects::tensor::ir::*;
use crate::ir::{fold, Fold, FoldCx, FoldOut, Folded, Folding};

/// A slice as large as its input starts at 0 whatever `starts` holds, so it is the input.
#[fold]
impl Fold for DynamicSlice {
    fn fold(
        &self,
        ins: DynamicSliceOperands<Folding>,
        cx: &FoldCx,
    ) -> Option<DynamicSliceResults<FoldOut>> {
        (ins.input.ty == cx.result_type(0)).then_some(DynamicSliceResults {
            output: Folded::Value(ins.input.value),
        })
    }
}

/// A patch as large as the base replaces all of it, so the result is the patch.
#[fold]
impl Fold for DynamicUpdateSlice {
    fn fold(
        &self,
        ins: DynamicUpdateSliceOperands<Folding>,
        _cx: &FoldCx,
    ) -> Option<DynamicUpdateSliceResults<FoldOut>> {
        (ins.patch.ty == ins.base.ty).then_some(DynamicUpdateSliceResults {
            output: Folded::Value(ins.patch.value),
        })
    }
}
