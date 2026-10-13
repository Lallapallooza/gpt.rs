//! The folders of the layout ops.

use crate::dialects::tensor::ir::*;
use crate::ir::{fold, Fold, FoldCx, FoldOut, Folded, Folding};

/// A bitcast to the input's own type is the input.
#[fold]
impl Fold for Bitcast {
    fn fold(
        &self,
        ins: BitcastOperands<Folding<'_>>,
        cx: &FoldCx<'_>,
    ) -> Option<BitcastResults<FoldOut>> {
        Some(BitcastResults {
            output: cx.identity(ins.input)?,
        })
    }
}

/// A reshape to the input's own type is the input, and a reshape of a constant is that constant
/// with the new extents, its elements in the same row-major order.
#[fold]
impl Fold for Reshape {
    fn fold(
        &self,
        ins: ReshapeOperands<Folding<'_>>,
        cx: &FoldCx<'_>,
    ) -> Option<ReshapeResults<FoldOut>> {
        let input = ins.input;
        let output = match cx.identity(input) {
            Some(output) => output,
            None => Folded::Constant(input.constant?.reshaped(cx.result_dims(0)?)),
        };
        Some(ReshapeResults { output })
    }
}

/// A transpose by the identity `perm` is the input.
#[fold]
impl Fold for Transpose {
    fn fold(
        &self,
        ins: TransposeOperands<Folding<'_>>,
        _cx: &FoldCx<'_>,
    ) -> Option<TransposeResults<FoldOut>> {
        self.perm.is_identity().then_some(TransposeResults {
            output: Folded::Value(ins.input.value),
        })
    }
}

/// A broadcast to the input's own type is the input.
#[fold]
impl Fold for BroadcastTo {
    fn fold(
        &self,
        ins: BroadcastToOperands<Folding<'_>>,
        cx: &FoldCx<'_>,
    ) -> Option<BroadcastToResults<FoldOut>> {
        Some(BroadcastToResults {
            output: cx.identity(ins.input)?,
        })
    }
}

/// A slice with the input's extents covers the whole input, so it is the input.
#[fold]
impl Fold for Slice {
    fn fold(
        &self,
        ins: SliceOperands<Folding<'_>>,
        cx: &FoldCx<'_>,
    ) -> Option<SliceResults<FoldOut>> {
        Some(SliceResults {
            output: cx.identity(ins.input)?,
        })
    }
}

/// A concat of one input is that input.
#[fold]
impl Fold for Concat {
    fn fold(
        &self,
        ins: ConcatOperands<Folding<'_>>,
        _cx: &FoldCx<'_>,
    ) -> Option<ConcatResults<FoldOut>> {
        let [input] = ins.inputs.as_slice() else {
            return None;
        };
        Some(ConcatResults {
            output: (*input).into(),
        })
    }
}
