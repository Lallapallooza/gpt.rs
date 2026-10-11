//! The type rules of the window ops: the types each op gives, and what it rejects.

use anyhow::{ensure, Result};

use super::contraction::ensure_reduction;
use crate::dialects::tensor::ir::*;
use crate::dialects::tensor::Accumulates;
use crate::op::{NoRegions, OpDef, OwnedTypes, TypeCx, Types};
use crate::types::TensorType;

impl OpDef for ExtractPatches {
    fn result_types(
        &self,
        ins: ExtractPatchesOperands<Types<'_>>,
        _cx: &TypeCx<'_, NoRegions>,
    ) -> Result<ExtractPatchesResults<OwnedTypes>> {
        let (_, dims) = self.geometry(ins.input)?;
        ensure!(
            ins.input.dtype.represents(self.pad_value.0),
            "pad_value {:?} is not representable in {}",
            self.pad_value.0,
            ins.input.dtype
        );
        Ok(ExtractPatchesResults {
            output: TensorType::new(ins.input.dtype, dims),
        })
    }
}

/// The type rule of every [`ReduceWindow`] op.
pub(crate) fn reduce_window<T: ReduceWindow>(
    op: &T,
    ins: ReduceWindowOperands<Types<'_>>,
    _cx: &TypeCx<'_, NoRegions>,
) -> Result<ReduceWindowResults<OwnedTypes>> {
    let (_, dims) = op.geometry(ins.input)?;
    ensure_reduction::<T::Combiner>(ins.input, op.accum, op.out)?;
    Ok(ReduceWindowResults {
        output: TensorType::new(op.out_dtype(ins.input.dtype), dims),
    })
}
