//! The type rules of the indexing ops: the types each op gives, and what it rejects.

use anyhow::{anyhow, ensure, Result};

use crate::dialects::tensor::ir::*;
use crate::dialects::tensor::utils::class::{
    ensure_class, ensure_dtype_class, ensure_index_dtype, ARITHMETIC,
};
use crate::dialects::tensor::utils::shape::{axis_index, check_size};
use crate::ir::{ranked, DType, Dim, NoRegions, OpDef, OwnedTypes, TensorType, TypeCx, Types};

impl OpDef for Iota {
    fn result_types(
        &self,
        _ins: IotaOperands<Types>,
        _cx: &TypeCx<NoRegions>,
    ) -> Result<IotaResults<OwnedTypes>> {
        axis_index(self.axis, self.shape.len())?;
        ensure_dtype_class("dtype", self.dtype, ARITHMETIC)?;
        Ok(IotaResults {
            output: TensorType {
                dtype: self.dtype,
                dims: self.shape.clone(),
            },
        })
    }
}

impl OpDef for Take {
    fn result_types(
        &self,
        ins: TakeOperands<Types>,
        _cx: &TypeCx<NoRegions>,
    ) -> Result<TakeResults<OwnedTypes>> {
        ensure!(
            ins.table.rank() >= 1,
            "the table must have rank at least 1, got {}",
            ins.table
        );
        ensure_index_dtype("indices", ins.indices.dtype)?;
        let dims = [&ins.indices.dims[..], &ins.table.dims[1..]].concat();
        Ok(TakeResults {
            output: ins.table.with_dims(ranked(dims)?),
        })
    }
}

impl OpDef for DynamicSlice {
    fn result_types(
        &self,
        ins: DynamicSliceOperands<Types>,
        _cx: &TypeCx<NoRegions>,
    ) -> Result<DynamicSliceResults<OwnedTypes>> {
        check_window_type(ins.input, ins.starts, &self.sizes)?;
        Ok(DynamicSliceResults {
            output: TensorType::new(ins.input.dtype, self.sizes.iter().copied()),
        })
    }
}

impl OpDef for DynamicUpdateSlice {
    fn result_types(
        &self,
        ins: DynamicUpdateSliceOperands<Types>,
        _cx: &TypeCx<NoRegions>,
    ) -> Result<DynamicUpdateSliceResults<OwnedTypes>> {
        let (base, patch) = (ins.base, ins.patch);
        ensure!(
            patch.dtype == base.dtype,
            "patch must have dtype {}, got {patch}",
            base.dtype
        );
        ensure!(
            patch.rank() == base.rank(),
            "patch must have rank {}, got {patch}",
            base.rank()
        );
        let sizes = patch
            .static_dims()
            .ok_or_else(|| anyhow!("patch must have static extents, got {patch}"))?;
        check_window_type(base, ins.starts, &sizes)?;
        Ok(DynamicUpdateSliceResults {
            output: base.clone(),
        })
    }
}

impl OpDef for Gather {
    fn result_types(
        &self,
        ins: GatherOperands<Types>,
        _cx: &TypeCx<NoRegions>,
    ) -> Result<GatherResults<OwnedTypes>> {
        let (input, indices) = (ins.input, ins.indices);
        ensure_index_shape(self.axis, input, indices)?;
        Ok(GatherResults {
            output: input.with_dims(indices.dims.clone()),
        })
    }
}

impl OpDef for ScatterReplace {
    fn result_types(
        &self,
        ins: ScatterReplaceOperands<Types>,
        _cx: &TypeCx<NoRegions>,
    ) -> Result<ScatterReplaceResults<OwnedTypes>> {
        Ok(ScatterReplaceResults {
            output: scatter_types(self.axis, ins.input, ins.indices, ins.updates)?,
        })
    }
}

/// The type rule of every [`Scatter`] op.
pub(crate) fn scatter<T: Scatter>(
    op: &T,
    ins: ScatterOperands<Types>,
    _cx: &TypeCx<NoRegions>,
) -> Result<ScatterResults<OwnedTypes>> {
    ensure_class("the input", ins.input, ARITHMETIC)?;
    Ok(ScatterResults {
        output: scatter_types(op.axis, ins.input, ins.indices, ins.updates)?,
    })
}

/// The type rule shared by the dynamic slice ops: `sizes` has one extent per axis of `input`,
/// each fitting its static extent, and `starts` holds one si32 or si64 start per axis.
fn check_window_type(input: &TensorType, starts: &TensorType, sizes: &[usize]) -> Result<()> {
    let rank = input.rank();
    ensure!(
        sizes.len() == rank,
        "sizes {sizes:?} do not match rank {rank}"
    );
    ensure!(
        matches!(starts.dtype, DType::Si32 | DType::Si64) && starts.dims[..] == [Dim::Static(rank)],
        "starts must be si32 or si64 of shape [{rank}], got {starts}"
    );
    for (axis, (dim, &size)) in input.dims.iter().zip(sizes).enumerate() {
        if let Dim::Static(extent) = *dim {
            check_size(axis, size, extent)?;
        }
    }
    Ok(())
}

/// Checks the operand types of a scatter along `axis`, and gives its result type: the input's.
fn scatter_types(
    axis: i64,
    input: &TensorType,
    indices: &TensorType,
    updates: &TensorType,
) -> Result<TensorType> {
    ensure_index_shape(axis, input, indices)?;
    ensure!(
        updates.dtype == input.dtype && updates.dims == indices.dims,
        "updates must have the dtype of {input} and the shape of {indices}, got {updates}"
    );
    Ok(input.clone())
}

/// Checks that `indices` are si32 or si64 of the rank of `input`, with the extents of `input` on
/// every axis but `axis`.
fn ensure_index_shape(axis: i64, input: &TensorType, indices: &TensorType) -> Result<()> {
    let rank = input.rank();
    let axis_position = axis_index(axis, rank)?;
    ensure_index_dtype("indices", indices.dtype)?;
    ensure!(
        indices.rank() == rank
            && (0..rank)
                .all(|other| other == axis_position || indices.dims[other] == input.dims[other]),
        "indices {indices} must match {input} on every axis but {axis_position}"
    );
    Ok(())
}
