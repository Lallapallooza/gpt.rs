//! Ops that address elements by index. The dynamic slice ops take their starts at run time, as a
//! decode step does to read and write its cache.

use anyhow::{anyhow, ensure, Result};
use smallvec::SmallVec;

use super::class::{ensure_class, ensure_dtype_class, ARITHMETIC};
use super::shape::{axis_index, check_size};
use crate::op::{
    fold, Fold, FoldCx, FoldOut, Folded, Folding, NoRegions, OpDef, OwnedTypes, TypeCx, Types,
};
use crate::traits::Movement;
use crate::types::{DType, Dim, TensorType};
use crate::Attr;

/// `tensor.iota`: a tensor of `shape` whose elements count up along `axis`.
///
/// - The value increases along `axis`, a negative axis counting from the end, from 0 to
///   `shape[axis] - 1`, and repeats along every other axis.
/// - Each value is its integer index converted to `dtype` as `tensor.cast` converts an integer: a
///   float rounds it once, to nearest-even, and a narrow integer saturates it.
#[derive(crate::Op, Debug, Clone, PartialEq, Eq, Hash)]
#[op("tensor.iota", results(output: Tensor))]
pub struct Iota {
    pub shape: SmallVec<[Dim; 4]>,
    pub dtype: DType,
    pub axis: i64,
}

impl OpDef for Iota {
    fn result_types(
        &self,
        _ins: IotaOperands<Types<'_>>,
        _cx: &TypeCx<'_, NoRegions>,
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

/// `tensor.take`: the rows of `table` at `indices`. The result shape is the indices' shape
/// followed by the table's trailing axes.
///
/// - Equivalent to `index_select` on axis 0, the common embedding primitive.
/// - `indices` are `si32` or `si64`, and each MUST satisfy `0 <= index < N`, where `N` is the
///   extent of the table's leading axis.
#[derive(crate::Op, Debug, Clone, PartialEq, Eq, Hash)]
#[op(
    "tensor.take",
    operands(table: Tensor, indices: Tensor),
    results(output: Tensor),
    traits(Movement::leading(1))
)]
pub struct Take;

impl OpDef for Take {
    fn result_types(
        &self,
        ins: TakeOperands<Types<'_>>,
        _cx: &TypeCx<'_, NoRegions>,
    ) -> Result<TakeResults<OwnedTypes>> {
        ensure!(
            ins.table.rank() >= 1,
            "the table must have rank at least 1, got {}",
            ins.table
        );
        ensure_index_dtype(ins.indices)?;
        let dims = ins
            .indices
            .dims
            .iter()
            .chain(&ins.table.dims[1..])
            .cloned()
            .collect();
        Ok(TakeResults {
            output: ins.table.with_dims(dims),
        })
    }
}

/// `tensor.dynamic_slice`: the block of extents `sizes` starting at the run-time `starts`, with
/// unit stride.
///
/// - `sizes` has one extent per axis, and `starts` holds one `si32` or `si64` start per axis, of
///   shape `[rank]`.
/// - On each axis, the start is clamped into `[0, extent - size]`, so the block lies inside the
///   input.
/// - If a size exceeds its axis's extent at run time, backends MUST reject the op as a spec
///   violation.
#[derive(crate::Op, Debug, Clone, PartialEq, Eq, Hash)]
#[op(
    "tensor.dynamic_slice",
    operands(input: Tensor, starts: Tensor),
    results(output: Tensor),
    traits(Movement::leading(1))
)]
pub struct DynamicSlice {
    pub sizes: SmallVec<[usize; 4]>,
}

impl OpDef for DynamicSlice {
    fn result_types(
        &self,
        ins: DynamicSliceOperands<Types<'_>>,
        _cx: &TypeCx<'_, NoRegions>,
    ) -> Result<DynamicSliceResults<OwnedTypes>> {
        check_window_type(ins.input, ins.starts, &self.sizes)?;
        Ok(DynamicSliceResults {
            output: TensorType::new(ins.input.dtype, self.sizes.iter().copied()),
        })
    }
}

/// `tensor.dynamic_update_slice`: `base` with the block at the run-time `starts` replaced by
/// `update`.
///
/// - The block's extents are the static extents of `update`, which has the dtype and rank of
///   `base`.
/// - `starts` holds one `si32` or `si64` start per axis, of shape `[rank]`, clamped as in
///   `tensor.dynamic_slice`: into `[0, extent - size]`.
#[derive(crate::Op, Debug, Clone, PartialEq, Eq, Hash)]
#[op(
    "tensor.dynamic_update_slice",
    operands(base: Tensor, update: Tensor, starts: Tensor),
    results(output: Tensor),
    traits(Movement::leading(2))
)]
pub struct DynamicUpdateSlice;

impl OpDef for DynamicUpdateSlice {
    fn result_types(
        &self,
        ins: DynamicUpdateSliceOperands<Types<'_>>,
        _cx: &TypeCx<'_, NoRegions>,
    ) -> Result<DynamicUpdateSliceResults<OwnedTypes>> {
        let (base, update) = (ins.base, ins.update);
        ensure!(
            update.dtype == base.dtype,
            "update must have dtype {}, got {update}",
            base.dtype
        );
        ensure!(
            update.rank() == base.rank(),
            "update must have rank {}, got {update}",
            base.rank()
        );
        let sizes = update
            .static_dims()
            .ok_or_else(|| anyhow!("update must have static extents, got {update}"))?;
        check_window_type(base, ins.starts, &sizes)?;
        Ok(DynamicUpdateSliceResults {
            output: base.clone(),
        })
    }
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

/// A slice as large as its input starts at 0 whatever `starts` holds, so it is the input.
#[fold]
impl Fold for DynamicSlice {
    fn fold(
        &self,
        ins: DynamicSliceOperands<Folding<'_>>,
        cx: &FoldCx<'_>,
    ) -> Option<DynamicSliceResults<FoldOut>> {
        (ins.input.ty == cx.result_type(0)).then_some(DynamicSliceResults {
            output: Folded::Value(ins.input.value),
        })
    }
}

/// An update as large as the base replaces all of it, so the result is the update.
#[fold]
impl Fold for DynamicUpdateSlice {
    fn fold(
        &self,
        ins: DynamicUpdateSliceOperands<Folding<'_>>,
        _cx: &FoldCx<'_>,
    ) -> Option<DynamicUpdateSliceResults<FoldOut>> {
        (ins.update.ty == ins.base.ty).then_some(DynamicUpdateSliceResults {
            output: Folded::Value(ins.update.value),
        })
    }
}

/// `tensor.gather`: `torch.gather`. Element `i` of the result is element `i` of `input` with its
/// coordinate on `axis` replaced by `indices[i]`.
///
/// - `indices` are `si32` or `si64` and MUST have the rank of the input and its extents on every
///   axis but `axis`, a negative axis counting from the end; along `axis` they MAY differ.
/// - The result has the shape of `indices`.
/// - Each index MUST satisfy `0 <= index < in_shape[axis]`.
#[derive(crate::Op, Debug, Clone, PartialEq, Eq, Hash)]
#[op(
    "tensor.gather",
    operands(input: Tensor, indices: Tensor),
    results(output: Tensor)
)]
pub struct Gather {
    pub axis: i64,
}

impl OpDef for Gather {
    fn result_types(
        &self,
        ins: GatherOperands<Types<'_>>,
        _cx: &TypeCx<'_, NoRegions>,
    ) -> Result<GatherResults<OwnedTypes>> {
        let (input, indices) = (ins.input, ins.indices);
        ensure_index_shape(self.axis, input, indices)?;
        Ok(GatherResults {
            output: input.with_dims(indices.dims.clone()),
        })
    }
}

/// How a scatter combines each update into its element.
#[derive(Attr, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ScatterKind {
    /// Accumulates the sum of the element and its updates.
    Add,
    /// Accumulates the maximum of the element and its updates.
    Max,
    /// Accumulates the minimum of the element and its updates.
    Min,
    /// Replaces the element: of several updates to one element, the last in row-major order of
    /// `updates` wins.
    Replace,
}

/// `tensor.scatter_add`, `tensor.scatter_max`, ...: `torch.scatter_reduce` of the same-rank form.
///
/// - The result is `input` with element `i` of `updates` combined by `kind` into the element whose
///   coordinate on `axis` is `indices[i]`, in row-major order of `updates`.
/// - `indices` are `si32` or `si64`, and `updates` MUST have their shape and the input's dtype.
///   Both MUST have the input's rank and its extents on every axis but `axis`, a negative axis
///   counting from the end; along `axis` they MAY differ.
/// - Each index MUST be in range.
/// - `scatter_add`, `scatter_max` and `scatter_min` accumulate in the default accumulator of the
///   dtype, then return the dtype.
#[derive(crate::Op, Debug, Clone, PartialEq, Eq, Hash)]
#[op(
    "tensor.scatter_{kind}",
    operands(input: Tensor, indices: Tensor, updates: Tensor),
    results(output: Tensor)
)]
pub struct Scatter {
    pub kind: ScatterKind,
    pub axis: i64,
}

impl OpDef for Scatter {
    fn result_types(
        &self,
        ins: ScatterOperands<Types<'_>>,
        _cx: &TypeCx<'_, NoRegions>,
    ) -> Result<ScatterResults<OwnedTypes>> {
        let (input, indices, updates) = (ins.input, ins.indices, ins.updates);
        if self.kind != ScatterKind::Replace {
            ensure_class("the input", input, ARITHMETIC)?;
        }
        ensure_index_shape(self.axis, input, indices)?;
        ensure!(
            updates.dtype == input.dtype && updates.dims == indices.dims,
            "updates must have the dtype of {input} and the shape of {indices}, got {updates}"
        );
        Ok(ScatterResults {
            output: input.clone(),
        })
    }
}

/// Checks that `indices` are si32 or si64 of the rank of `input`, with the extents of `input` on
/// every axis but `axis`.
fn ensure_index_shape(axis: i64, input: &TensorType, indices: &TensorType) -> Result<()> {
    let rank = input.rank();
    let axis = axis_index(axis, rank)?;
    ensure_index_dtype(indices)?;
    ensure!(
        indices.rank() == rank && (0..rank).all(|d| d == axis || indices.dims[d] == input.dims[d]),
        "indices {indices} must match {input} on every axis but {axis}"
    );
    Ok(())
}

/// Checks that `indices` are si32 or si64.
pub(super) fn ensure_index_dtype(indices: &TensorType) -> Result<()> {
    ensure!(
        matches!(indices.dtype, DType::Si32 | DType::Si64),
        "indices must be si32 or si64, got {indices}"
    );
    Ok(())
}
