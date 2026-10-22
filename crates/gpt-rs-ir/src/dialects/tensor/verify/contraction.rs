//! The type rules of the contraction ops: the types each op gives, and what it rejects.

use anyhow::{ensure, Result};

use crate::dialects::arith;
use crate::dialects::arith::math::Combiner;
use crate::dialects::tensor::ir::*;
use crate::dialects::tensor::utils::class::{
    ensure_class, ensure_dtype_class, ensure_index_dtype, Class, ARITHMETIC,
};
use crate::dialects::tensor::utils::contraction::Side;
use crate::dialects::tensor::utils::shape::axis_index;
use crate::dialects::tensor::Accumulates;
use crate::ir::{ranked, DType, Dim, NoRegions, OpDef, OwnedTypes, TensorType, TypeCx, Types};

impl OpDef for DotGeneral {
    fn result_types(
        &self,
        ins: DotGeneralOperands<Types>,
        _cx: &TypeCx<NoRegions>,
    ) -> Result<DotGeneralResults<OwnedTypes>> {
        let (lhs, rhs) = (ins.lhs, ins.rhs);
        // Integers multiply integers, and floats, narrow ones included, multiply floats.
        let family: &[Class] = match Class::of(lhs.dtype) {
            Class::Int => &[Class::Int],
            Class::Float | Class::Narrow => &[Class::Float, Class::Narrow],
            _ => &[Class::Int, Class::Float, Class::Narrow],
        };
        ensure_class("lhs", lhs, family)?;
        ensure_class("rhs", rhs, family)?;
        ensure!(
            lhs.dtype == rhs.dtype || (self.accum.is_some() && self.out.is_some()),
            "operands of different dtypes, {lhs} and {rhs}, need accum and out"
        );
        if let Some(accum) = self.accum {
            ensure_dtype_class("accum", accum, &family[..1])?;
        }
        if let Some(out) = self.out {
            ensure_dtype_class("out", out, family)?;
        }
        ensure!(
            self.batch_lhs.len() == self.batch_rhs.len(),
            "batch_lhs {:?} and batch_rhs {:?} differ in length",
            self.batch_lhs,
            self.batch_rhs
        );
        ensure!(
            self.contract_lhs.len() == self.contract_rhs.len(),
            "contract_lhs {:?} and contract_rhs {:?} differ in length",
            self.contract_lhs,
            self.contract_rhs
        );
        let axes = self.axes(lhs.rank(), rhs.rank())?;
        for (group, lhs_axes, rhs_axes) in [
            ("batch", &axes.batch_lhs, &axes.batch_rhs),
            ("contract", &axes.contract_lhs, &axes.contract_rhs),
        ] {
            for (&lhs_axis, &rhs_axis) in lhs_axes.iter().zip(rhs_axes) {
                ensure!(
                    lhs.dims[lhs_axis] == rhs.dims[rhs_axis],
                    "{group}_lhs axis {lhs_axis} has extent {}, but {group}_rhs axis {rhs_axis} has \
                     extent {}",
                    lhs.dims[lhs_axis],
                    rhs.dims[rhs_axis]
                );
            }
        }
        let lhs_result_axes = [&axes.batch_lhs[..], &axes.free(Side::Lhs)[..]].concat();
        let mut dims = Vec::new();
        for &axis in &lhs_result_axes {
            dims.push(lhs.dims[axis].clone());
        }
        for &axis in &axes.free(Side::Rhs) {
            dims.push(rhs.dims[axis].clone());
        }
        Ok(DotGeneralResults {
            output: TensorType {
                dtype: self.out_dtype(lhs.dtype),
                dims: ranked(dims)?,
            },
        })
    }
}

impl OpDef for ArgMax {
    fn result_types(
        &self,
        ins: ArgMaxOperands<Types>,
        _cx: &TypeCx<NoRegions>,
    ) -> Result<ArgMaxResults<OwnedTypes>> {
        let input = ins.input;
        ensure_class("the input", input, ARITHMETIC)?;
        ensure_index_dtype("indices", self.out)?;
        let axis = axis_index(self.axis, input.rank())?;
        ensure!(
            input.dims[axis] != Dim::Static(0),
            "axis {} of {input} is empty",
            self.axis
        );
        let mut dims = input.dims.clone();
        if self.keepdims {
            dims[axis] = Dim::Static(1);
        } else {
            dims.remove(axis);
        }
        Ok(ArgMaxResults {
            output: TensorType {
                dtype: self.out,
                dims,
            },
        })
    }
}

/// The type rule of every [`Reduce`] op.
pub(crate) fn reduce<T: Reduce>(
    op: &T,
    ins: ReduceOperands<Types>,
    _cx: &TypeCx<NoRegions>,
) -> Result<ReduceResults<OwnedTypes>> {
    let input = ins.input;
    ensure_reduction::<T::Combiner>(input, op.accum, op.out)?;
    let axes = op.reduced_axes(input.rank())?;
    Ok(ReduceResults {
        output: TensorType {
            dtype: op.out_dtype(input.dtype),
            dims: op.reduced_dims(&axes, &input.dims, Dim::Static(1)),
        },
    })
}

/// The type rule of every [`Segment`] op.
pub(crate) fn segment<T: Segment>(
    op: &T,
    ins: SegmentOperands<Types>,
    _cx: &TypeCx<NoRegions>,
) -> Result<SegmentResults<OwnedTypes>> {
    let (data, ids) = (ins.data, ins.segment_ids);
    ensure_reduction::<T::Combiner>(data, None, None)?;
    ensure!(
        data.rank() >= 1,
        "data must have rank at least 1, got {data}"
    );
    ensure!(
        matches!(ids.dtype, DType::Si32 | DType::Si64)
            && ids.rank() == 1
            && ids.dims[0] == data.dims[0],
        "segment ids must be si32 or si64 with one id per row of {data}, got {ids}"
    );
    let mut dims = data.dims.clone();
    dims[0] = Dim::Static(op.num_segments);
    Ok(SegmentResults {
        output: data.with_dims(dims),
    })
}

/// Checks the dtypes of a reduction that folds `input` elements with `C`, accumulated in `accum`
/// and returned as `out`: the input is of a class `C` computes on, and `accum` and `out` are of
/// the input's class.
pub(super) fn ensure_reduction<C: Combiner>(
    input: &TensorType,
    accum: Option<DType>,
    out: Option<DType>,
) -> Result<()> {
    ensure_class("the input", input, <C::Scalar as arith::Binary>::CLASSES)?;
    let class = [Class::of(input.dtype)];
    for (what, chosen) in [("accum", accum), ("out", out)] {
        if let Some(dtype) = chosen {
            ensure_dtype_class(what, dtype, &class)?;
        }
    }
    Ok(())
}

/// The type rule of every [`SelectK`] op.
pub(crate) fn select_k<T: SelectK>(
    op: &T,
    ins: SelectKOperands<Types>,
    _cx: &TypeCx<NoRegions>,
) -> Result<SelectKResults<OwnedTypes>> {
    let input = ins.input;
    ensure_class("the input", input, ARITHMETIC)?;
    ensure_index_dtype("indices", op.indices)?;
    let axis = axis_index(op.axis, input.rank())?;
    if let Some(extent) = input.static_dim(axis) {
        ensure!(
            op.k <= extent,
            "k = {} exceeds axis {} of {input}",
            op.k,
            op.axis
        );
    }
    let mut dims = input.dims.clone();
    dims[axis] = Dim::Static(op.k);
    Ok(SelectKResults {
        values: input.with_dims(dims.clone()),
        indices: TensorType {
            dtype: op.indices,
            dims,
        },
    })
}
