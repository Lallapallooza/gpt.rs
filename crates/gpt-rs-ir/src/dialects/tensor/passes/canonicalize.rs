//! The canonicalization patterns of the `tensor` ops.

use std::marker::PhantomData;

use anyhow::Result;
use smallvec::SmallVec;

use crate::dialects::tensor::ir::shape::axis_index;
use crate::dialects::tensor::{
    ArgMax, BroadcastTo, Concat, DotGeneral, Gather, Iota, Reduce, ReduceWindow, Reshape, Scatter,
    Slice, TopK, Transpose,
};
use crate::module::Value;
use crate::op::{InOp, Matched};
use crate::rewrite::{pattern, Pattern, Rewriter};
use crate::types::DType;

/// `reshape(reshape(x))` is one reshape of `x`: both keep the row-major order.
#[pattern(canonical)]
struct CollapseReshapeChain;

impl Pattern for CollapseReshapeChain {
    type Root = Reshape;
    fn rewrite(&self, root: Matched<Reshape>, rw: &mut Rewriter<'_>) -> Result<bool> {
        let Some(first) = rw.def_as::<Reshape>(root.operands.input) else {
            return Ok(false);
        };
        let one = rw.insert_before(root.inst, root.attrs, first.operands.input)?;
        rw.replace(root.inst, &[one.output])?;
        Ok(true)
    }
}

/// `broadcast_to(broadcast_to(x, s1), s2)` is `broadcast_to(x, s2)`: every axis of `x` that
/// broadcasts to `s1` broadcasts to `s2` too.
#[pattern(canonical)]
struct CollapseBroadcastChain;

impl Pattern for CollapseBroadcastChain {
    type Root = BroadcastTo;
    fn rewrite(&self, root: Matched<BroadcastTo>, rw: &mut Rewriter<'_>) -> Result<bool> {
        let Some(first) = rw.def_as::<BroadcastTo>(root.operands.input) else {
            return Ok(false);
        };
        let one = rw.insert_before(root.inst, root.attrs, first.operands.input)?;
        rw.replace(root.inst, &[one.output])?;
        Ok(true)
    }
}

/// Steps of 1 on every axis are written as no strides.
#[pattern(canonical)]
struct UnitStrides;

impl Pattern for UnitStrides {
    type Root = Slice;
    fn rewrite(&self, root: Matched<Slice>, rw: &mut Rewriter<'_>) -> Result<bool> {
        let attrs = root.attrs;
        if attrs.strides.is_empty() || attrs.strides.iter().any(|&stride| stride != 1) {
            return Ok(false);
        }
        rw.set_op(root.inst, Slice::new(attrs.starts, attrs.sizes));
        Ok(true)
    }
}

/// `slice(slice(x, s1), s2)` is one slice of `x`: element `k` of the second slice is element
/// `s2.starts + k * s2.strides` of the first, which is element
/// `s1.starts + (s2.starts + k * s2.strides) * s1.strides` of `x`.
#[pattern(canonical)]
struct CollapseSliceChain;

impl Pattern for CollapseSliceChain {
    type Root = Slice;
    fn rewrite(&self, root: Matched<Slice>, rw: &mut Rewriter<'_>) -> Result<bool> {
        let Some(first) = rw.def_as::<Slice>(root.operands.input) else {
            return Ok(false);
        };
        let (outer, inner) = (&first.attrs, &root.attrs);
        let rank = inner.sizes.len();
        let starts = (0..rank)
            .map(|axis| outer.starts[axis] + inner.starts[axis] * outer.stride(axis))
            .collect();
        let strides: SmallVec<[usize; 4]> = (0..rank)
            .map(|axis| outer.stride(axis) * inner.stride(axis))
            .collect();
        let strides = if strides.iter().all(|&stride| stride == 1) {
            SmallVec::new()
        } else {
            strides
        };
        let one = Slice {
            starts,
            sizes: root.attrs.sizes,
            strides,
        };
        let one = rw.insert_before(root.inst, one, first.operands.input)?;
        rw.replace(root.inst, &[one.output])?;
        Ok(true)
    }
}

/// `transpose(transpose(x, p1), p2)` is `transpose(x, q)` with `q[i] = p1[p2[i]]`.
#[pattern(canonical)]
struct CollapseTransposeChain;

impl Pattern for CollapseTransposeChain {
    type Root = Transpose;
    fn rewrite(&self, root: Matched<Transpose>, rw: &mut Rewriter<'_>) -> Result<bool> {
        let Some(first) = rw.def_as::<Transpose>(root.operands.input) else {
            return Ok(false);
        };
        let perm = root
            .attrs
            .perm
            .iter()
            .map(|&axis| first.attrs.perm[axis])
            .collect();
        let composed = rw.insert_before(root.inst, Transpose { perm }, first.operands.input)?;
        rw.replace(root.inst, &[composed.output])?;
        Ok(true)
    }
}

/// `(out, accum)` without the dtypes an op on `input` elements uses by default: the input dtype
/// for `out`, and the input's default accumulator for `accum`. `None` when neither changes.
fn drop_defaults(
    out: Option<DType>,
    accum: Option<DType>,
    input: DType,
) -> Option<(Option<DType>, Option<DType>)> {
    let new_out = out.filter(|&d| d != input);
    let new_accum = accum.filter(|&d| d != input.default_accumulator());
    ((new_out, new_accum) != (out, accum)).then_some((new_out, new_accum))
}

/// The dtype of the elements of `value`.
fn dtype(rw: &Rewriter<'_>, value: Value) -> DType {
    rw.module.ty(value).as_tensor().expect("a tensor").dtype
}

/// An op with one `axis` attribute, an axis of its first operand, or of its result when it has
/// no operands.
trait OneAxis: InOp {
    fn axis_mut(&mut self) -> &mut i64;
}

/// A negative `axis` is written as the axis it counts back to.
#[pattern(canonical(ArgMax, TopK, Concat, Iota, Gather, Scatter))]
struct NonNegativeAxis<T>(PhantomData<T>);

impl<T: OneAxis> Pattern for NonNegativeAxis<T> {
    type Root = T;
    fn rewrite(&self, root: Matched<T>, rw: &mut Rewriter<'_>) -> Result<bool> {
        let mut op = root.attrs;
        let axis = *op.axis_mut();
        if axis >= 0 {
            return Ok(false);
        }
        let module = &*rw.module;
        let operands = module.operands(root.inst);
        let counted = operands.first().unwrap_or(&module.results(root.inst)[0]);
        let rank = module.ty(*counted).as_tensor().expect("a tensor").rank();
        *op.axis_mut() = axis_index(axis, rank)? as i64;
        rw.set_op(root.inst, op);
        Ok(true)
    }
}

macro_rules! one_axis {
    ($($op:ty),*) => {$(
        impl OneAxis for $op {
            fn axis_mut(&mut self) -> &mut i64 {
                &mut self.axis
            }
        }
    )*};
}

one_axis!(ArgMax, TopK, Concat, Iota, Gather, Scatter);

/// Negative batch and contract axes are written as the axes they count back to.
#[pattern(canonical)]
struct NonNegativeDotAxes;

impl Pattern for NonNegativeDotAxes {
    type Root = DotGeneral;
    fn rewrite(&self, root: Matched<DotGeneral>, rw: &mut Rewriter<'_>) -> Result<bool> {
        let op = &root.attrs;
        let lists = [
            &op.batch_lhs,
            &op.batch_rhs,
            &op.contract_lhs,
            &op.contract_rhs,
        ];
        if lists.iter().all(|axes| axes.iter().all(|&axis| axis >= 0)) {
            return Ok(false);
        }
        let rank = |value| rw.module.ty(value).as_tensor().expect("a tensor").rank();
        let axes = op.axes(rank(root.operands.lhs), rank(root.operands.rhs))?;
        let signed =
            |axes: SmallVec<[usize; 4]>| axes.into_iter().map(|axis| axis as i64).collect();
        rw.set_op(
            root.inst,
            DotGeneral {
                batch_lhs: signed(axes.batch_lhs),
                batch_rhs: signed(axes.batch_rhs),
                contract_lhs: signed(axes.contract_lhs),
                contract_rhs: signed(axes.contract_rhs),
                ..root.attrs
            },
        );
        Ok(true)
    }
}

/// An `out` equal to the operands' dtype, or an `accum` equal to their default accumulator, is left
/// out.
#[pattern(canonical)]
struct DropDefaultDotDtypes;

impl Pattern for DropDefaultDotDtypes {
    type Root = DotGeneral;
    fn rewrite(&self, root: Matched<DotGeneral>, rw: &mut Rewriter<'_>) -> Result<bool> {
        let input = dtype(rw, root.operands.lhs);
        let Some((out, accum)) = drop_defaults(root.attrs.out, root.attrs.accum, input) else {
            return Ok(false);
        };
        rw.set_op(
            root.inst,
            DotGeneral {
                out,
                accum,
                ..root.attrs
            },
        );
        Ok(true)
    }
}

/// The reduced axes are written non-negative and in increasing order.
#[pattern(canonical)]
struct SortedReduceAxes;

impl Pattern for SortedReduceAxes {
    type Root = Reduce;
    fn rewrite(&self, root: Matched<Reduce>, rw: &mut Rewriter<'_>) -> Result<bool> {
        let rank = rw
            .module
            .ty(root.operands.input)
            .as_tensor()
            .expect("a tensor")
            .rank();
        let axes: SmallVec<[i64; 4]> = root
            .attrs
            .reduced_axes(rank)?
            .into_iter()
            .map(|axis| axis as i64)
            .collect();
        if axes == root.attrs.axes {
            return Ok(false);
        }
        rw.set_op(root.inst, Reduce { axes, ..root.attrs });
        Ok(true)
    }
}

/// An `out` equal to the input's dtype, or an `accum` equal to its default accumulator, is left out.
#[pattern(canonical)]
struct DropDefaultReduceDtypes;

impl Pattern for DropDefaultReduceDtypes {
    type Root = Reduce;
    fn rewrite(&self, root: Matched<Reduce>, rw: &mut Rewriter<'_>) -> Result<bool> {
        let input = dtype(rw, root.operands.input);
        let Some((out, accum)) = drop_defaults(root.attrs.out, root.attrs.accum, input) else {
            return Ok(false);
        };
        rw.set_op(
            root.inst,
            Reduce {
                out,
                accum,
                ..root.attrs
            },
        );
        Ok(true)
    }
}

/// An `accum` equal to the input's default accumulator is left out.
#[pattern(canonical)]
struct DropDefaultWindowAccum;

impl Pattern for DropDefaultWindowAccum {
    type Root = ReduceWindow;
    fn rewrite(&self, root: Matched<ReduceWindow>, rw: &mut Rewriter<'_>) -> Result<bool> {
        let input = dtype(rw, root.operands.input);
        let Some((_, accum)) = drop_defaults(None, root.attrs.accum, input) else {
            return Ok(false);
        };
        rw.set_op(
            root.inst,
            ReduceWindow {
                accum,
                ..root.attrs
            },
        );
        Ok(true)
    }
}
