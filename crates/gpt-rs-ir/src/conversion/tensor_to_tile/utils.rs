//! Blocks the patterns of `tensor-to-tile` build: elementwise maps, casts, copies, fills, and the
//! iterators of a reduction.

use crate::dialects::tile::{par_iters, Block, Body, Fold, Inputs, Iter, IterKind, Read, Write};
use crate::host::HostScalar;
use crate::ir::{AccessExpr, DType, Dim, IndexExpr, TensorType, Value};
use crate::rewrite::Rewriter;

/// One parallel block over the extents of `output` that reads each of `inputs` at the same point
/// and computes `compute` on the values.
pub(super) fn elementwise(
    rewriter: &mut Rewriter<'_>,
    inputs: &[Value],
    output: &TensorType,
    compute: impl FnOnce(&mut Body<'_, '_>, &[Value]) -> Value,
) -> Value {
    let (iters, coords) = par_iters(&output.dims);
    let mut reads = Inputs::new(Vec::new());
    for &input in inputs {
        reads.read(input, coords.clone());
    }
    let block = Block::new(iters, reads.reads, vec![Fold::plain(Write::new(coords))]);
    block.build(rewriter, reads.values, |body, args| {
        vec![compute(body, args)]
    })[0]
}

/// `value`, of type `value_type`, cast to `to` elementwise, or `value` itself when it is a `to`
/// already.
pub(super) fn cast_to(
    rewriter: &mut Rewriter<'_>,
    value: Value,
    value_type: &TensorType,
    to: DType,
) -> Value {
    if value_type.dtype == to {
        return value;
    }
    let output = value_type.with_dtype(to);
    elementwise(rewriter, &[value], &output, |body, args| {
        body.cast(args[0], to)
    })
}

/// A copy block: over `iters`, it writes at `coords` the element that `read` gives.
pub(super) fn copy(
    rewriter: &mut Rewriter<'_>,
    inputs: Vec<Value>,
    iters: Vec<Iter>,
    coords: Vec<IndexExpr>,
    read: Read,
) -> Value {
    let block = Block::new(iters, vec![read], vec![Fold::plain(Write::new(coords))]);
    block.build(rewriter, inputs, |_, args| vec![args[0]])[0]
}

/// The coordinates `coords`, in the form of an access, to which reads of operands can be added.
pub(super) fn access(coords: &[IndexExpr]) -> Vec<AccessExpr> {
    let mut accesses = Vec::with_capacity(coords.len());
    for coord in coords {
        accesses.push(coord.clone().into());
    }
    accesses
}

/// A block filling a tensor of `dims` with `value`.
pub(super) fn fill(rewriter: &mut Rewriter<'_>, dims: &[Dim], value: HostScalar) -> Value {
    let (iters, coords) = par_iters(dims);
    let block = Block::new(iters, Vec::new(), vec![Fold::plain(Write::new(coords))]);
    block.build(rewriter, Vec::new(), |body, _| vec![body.lit(value)])[0]
}

/// Iterators over the axes of `input` with `axes` reductions, the coordinates naming them, and
/// the write of a result without those axes, or with each as `0` under `keepdims`.
pub(super) fn along(
    input: &TensorType,
    axes: &[usize],
    keepdims: bool,
) -> (Vec<Iter>, Vec<IndexExpr>, Write) {
    let (mut iters, coords) = par_iters(&input.dims);
    let mut kept = Vec::new();
    for (axis, coord) in coords.iter().enumerate() {
        if !axes.contains(&axis) {
            kept.push(coord.clone());
            continue;
        }
        iters[axis].kind = IterKind::Red;
        if keepdims {
            kept.push(IndexExpr::constant(0));
        }
    }
    (iters, coords, Write::new(kept))
}
