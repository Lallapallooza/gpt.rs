use std::marker::PhantomData;

use super::utils::{along, cast_to, copy, fill};
use crate::dialects::arith;
use crate::dialects::tensor::utils::shape::axis_index;
use crate::dialects::tensor::{Accumulates, ArgMax, DotGeneral, Reduce, Segment, SelectK, Side};
use crate::dialects::tile::{par_iters, push_iter, Block, CombineKind, Fold, KBest};
use crate::index::{AccessExpr, IndexExpr, IterKind, Read, Write};
use crate::op::Matched;
use crate::rewrite::{pattern, NoMatch, Pattern, Rewriter};
use crate::scalar;
use crate::scalar::Combiner;
use crate::types::Dim;

/// `tensor.dot_general` is a block with one reduction iterator per contracted pair, in
/// `contract_lhs` order, that sums the products in the accumulator dtype; a parallel block casts
/// the sums to the result dtype.
#[pattern]
pub(super) struct DotToTile;

impl Pattern for DotToTile {
    type Root = DotGeneral;

    fn rewrite(
        &self,
        dot: Matched<'_, DotGeneral>,
        rewriter: &mut Rewriter<'_>,
    ) -> Result<(), NoMatch> {
        let (lhs, rhs) = (dot.lhs().tensor_type(), dot.rhs().tensor_type());
        let axes = dot.axes();
        let accum = dot.accum_dtype(lhs.dtype);
        let lhs_free = axes.free(Side::Lhs);
        let rhs_free = axes.free(Side::Rhs);
        let mut iters = Vec::new();
        let mut lhs_coords = vec![IndexExpr::constant(0); lhs.rank()];
        let mut rhs_coords = vec![IndexExpr::constant(0); rhs.rank()];
        let mut result_coords = Vec::new();
        for (&lhs_axis, &rhs_axis) in axes.batch_lhs.iter().zip(&axes.batch_rhs) {
            let batch = push_iter(&mut iters, IterKind::Par, lhs.dims[lhs_axis].clone());
            lhs_coords[lhs_axis] = batch.clone();
            rhs_coords[rhs_axis] = batch.clone();
            result_coords.push(batch);
        }
        for &lhs_axis in &lhs_free {
            let row = push_iter(&mut iters, IterKind::Par, lhs.dims[lhs_axis].clone());
            lhs_coords[lhs_axis] = row.clone();
            result_coords.push(row);
        }
        for &rhs_axis in &rhs_free {
            let column = push_iter(&mut iters, IterKind::Par, rhs.dims[rhs_axis].clone());
            rhs_coords[rhs_axis] = column.clone();
            result_coords.push(column);
        }
        for (&lhs_axis, &rhs_axis) in axes.contract_lhs.iter().zip(&axes.contract_rhs) {
            let term = push_iter(&mut iters, IterKind::Red, lhs.dims[lhs_axis].clone());
            lhs_coords[lhs_axis] = term.clone();
            rhs_coords[rhs_axis] = term;
        }
        let reads = vec![Read::new(0, lhs_coords), Read::new(1, rhs_coords)];
        let sum = Fold::new(CombineKind::Sum, vec![Write::new(result_coords)]);
        let block = Block::new(iters, reads, vec![sum]);
        let inputs = vec![dot.lhs().id(), dot.rhs().id()];
        let sums = block.build(rewriter, inputs, |body, args| {
            let lhs_term = body.cast(args[0], accum);
            let rhs_term = body.cast(args[1], accum);
            vec![body.create(arith::Mul, (lhs_term, rhs_term))]
        })[0];
        let output = dot.output().tensor_type();
        let result = cast_to(rewriter, sums, &output.with_dtype(accum), output.dtype);
        rewriter.replace_op(&dot, result);
        Ok(())
    }
}

/// A `tensor.reduce_*` op is a block whose reduced axes are reduction iterators, in axis order,
/// folding with the combine of its `arith` op in the accumulator dtype; a parallel block casts to
/// the result dtype.
#[pattern]
pub(super) struct ReduceToTile<T>(pub(super) PhantomData<T>);

impl<T: Reduce> Pattern for ReduceToTile<T> {
    type Root = T;

    fn rewrite(&self, reduce: Matched<'_, T>, rewriter: &mut Rewriter<'_>) -> Result<(), NoMatch> {
        let input = reduce.input().tensor_type();
        let axes = reduce.reduced_axes();
        let accum = reduce.accum_dtype(input.dtype);
        let (iters, coords, write) = along(input, &axes, reduce.keepdims);
        let fold = Fold::new(T::Combiner::COMBINE, vec![write]);
        let block = Block::new(iters, vec![Read::new(0, coords)], vec![fold]);
        let inputs = vec![reduce.input().id()];
        let folded = block.build(rewriter, inputs, |body, args| {
            vec![body.cast(args[0], accum)]
        })[0];
        let output = reduce.output().tensor_type();
        let result = cast_to(rewriter, folded, &output.with_dtype(accum), output.dtype);
        rewriter.replace_op(&reduce, result);
        Ok(())
    }
}

/// `tensor.argmax` is a block that folds its axis with `argmax` over each element and its
/// index.
#[pattern]
pub(super) struct ArgMaxToTile;

impl Pattern for ArgMaxToTile {
    type Root = ArgMax;

    fn rewrite(
        &self,
        argmax: Matched<'_, ArgMax>,
        rewriter: &mut Rewriter<'_>,
    ) -> Result<(), NoMatch> {
        let input = argmax.input().tensor_type();
        let axis = axis_index(argmax.axis, input.rank()).expect("a verified axis");
        let (iters, coords, write) = along(input, &[axis], argmax.keepdims);
        let position = coords[axis].clone();
        let index_dtype = argmax.out;
        let fold = Fold::new(CombineKind::Argmax, vec![write.clone(), write]);
        let block = Block::new(iters, vec![Read::new(0, coords)], vec![fold]);
        let inputs = vec![argmax.input().id()];
        let results = block.build(rewriter, inputs, |body, args| {
            vec![args[0], body.index(position, index_dtype)]
        });
        rewriter.replace_op(&argmax, results[1]);
        Ok(())
    }
}

/// `tensor.top_k` and `tensor.bottom_k` are a block that folds their axis with `top_k`, which
/// appends `k` as the last axis; for another axis, copy blocks move it back.
#[pattern]
pub(super) struct SelectKToTile<T>(pub(super) PhantomData<T>);

impl<T: SelectK> Pattern for SelectKToTile<T> {
    type Root = T;

    fn rewrite(&self, top_k: Matched<'_, T>, rewriter: &mut Rewriter<'_>) -> Result<(), NoMatch> {
        let input = top_k.input().tensor_type();
        let axis = axis_index(top_k.axis, input.rank()).expect("a verified axis");
        let (iters, coords, write) = along(input, &[axis], false);
        let position = coords[axis].clone();
        let index_dtype = top_k.indices;
        let combine = if T::LARGEST {
            CombineKind::TopK(KBest { k: top_k.k })
        } else {
            CombineKind::BottomK(KBest { k: top_k.k })
        };
        let fold = Fold::new(combine, vec![write.clone(), write]);
        let block = Block::new(iters, vec![Read::new(0, coords)], vec![fold]);
        let inputs = vec![top_k.input().id()];
        let found = block.build(rewriter, inputs, |body, args| {
            vec![args[0], body.index(position, index_dtype)]
        });
        if axis + 1 == input.rank() {
            rewriter.replace_op(&top_k, found);
            return Ok(());
        }
        // `top_k` appended `k` as the last axis; copy blocks move it back to `axis`.
        let mut dims = input.dims.clone();
        dims[axis] = Dim::Static(top_k.k);
        let mut moved = Vec::new();
        for result in found {
            let (iters, coords) = par_iters(&dims);
            let mut source = coords.clone();
            let top_k_coord = source.remove(axis);
            source.push(top_k_coord);
            moved.push(copy(
                rewriter,
                vec![result],
                iters,
                coords,
                Read::new(0, source),
            ));
        }
        rewriter.replace_op(&top_k, moved);
        Ok(())
    }
}

/// A `tensor.segment_*` op folds each row of `data` into a tensor filled with the initial value
/// of its fold, at the row its id names, in the accumulator dtype; a parallel block casts back.
#[pattern]
pub(super) struct SegmentToTile<T>(pub(super) PhantomData<T>);

impl<T: Segment> Pattern for SegmentToTile<T> {
    type Root = T;

    fn rewrite(&self, segment: Matched<'_, T>, rewriter: &mut Rewriter<'_>) -> Result<(), NoMatch> {
        let data = segment.data().tensor_type();
        let accum = data.dtype.default_accumulator();
        let mut dims = data.dims.clone();
        dims[0] = Dim::Static(segment.num_segments);
        let identity = scalar::fold_init::<T::Combiner>(accum).expect("a fold of the accumulator");
        let filled = fill(rewriter, &dims, identity);
        let (iters, coords) = par_iters(&data.dims);
        let mut target = vec![AccessExpr::read(1, [coords[0].clone()])];
        for coord in &coords[1..] {
            target.push(coord.clone().into());
        }
        let write = Write::update(2, target);
        let fold = Fold::new(T::Combiner::COMBINE, vec![write]);
        let block = Block::new(iters, vec![Read::new(0, coords)], vec![fold]);
        let inputs = vec![segment.data().id(), segment.segment_ids().id(), filled];
        let folded = block.build(rewriter, inputs, |body, args| {
            vec![body.cast(args[0], accum)]
        })[0];
        let output = segment.output().tensor_type();
        let result = cast_to(rewriter, folded, &output.with_dtype(accum), data.dtype);
        rewriter.replace_op(&segment, result);
        Ok(())
    }
}
