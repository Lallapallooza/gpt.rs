use std::marker::PhantomData;

use super::utils::{access, cast_to, copy};
use crate::dialects::arith::math::Combiner;
use crate::dialects::tensor::utils::shape::axis_index;
use crate::dialects::tensor::{
    DynamicSlice, DynamicUpdateSlice, Gather, Iota, Scatter, ScatterReplace, Take,
};
use crate::dialects::tile::{par_iters, Block, CombineKind, Fold, Read, Write};
use crate::ir::{AccessExpr, Dim, Matched, TensorType};
use crate::require;
use crate::rewrite::{pattern, NoMatch, Pattern, Rewriter};

/// `tensor.iota` is a block that yields `tile.index` of its axis.
#[pattern]
pub(super) struct IotaToTile;

impl Pattern for IotaToTile {
    type Root = Iota;

    fn rewrite(&self, iota: Matched<'_, Iota>, rewriter: &mut Rewriter<'_>) -> Result<(), NoMatch> {
        let axis = axis_index(iota.axis, iota.shape.len()).expect("a verified axis");
        let (iters, coords) = par_iters(&iota.shape);
        let position = coords[axis].clone();
        let dtype = iota.dtype;
        let block = Block::new(iters, Vec::new(), vec![Fold::plain(Write::new(coords))]);
        let result = block.build(rewriter, Vec::new(), |body, _| {
            vec![body.index(position, dtype)]
        })[0];
        rewriter.replace_op(&iota, result);
        Ok(())
    }
}

/// `tensor.take` is a copy block that reads the table at the row its index operand names.
#[pattern]
pub(super) struct TakeToTile;

impl Pattern for TakeToTile {
    type Root = Take;

    fn rewrite(&self, take: Matched<'_, Take>, rewriter: &mut Rewriter<'_>) -> Result<(), NoMatch> {
        let output = take.output().tensor_type();
        let index_rank = take.indices().tensor_type().rank();
        let (iters, coords) = par_iters(&output.dims);
        let mut source = vec![AccessExpr::read(1, coords[..index_rank].to_vec())];
        source.extend(access(&coords[index_rank..]));
        let inputs = vec![take.table().id(), take.indices().id()];
        let result = copy(rewriter, inputs, iters, coords, Read::new(0, source));
        rewriter.replace_op(&take, result);
        Ok(())
    }
}

/// `tensor.gather` is a copy block whose coordinate on the axis is the index operand's element.
#[pattern]
pub(super) struct GatherToTile;

impl Pattern for GatherToTile {
    type Root = Gather;

    fn rewrite(
        &self,
        gather: Matched<'_, Gather>,
        rewriter: &mut Rewriter<'_>,
    ) -> Result<(), NoMatch> {
        let input = gather.input().tensor_type();
        let output = gather.output().tensor_type();
        let axis = axis_index(gather.axis, input.rank()).expect("a verified axis");
        let (iters, coords) = par_iters(&output.dims);
        let mut source = access(&coords);
        source[axis] = AccessExpr::read(1, coords.clone());
        let inputs = vec![gather.input().id(), gather.indices().id()];
        let result = copy(rewriter, inputs, iters, coords, Read::new(0, source));
        rewriter.replace_op(&gather, result);
        Ok(())
    }
}

/// The clamped start of a dynamic slice of `size` along `axis` of `input`, whose starts are
/// operand `starts`: `min(max(starts[axis], 0), extent - size)`.
fn clamped_start(input: &TensorType, starts: usize, axis: usize, size: usize) -> AccessExpr {
    let start = AccessExpr::read(starts, [AccessExpr::constant(axis as i64)]);
    let last = AccessExpr::dim(&input.dims[axis]) - AccessExpr::constant(size as i64);
    AccessExpr::min(
        AccessExpr::max(start, AccessExpr::constant(0)),
        last.simplify(),
    )
}

/// `tensor.dynamic_slice` is a copy block from the clamped start,
/// `min(max(start, 0), extent - size)`.
#[pattern]
pub(super) struct DynamicSliceToTile;

impl Pattern for DynamicSliceToTile {
    type Root = DynamicSlice;

    fn rewrite(
        &self,
        slice: Matched<'_, DynamicSlice>,
        rewriter: &mut Rewriter<'_>,
    ) -> Result<(), NoMatch> {
        let input = slice.input().tensor_type();
        let dims: Vec<Dim> = slice.sizes.iter().map(|&size| Dim::Static(size)).collect();
        let (iters, coords) = par_iters(&dims);
        let mut source = Vec::new();
        for (axis, coord) in coords.iter().enumerate() {
            source.push(clamped_start(input, 1, axis, slice.sizes[axis]) + coord.clone().into());
        }
        let inputs = vec![slice.input().id(), slice.starts().id()];
        let result = copy(rewriter, inputs, iters, coords, Read::new(0, source));
        rewriter.replace_op(&slice, result);
        Ok(())
    }
}

/// `tensor.dynamic_update_slice` is a block that writes the update into the base at the clamped
/// start.
#[pattern]
pub(super) struct DynamicUpdateSliceToTile;

impl Pattern for DynamicUpdateSliceToTile {
    type Root = DynamicUpdateSlice;

    fn rewrite(
        &self,
        update_slice: Matched<'_, DynamicUpdateSlice>,
        rewriter: &mut Rewriter<'_>,
    ) -> Result<(), NoMatch> {
        let base = update_slice.base().tensor_type();
        let update = update_slice.update().tensor_type();
        require!(let Some(sizes) = update.static_dims(), "the update has a dynamic extent");
        let (iters, coords) = par_iters(&update.dims);
        let mut target = Vec::new();
        for (axis, coord) in coords.iter().enumerate() {
            target.push(clamped_start(base, 2, axis, sizes[axis]) + coord.clone().into());
        }
        let write = Write::update(0, target);
        let block = Block::new(iters, vec![Read::new(1, coords)], vec![Fold::plain(write)]);
        let inputs = vec![
            update_slice.base().id(),
            update_slice.update().id(),
            update_slice.starts().id(),
        ];
        let result = block.build(rewriter, inputs, |_, args| vec![args[0]])[0];
        rewriter.replace_op(&update_slice, result);
        Ok(())
    }
}

/// A `tensor.scatter_*` op that folds is a block that folds each update into the input at the
/// element its index names, in row-major order of the updates, with the combine of its `arith`
/// op, in the accumulator dtype between cast blocks.
#[pattern]
pub(super) struct ScatterToTile<T>(pub(super) PhantomData<T>);

impl<T: Scatter> Pattern for ScatterToTile<T> {
    type Root = T;

    fn rewrite(&self, scatter: Matched<'_, T>, rewriter: &mut Rewriter<'_>) -> Result<(), NoMatch> {
        let input = scatter.input().tensor_type();
        let accum = input.dtype.default_accumulator();
        let base = cast_to(rewriter, scatter.input().id(), input, accum);
        let updates = scatter.updates().tensor_type();
        let block = scatter_block(scatter.axis, input, updates, T::Combiner::COMBINE);
        let inputs = vec![base, scatter.indices().id(), scatter.updates().id()];
        let folded = block.build(rewriter, inputs, |body, args| {
            vec![body.cast(args[0], accum)]
        })[0];
        let result = cast_to(rewriter, folded, &input.with_dtype(accum), input.dtype);
        rewriter.replace_op(&scatter, result);
        Ok(())
    }
}

/// `tensor.scatter_replace` is a block that writes each update into the input at the element its
/// index names, in row-major order of the updates, so the last one wins.
#[pattern]
pub(super) struct ScatterReplaceToTile;

impl Pattern for ScatterReplaceToTile {
    type Root = ScatterReplace;

    fn rewrite(
        &self,
        scatter: Matched<'_, ScatterReplace>,
        rewriter: &mut Rewriter<'_>,
    ) -> Result<(), NoMatch> {
        let input = scatter.input().tensor_type();
        let updates = scatter.updates().tensor_type();
        let block = scatter_block(scatter.axis, input, updates, CombineKind::None);
        let inputs = vec![
            scatter.input().id(),
            scatter.indices().id(),
            scatter.updates().id(),
        ];
        let written = block.build(rewriter, inputs, |_, args| vec![args[0]])[0];
        rewriter.replace_op(&scatter, written);
        Ok(())
    }
}

/// The block of a scatter along `axis` of `input`: it reads update `i` of operand 2 and writes it
/// into operand 0 at the coordinate that operand 1, the indices, names.
fn scatter_block(
    axis: i64,
    input: &TensorType,
    updates: &TensorType,
    combine: CombineKind,
) -> Block {
    let axis = axis_index(axis, input.rank()).expect("a verified axis");
    let (iters, coords) = par_iters(&updates.dims);
    let mut target = access(&coords);
    target[axis] = AccessExpr::read(1, coords.clone());
    let write = Write::update(0, target);
    Block::new(
        iters,
        vec![Read::new(2, coords)],
        vec![Fold::new(combine, vec![write])],
    )
}
