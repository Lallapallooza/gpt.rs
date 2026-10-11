use std::marker::PhantomData;

use super::utils::{cast_to, copy};
use crate::dialects::tensor::{Accumulates, ExtractPatches, ReduceWindow};
use crate::dialects::tile::{par_iters, push_iter, Block, Fold};
use crate::index::{Constraint, IndexExpr, IterKind, Read, Relation, Write};
use crate::op::Matched;
use crate::require;
use crate::rewrite::{pattern, NoMatch, Pattern, Rewriter};
use crate::scalar::Combiner;
use crate::types::Dim;

/// `tensor.extract_patches` is a copy block that reads channel `c` of window element `k` at
/// position `o` into column `k * C + c`, padded outside the input.
#[pattern]
pub(super) struct ExtractPatchesToTile;

impl Pattern for ExtractPatchesToTile {
    type Root = ExtractPatches;

    fn rewrite(
        &self,
        patches: Matched<'_, ExtractPatches>,
        rewriter: &mut Rewriter<'_>,
    ) -> Result<(), NoMatch> {
        let input = patches.input().tensor_type();
        let output = patches.output().tensor_type();
        let rank = input.rank();
        let channels = IndexExpr::dim(&input.dims[rank - 1]);
        let (iters, coords) = par_iters(&output.dims);
        let column = coords[rank - 1].clone();
        let window_element = IndexExpr::floordiv(column.clone(), channels.clone());
        let window: Vec<Dim> = patches
            .window
            .iter()
            .map(|&extent| Dim::Static(extent))
            .collect();
        let offsets = window_element.delinearize(&window);
        let mut source = vec![coords[0].clone()];
        for (axis, offset) in offsets.into_iter().enumerate() {
            let position =
                coords[1 + axis].clone() * IndexExpr::constant(patches.strides[axis] as i64);
            let shift = offset * IndexExpr::constant(patches.dilation[axis] as i64);
            let low_padding = IndexExpr::constant(patches.padding[axis].0 as i64);
            source.push(((position + shift) - low_padding).simplify());
        }
        source.push(IndexExpr::modulo(column, channels));
        let read = Read {
            pad: Some(patches.pad_value),
            ..Read::new(0, source)
        };
        let inputs = vec![patches.input().id()];
        let result = copy(rewriter, inputs, iters, coords, read);
        rewriter.replace_op(&patches, result);
        Ok(())
    }
}

/// A `tensor.reduce_window_*` op is a block with the window offsets as reduction iterators after
/// the positions, folding with the combine of its `arith` op. A window element in the padding or
/// between dilated input elements fails one of the block's constraints.
#[pattern]
pub(super) struct ReduceWindowToTile<T>(pub(super) PhantomData<T>);

impl<T: ReduceWindow> Pattern for ReduceWindowToTile<T> {
    type Root = T;

    fn rewrite(&self, window: Matched<'_, T>, rewriter: &mut Rewriter<'_>) -> Result<(), NoMatch> {
        let input = window.input().tensor_type();
        let output = window.output().tensor_type();
        let accum = window.accum_dtype(input.dtype);
        require!(let Some(extents) = input.static_dims(), "{input} has a dynamic extent");
        let (mut iters, positions) = par_iters(&output.dims);
        let mut source = Vec::new();
        let mut constraints = Vec::new();
        for (axis, position) in positions.iter().enumerate() {
            let window_offset = push_iter(
                &mut iters,
                IterKind::Red,
                Dim::Static(window.window_dims[axis]),
            );
            let start = position.clone() * IndexExpr::constant(window.strides[axis] as i64);
            let offset = window_offset * IndexExpr::constant(window.window_dilation[axis] as i64);
            let low_padding = IndexExpr::constant(window.padding[axis].0 as i64);
            let dilated = ((start + offset) - low_padding).simplify();
            let dilation = window.base_dilation[axis] as i64;
            let (low, high) = window.padding[axis];
            if low > 0 {
                let zero = IndexExpr::constant(0);
                constraints.push(Constraint::cmp(zero, Relation::LessEqual, dilated.clone()));
            }
            if high > 0 {
                let last = IndexExpr::constant((extents[axis] as i64 - 1).max(0) * dilation);
                constraints.push(Constraint::cmp(dilated.clone(), Relation::LessEqual, last));
            }
            if dilation > 1 {
                let remainder = IndexExpr::modulo(dilated.clone(), IndexExpr::constant(dilation));
                let zero = IndexExpr::constant(0);
                constraints.push(Constraint::cmp(remainder, Relation::Equal, zero));
                source.push(IndexExpr::floordiv(dilated, IndexExpr::constant(dilation)));
            } else {
                source.push(dilated);
            }
        }
        let fold = Fold::new(T::Combiner::COMBINE, vec![Write::new(positions)]);
        let block = Block::new(iters, vec![Read::new(0, source)], vec![fold]).restrict(constraints);
        let inputs = vec![window.input().id()];
        let folded = block.build(rewriter, inputs, |body, args| {
            vec![body.cast(args[0], accum)]
        })[0];
        let result = cast_to(rewriter, folded, &output.with_dtype(accum), output.dtype);
        rewriter.replace_op(&window, result);
        Ok(())
    }
}
