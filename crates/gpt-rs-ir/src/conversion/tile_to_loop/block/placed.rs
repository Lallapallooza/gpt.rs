//! The blocks placed in the loops of the first block of a run: each runs where the block before it
//! finishes the elements it reads, at the point of an element it reads alone, or over the piece
//! once the piece is finished, reading that block's elements there from scratch.

use std::rc::Rc;

use anyhow::Result;

use super::body::Input;
use super::plan::Node;
use super::scope::{Point, Scope};
use super::BlockLowering;
use crate::conversion::tile_to_loop::utils::{
    alloc, constant, each_point, index, offset_from, points_between, store,
};
use crate::dialects::tile::Block;
use crate::ir::{Dim, IndexExpr, Ranked, Space, Value, MAX_ITERS};
use crate::rewrite::Builder;

/// The elements of the result of a block of a run, kept for the block after it, `reader`, which
/// reads them across a piece of the run's loops: a buffer of private scratch with an axis for
/// each axis of the result that an iterator is the coordinate of, from the piece's start along it.
pub(super) struct Piece {
    reader: usize,
    pub(super) buffer: Value,
    /// For each axis of the buffer, the axis of the result it holds and the iterator that is the
    /// result's coordinate there.
    axes: Ranked<(usize, usize)>,
    /// Where the piece starts along each iterator that an axis of the buffer holds, by position.
    starts: Point,
}

impl Piece {
    /// The position in the piece's buffer of the element of the result at `coords`.
    pub(super) fn position(&self, builder: &mut Builder, coords: &[Value]) -> Vec<Value> {
        let mut at = Vec::with_capacity(self.axes.len());
        for &(axis, position) in &self.axes {
            let start = self.starts[position].expect("the piece starts along each of its axes");
            at.push(offset_from(builder, coords[axis], start));
        }
        at
    }
}

/// Where the last block of a run's blocks that run at one point puts its element: in the run's
/// result, at these coordinates, or in the piece that the next block reads across, at this
/// position.
enum Kept {
    Result(Value, Vec<Value>),
    Piece(Value, Vec<Value>),
}

impl BlockLowering<'_> {
    /// The block of the run at `index`: the first, at 0, then the blocks placed after it.
    fn run_block(&self, index: usize) -> &BlockLowering<'_> {
        match index {
            0 => self,
            _ => &self.placed[index - 1],
        }
    }

    /// Gives `value`, the element that block `index` of the run computes at the point of `scope`,
    /// to the blocks after it: each that runs at that point computes on the element of the block
    /// before it in turn, and the last of them puts its element into the run's result, or into
    /// the piece that the next block reads across.
    pub(super) fn give(
        &self,
        builder: &mut Builder,
        scope: &Scope,
        index: usize,
        value: Value,
    ) -> Result<()> {
        let mut last = index;
        while self.placed.get(last).is_some_and(|next| next.at_point) {
            last += 1;
        }
        let giver = self.run_block(last);
        let write = &giver.block.folds[0].writes[0];
        let coords = giver.coords(builder, &write.coords, &scope.point);
        let kept = match self.placed.get(last) {
            None => Kept::Result(giver.results[0], coords),
            Some(_) => {
                let piece = scope
                    .pieces
                    .iter()
                    .find(|piece| piece.reader == last + 1)
                    .expect("a block that reads across a piece has one");
                Kept::Piece(piece.buffer, piece.position(builder, &coords))
            }
        };
        let mut element = value;
        for next in &self.placed[index..last] {
            let next_scope = next.placed_scope(scope);
            element = next.point(builder, &next_scope, Input::Element(element))?[0];
        }
        match kept {
            Kept::Result(buffer, at) | Kept::Piece(buffer, at) => {
                store(builder, element, buffer, at)
            }
        }
        Ok(())
    }

    /// The scope of this block, one placed after the first, at the point of `scope`, the run's:
    /// it reads its other operands in place, with their pads.
    pub(super) fn placed_scope(&self, scope: &Scope) -> Scope {
        Scope {
            clips: Rc::default(),
            staged: vec![None; self.operands.len()],
            ..scope.clone()
        }
    }

    /// The nodes `inner` in a piece of the run's loops, whose tile of each iterator is `tiles`:
    /// the piece of each block placed after this one that reads across it, kept while `inner`
    /// computes the elements, then that block on each of its points in the piece, in order.
    pub(super) fn emit_piece(
        &self,
        builder: &mut Builder,
        tiles: &[Dim],
        inner: &[Node],
        scope: &Scope,
    ) -> Result<()> {
        let mut pieces = Vec::new();
        for (index, reader) in self.placed.iter().enumerate() {
            if reader.at_point {
                continue;
            }
            let giver = self.run_block(index);
            let write = &giver.block.folds[0].writes[0];
            let dtype = Block::yield_types(self.module, giver.body)?[0];
            let mut axes = Ranked::new();
            let mut starts = [None; MAX_ITERS];
            let mut extents: Ranked<usize> = Ranked::new();
            for (axis, coord) in write.coords.iter().enumerate() {
                let Some(position) = coord.as_iter() else {
                    continue;
                };
                axes.push((axis, position));
                starts[position] = Some(self.piece_start(builder, scope, position));
                let extent = tiles[position].as_static();
                extents.push(extent.expect("the verifier keeps a piece read across static"));
            }
            let buffer = alloc(builder, &extents, dtype, Space::Private);
            pieces.push(Piece {
                reader: index + 1,
                buffer,
                axes,
                starts,
            });
        }
        let in_piece = Scope {
            pieces: Rc::new(pieces),
            ..scope.clone()
        };
        self.emit(builder, inner, &in_piece)?;
        for piece in in_piece.pieces.iter() {
            let reader = &self.placed[piece.reader - 1];
            let positions = reader.block.iters.len();
            let mut starts = Vec::with_capacity(positions);
            let mut counts = Vec::with_capacity(positions);
            for position in 0..positions {
                let (start, count) = self.piece_span(builder, scope, position);
                starts.push(start);
                counts.push(count);
            }
            each_point(builder, &counts, None, &mut |point_builder, offsets| {
                let mut at = in_piece.clone();
                for (position, (&start, &offset)) in starts.iter().zip(offsets).enumerate() {
                    let value = index(
                        point_builder,
                        IndexExpr::iter(0) + IndexExpr::iter(1),
                        &[start, offset],
                    );
                    at.point[position] = Some(value);
                }
                let yielded = reader.point(
                    point_builder,
                    &reader.placed_scope(&at),
                    Input::Piece(piece),
                )?;
                self.give(point_builder, &at, piece.reader, yielded[0])
            })?;
        }
        Ok(())
    }

    /// Where the piece of the run's loops in `scope` starts along the iterator at `position`: at
    /// the point the loops around stand at, or the start of its span.
    fn piece_start(&self, builder: &mut Builder, scope: &Scope, position: usize) -> Value {
        match scope.point[position] {
            Some(point) => point,
            None => self.range(builder, scope, position).start,
        }
    }

    /// Where the piece of the run's loops in `scope` starts along the iterator at `position`, and
    /// how many of its points it holds: one, where the loops around stand at a point of it.
    fn piece_span(&self, builder: &mut Builder, scope: &Scope, position: usize) -> (Value, Value) {
        if let Some(point) = scope.point[position] {
            return (point, constant(builder, 1));
        }
        let span = self.range(builder, scope, position);
        (span.start, points_between(builder, span.start, span.end))
    }
}
