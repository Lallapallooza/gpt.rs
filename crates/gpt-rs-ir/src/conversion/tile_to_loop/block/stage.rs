//! The copies that the loops stage: an operand's elements that the loops inside a loop of the
//! nest read, copied into a buffer of the memory their holding names, in the order those loops
//! walk them.

use std::rc::Rc;

use anyhow::Result;

use super::nest::{clip_points, Tiling};
use super::scope::{Point, Ranges, Scope, Span};
use super::BlockLowering;
use crate::conversion::tile_to_loop::utils::{
    alloc, constant, each_point, for_range, index, offset_from, store,
};
use crate::dialects::tile::{CopyAxis, Source, Tail};
use crate::ir::{IndexExpr, Level, OperandRef, Space, Value, MAX_ITERS};
use crate::rewrite::Builder;

/// A staged copy of an operand: its buffer, its axes, and the span of each iterator around the
/// loop that makes it.
#[derive(Debug, Clone)]
pub(super) struct Staged {
    pub(super) buffer: Value,
    pub(super) axes: Vec<CopyAxis>,
    pub(super) around: Ranges,
}

/// A staged copy while its loops are built: the read it copies, its buffer and axes, the span of
/// each iterator that the loops around narrow it to, and the position in the copy so far.
struct StageCopy<'c> {
    read: usize,
    buffer: Value,
    axes: &'c [CopyAxis],
    spans: Ranges,
    at: Vec<Value>,
}

impl StageCopy<'_> {
    /// The span of the iterator at `position` that the loops of the copy stand in.
    fn span(&self, position: usize) -> Span {
        self.spans[position].expect("a copy spans each iterator of its block")
    }
}

impl BlockLowering<'_> {
    /// `scope` with the copies that the stages at loop `at` of the nest, or before it, make for
    /// the tiles of `scope`, which the reads inside take their elements from.
    pub(super) fn stage(
        &self,
        builder: &mut Builder,
        at: Option<usize>,
        scope: &Scope,
    ) -> Result<Scope> {
        let mut staged = scope.staged.clone();
        for copy in &self.walk.copies {
            if copy.at != at {
                continue;
            }
            let OperandRef(operand) = copy.operand;
            let space = self.copy_space(&copy.level);
            let buffer = self.copy_operand(builder, (operand, space), at, scope)?;
            staged[operand] = Some(Rc::new(buffer));
        }
        Ok(Scope {
            staged,
            ..scope.clone()
        })
    }

    /// The buffer space of a copy held for the memory `level`: the space that the dispatch's
    /// target gives that memory.
    fn copy_space(&self, level: &Level) -> Space {
        self.target
            .and_then(|target| target.holding_memory(level))
            .and_then(|memory| memory.space)
            .expect("the verifier holds a copy for a memory of the target with a buffer space")
    }

    /// The copy of operand `operand` in `space` that the stage at loop `at` makes for the tiles of
    /// `scope`.
    fn copy_operand(
        &self,
        builder: &mut Builder,
        (operand, space): (usize, Space),
        at: Option<usize>,
        scope: &Scope,
    ) -> Result<Staged> {
        let read = self
            .block
            .only_read_of(operand)
            .expect("the verifier holds a copy of an operand that one read takes alone");
        let position = self
            .block
            .reads
            .iter()
            .position(|other| other == read)
            .expect("the read is one of the block's");
        let dtype = self
            .module
            .ty(self.module.args(self.body)[position])
            .as_scalar()
            .expect("the verifier gives each read of a block a scalar");
        let axes = self.walk.copy_axes(self.block, read, at);
        let mut extents = Vec::with_capacity(axes.len());
        for axis in &axes {
            let extent = axis.extent().as_static();
            extents.push(extent.expect("the verifier keeps the axes of a copy static"));
        }
        let buffer = alloc(builder, &extents, dtype, space);
        let mut spans = [None; MAX_ITERS];
        for (iter, span) in spans.iter_mut().enumerate().take(self.block.iters.len()) {
            *span = Some(self.range(builder, scope, iter));
        }
        // The iterators by which a read picks its operand from a list stand at one point here,
        // the start of their span.
        let mut point = scope.point;
        if let Source::List { pick, .. } = &read.source {
            for iter in pick.iters() {
                if point[iter].is_none() {
                    point[iter] = spans[iter].map(|span| span.start);
                }
            }
        }
        let mut copy = StageCopy {
            read: position,
            buffer,
            axes: &axes,
            spans,
            at: Vec::new(),
        };
        self.copy_in(builder, scope, &mut copy, 0, &mut point)?;
        Ok(Staged {
            buffer,
            axes,
            around: scope.ranges,
        })
    }

    /// Copies what the read of `copy` gives at each point of its axes from the one at `depth` on
    /// into its buffer: a loop over the tiles of each loop of the nest, each clipped to the span
    /// around it, and over the points of each innermost tile.
    fn copy_in(
        &self,
        builder: &mut Builder,
        scope: &Scope,
        copy: &mut StageCopy,
        depth: usize,
        point: &mut Point,
    ) -> Result<()> {
        let Some(axis) = copy.axes.get(depth) else {
            let value = self.read_in_place(builder, scope, copy.read, point);
            store(builder, value, copy.buffer, copy.at.clone());
            return Ok(());
        };
        match *axis {
            CopyAxis::Tiles { index: nested, .. } => {
                self.copy_tiles(builder, scope, copy, depth, nested, point)
            }
            CopyAxis::Points {
                position,
                extent: ref dim,
            } => {
                let extent = dim.as_static().expect("a copy's axes are static");
                self.copy_points(builder, scope, copy, depth, (position, extent), point)
            }
        }
    }

    /// The copy's axis at `depth`, over the tiles of loop `nested` of the nest, each clipped to
    /// the span around it, with the last tile built as the nest's loop over the iterator builds
    /// it ([`crate::dialects::tile::Walk::tail`]).
    fn copy_tiles(
        &self,
        builder: &mut Builder,
        scope: &Scope,
        copy: &mut StageCopy,
        depth: usize,
        nested: usize,
        point: &mut Point,
    ) -> Result<()> {
        let entry = &self.walk.nest[nested];
        let position = entry.iter.position();
        let around = copy.span(position);
        let span = Span {
            points: None,
            ..around
        };
        let peel = self.walk.tail(self.block, position) == Tail::Peel;
        let tiling = Tiling::of(builder, span, entry.extent, peel);
        let zero = constant(builder, 0);
        tiling.each(
            builder,
            zero,
            tiling.tiles,
            &mut |loop_builder, tile, status| {
                copy.spans[position] = Some(tiling.span(loop_builder, tile, status));
                copy.at.push(tile);
                let copied = self.copy_in(loop_builder, scope, copy, depth + 1, point);
                copy.at.pop();
                copied
            },
        )?;
        copy.spans[position] = Some(around);
        Ok(())
    }

    /// The copy's axis at `depth`, over the points of iterator `position` in its innermost tile of
    /// `extent` points: the last axis is one `loop.atom` when the copy moves whole vectors.
    fn copy_points(
        &self,
        builder: &mut Builder,
        scope: &Scope,
        copy: &mut StageCopy,
        depth: usize,
        (position, extent): (usize, usize),
        point: &mut Point,
    ) -> Result<()> {
        let tile = copy.span(position);
        let last = depth + 1 == copy.axes.len();
        let read = &self.block.reads[copy.read];
        if self.walk.atom.is_some() && last && self.walk.copies_vectors(read, copy.axes) {
            let span = Span {
                points: None,
                ..tile
            };
            let atom = self.atom_box(builder, &[(position, span, extent)]);
            each_point(builder, &[], Some(&atom), &mut |atom_builder, offsets| {
                let value = index(
                    atom_builder,
                    IndexExpr::iter(0) + IndexExpr::iter(1),
                    &[tile.start, offsets[0]],
                );
                point[position] = Some(value);
                let element = self.read_in_place(atom_builder, scope, copy.read, point);
                let at = [&copy.at[..], &offsets[..1]].concat();
                store(atom_builder, element, copy.buffer, at);
                Ok(())
            })?;
            point[position] = None;
            return Ok(());
        }
        let zero = constant(builder, 0);
        let clipped = clip_points(builder, extent, tile.start, tile.end);
        for_range(builder, zero, clipped, |loop_builder, offset| {
            let value = index(
                loop_builder,
                IndexExpr::iter(0) + IndexExpr::iter(1),
                &[tile.start, offset],
            );
            point[position] = Some(value);
            copy.at.push(offset);
            let copied = self.copy_in(loop_builder, scope, copy, depth + 1, point);
            copy.at.pop();
            copied
        })?;
        point[position] = None;
        Ok(())
    }

    /// The position in `staged` of the element at the point of `scope`: the tile that each loop
    /// over tiles of the scope stands at, and the point's offset in the copy's innermost tile of
    /// each iterator: the scope's, inside the loops over the iterator's tiles that the copy has
    /// an axis for, or else the span of the iterator around the copy.
    pub(super) fn position_in_copy(
        &self,
        builder: &mut Builder,
        scope: &Scope,
        staged: &Staged,
    ) -> Vec<Value> {
        let mut tiled = [false; MAX_ITERS];
        let mut at = Vec::with_capacity(staged.axes.len());
        for axis in &staged.axes {
            let value = match *axis {
                CopyAxis::Tiles { index: nested, .. } => {
                    tiled[self.walk.nest[nested].iter.position()] = true;
                    scope.tiles[nested].expect("a copy is read inside the loops of its tiles")
                }
                CopyAxis::Points { position, .. } => {
                    let value = scope.point[position].expect("a point of the copy's tile");
                    let tile_start = match tiled[position] {
                        true => scope.starts[position],
                        false => staged.around[position].map(|span| span.start),
                    };
                    match tile_start {
                        Some(origin) => offset_from(builder, value, origin),
                        None => value,
                    }
                }
            };
            at.push(value);
        }
        at
    }
}
