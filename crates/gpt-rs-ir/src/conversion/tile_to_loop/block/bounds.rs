//! Where the coordinates of padded reads stay inside their operands: the points of a loop that
//! keep a coordinate in range, which a clip narrows the loop to.

use super::scope::{Scope, Span};
use super::BlockLowering;
use crate::conversion::tile_to_loop::utils::{constant, extent, index};
use crate::dialects::tile::Source;
use crate::ir::{IndexExpr, Value};
use crate::rewrite::Builder;

/// The points `p` of an iterator, `first` up to `after`, where a coordinate that moves by `step`
/// with it stays in range: `base + step * p >= 0` and `base + step * p < bound`, `base` the
/// coordinate at `p = 0`.
fn in_range_points(builder: &mut Builder, base: Value, step: i64, bound: Value) -> (Value, Value) {
    let size = IndexExpr::constant(step.abs());
    match step > 0 {
        true => (
            index(
                builder,
                IndexExpr::floordiv(
                    IndexExpr::constant(step - 1) - IndexExpr::iter(0),
                    size.clone(),
                ),
                &[base],
            ),
            index(
                builder,
                IndexExpr::floordiv(
                    IndexExpr::iter(1) - IndexExpr::constant(1) - IndexExpr::iter(0),
                    size,
                ) + IndexExpr::constant(1),
                &[base, bound],
            ),
        ),
        false => (
            index(
                builder,
                IndexExpr::floordiv(
                    IndexExpr::iter(0) - IndexExpr::iter(1) + size.clone(),
                    size.clone(),
                ),
                &[base, bound],
            ),
            index(
                builder,
                IndexExpr::floordiv(IndexExpr::iter(0), size) + IndexExpr::constant(1),
                &[base],
            ),
        ),
    }
}

/// `first` and `after` narrowed by the points `(low, high)`: the larger start and the smaller end.
fn narrowed(
    builder: &mut Builder,
    (first, after): (Value, Value),
    (low, high): (Value, Value),
) -> (Value, Value) {
    let largest = IndexExpr::max(IndexExpr::iter(0), IndexExpr::iter(1));
    let narrowed_first = index(builder, largest, &[first, low]);
    let smallest = IndexExpr::min(IndexExpr::iter(0), IndexExpr::iter(1));
    (narrowed_first, index(builder, smallest, &[after, high]))
}

/// The points from `first` up to `after` kept within a span that ends at `end`: empty where no
/// point is in range.
fn within(builder: &mut Builder, first: Value, after: Value, end: Value) -> (Value, Value) {
    let start = index(
        builder,
        IndexExpr::min(IndexExpr::iter(0), IndexExpr::iter(1)),
        &[first, end],
    );
    let stop = index(
        builder,
        IndexExpr::max(IndexExpr::iter(0), IndexExpr::iter(1)),
        &[after, start],
    );
    (start, stop)
}

impl BlockLowering<'_> {
    /// The span of iterator `position` in `scope`, or its whole extent.
    pub(super) fn range(&self, builder: &mut Builder, scope: &Scope, position: usize) -> Span {
        if let Some(span) = scope.ranges[position] {
            return span;
        }
        let dim = &self.block.iters[position].extent;
        Span {
            start: constant(builder, 0),
            end: extent(builder, dim),
            points: dim.as_static(),
        }
    }

    /// `span`, the points of the iterator at `position`, narrowed to those where each of
    /// `coords`, coordinates `(read, axis)` that take it, stays inside its operand, every other
    /// iterator they take standing at its one point: in the scope's point, or the start of its
    /// span.
    pub(super) fn clip(
        &self,
        builder: &mut Builder,
        position: usize,
        span: Span,
        coords: &[(usize, usize)],
        scope: &Scope,
    ) -> Span {
        let mut points = (span.start, span.end);
        for &(read_index, axis) in coords {
            let read = &self.block.reads[read_index];
            let Source::Operand(operand) = read.source else {
                panic!("a clip narrows the loops for a read of one operand");
            };
            let coord = &read.coords[axis];
            let step = coord
                .coefficient(position)
                .expect("a clip narrows the loops for an affine coordinate");
            // The coordinate where the iterator is 0.
            let mut at = scope.point;
            for other in coord.iters() {
                if at[other].is_none() {
                    at[other] = Some(self.range(builder, scope, other).start);
                }
            }
            at[position] = Some(constant(builder, 0));
            let mut operands = Vec::new();
            let base_expr = self.lower_expr(builder, coord, &at, &mut operands);
            let base = index(builder, base_expr, &operands);
            let bound = extent(builder, &self.operands[operand].dims[axis]);
            let in_range = in_range_points(builder, base, step, bound);
            points = narrowed(builder, points, in_range);
        }
        let (start, end) = within(builder, points.0, points.1, span.end);
        Span {
            start,
            end,
            points: None,
        }
    }

    /// The fewest of `most` points of a tile of the iterator at `position` that [`Self::clip`]
    /// keeps for `coords`: each coordinate drops at most the points of the iterator's extent where
    /// its least value, over the extents of its other iterators, falls before its axis, and those
    /// where its greatest falls past it. One where an extent or a step is not known before the run.
    pub(super) fn least_kept(
        &self,
        position: usize,
        most: usize,
        coords: &[(usize, usize)],
    ) -> usize {
        let mut dropped = 0;
        for &coord in coords {
            let Some(points) = self.droppable(position, coord) else {
                return 1;
            };
            dropped += points.iter().filter(|&&drops| drops).count();
        }
        most.saturating_sub(dropped).max(1)
    }

    /// The most points that [`Self::clip`] cuts from the start and from the end of a tile of the
    /// iterator at `position` for `coords`: the longest runs of points each coordinate can drop at
    /// the start and at the end of the iterator's extent, where the points it keeps are one run.
    /// None where an extent or a step is not known before the run.
    pub(super) fn cut_ends(
        &self,
        position: usize,
        coords: &[(usize, usize)],
    ) -> Option<(usize, usize)> {
        let (mut first, mut last) = (0, 0);
        for &coord in coords {
            let points = self.droppable(position, coord)?;
            let leading = points.iter().take_while(|&&drops| drops).count();
            let trailing = points.iter().rev().take_while(|&&drops| drops).count();
            first = first.max(leading);
            last = last.max(trailing);
        }
        Some((first, last))
    }

    /// Whether the coordinate `(read, axis)` can fall outside its axis at each point of the
    /// iterator at `position`: where its bounds there, over the extents of its other iterators
    /// ([`crate::ir::Expr::bounds`]), reach before the axis or past it. None where an extent is
    /// not known before the run.
    fn droppable(&self, position: usize, (read_index, axis): (usize, usize)) -> Option<Vec<bool>> {
        let iters = &self.block.iters;
        let extent = iters[position].extent.as_static()?;
        let read = &self.block.reads[read_index];
        let Source::Operand(operand) = read.source else {
            return None;
        };
        let coord = &read.coords[axis];
        let bound = self.operands[operand].dims[axis].as_static()? as i64;
        let mut points = Vec::with_capacity(extent);
        for point in 0..extent as i64 {
            let range = |iter: usize| match iter == position {
                true => Some((point, point)),
                false => Some((0, iters.get(iter)?.extent.as_static()? as i64 - 1)),
            };
            let (low, high) = coord.bounds(&range)?;
            points.push(low < 0 || high >= bound);
        }
        Some(points)
    }
}
