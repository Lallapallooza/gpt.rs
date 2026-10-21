//! The loops of a fold, as their plan's nodes say: a loop over the tiles of each loop of the
//! nest, or over its points for a loop of one point at a time, and the register tile as one
//! `loop.atom`, each narrowed where the fold's clips say, and the register tile cut at the points
//! of the loops where they cut it.

use std::rc::Rc;

use anyhow::Result;

use super::plan::Node;
use super::scope::{Scope, Span};
use super::BlockLowering;
use crate::conversion::tile_to_loop::utils::{
    constant, each_point, for_range, forall_range, if_then, index, points_between, predicate,
    AtomBox,
};
use crate::dialects::tile::{Parallelism, Step, Tail};
use crate::ir::{Constraint, IndexExpr, Level, Ranked, Relation, Value, MAX_GROUP_RANK};
use crate::rewrite::Builder;

/// The tile that a loop over tiles builds its body for: one of its full tiles; the last one,
/// which holds the points the full ones leave; or any tile, clipped to the span.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum TileStatus {
    Full,
    Rest,
    Clipped,
}

/// The last of the tiles of a span.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LastTile {
    /// Full, as every tile is.
    Full,
    /// Holding these points, fewer than a tile's.
    Partial(usize),
    /// Holding a count of points that is known only when the code runs, which may fill it.
    Unknown,
}

impl LastTile {
    /// The points of the last tile when it holds fewer than a tile's, known before the run.
    fn partial_points(self) -> Option<usize> {
        match self {
            LastTile::Partial(points) => Some(points),
            LastTile::Full | LastTile::Unknown => None,
        }
    }
}

/// The tiles of `size` points of a span: how many there are, what the last holds, and, when that
/// last tile builds a body of its own, how many tiles are full.
pub(super) struct Tiling {
    span: Span,
    size: usize,
    pub(super) tiles: Value,
    last: LastTile,
    /// The number of full tiles, when the last tile builds a body of its own: a register tile's
    /// extents follow from the points of the spans it covers.
    full: Option<Value>,
}

impl Tiling {
    /// The tiles of `size` points of `span`, whose last tile, when not full, builds a body of its
    /// own when `peel` holds.
    pub(super) fn of(builder: &mut Builder, span: Span, size: usize, peel: bool) -> Self {
        let size_expr = IndexExpr::constant(size as i64);
        let bounds = [span.start, span.end];
        let points_expr = IndexExpr::iter(1) - IndexExpr::iter(0);
        let (tiles, last) = match span.points {
            Some(points) => (
                constant(builder, points.div_ceil(size) as i64),
                match points % size {
                    0 => LastTile::Full,
                    rest => LastTile::Partial(rest),
                },
            ),
            None => {
                let rounded_up = points_expr.clone() + IndexExpr::constant(size as i64 - 1);
                let tiles = IndexExpr::floordiv(rounded_up, size_expr.clone());
                (index(builder, tiles, &bounds), LastTile::Unknown)
            }
        };
        let full = match (peel, last, span.points) {
            (false, _, _) | (true, LastTile::Full, _) => None,
            (true, _, Some(points)) => Some(constant(builder, (points / size) as i64)),
            (true, _, None) => Some(index(
                builder,
                IndexExpr::floordiv(points_expr, size_expr),
                &bounds,
            )),
        };
        Self {
            span,
            size,
            tiles,
            last,
            full,
        }
    }

    /// The status of every tile when one body serves them all: every tile is full, or the last
    /// is not peeled and each is clipped.
    fn one_body(&self) -> Option<TileStatus> {
        match (self.last, self.full) {
            (LastTile::Full, _) => Some(TileStatus::Full),
            (_, None) => Some(TileStatus::Clipped),
            (_, Some(_)) => None,
        }
    }

    /// The span of tile `tile`, whose status is `status`.
    pub(super) fn span(&self, builder: &mut Builder, tile: Value, status: TileStatus) -> Span {
        let size = self.size as i64;
        let start = index(
            builder,
            IndexExpr::iter(0) + IndexExpr::iter(1) * IndexExpr::constant(size),
            &[self.span.start, tile],
        );
        match status {
            TileStatus::Full => Span {
                start,
                end: index(
                    builder,
                    IndexExpr::iter(0) + IndexExpr::constant(size),
                    &[start],
                ),
                points: Some(self.size),
            },
            TileStatus::Rest => Span {
                start,
                end: self.span.end,
                points: self.last.partial_points(),
            },
            TileStatus::Clipped => Span {
                start,
                end: index(
                    builder,
                    IndexExpr::min(
                        IndexExpr::iter(0) + IndexExpr::constant(size),
                        IndexExpr::iter(1),
                    ),
                    &[start, self.span.end],
                ),
                points: None,
            },
        }
    }

    /// Builds `tile` for each of the tiles from `first` up to `end`: when every tile is full or
    /// the last is not peeled, one loop; else a loop over those that are full, then one over the
    /// last when it is among them.
    pub(super) fn each(
        &self,
        builder: &mut Builder,
        first: Value,
        end: Value,
        tile: &mut dyn FnMut(&mut Builder, Value, TileStatus) -> Result<()>,
    ) -> Result<()> {
        let Some(full) = self.full else {
            let status = self.one_body().expect("one body serves every tile");
            return for_range(builder, first, end, |loop_builder, value| {
                tile(loop_builder, value, status)
            });
        };
        let full_end = index(
            builder,
            IndexExpr::min(IndexExpr::iter(0), IndexExpr::iter(1)),
            &[end, full],
        );
        for_range(builder, first, full_end, |loop_builder, value| {
            tile(loop_builder, value, TileStatus::Full)
        })?;
        let rest_start = index(
            builder,
            IndexExpr::max(IndexExpr::iter(0), IndexExpr::iter(1)),
            &[first, full],
        );
        for_range(builder, rest_start, end, |loop_builder, value| {
            tile(loop_builder, value, TileStatus::Rest)
        })
    }
}

/// The points of a tile of `extent` points from `start` that lie before `end`.
pub(super) fn clip_points(builder: &mut Builder, extent: usize, start: Value, end: Value) -> Value {
    index(
        builder,
        IndexExpr::min(
            IndexExpr::constant(extent as i64),
            IndexExpr::iter(1) - IndexExpr::iter(0),
        ),
        &[start, end],
    )
}

/// A loop over tiles while its tiles are built: loop `nested` of the nest, over the iterator at
/// `iter`, the coordinates it cuts each tile to, its tail, its tiling, and the nodes inside it.
struct TileRun<'n, 'p> {
    nested: usize,
    iter: usize,
    cut: Option<&'n [(usize, usize)]>,
    tail: Tail,
    tiling: Tiling,
    inner: &'n [Node<'p>],
}

impl BlockLowering<'_> {
    /// Builds the first of `nodes`, which wraps the ones after it, from `scope`.
    pub(super) fn emit(&self, builder: &mut Builder, nodes: &[Node], scope: &Scope) -> Result<()> {
        let (node, inner) = nodes
            .split_first()
            .expect("the loops of a fold end in the body at a point");
        match node {
            Node::Step(Step::Tiles(nested)) => self.emit_tiles(builder, *nested, inner, scope),
            Node::Step(Step::Reg) => self.emit_register(builder, inner, scope),
            Node::State(state) => self.emit_state(builder, state, inner, scope),
            Node::Prefix(position) => self.emit_prefix(builder, *position, inner, scope),
            Node::Copies(fold) => self.emit_copies(builder, *fold, inner, scope),
            Node::Piece(tiles) => self.emit_piece(builder, tiles, inner, scope),
            Node::Body(leaf) => self.emit_leaf(builder, leaf, scope),
        }
    }

    /// Loop `nested` of the nest, over its tiles around `inner`: one after another, or one tile
    /// for each worker of its level. Over an iterator of the register tile, the fold's clips cut
    /// each tile. Over any other, they narrow the loop's span.
    fn emit_tiles(
        &self,
        builder: &mut Builder,
        nested: usize,
        inner: &[Node],
        scope: &Scope,
    ) -> Result<()> {
        let entry = &self.walk.nest[nested];
        let iter = entry.iter.position();
        let clipped = scope.clips.steps.get(&Step::Tiles(nested));
        let cuts = self.walk.in_reg(iter);
        if entry.extent == 1 && !(cuts && clipped.is_some()) {
            return self.emit_points(builder, nested, inner, scope);
        }
        let whole = self.range(builder, scope, iter);
        let (span, cut) = match (clipped, cuts) {
            (Some(coords), false) => (self.clip(builder, iter, whole, coords, scope), None),
            (Some(coords), true) => (whole, Some(coords.as_slice())),
            (None, _) => (whole, None),
        };
        let tail = self.walk.tail(self.block, iter);
        let tiles = TileRun {
            nested,
            iter,
            cut,
            tail,
            tiling: Tiling::of(builder, span, entry.extent, tail == Tail::Peel),
            inner,
        };
        if let Some(unit) = scope.entered[nested] {
            return self.enter_tile(builder, &tiles, unit, scope);
        }
        let Parallelism::Par(level) = &entry.parallelism else {
            let mut tile = |tile_builder: &mut Builder<'_>, tile: Value, status: TileStatus| {
                self.build_tile(tile_builder, &tiles, (tile, status), scope)
            };
            let zero = constant(builder, 0);
            return tiles
                .tiling
                .each(builder, zero, tiles.tiling.tiles, &mut tile);
        };
        let count = tiles.tiling.tiles;
        self.on_units(
            builder,
            level,
            (nested, count),
            (inner, scope),
            |unit_builder, unit, units| self.enter_tile(unit_builder, &tiles, unit, units),
        )
    }

    /// Tile `unit` of the loop over tiles `tiles`, alone: its body for its status, or a body for
    /// each status it can have.
    fn enter_tile(
        &self,
        builder: &mut Builder,
        tiles: &TileRun,
        unit: Value,
        scope: &Scope,
    ) -> Result<()> {
        let mut tile = |tile_builder: &mut Builder<'_>, tile: Value, status: TileStatus| {
            self.build_tile(tile_builder, tiles, (tile, status), scope)
        };
        if let Some(status) = tiles.tiling.one_body() {
            return tile(builder, unit, status);
        }
        let next = index(
            builder,
            IndexExpr::iter(0) + IndexExpr::constant(1),
            &[unit],
        );
        tiles.tiling.each(builder, unit, next, &mut tile)
    }

    /// Loop `nested` of the nest, of one point at a time, over the points of its span around
    /// `inner`, narrowed where the fold's clips say: one after another, or one for each worker of
    /// its level. It stands at each point, with the copies that the holdings make at it, and the
    /// register tile cut there where the clips cut it.
    fn emit_points(
        &self,
        builder: &mut Builder,
        nested: usize,
        inner: &[Node],
        scope: &Scope,
    ) -> Result<()> {
        let entry = &self.walk.nest[nested];
        let iter = entry.iter.position();
        let whole = self.range(builder, scope, iter);
        let span = match scope.clips.steps.get(&Step::Tiles(nested)) {
            Some(coords) => self.clip(builder, iter, whole, coords, scope),
            None => whole,
        };
        let clips = Rc::clone(&scope.clips);
        let cuts = clips.cuts.get(&Step::Tiles(nested));
        let point = |loop_builder: &mut Builder<'_>, value: Value, around: &Scope| {
            let at_point = around.at(iter, value);
            let with_copies = self.stage(loop_builder, Some(nested), &at_point)?;
            match cuts {
                Some(coords) => self.cut_register(loop_builder, coords, inner, &with_copies),
                None => self.emit(loop_builder, inner, &with_copies),
            }
        };
        let from_start = |point_builder: &mut Builder<'_>, unit: Value| {
            index(
                point_builder,
                IndexExpr::iter(0) + IndexExpr::iter(1),
                &[span.start, unit],
            )
        };
        if let Some(unit) = scope.entered[nested] {
            let value = from_start(builder, unit);
            return point(builder, value, scope);
        }
        match &entry.parallelism {
            Parallelism::Seq => for_range(builder, span.start, span.end, |loop_builder, value| {
                point(loop_builder, value, scope)
            }),
            Parallelism::Par(level) => {
                let count = points_between(builder, span.start, span.end);
                self.on_units(
                    builder,
                    level,
                    (nested, count),
                    (inner, scope),
                    |unit_builder, unit, units| {
                        let value = from_start(unit_builder, unit);
                        point(unit_builder, value, units)
                    },
                )
            }
        }
    }

    /// Loop `nested` of the nest, of `count` tiles or points on the workers of `level`, and the
    /// loops on the same workers that the nodes inside it start with, over other iterators and
    /// unclipped, as one `loop.forall` over the product of their counts. Each worker enters its
    /// index of the first loop through `enter`, and stands at its index of each other, which that
    /// loop builds alone when the nodes reach it ([`Scope::entered`]).
    fn on_units(
        &self,
        builder: &mut Builder,
        level: &Level,
        (nested, count): (usize, Value),
        (inner, scope): (&[Node], &Scope),
        enter: impl FnOnce(&mut Builder, Value, &Scope) -> Result<()>,
    ) -> Result<()> {
        let mut run = vec![(nested, count)];
        let mut iters = vec![self.walk.nest[nested].iter.position()];
        let mut next = nested + 1;
        while let Some(entry) = self.walk.nest.get(next) {
            let iter = entry.iter.position();
            let on_level = entry.parallelism == Parallelism::Par(level.clone());
            let clipped = scope.clips.steps.contains_key(&Step::Tiles(next));
            let right_inside = matches!(
                inner.get(next - nested - 1),
                Some(Node::Step(Step::Tiles(step))) if *step == next
            );
            if !on_level || clipped || iters.contains(&iter) || !right_inside {
                break;
            }
            run.push((next, self.unit_count(builder, next, scope)));
            iters.push(iter);
            next += 1;
        }
        let mut total = run[0].1;
        for &(_, later) in &run[1..] {
            total = index(
                builder,
                IndexExpr::iter(0) * IndexExpr::iter(1),
                &[total, later],
            );
        }
        let zero = constant(builder, 0);
        forall_range(builder, level.clone(), zero, total, |unit_builder, unit| {
            let mut units = scope.clone();
            let mut rest = unit;
            for &(later, later_count) in run[1..].iter().rev() {
                let at = index(
                    unit_builder,
                    IndexExpr::modulo(IndexExpr::iter(0), IndexExpr::iter(1)),
                    &[rest, later_count],
                );
                units.entered[later] = Some(at);
                rest = index(
                    unit_builder,
                    IndexExpr::floordiv(IndexExpr::iter(0), IndexExpr::iter(1)),
                    &[rest, later_count],
                );
            }
            enter(unit_builder, rest, &units)
        })
    }

    /// The tiles, or points, that loop `nested` of the nest walks in `scope`, unclipped.
    fn unit_count(&self, builder: &mut Builder, nested: usize, scope: &Scope) -> Value {
        let entry = &self.walk.nest[nested];
        let iter = entry.iter.position();
        let span = self.range(builder, scope, iter);
        if entry.extent == 1 {
            return points_between(builder, span.start, span.end);
        }
        let peel = self.walk.tail(self.block, iter) == Tail::Peel;
        Tiling::of(builder, span, entry.extent, peel).tiles
    }

    /// Tile `tile` of a loop over tiles, whose status is `status`: its span, cut to its points in
    /// range where the fold's clips cut a tile of the register tile.
    fn build_tile(
        &self,
        builder: &mut Builder,
        tiles: &TileRun,
        (tile, status): (Value, TileStatus),
        scope: &Scope,
    ) -> Result<()> {
        let tile_span = tiles.tiling.span(builder, tile, status);
        let Some(coords) = tiles.cut else {
            return self.in_tile(builder, tiles, (tile, tile_span), scope);
        };
        let kept = self.clip(builder, tiles.iter, tile_span, coords, scope);
        if tiles.tail != Tail::Peel {
            return self.in_tile(builder, tiles, (tile, kept), scope);
        }
        // The tile cut to its points in range, built once for each count they can have.
        let most = tile_span.points.unwrap_or(tiles.tiling.size);
        let least = match tile_span.points {
            Some(_) => self.least_kept(tiles.iter, most, coords),
            None => 1,
        };
        let count = points_between(builder, kept.start, kept.end);
        for points in least..=most {
            let equal = Constraint::cmp(
                IndexExpr::iter(0),
                Relation::Equal,
                IndexExpr::constant(points as i64),
            );
            let holds = predicate(builder, &[equal], &[count]);
            let counted = Span {
                points: Some(points),
                ..kept
            };
            if_then(builder, holds, |then_builder| {
                self.in_tile(then_builder, tiles, (tile, counted), scope)
            })?;
        }
        Ok(())
    }

    /// The nodes inside tile `tile`, of span `tile_span`, of a loop over tiles: the copies that
    /// the holdings make at its loop of the nest, then the nodes inside.
    fn in_tile(
        &self,
        builder: &mut Builder,
        tiles: &TileRun,
        (tile, tile_span): (Value, Span),
        scope: &Scope,
    ) -> Result<()> {
        let mut at_tile = scope.clone();
        at_tile.ranges[tiles.iter] = Some(tile_span);
        at_tile.tiles[tiles.nested] = Some(tile);
        at_tile.starts[tiles.iter] = Some(tile_span.start);
        let with_copies = self.stage(builder, Some(tiles.nested), &at_tile)?;
        self.emit(builder, tiles.inner, &with_copies)
    }

    /// The nodes `inner`, inside which the register tile runs on the points of its tile in the
    /// scope where each of `coords` stays in range ([`crate::dialects::tile::Walk::cut_step`]).
    /// They are built once for each count of points the cut can drop from the start of the tile
    /// and keep: the tile cut so starts a known count of points into the state's registers. Where
    /// the reach of a coordinate is not known before the run, the cut may drop any of the points.
    fn cut_register(
        &self,
        builder: &mut Builder,
        coords: &[(usize, usize)],
        inner: &[Node],
        scope: &Scope,
    ) -> Result<()> {
        let &(read, axis) = coords.first().expect("a cut has a coordinate");
        let lanes = self.walk.atom.as_ref().map(|atom| atom.iter.position());
        let position = self.block.reads[read].coords[axis]
            .iters()
            .into_iter()
            .find(|&taken| self.walk.in_reg(taken) && Some(taken) != lanes)
            .expect("a cut coordinate takes an iterator of the register tile");
        let tile = self.range(builder, scope, position);
        let points = tile
            .points
            .expect("a cut iterator's extent and its tiles' points are known before the run");
        let (first, last) = self.cut_ends(position, coords).unwrap_or((points, points));
        let kept = self.clip(builder, position, tile, coords, scope);
        let skipped = index(
            builder,
            IndexExpr::iter(0) - IndexExpr::iter(1),
            &[kept.start, tile.start],
        );
        let count = points_between(builder, kept.start, kept.end);
        for dropped in 0..=first.min(points) {
            let fewest = points.saturating_sub(dropped + last).max(1);
            for counted in fewest..=points - dropped {
                let holds = predicate(
                    builder,
                    &[
                        Constraint::cmp(
                            IndexExpr::iter(0),
                            Relation::Equal,
                            IndexExpr::constant(dropped as i64),
                        ),
                        Constraint::cmp(
                            IndexExpr::iter(1),
                            Relation::Equal,
                            IndexExpr::constant(counted as i64),
                        ),
                    ],
                    &[skipped, count],
                );
                if_then(builder, holds, |then_builder| {
                    let start = index(
                        then_builder,
                        IndexExpr::iter(0) + IndexExpr::constant(dropped as i64),
                        &[tile.start],
                    );
                    let end = index(
                        then_builder,
                        IndexExpr::iter(0) + IndexExpr::constant(counted as i64),
                        &[start],
                    );
                    let mut cut = scope.clone();
                    cut.ranges[position] = Some(Span {
                        start,
                        end,
                        points: Some(counted),
                    });
                    self.emit(then_builder, inner, &cut)
                })?;
            }
        }
        Ok(())
    }

    /// Builds the register tile as one `loop.atom` over the innermost tile of each of its
    /// iterators in the scope, clipped to its range, and inside it the nodes `inner`.
    fn emit_register(&self, builder: &mut Builder, inner: &[Node], scope: &Scope) -> Result<()> {
        let mut spans = Vec::with_capacity(self.walk.reg.len());
        for dim in &self.walk.reg {
            let position = dim.iter.position();
            spans.push((position, self.range(builder, scope, position), dim.extent));
        }
        let atom = self.atom_box(builder, &spans);
        each_point(builder, &[], Some(&atom), &mut |atom_builder, offsets| {
            let mut in_atom = scope.clone();
            for (&(position, span, _), &offset) in spans.iter().zip(offsets) {
                let value = index(
                    atom_builder,
                    IndexExpr::iter(0) + IndexExpr::iter(1),
                    &[span.start, offset],
                );
                in_atom.point[position] = Some(value);
            }
            self.emit(atom_builder, inner, &in_atom)
        })
    }

    /// The `loop.atom` over the tiles `spans`, each the span of the iterator at a position and the
    /// points of its tile, in vectors of the register tile's lanes along the last.
    pub(super) fn atom_box(
        &self,
        builder: &mut Builder,
        spans: &[(usize, Span, usize)],
    ) -> AtomBox {
        let lanes = self.lanes();
        let last = spans.len() - 1;
        let mut atom = AtomBox {
            tile: Ranked::new(),
            lanes,
            counts: [None; MAX_GROUP_RANK],
        };
        for (axis, &(_, span, extent)) in spans.iter().enumerate() {
            let lane_axis = (axis == last).then_some(lanes);
            let (points, count) = self.atom_axis(builder, span, extent, lane_axis);
            atom.tile.push(points);
            atom.counts[axis] = Some(count);
        }
        atom
    }

    /// The lanes of the register tile's vectors.
    pub(super) fn lanes(&self) -> usize {
        self.walk
            .atom
            .as_ref()
            .expect("a register tile has an atom")
            .lanes
    }

    /// The extent of the register tile's axis over `span`, whose tile holds `extent` points, and
    /// the value of the points it computes: the span's count when it is known, rounded up to
    /// whole vectors along `lanes`, the lane axis; else `extent`, clipped to the span.
    pub(super) fn atom_axis(
        &self,
        builder: &mut Builder,
        span: Span,
        extent: usize,
        lanes: Option<usize>,
    ) -> (usize, Value) {
        let Some(points) = span.points else {
            return (extent, clip_points(builder, extent, span.start, span.end));
        };
        assert!(
            points <= extent,
            "the verifier holds no more points in a tile of the register tile's iterator than the \
             register tile, not {points} in {extent}"
        );
        let tile = match lanes {
            Some(lane_count) => points.div_ceil(lane_count) * lane_count,
            None => points,
        };
        (tile, constant(builder, points as i64))
    }
}
