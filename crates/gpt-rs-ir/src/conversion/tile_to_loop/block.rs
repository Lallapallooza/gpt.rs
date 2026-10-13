//! A `tile.block` in a `loop.func`: each of its folds as a loop nest that runs the block's reads,
//! constraints and body at each point.

use std::collections::HashMap;
use std::ops::Range;

use anyhow::{anyhow, bail, Result};
use smallvec::SmallVec;

use super::combine::{fold_of, FoldLowering, State};
use super::utils::{
    alloc, constant, extent, for_extent, for_range, forall_range, if_then, index, load, predicate,
    store,
};
use crate::dialects::arith;
use crate::dialects::tile::{
    Block, CombineKind, Fold, Index, IterKind, Parallelism, Read, Schedule, Source, Step,
};
use crate::ir::{
    AccessExpr, Cmp, Constraint, DType, Dim, IndexExpr, Module, OperandRef, Region, Relation, Root,
    Value,
};
use crate::rewrite::Builder;
use crate::traits::Terminator;

/// The value of each iterator in scope, by position.
type Point = Vec<Option<Value>>;

/// The tile of each iterator that the loops around have narrowed it to, as its start and end, or
/// none for its whole extent.
type Ranges = Vec<Option<(Value, Value)>>;

/// What a loop over the elements of a state builds at each: from the builder, the iterators'
/// values and the element's position in the state.
type ElementBody<'a> = dyn FnMut(&mut Builder<'_>, &mut Point, Vec<Value>) -> Result<()> + 'a;

/// A tile of a parallel iterator that a fold's state covers: the iterator, the tile's extent, and
/// its start where the state begins.
struct StateTile {
    position: usize,
    extent: usize,
    origin: Value,
}

/// One block of a dispatch, with the buffers of the `loop.func` it reads and writes, and the
/// schedule of the dispatch.
pub(super) struct BlockLowering<'m> {
    pub(super) module: &'m Module,
    pub(super) block: &'m Block,
    pub(super) body: Region,
    /// The buffer of each operand, and its extents.
    pub(super) operands: Vec<(Value, Vec<Dim>)>,
    /// The buffer of each result.
    pub(super) results: Vec<Value>,
    pub(super) schedule: &'m Schedule,
}

/// One fold of a block: its kind, the yielded values it takes, and the results it gives.
struct Part<'f> {
    fold: &'f Fold,
    yields: Range<usize>,
    results: Range<usize>,
}

/// The position of `value` in `operands`, which it joins when it is not there yet.
fn slot(operands: &mut Vec<Value>, value: Value) -> usize {
    match operands.iter().position(|&operand| operand == value) {
        Some(position) => position,
        None => {
            operands.push(value);
            operands.len() - 1
        }
    }
}

/// The buffers of the state components of `folding` for yielded values of `dtypes`, each with
/// `extents` of elements in front of its own.
fn state_buffers(
    builder: &mut Builder<'_>,
    folding: &dyn FoldLowering,
    dtypes: &[DType],
    extents: &[usize],
) -> Vec<Value> {
    let mut buffers = Vec::new();
    for (dtype, shape) in folding.components(dtypes) {
        buffers.push(alloc(builder, &[extents, &shape[..]].concat(), dtype));
    }
    buffers
}

impl BlockLowering<'_> {
    /// Lowers each fold of the block, in order.
    pub(super) fn lower(&self, builder: &mut Builder<'_>) -> Result<()> {
        let dtypes = Block::yield_types(self.module, self.body)?;
        let (mut first_yield, mut first_result) = (0, 0);
        for fold in &self.block.folds {
            let rule = fold.combine.rule();
            let part = Part {
                fold,
                yields: first_yield..first_yield + rule.yields(),
                results: first_result..first_result + rule.results(),
            };
            first_yield += rule.yields();
            first_result += rule.results();
            match (fold.writes[0].into, &fold.combine) {
                (Some(OperandRef(operand)), _) => {
                    self.lower_into(builder, &part, operand, &dtypes)?
                }
                (None, CombineKind::None) => self.lower_map(builder, &part)?,
                (None, _) => self.lower_fold(builder, &part, &dtypes)?,
            }
        }
        Ok(())
    }

    /// `none` into a new result: every point writes its element.
    fn lower_map(&self, builder: &mut Builder<'_>, part: &Part<'_>) -> Result<()> {
        let steps = self.schedule.steps(self.block, false);
        let count = self.block.iters.len();
        let (mut ranges, mut point) = (vec![None; count], vec![None; count]);
        self.walk(
            builder,
            &steps,
            &mut ranges,
            &mut point,
            &mut |builder, _, point| {
                self.guarded(builder, point, &mut |builder, point| {
                    let yielded = self.point(builder, point)?;
                    let coords = self.coords(builder, &part.fold.writes[0].coords, point)?;
                    let result = self.results[part.results.start];
                    store(builder, yielded[part.yields.start], result, coords);
                    Ok(())
                })
            },
        )
    }

    /// A fold into a copy of operand `operand`: the copy first, then every point in order, which
    /// writes its value, or folds it into the element it names.
    fn lower_into(
        &self,
        builder: &mut Builder<'_>,
        part: &Part<'_>,
        operand: usize,
        dtypes: &[DType],
    ) -> Result<()> {
        let result = self.results[part.results.start];
        let (source, dims) = &self.operands[operand];
        copy(builder, *source, result, dims)?;
        let folding = match part.fold.combine {
            CombineKind::None => None,
            ref kind => Some(fold_of(kind)?),
        };
        let steps = self.schedule.steps(self.block, false);
        let count = self.block.iters.len();
        let (mut ranges, mut point) = (vec![None; count], vec![None; count]);
        self.walk(
            builder,
            &steps,
            &mut ranges,
            &mut point,
            &mut |builder, _, point| {
                self.guarded(builder, point, &mut |builder, point| {
                    let yielded = self.point(builder, point)?;
                    let coords = self.coords(builder, &part.fold.writes[0].coords, point)?;
                    let value = yielded[part.yields.start];
                    let Some(folding) = &folding else {
                        store(builder, value, result, coords);
                        return Ok(());
                    };
                    let current = load(builder, result, coords.clone(), None);
                    let yield_dtypes = &dtypes[part.yields.clone()];
                    let state = State {
                        buffers: state_buffers(builder, folding.as_ref(), yield_dtypes, &[]),
                        at: Vec::new(),
                    };
                    folding.start(builder, &state, yield_dtypes, &[current])?;
                    folding.push(builder, &state, 0, &yielded[part.yields.clone()])?;
                    let folded = folding.results(builder, &state);
                    store(builder, folded[0][0], result, coords);
                    Ok(())
                })
            },
        )
    }

    /// A fold into new results. The loops before the first loop over a reduced iterator walk the
    /// elements. Before that loop, the state of each element of the tiles inside starts. That loop
    /// and the loops inside it fold every point into the state, once per sweep of the fold. Then
    /// each element's results go to their places.
    fn lower_fold(
        &self,
        builder: &mut Builder<'_>,
        part: &Part<'_>,
        dtypes: &[DType],
    ) -> Result<()> {
        let folding = fold_of(&part.fold.combine)?;
        let yield_dtypes = &dtypes[part.yields.clone()];
        let iters = &self.block.iters;
        let prefix = iters.iter().position(|iter| iter.kind == IterKind::Prefix);
        let steps = self.schedule.steps(self.block, true);
        let (first_reduced, tiles) = self.schedule.state_tiles(self.block, &steps);
        let mut extents = Vec::with_capacity(tiles.len());
        for (position, extent) in &tiles {
            extents.push(extent.as_static().ok_or_else(|| {
                anyhow!(
                    "the state keeps the tile of d{position}, whose extent {extent} is not static"
                )
            })?);
        }
        let count = iters.len();
        let (mut ranges, mut point) = (vec![None; count], vec![None; count]);
        let (outer, inner) = steps.split_at(first_reduced);
        self.walk(
            builder,
            outer,
            &mut ranges,
            &mut point,
            &mut |builder, ranges, point| {
                let buffers = state_buffers(builder, folding.as_ref(), yield_dtypes, &extents);
                let mut state_tiles = Vec::with_capacity(tiles.len());
                for (&(position, _), &extent) in tiles.iter().zip(&extents) {
                    let origin = self.range(builder, ranges, position).0;
                    state_tiles.push(StateTile {
                        position,
                        extent,
                        origin,
                    });
                }
                let state_at = |at: Vec<Value>| State {
                    buffers: buffers.clone(),
                    at,
                };
                self.each_element(
                    builder,
                    &state_tiles,
                    ranges,
                    point,
                    &mut |builder, point, at| {
                        let mut init = Vec::with_capacity(part.fold.init.len());
                        for (component, &OperandRef(operand)) in part.fold.init.iter().enumerate() {
                            let write = &part.fold.writes[component];
                            let coords = self.coords(builder, &write.coords, point)?;
                            init.push(load(builder, self.operands[operand].0, coords, None));
                        }
                        folding.start(builder, &state_at(at), yield_dtypes, &init)
                    },
                )?;
                for sweep in 0..folding.sweeps() {
                    if sweep > 0 {
                        self.each_element(
                            builder,
                            &state_tiles,
                            ranges,
                            point,
                            &mut |builder, _, at| folding.next_sweep(builder, &state_at(at), sweep),
                        )?;
                    }
                    let mut push =
                        |builder: &mut Builder<'_>, _: &mut Ranges, point: &mut Point| {
                            self.guarded(builder, point, &mut |builder, point| {
                                let yielded = self.point(builder, point)?;
                                let mut at = Vec::with_capacity(tiles.len());
                                for tile in &state_tiles {
                                    let origin = tile.origin;
                                    let value = point[tile.position]
                                        .expect("an element of the state's tile");
                                    at.push(index(
                                        builder,
                                        IndexExpr::iter(0) - IndexExpr::iter(1),
                                        &[value, origin],
                                    ));
                                }
                                folding.push(
                                    builder,
                                    &state_at(at),
                                    sweep,
                                    &yielded[part.yields.clone()],
                                )
                            })
                        };
                    match prefix {
                        // The prefix of the element: its points up to and including the element's.
                        Some(position) => {
                            let at = point[position].expect("the prefix iterator is outside");
                            let start = constant(builder, 0);
                            let end =
                                index(builder, IndexExpr::iter(0) + IndexExpr::constant(1), &[at]);
                            for_range(builder, start, end, |builder, step| {
                                point[position] = Some(step);
                                let pushed = push(builder, ranges, point);
                                point[position] = Some(at);
                                pushed
                            })?;
                        }
                        None => self.walk(builder, inner, ranges, point, &mut push)?,
                    }
                }
                let rule = part.fold.combine.rule();
                self.each_element(
                    builder,
                    &state_tiles,
                    ranges,
                    point,
                    &mut |builder, point, at| {
                        let finished = folding.results(builder, &state_at(at));
                        for (component, values) in finished.into_iter().enumerate() {
                            let write = &part.fold.writes[component];
                            let result = self.results[part.results.start + component];
                            let coords = self.coords(builder, &write.coords, point)?;
                            if rule.extra_dims(component).is_empty() {
                                store(builder, values[0], result, coords);
                                continue;
                            }
                            // Along the dimension the kind appends.
                            for (rank, value) in values.into_iter().enumerate() {
                                let at = constant(builder, rank as i64);
                                store(builder, value, result, [coords.clone(), vec![at]].concat());
                            }
                        }
                        Ok(())
                    },
                )
            },
        )
    }

    /// The start and end of the tile of iterator `position` in `ranges`, or of its whole extent.
    fn range(&self, builder: &mut Builder<'_>, ranges: &Ranges, position: usize) -> (Value, Value) {
        match ranges[position] {
            Some(range) => range,
            None => {
                let start = constant(builder, 0);
                let end = extent(builder, &self.block.iters[position].extent);
                (start, end)
            }
        }
    }

    /// Builds the loops of `steps`, outermost first, and `inner` in the innermost, with the
    /// iterators' tiles in `ranges` and the values of those that loops walk in `point`.
    fn walk(
        &self,
        builder: &mut Builder<'_>,
        steps: &[Step],
        ranges: &mut Ranges,
        point: &mut Point,
        inner: &mut dyn FnMut(&mut Builder<'_>, &mut Ranges, &mut Point) -> Result<()>,
    ) -> Result<()> {
        let Some((&step, rest)) = steps.split_first() else {
            return inner(builder, ranges, point);
        };
        let position = self.schedule.iter_of(step);
        let (start, end) = self.range(builder, ranges, position);
        let Step::Tiles(nested) = step else {
            for_range(builder, start, end, |builder, value| {
                point[position] = Some(value);
                self.walk(builder, rest, ranges, point, inner)
            })?;
            point[position] = None;
            return Ok(());
        };
        let entry = &self.schedule.nest[nested];
        let size = entry.extent as i64;
        let first = IndexExpr::iter(1) - IndexExpr::iter(0) + IndexExpr::constant(size - 1);
        let tiles = index(
            builder,
            IndexExpr::floordiv(first, IndexExpr::constant(size)),
            &[start, end],
        );
        let mut tile = |builder: &mut Builder<'_>, tile: Value| -> Result<()> {
            let tile_start = index(
                builder,
                IndexExpr::iter(0) + IndexExpr::iter(1) * IndexExpr::constant(size),
                &[start, tile],
            );
            let tile_end = index(
                builder,
                IndexExpr::min(
                    IndexExpr::iter(0) + IndexExpr::constant(size),
                    IndexExpr::iter(1),
                ),
                &[tile_start, end],
            );
            let around = ranges[position].replace((tile_start, tile_end));
            let walked = self.walk(builder, rest, ranges, point, inner);
            ranges[position] = around;
            walked
        };
        let zero = constant(builder, 0);
        let Parallelism::Par(par) = &entry.parallelism else {
            return for_range(builder, zero, tiles, |builder, value| tile(builder, value));
        };
        let Some(units) = par.units else {
            return forall_range(builder, par.level, zero, tiles, |builder, value| {
                tile(builder, value)
            });
        };
        // Each of the units runs a contiguous run of the tiles.
        let units = units as i64;
        let unit_count = constant(builder, units);
        forall_range(builder, par.level, zero, unit_count, |builder, unit| {
            let tiles_per_unit = IndexExpr::floordiv(
                IndexExpr::iter(0) + IndexExpr::constant(units - 1),
                IndexExpr::constant(units),
            );
            let per_unit = index(builder, tiles_per_unit, &[tiles]);
            let run_start = index(
                builder,
                IndexExpr::iter(0) * IndexExpr::iter(1),
                &[unit, per_unit],
            );
            let run_end = index(
                builder,
                IndexExpr::min(IndexExpr::iter(0) + IndexExpr::iter(1), IndexExpr::iter(2)),
                &[run_start, per_unit, tiles],
            );
            for_range(builder, run_start, run_end, |builder, value| {
                tile(builder, value)
            })
        })
    }

    /// Loops over the elements of the state's tiles, each clipped to its iterator's range, and
    /// builds `inner` at each, with the iterators' values in `point` and the element's position in
    /// the state.
    fn each_element(
        &self,
        builder: &mut Builder<'_>,
        tiles: &[StateTile],
        ranges: &Ranges,
        point: &mut Point,
        inner: &mut ElementBody<'_>,
    ) -> Result<()> {
        self.each_from(builder, tiles, ranges, point, &mut Vec::new(), inner)
    }

    fn each_from(
        &self,
        builder: &mut Builder<'_>,
        tiles: &[StateTile],
        ranges: &Ranges,
        point: &mut Point,
        at: &mut Vec<Value>,
        inner: &mut ElementBody<'_>,
    ) -> Result<()> {
        let Some((tile, rest)) = tiles.split_first() else {
            return inner(builder, point, at.clone());
        };
        let (_, end) = self.range(builder, ranges, tile.position);
        let start = constant(builder, 0);
        let clipped = index(
            builder,
            IndexExpr::min(
                IndexExpr::constant(tile.extent as i64),
                IndexExpr::iter(1) - IndexExpr::iter(0),
            ),
            &[tile.origin, end],
        );
        for_range(builder, start, clipped, |builder, offset| {
            let value = index(
                builder,
                IndexExpr::iter(0) + IndexExpr::iter(1),
                &[tile.origin, offset],
            );
            point[tile.position] = Some(value);
            at.push(offset);
            let built = self.each_from(builder, rest, ranges, point, at, inner);
            at.pop();
            built
        })?;
        point[tile.position] = None;
        Ok(())
    }

    /// Builds `body` where every constraint of the block holds at `point`: inside one
    /// `loop.if` per constraint, each checked once the ones before it hold.
    fn guarded(
        &self,
        builder: &mut Builder<'_>,
        point: &mut Point,
        body: &mut dyn FnMut(&mut Builder<'_>, &mut Point) -> Result<()>,
    ) -> Result<()> {
        self.guarded_from(builder, 0, point, body)
    }

    fn guarded_from(
        &self,
        builder: &mut Builder<'_>,
        first: usize,
        point: &mut Point,
        body: &mut dyn FnMut(&mut Builder<'_>, &mut Point) -> Result<()>,
    ) -> Result<()> {
        let Some(constraint) = self.block.constraints.get(first) else {
            return body(builder, point);
        };
        let mut operands = Vec::new();
        let mut cmps = SmallVec::new();
        for cmp in &constraint.0 {
            let lhs = self.lower_expr(builder, &cmp.lhs, point, &mut operands)?;
            let rhs = self.lower_expr(builder, &cmp.rhs, point, &mut operands)?;
            cmps.push(Cmp::new(lhs, cmp.op, rhs));
        }
        let holds = predicate(builder, &[Constraint(cmps)], &operands);
        if_then(builder, holds, |builder| {
            self.guarded_from(builder, first + 1, point, body)
        })
    }

    /// Builds the reads and the body of the block at `point`, and gives the values it yields.
    fn point(&self, builder: &mut Builder<'_>, point: &Point) -> Result<Vec<Value>> {
        let module = self.module;
        let mut map: HashMap<Value, Value> = HashMap::new();
        for (&arg, read) in module.args(self.body).iter().zip(&self.block.reads) {
            map.insert(arg, self.read(builder, read, point)?);
        }
        for inst in module.insts(self.body) {
            let op = module.op(inst);
            if let Some(position) = op.get::<Index>() {
                let mut operands = Vec::new();
                let access = AccessExpr::from(position.expr.clone());
                let lowered = self.lower_expr(builder, &access, point, &mut operands)?;
                let at = index(builder, lowered, &operands);
                let value = builder.create(
                    arith::FromIndex {
                        dtype: position.dtype,
                    },
                    at,
                );
                map.insert(module.results(inst)[0], value);
                continue;
            }
            if op.has_trait::<Terminator>() {
                return Ok(module
                    .operands(inst)
                    .iter()
                    .map(|value| map[value])
                    .collect());
            }
            builder.create_mapped(
                op.clone(),
                module.operands(inst),
                module.operand_segments(inst).map(<[usize]>::to_vec),
                module.results(inst),
                &mut map,
            )?;
        }
        bail!("the body of the block ends without tile.yield")
    }

    /// The value that `read` gives at `point`.
    fn read(&self, builder: &mut Builder<'_>, read: &Read, point: &Point) -> Result<Value> {
        let coords = self.coords(builder, &read.coords, point)?;
        let (first, count, pick) = match &read.source {
            Source::Operand(operand) => {
                return Ok(load(builder, self.operands[*operand].0, coords, read.pad));
            }
            Source::List { first, count, pick } => (*first, *count, pick),
        };
        // Each operand of the list read at the coordinates, the one `pick` names chosen.
        let mut operands = Vec::new();
        let pick = self.lower_expr(builder, pick, point, &mut operands)?;
        let picked = index(builder, pick, &operands);
        let mut value = load(builder, self.operands[first].0, coords.clone(), read.pad);
        for offset in 1..count {
            let candidate = load(
                builder,
                self.operands[first + offset].0,
                coords.clone(),
                read.pad,
            );
            let named = Constraint::cmp(
                IndexExpr::iter(0),
                Relation::Equal,
                IndexExpr::constant(offset as i64),
            );
            let chosen = predicate(builder, &[named], &[picked]);
            value = builder.create(arith::Select, (chosen, candidate, value));
        }
        Ok(value)
    }

    /// The index values of `exprs` at `point`.
    fn coords(
        &self,
        builder: &mut Builder<'_>,
        exprs: &[AccessExpr],
        point: &Point,
    ) -> Result<Vec<Value>> {
        let mut operands = Vec::new();
        let mut lowered = Vec::with_capacity(exprs.len());
        for expr in exprs {
            lowered.push(self.lower_expr(builder, expr, point, &mut operands)?);
        }
        let mut values = Vec::with_capacity(lowered.len());
        for expr in lowered {
            values.push(index(builder, expr, &operands));
        }
        Ok(values)
    }

    /// `expr` at `point` as an expression whose `di` is `operands[i]`: each iterator becomes its
    /// value, and each element it reads a load, which joins `operands` as an index.
    fn lower_expr(
        &self,
        builder: &mut Builder<'_>,
        expr: &AccessExpr,
        point: &Point,
        operands: &mut Vec<Value>,
    ) -> Result<IndexExpr> {
        Ok(match expr.root() {
            Root::Const(value) => IndexExpr::constant(value),
            Root::Sym(sym) => IndexExpr::sym(sym),
            Root::Iter(position) => {
                let value = point
                    .get(position)
                    .copied()
                    .flatten()
                    .ok_or_else(|| anyhow!("d{position} has no value here"))?;
                IndexExpr::iter(slot(operands, value))
            }
            Root::Binary(op, lhs, rhs) => {
                let lhs = self.lower_expr(builder, &lhs, point, operands)?;
                let rhs = self.lower_expr(builder, &rhs, point, operands)?;
                IndexExpr::binary(op, lhs, rhs)
            }
            Root::Neg(operand) => -self.lower_expr(builder, &operand, point, operands)?,
            Root::Read(operand, coords) => {
                let mut values = Vec::with_capacity(coords.len());
                for coord in &coords {
                    let lowered = self.lower_expr(builder, coord, point, operands)?;
                    values.push(index(builder, lowered, operands));
                }
                let element = load(builder, self.operands[operand].0, values, None);
                let value = builder.create(arith::ToIndex, element);
                IndexExpr::iter(slot(operands, value))
            }
        })
    }
}

/// Copies every element of `from` into `to`, two buffers of the extents `dims`.
pub(super) fn copy(builder: &mut Builder<'_>, from: Value, to: Value, dims: &[Dim]) -> Result<()> {
    copy_from(builder, from, to, dims, &mut Vec::new())
}

fn copy_from(
    builder: &mut Builder<'_>,
    from: Value,
    to: Value,
    dims: &[Dim],
    coords: &mut Vec<Value>,
) -> Result<()> {
    let Some((first, rest)) = dims.split_first() else {
        let element = load(builder, from, coords.clone(), None);
        store(builder, element, to, coords.clone());
        return Ok(());
    };
    for_extent(builder, first, |builder, value| {
        coords.push(value);
        let copied = copy_from(builder, from, to, rest, coords);
        coords.pop();
        copied
    })
}
