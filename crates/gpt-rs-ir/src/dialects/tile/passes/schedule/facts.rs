//! What the rules of the schedule read: the block's iterators, reads and fold, and the target's
//! registers, memories, workers and the issue of the ops a register tile runs. Each is a fact the
//! IR or the target states; a fact the target lacks is an error that names it.

use std::cell::RefCell;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::ops::Range;

use anyhow::{anyhow, bail, Result};
use tinyvec::ArrayVec;

use crate::dialects::isa::{self, Broadcast, Fma, Load};
use crate::dialects::tile::{
    Block, CombineKind, Dispatch, Index, IterKind, ModeKind, Read, Source,
};
use crate::dialects::{arith, host};
use crate::ir::{gcd, DType, Inst, Module, OpKind, Ranked, Value, MAX_ITERS};
use crate::targets::{InstructionCost, MemoryKind, TargetDesc};
use crate::Op;

/// The first block of a dispatch as the rules count it. A fact of each iterator is an array of
/// [`MAX_ITERS`] entries; past the block's iterators it holds what an iterator of one point that
/// nothing takes would: one point, not reduced, taken by no read or op.
pub(super) struct Facts {
    /// The points of each iterator, and whether it is reduced.
    pub extents: [usize; MAX_ITERS],
    pub reduced: [bool; MAX_ITERS],
    /// The iterator the lanes run along and the points of one vector.
    pub lanes: Option<(usize, usize)>,
    /// Whether a `tile.index` of the body takes each iterator.
    pub indexed: [bool; MAX_ITERS],
    pub reads: Vec<ReadFacts>,
    /// The fold's state, when the block folds into one that the register tile's accumulators
    /// hold.
    pub fold: Option<FoldFacts>,
    /// The vector loads and broadcasts one worker issues per cycle.
    pub load_issue: f64,
    pub broadcast_issue: f64,
    /// The vector registers of a worker.
    pub registers: usize,
    /// The registers the blocks placed at the register tile hold: of those that run at each
    /// point on the element it finishes, while the tile holds its accumulators, and of those that
    /// run over the finished tile, which the code keeps in memory for them.
    pub placed_at_point: usize,
    pub placed_after: usize,
    /// The values the body computes beside the one it yields.
    pub temporaries: Vec<Temporary>,
    /// The pads of the padded reads, one register each, which the code holds across its loops to
    /// fill the lanes that fall outside an operand.
    pub pads: usize,
    /// By iterator, the body's ops whose values vary along it, which each row of a register tile
    /// along it computes; the others it computes once for all rows.
    pub ops_along: [usize; MAX_ITERS],
    /// The body's ops whose values vary along some iterator; the others, such as constants, the
    /// vector code computes once for the block.
    pub varying_ops: usize,
    /// The memories after the registers, innermost first.
    pub memories: Vec<Store>,
    /// The memories that hold reads in place, outermost first: the caches.
    pub caches: Vec<usize>,
    /// The workers that share the work.
    pub units: usize,
    /// The sum of each read's lines in its most crowded set, of each memory that places lines by
    /// address, by the points of each iterator they are read over, as the rules ask it again
    /// for the same pieces of many schedules.
    pub crowded: RefCell<BTreeMap<(usize, [usize; MAX_ITERS]), u64>>,
}

/// The elements of a value that one worker wrote: a range of coordinates along each axis.
pub(super) type Part = Ranked<Range<i64>>;

/// The fold of a block whose state the register tile's accumulators hold: the latency of the op
/// that steps the state, in cycles, and its issue rate, the steps one worker starts per cycle.
pub(super) struct FoldFacts {
    pub latency: f64,
    pub step_issue: f64,
    /// The dtype of its state.
    pub state: DType,
}

/// A value the body computes beside the one it yields: the iterators it varies along, and the
/// positions of the body's ops while it is held, from the op that computes it to its last use.
pub(super) struct Temporary {
    pub varies: [bool; MAX_ITERS],
    pub held: Range<usize>,
}

/// A read of the block.
pub(super) struct ReadFacts {
    pub element_bits: u64,
    pub axes: Ranked<Axis>,
    /// The constant of each axis's coordinate, where the coordinate has one: none where it takes a
    /// symbol or a read.
    pub offsets: Ranked<Option<i64>>,
    /// For each call of the dispatch, when the read's operand is a parameter of the dispatch, the
    /// part of the call's argument that each worker of its writer wrote, in the order of the
    /// workers ([`worker_parts`]); none where no dispatch of the run wrote it ([`written`]).
    pub written: Vec<Vec<Part>>,
    /// Whether it depends on each iterator.
    pub depends: [bool; MAX_ITERS],
    /// The elements between the points of each iterator: the coefficients of the coordinates
    /// times the row-major strides of their axes.
    pub strides: [i64; MAX_ITERS],
}

/// An axis of the operand of a read: its extent and the iterators its coordinate takes, with their
/// coefficients, by ascending magnitude of the coefficient ([`axis_points`]).
#[derive(Debug, Clone, Copy, Default)]
pub(super) struct Axis {
    pub extent: usize,
    pub terms: ArrayVec<[(usize, i64); MAX_ITERS]>,
}

/// The sets of a cache: the bytes of a line, how many sets, and the lines of each.
#[derive(Debug, Clone, Copy)]
pub(super) struct Sets {
    pub line: u64,
    pub count: u64,
    pub ways: u64,
}

/// A memory after the registers.
pub(super) struct Store {
    pub name: String,
    /// The sets of a cache that places each line by its address within a page, where the target
    /// states its page, and the lines each set holds: none for one whose ways hold more than a
    /// page each, as the address of a line in the memory it reads decides its set.
    pub sets: Option<Sets>,
    /// The bytes one worker may keep in it: of a cache all ways but one, shared among the workers
    /// that share it; of scratch its bytes, shared the same way.
    pub capacity: u64,
    /// Whether it holds only what the code copies into it.
    pub copies_only: bool,
    /// The bytes per cycle one worker reads into it from the memory outside while every worker
    /// reads, where the target measured them.
    pub fill_rate: Option<f64>,
    /// The workers that share it, consecutive ones.
    pub sharing: usize,
    /// Whether the memory it fills from is a cache, whose contents the counts know.
    pub filled_from_cache: bool,
}

impl Facts {
    /// The facts of the block of `inst`, the first of the dispatch `dispatch`, on `target`. The
    /// blocks `placed` run after it, at its register tile, and a product and the sum it enters
    /// round once when `contract`. It is `None` where the block takes no schedule: an extent is
    /// not known before the run, an iterator is a prefix or an ordered reduction, a read picks its
    /// operand from a list or has a coordinate that is not affine, the block folds other than one
    /// value into one state, it writes an operand where its points may collide
    /// ([`Fold::updates_in_order`](crate::dialects::tile::Fold::updates_in_order)), or its body
    /// has an op without a lane-wise form ([`Block::vector_work`]). It fails where the target
    /// lacks a fact that the rules read.
    pub fn of(
        module: &Module,
        dispatch: Inst,
        inst: Inst,
        placed: &[Inst],
        target: &TargetDesc,
        contract: bool,
    ) -> Result<Option<Facts>> {
        let Some(block) = module.op(inst).get::<Block>() else {
            return Ok(None);
        };
        let Some((extents, reduced)) = iterators(block) else {
            return Ok(None);
        };
        let [fold] = block.folds.as_slice() else {
            return Ok(None);
        };
        // Points that update an operand where they may collide are written in the block's order.
        if fold.updates_in_order(&block.iters) {
            return Ok(None);
        }
        let Some(reads) = read_facts(module, dispatch, inst) else {
            return Ok(None);
        };
        let width = target.max_vector_bits();
        let result_dtype = module
            .results(inst)
            .first()
            .and_then(|&result| module.ty(result).dtype());
        let Some(state_dtype) = result_dtype else {
            return Ok(None);
        };
        let fold_facts = match fold.combine {
            CombineKind::None => None,
            ref combine => {
                let fusable = contract && target.has_fma();
                let Some(step) = step_op(module, inst, combine, fusable) else {
                    return Ok(None);
                };
                Some(FoldFacts::of(target, &step, width, state_dtype)?)
            }
        };
        let load_issue = cost(target, op_name::<Load>(), width, state_dtype)?.issue;
        let broadcast_issue = cost(target, op_name::<Broadcast>(), width, state_dtype)?.issue;
        let (placed_at_point, placed_after) = placed_registers(module, inst, placed);
        let varies = varies_along(module, inst, block);
        let temporaries = temporaries(module, inst, &varies);
        let (ops_along, varying_ops) = ops_along(module, inst, &varies);
        let lanes = block
            .walk()
            .atom
            .map(|atom| (atom.iter.position(), atom.lanes));
        if let Some((lane, _)) = lanes {
            if Block::vector_work(module, inst, lane).is_err() {
                return Ok(None);
            }
        }
        let indexed = indexed_iterators(module, inst);
        let memories = stores(target)?;
        let caches = caches(&memories);
        Ok(Some(Facts {
            extents,
            reduced,
            lanes,
            indexed,
            reads,
            fold: fold_facts,
            load_issue,
            broadcast_issue,
            registers: target.isa.vector_registers as usize,
            placed_at_point,
            placed_after,
            temporaries,
            pads: pads(block),
            ops_along,
            varying_ops,
            memories,
            caches,
            units: workers(target)?,
            crowded: RefCell::default(),
        }))
    }
}

/// The facts of the reads of the block of `inst`, the first of `dispatch`, with the parts of their
/// operands that the workers of other dispatches wrote: none where one of them has none
/// ([`ReadFacts::of`]).
fn read_facts(module: &Module, dispatch: Inst, inst: Inst) -> Option<Vec<ReadFacts>> {
    let block = module.op(inst).get::<Block>()?;
    let calls = Dispatch::callers(module, dispatch);
    let mut reads = Vec::with_capacity(block.reads.len());
    for read in &block.reads {
        let mut facts = ReadFacts::of(module, inst, read)?;
        let placed = facts.offsets.iter().all(Option::is_some);
        if let (&Source::Operand(operand), true) = (&read.source, placed) {
            facts.written = written(module, (dispatch, &calls), inst, operand);
        }
        reads.push(facts);
    }
    Some(reads)
}

/// For each of `calls`, the calls of `dispatch`, the part of an argument that each worker of its
/// writer wrote ([`worker_parts`]). The argument is the one that operand `operand` of the first
/// block, the block of `inst`, takes, when that operand is a parameter of the dispatch. The parts
/// are none for a value computed before the run or an argument of the function, which no worker of
/// the run wrote. A call is left out where the dispatch that wrote the argument has no workers yet,
/// because it is scheduled after this one in module order, or not at all.
fn written(
    module: &Module,
    (dispatch, calls): (Inst, &[Inst]),
    inst: Inst,
    operand: usize,
) -> Vec<Vec<Part>> {
    let body = module.regions(dispatch)[0];
    let value = module.operands(inst)[operand];
    let Some(arg) = module.args(body).iter().position(|&param| param == value) else {
        return Vec::new();
    };
    let mut written = Vec::new();
    for &call in calls {
        let argument = module.operands(call)[arg];
        let argument_ref = module.value_ref(argument);
        let defining = argument_ref.defining_inst();
        let in_run = defining.filter(|&defined| {
            module.op(defined).get::<host::Dispatch>().is_some() && !argument_ref.known_before_run()
        });
        let Some(producer) = in_run else {
            written.push(Vec::new());
            continue;
        };
        if let Some(parts) = worker_parts(module, producer, argument) {
            written.push(parts);
        }
    }
    written
}

/// The part of `value`, a result of the `host.dispatch` `producer`, that each worker of the called
/// dispatch wrote, in the order of the workers. The workers take consecutive iterations of its
/// modes of workers, the last the fastest. It holds where the first block of the dispatch has modes
/// of workers, each over an iterator of its own, and the value is the result of that block, or of a
/// block of its type placed at its register tile, so that each worker wrote it at the first
/// block's points.
fn worker_parts(module: &Module, producer: Inst, value: Value) -> Option<Vec<Part>> {
    let callee = &module.op(producer).get::<host::Dispatch>()?.callee;
    let dispatch = Dispatch::named(module, callee)?;
    let result = module
        .results(producer)
        .iter()
        .position(|&each| each == value)?;
    let body = module.regions(dispatch)[0];
    let blocks = Dispatch::blocks(module, body);
    let &(first, block) = blocks.first()?;
    let returned = module.operands(module.insts(body).last()?)[result];
    let writer = module.value_ref(returned).defining_inst()?;
    let placed = super::placed(module, &blocks);
    if writer != first && !placed.contains(&writer) {
        return None;
    }
    if module.ty(returned) != module.ty(*module.results(first).first()?) {
        return None;
    }
    let write = block.folds.first()?.writes.first()?;
    let mut extents = [1; MAX_ITERS];
    for (position, iter) in block.iters.iter().enumerate() {
        extents[position] = iter.extent.as_static()?;
    }
    // Each split: its iterator, the points of each part and the parts.
    let mut splits: ArrayVec<[(usize, usize, usize); MAX_ITERS]> = ArrayVec::new();
    for mode in &block.loops {
        if !matches!(mode.kind, Some(ModeKind::Workers(_))) {
            continue;
        }
        let position = mode.iter.position();
        if splits.iter().any(|&(seen, ..)| seen == position) {
            return None;
        }
        splits.push((position, mode.size, extents[position].div_ceil(mode.size)));
    }
    if splits.is_empty() {
        return None;
    }
    let mut workers = 1;
    for &(_, _, count) in &splits {
        workers *= count;
    }
    let mut parts = Vec::with_capacity(workers);
    for worker in 0..workers {
        let mut ranges = [(0, 0); MAX_ITERS];
        for (position, &extent) in extents.iter().enumerate() {
            ranges[position] = (0, extent as i64 - 1);
        }
        let mut rest = worker;
        for &(position, size, count) in splits.iter().rev() {
            let digit = rest % count;
            rest /= count;
            let low = digit * size;
            let high = ((digit + 1) * size).min(extents[position]) - 1;
            ranges[position] = (low as i64, high as i64);
        }
        let mut part = Part::new();
        for coord in write.coords.iter() {
            let (low, high) = coord.bounds(&|position| ranges.get(position).copied())?;
            part.push(low..high + 1);
        }
        parts.push(part);
    }
    Some(parts)
}

/// Whether a `tile.index` of the body of the block of `inst` takes each iterator.
fn indexed_iterators(module: &Module, inst: Inst) -> [bool; MAX_ITERS] {
    let mut indexed = [false; MAX_ITERS];
    for body_inst in module.insts(module.regions(inst)[0]) {
        if let Some(index) = module.op(body_inst).get::<Index>() {
            for position in index.expr.iters() {
                indexed[position] = true;
            }
        }
    }
    indexed
}

/// The points of each iterator of `block` and whether it is reduced: none where an extent is not
/// known before the run, or an iterator is a prefix or an ordered reduction.
fn iterators(block: &Block) -> Option<([usize; MAX_ITERS], [bool; MAX_ITERS])> {
    let mut extents = [1; MAX_ITERS];
    let mut reduced = [false; MAX_ITERS];
    for (position, iter) in block.iters.iter().enumerate() {
        if matches!(iter.kind, IterKind::Prefix | IterKind::RedOrdered) {
            return None;
        }
        extents[position] = iter.extent.as_static()?;
        reduced[position] = iter.kind.is_reduced();
    }
    Some((extents, reduced))
}

/// The pads of the padded reads of `block`, each value once.
fn pads(block: &Block) -> usize {
    let mut pad_values = Vec::new();
    for read in &block.reads {
        if let Some(pad) = read.pad {
            if !pad_values.contains(&pad.0.to_bits()) {
                pad_values.push(pad.0.to_bits());
            }
        }
    }
    pad_values.len()
}

/// The memories of `memories` that hold reads in place, outermost first: the caches.
fn caches(memories: &[Store]) -> Vec<usize> {
    let mut caches = Vec::with_capacity(memories.len());
    for (index, store) in memories.iter().enumerate().rev() {
        if !store.copies_only {
            caches.push(index);
        }
    }
    caches
}

impl FoldFacts {
    /// The fold whose state the op `step` steps, at `width` bits for `dtype`, or an error that
    /// names the fact the target lacks.
    fn of(target: &TargetDesc, step: &Op, width: u32, dtype: DType) -> Result<FoldFacts> {
        let name = step.name();
        let row = cost(target, &name, width, dtype)?;
        let latency = row.latency.ok_or_else(|| {
            anyhow!("the target states no latency of {name} at {width} bits for {dtype}")
        })?;
        Ok(FoldFacts {
            latency,
            step_issue: row.issue,
            state: dtype,
        })
    }
}

impl ReadFacts {
    /// The facts of `read`, a read of the block of `inst`: none where it picks its operand from a
    /// list or a coordinate is not affine.
    fn of(module: &Module, inst: Inst, read: &Read) -> Option<ReadFacts> {
        let Source::Operand(source) = read.source else {
            return None;
        };
        let ty = module.ty(module.operands(inst)[source]).as_tensor()?;
        let dims = ty.static_dims()?;
        let mut axes = Ranked::new();
        let mut offsets = Ranked::new();
        let mut depends = [false; MAX_ITERS];
        let mut strides = [0_i64; MAX_ITERS];
        let mut axis_stride = 1_i64;
        for axis in (0..read.coords.len()).rev() {
            let coord = &read.coords[axis];
            let mut terms: ArrayVec<[(usize, i64); MAX_ITERS]> = ArrayVec::new();
            for position in coord.iters() {
                let coefficient = coord.coefficient(position)?;
                if coefficient != 0 {
                    terms.push((position, coefficient));
                    depends[position] = true;
                    strides[position] += coefficient * axis_stride;
                }
            }
            terms.sort_by_key(|&(_, coefficient)| coefficient.unsigned_abs());
            axes.push(Axis {
                extent: dims[axis],
                terms,
            });
            offsets.push(coord.bounds(&|_| Some((0, 0))).map(|(offset, _)| offset));
            axis_stride *= dims[axis] as i64;
        }
        axes.reverse();
        offsets.reverse();
        Some(ReadFacts {
            element_bits: ty.dtype.bits() as u64,
            axes,
            offsets,
            written: Vec::new(),
            depends,
            strides,
        })
    }

    /// The bytes of the elements it reads over `points` of each iterator: along each axis the
    /// distinct coordinates, at most its extent.
    pub fn footprint(&self, points: &[usize]) -> u64 {
        let mut elements = 1_u64;
        for axis in &self.axes {
            elements *= axis_points(&axis.terms, points).min(axis.extent as u64);
        }
        (elements * self.element_bits).div_ceil(8)
    }

    /// The most of its lines over `points` of each iterator that fall in one of the sets `sets`,
    /// each set holding the lines whose address within a way of the cache names it. Along each axis
    /// of its operand, it reads the distinct coordinates that the footprint counts
    /// ([`axis_points`]), spaced by the least coefficient of the iterators that move it. Axes whose
    /// steps continue the run of the axes before them, the smallest first, or step within one line,
    /// make runs whose every line it reads. The other axes place the runs. Where the runs start
    /// within a way is counted in bins of the steps' common divisor. Each run's lines then fall in
    /// consecutive sets from the set of its first.
    pub fn most_lines_in_a_set(&self, points: &[usize], sets: Sets) -> u64 {
        let line_bits = sets.line * 8;
        let way_bits = line_bits * sets.count;
        let mut steps: Ranked<(u64, u64)> = Ranked::new();
        let mut axis_bits = self.element_bits;
        for axis in self.axes.iter().rev() {
            let terms = &axis.terms;
            let count = axis_points(terms, points).min(axis.extent as u64);
            let mut least = None;
            for &(position, coefficient) in terms {
                if points[position] > 1 {
                    let magnitude = coefficient.unsigned_abs();
                    least = Some(least.map_or(magnitude, |seen: u64| seen.min(magnitude)));
                }
            }
            if let (Some(step), true) = (least, count > 1) {
                steps.push((step * axis_bits, count));
            }
            axis_bits *= axis.extent as u64;
        }
        steps.sort_unstable();
        let mut run = self.element_bits;
        let mut apart: Ranked<(u64, u64)> = Ranked::new();
        for (step, count) in steps {
            match apart.is_empty() && step <= run.max(line_bits) {
                true => run += step * (count - 1),
                false => apart.push((step, count)),
            }
        }
        let mut grain = way_bits;
        for &(step, _) in &apart {
            grain = gcd(grain, step % way_bits);
        }
        let bins = (way_bits / grain) as usize;
        let mut starts = vec![0_u64; bins];
        starts[0] = 1;
        for (step, count) in apart {
            let shift = ((step % way_bits) / grain) as usize;
            starts = spread(&starts, shift, count);
        }
        // Each run's lines in consecutive sets: `rounds` lines in every set where the run wraps
        // the sets, and one more in the first `rest` from its first, as a circular difference.
        let set_count = sets.count as usize;
        let (mut everywhere, mut difference) = (0_u64, vec![0_i64; set_count + 1]);
        for (bin, &held) in starts.iter().enumerate() {
            if held == 0 {
                continue;
            }
            let start = bin as u64 * grain;
            let first_line = start / line_bits;
            let lines = (start + run - 1) / line_bits - first_line + 1;
            everywhere += held * (lines / sets.count);
            let first = (first_line % sets.count) as usize;
            let end = first + (lines % sets.count) as usize;
            difference[first] += held as i64;
            difference[end.min(set_count)] -= held as i64;
            if end > set_count {
                difference[0] += held as i64;
                difference[end - set_count] -= held as i64;
            }
        }
        let (mut most, mut running) = (0_u64, 0_i64);
        for &change in &difference[..set_count] {
            running += change;
            most = most.max(everywhere + running as u64);
        }
        most
    }

    /// Whether it reads the same elements over the ranges `first` and `second` of each iterator:
    /// where they agree on every iterator it takes.
    pub fn same_footprint(&self, first: &[Range<usize>], second: &[Range<usize>]) -> bool {
        let mut agree = true;
        for (position, &depends) in self.depends.iter().enumerate() {
            agree &= !depends || first[position] == second[position];
        }
        agree
    }

    /// The bytes of the elements it reads over the range `ranges` of each iterator that lie in
    /// `part`: along each axis, its distinct coordinates over the ranges times the share of the
    /// coordinates they span that the part holds, at most the axis's extent; none where a
    /// coordinate has no constant.
    pub fn bytes_within(&self, ranges: &[Range<usize>], part: &Part) -> u64 {
        let mut points = [1; MAX_ITERS];
        for (position, range) in ranges.iter().enumerate() {
            points[position] = range.len();
        }
        let mut elements = 1_u64;
        for (axis_index, axis) in self.axes.iter().enumerate() {
            let terms = &axis.terms;
            let Some(offset) = self.offsets[axis_index] else {
                return 0;
            };
            let (mut low, mut high) = (offset, offset);
            for &(position, coefficient) in terms {
                let first = coefficient * ranges[position].start as i64;
                let last = coefficient * (ranges[position].end as i64 - 1);
                low += first.min(last);
                high += first.max(last);
            }
            let held = (high + 1).min(part[axis_index].end) - low.max(part[axis_index].start);
            if held <= 0 {
                return 0;
            }
            let span = (high - low + 1) as u64;
            let taken = axis_points(terms, &points) * held as u64 / span;
            elements *= taken.min(axis.extent as u64);
        }
        (elements * self.element_bits).div_ceil(8)
    }
}

/// The starts `starts`, counted by bin of a way, moved by `count` steps of `shift` bins each: each
/// bin takes the starts that land on it. The steps visit the bins of one class modulo the shift's
/// common divisor with the bins in a cycle, so a bin takes every start of its class once for each
/// whole cycle of the steps, and those of the window of the remaining steps behind it.
fn spread(starts: &[u64], shift: usize, count: u64) -> Vec<u64> {
    let bins = starts.len();
    let classes = gcd(bins as u64, shift as u64) as usize;
    let period = bins / classes;
    let (rounds, rest) = (count / period as u64, (count % period as u64) as usize);
    let mut moved = vec![0_u64; bins];
    let mut cycle = Vec::with_capacity(period);
    for class in 0..classes {
        cycle.clear();
        for step in 0..period {
            cycle.push((class + step * shift) % bins);
        }
        let mut total = 0;
        for &bin in &cycle {
            total += starts[bin];
        }
        // The starts of the `rest` bins up to the first of the cycle, walking back.
        let mut window = 0;
        for back in 0..rest {
            window += starts[cycle[(period - back) % period]];
        }
        for position in 0..period {
            moved[cycle[position]] = rounds * total + window;
            if rest > 0 {
                window += starts[cycle[(position + 1) % period]];
                window -= starts[cycle[(position + 1 + period - rest) % period]];
            }
        }
    }
    moved
}

/// The distinct coordinates an axis whose iterators `terms` take, with their coefficients by
/// ascending magnitude, over `points` of each iterator: walked by ascending step, a step past what
/// the smaller ones span multiplies what they take; one within it extends a contiguous range.
fn axis_points(terms: &[(usize, i64)], points: &[usize]) -> u64 {
    let (mut count, mut span) = (1_u64, 1_u64);
    for &(position, coefficient) in terms {
        let step = coefficient.unsigned_abs();
        let taken = points[position] as u64;
        let past = step >= span;
        span += step * (taken - 1);
        match past {
            true => count *= taken,
            false => count = span,
        }
    }
    count
}

/// The op that steps the state of the fold `combine` of the block of `inst` with each point's
/// value, as the vector code runs it: the lane-wise form of the fold's step, or one fused
/// multiply-add where `contract` and the body yields a product that nothing else takes. None for
/// a fold whose state one op does not step.
fn step_op(module: &Module, inst: Inst, combine: &CombineKind, contract: bool) -> Option<Op> {
    let step = combine.rule().step()?;
    let body = module.regions(inst)[0];
    let terminator = module.insts(body).last()?;
    let yielded = *module.operands(terminator).first()?;
    let sums = step.get::<arith::Add>().is_some();
    if contract && sums && arith::lone_product(module, yielded).is_some() {
        return Some(Fma.into());
    }
    isa::lanewise(&step)
}

/// The cost row of the op named `op` at `width` bits for `dtype`, or an error that names the
/// missing fact.
fn cost<'t>(
    target: &'t TargetDesc,
    op: &str,
    width: u32,
    dtype: DType,
) -> Result<&'t InstructionCost> {
    target
        .cost_of(op, width, dtype)
        .ok_or_else(|| anyhow!("the target states no cost of {op} at {width} bits for {dtype}"))
}

/// The memories of `target` after the registers that hold pieces of the work, innermost first.
fn stores(target: &TargetDesc) -> Result<Vec<Store>> {
    let units = workers(target)?;
    let mut stores = Vec::new();
    for (index, memory) in target.memories.iter().enumerate() {
        if !matches!(memory.kind, MemoryKind::Cache | MemoryKind::Scratch) {
            continue;
        }
        let Some(bytes) = memory.bytes else {
            bail!("the target states no bytes of memory {}", memory.name);
        };
        let held = match memory.kind {
            MemoryKind::Cache => {
                let ways = memory.ways.expect("a target states the ways of each cache");
                bytes * u64::from(ways - 1) / u64::from(ways)
            }
            _ => bytes,
        };
        let outside = target.memories.get(index + 1);
        let mut sets = None;
        if let (Some(line), Some(ways), Some(page)) = (memory.line, memory.ways, target.isa.page) {
            let way = bytes / u64::from(ways);
            if memory.kind == MemoryKind::Cache && way <= u64::from(page) {
                sets = Some(Sets {
                    line: u64::from(line),
                    count: way / u64::from(line),
                    ways: u64::from(ways),
                });
            }
        }
        stores.push(Store {
            name: memory.name.clone(),
            sets,
            capacity: held / target.units_sharing(memory) as u64,
            sharing: target.units_sharing(memory),
            filled_from_cache: outside.is_some_and(|outer| outer.kind == MemoryKind::Cache),
            copies_only: memory.kind == MemoryKind::Scratch,
            fill_rate: outside.and_then(|outer| target.read_bytes_per_cycle(outer, units)),
        });
    }
    Ok(stores)
}

/// The units of `target`'s parallel level.
fn workers(target: &TargetDesc) -> Result<usize> {
    let level = target.parallel_level();
    target
        .units(&level)
        .ok_or_else(|| anyhow!("the target states no units of level {level}"))
}

/// The iterators of `block`, the block of `inst`, that each value of its body varies along: those
/// that the coordinates of the reads it is computed from take and that the `tile.index` values it
/// is computed from read.
fn varies_along(module: &Module, inst: Inst, block: &Block) -> HashMap<Value, [bool; MAX_ITERS]> {
    let body = module.regions(inst)[0];
    let mut varies: HashMap<Value, [bool; MAX_ITERS]> = HashMap::new();
    for (read, &arg) in block.reads.iter().zip(module.args(body)) {
        let mut along = [false; MAX_ITERS];
        for coord in &read.coords {
            for position in coord.iters() {
                along[position] = true;
            }
        }
        varies.insert(arg, along);
    }
    for body_inst in module.insts(body) {
        let mut along = [false; MAX_ITERS];
        if let Some(index) = module.op(body_inst).get::<Index>() {
            for position in index.expr.iters() {
                along[position] = true;
            }
        }
        for operand in module.operands(body_inst) {
            if let Some(operand_along) = varies.get(operand) {
                for (position, &moves) in operand_along.iter().enumerate() {
                    along[position] |= moves;
                }
            }
        }
        for &result in module.results(body_inst) {
            varies.insert(result, along);
        }
    }
    varies
}

/// The values the body of the block of `inst` computes beside the one it yields, each with the
/// iterators it varies along, by `varies`, and the positions of the body's ops while it is held:
/// from the op that computes it to the last op that takes it, or to the end where none does.
fn temporaries(
    module: &Module,
    inst: Inst,
    varies: &HashMap<Value, [bool; MAX_ITERS]>,
) -> Vec<Temporary> {
    let body = module.regions(inst)[0];
    let insts: Vec<Inst> = module.insts(body).collect();
    let yielded: Vec<Value> = match insts.last() {
        Some(&terminator) => module.operands(terminator).to_vec(),
        None => Vec::new(),
    };
    let mut last_use = HashMap::new();
    for (index, &body_inst) in insts.iter().enumerate() {
        for &operand in module.operands(body_inst) {
            last_use.insert(operand, index);
        }
    }
    let mut temporaries = Vec::new();
    for (index, &body_inst) in insts.iter().enumerate() {
        for &result in module.results(body_inst) {
            if yielded.contains(&result) {
                continue;
            }
            let end = last_use.get(&result).map_or(insts.len(), |&last| last + 1);
            temporaries.push(Temporary {
                varies: varies[&result],
                held: index..end,
            });
        }
    }
    temporaries
}

/// The registers that the blocks `placed` after the block of `inst` hold: the most of those that
/// read the element of the block before them at the point where it finishes it
/// ([`Block::read_at_point`]), and the most of those that read across the finished tile.
fn placed_registers(module: &Module, inst: Inst, placed: &[Inst]) -> (usize, usize) {
    let (mut at_point, mut after) = (0, 0);
    let mut before = inst;
    for &later in placed {
        let registers = most_live(module, later, &|_| true);
        let earlier = module.op(before).get::<Block>();
        let block = module.op(later).get::<Block>();
        let epilogue_input = Block::epilogue_input(module, before, later);
        let reads_point = match (earlier, block, epilogue_input) {
            (Some(earlier_block), Some(later_block), Some(input)) => {
                earlier_block.read_at_point(later_block, input)
            }
            _ => false,
        };
        match reads_point {
            true => at_point = at_point.max(registers),
            false => after = after.max(registers),
        }
        before = later;
    }
    (at_point, after)
}

/// By iterator, the ops of the body of the block of `inst` whose values vary along it, by `varies`,
/// and the ops whose values vary along any.
fn ops_along(
    module: &Module,
    inst: Inst,
    varies: &HashMap<Value, [bool; MAX_ITERS]>,
) -> ([usize; MAX_ITERS], usize) {
    let mut counts = [0; MAX_ITERS];
    let mut varying = 0;
    for body_inst in module.insts(module.regions(inst)[0]) {
        let Some(&result) = module.results(body_inst).first() else {
            continue;
        };
        let along = &varies[&result];
        for (position, &moves) in along.iter().enumerate() {
            counts[position] += usize::from(moves);
        }
        varying += usize::from(along.contains(&true));
    }
    (counts, varying)
}

/// The most values of the body of the block of `inst` that `counted` takes that it holds at once,
/// in the order of its ops: its arguments from the start, and each value an op computes from that
/// op, until its last use.
fn most_live(module: &Module, inst: Inst, counted: &dyn Fn(Value) -> bool) -> usize {
    let body = module.regions(inst)[0];
    let insts: Vec<Inst> = module.insts(body).collect();
    let mut last_use = HashMap::new();
    for (index, &body_inst) in insts.iter().enumerate() {
        for &operand in module.operands(body_inst) {
            last_use.insert(operand, index);
        }
    }
    let mut live: HashSet<Value> = HashSet::new();
    for &arg in module.args(body) {
        if counted(arg) {
            live.insert(arg);
        }
    }
    let mut most = live.len();
    for (index, &body_inst) in insts.iter().enumerate() {
        for &result in module.results(body_inst) {
            if counted(result) {
                live.insert(result);
            }
        }
        most = most.max(live.len());
        for operand in module.operands(body_inst) {
            if last_use.get(operand) == Some(&index) {
                live.remove(operand);
            }
        }
    }
    most
}

/// The name of op `T`, as the target's cost table names it.
fn op_name<T: OpKind>() -> &'static str {
    T::INFO.name
}
