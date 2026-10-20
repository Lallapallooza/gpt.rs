//! The walk that a block's schedule gives: the loops of its modes, its register tile and atom, and
//! the copies and prefetches of its holdings, with the rules that follow from them: the loops of a
//! fold in order, where a fold's state starts and lives, how a loop over tiles builds a tile that
//! holds fewer points, the loops that clips narrow, the axes of a copy, and the piece that placed
//! blocks read. The verifier and `tile-to-loop` read them alike, so the IR states each decision
//! once and what follows from it is computed, never stored.

use std::collections::{BTreeSet, HashMap};

use tinyvec::ArrayVec;

use crate::dialects::tile::ir::*;
use crate::dialects::tile::verify::check_schedule;
use crate::ir::{AccessExpr, DType, Dim, Inst, IterRef, Level, Module, OperandRef, MAX_ITERS};
use crate::targets::TargetDesc;

/// How a block walks its points under its modes.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Walk {
    /// Loops over tiles of the iterators, outermost first: the block's modes but those of its
    /// register tile, or the modes of its reference walk when it has none
    /// ([`Block::reference_modes`]). Every iterator that the register tile leaves out ends in a
    /// loop of one point.
    pub nest: Vec<Loop>,
    /// The register tile: parallel iterators whose points, innermost of all, run at once in
    /// registers, each over its innermost tile, the lanes last.
    pub reg: Vec<RegDim>,
    /// How the register tile computes.
    pub atom: Option<AtomPlan>,
    /// The operands that the block reads from a copy rather than in place.
    pub copies: Vec<CopyAt>,
    /// The operands that the block reads in place and asks a cache for ahead of the reads.
    pub prefetches: Vec<PrefetchAt>,
    /// The iterators whose loops skip the points where a padded read falls outside its operand.
    pub clip: IterList,
    /// The loop of the nest that each of the block's modes is, none for a mode of the register
    /// tile.
    pub nest_of_mode: Vec<Option<usize>>,
}

/// One loop of a nest: over the tiles of `extent` points of iterator `iter`, within its tile of
/// the loop around it, or within its whole extent. The last tile of a range may be smaller.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Loop {
    pub iter: IterRef,
    pub extent: usize,
    pub parallelism: Parallelism,
}

/// How a loop of a nest runs its tiles.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Parallelism {
    /// One after another, in increasing order.
    Seq,
    /// One on each worker of an execution level, which may run at the same time.
    Par(Level),
}

/// One iterator of a register tile and the points of it the tile holds.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RegDim {
    pub iter: IterRef,
    pub extent: usize,
}

/// The atom of a register tile: vectors of `lanes` elements whose lanes run along the iterator
/// `iter`, the last of the tile.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AtomPlan {
    pub lanes: usize,
    pub iter: IterRef,
}

/// A copy of the elements of operand `operand` that the loops inside loop `at` of the nest read,
/// or the whole nest without `at`, which a unit makes in its scratch for each tile of that loop
/// and reads instead of the operand, sized for the memory `level`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CopyAt {
    pub operand: OperandRef,
    pub level: Level,
    pub at: Option<usize>,
}

/// A read of operand `operand` in place that asks the memory `level` for the elements of the next
/// tile of loop `at` of the nest, or for none without `at`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrefetchAt {
    pub operand: OperandRef,
    pub level: Level,
    pub at: Option<usize>,
}

/// An axis of a copy of an operand.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CopyAxis {
    /// The tiles of loop `index` of the nest within its tile of the loop around it: `count` of
    /// them.
    Tiles { index: usize, count: Dim },
    /// The points of the innermost tile of the iterator at `position`, of `extent` points.
    Points { position: usize, extent: Dim },
}

impl CopyAxis {
    /// How many elements the axis holds.
    pub fn extent(&self) -> &Dim {
        match self {
            CopyAxis::Tiles { count, .. } => count,
            CopyAxis::Points { extent, .. } => extent,
        }
    }
}

/// One loop of a block under its walk.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Step {
    /// Loop `index` of the nest, over tiles.
    Tiles(usize),
    /// Over the points of the register tile, all at once.
    Reg,
}

/// How a loop over tiles builds the tiles that do not hold a whole tile's points: the last tile
/// of its range, and a tile that a clip cuts to its points in range.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tail {
    /// Every tile of every loop over the iterator holds a whole tile's points, and no clip cuts
    /// one.
    Divisible,
    /// A tile whose points are known before the run builds a body of its own: the whole tiles
    /// one, the last tile another, and a cut tile one for each count of points it can keep.
    Peel,
    /// One body serves every tile, which masks the points past its range.
    Mask,
}

/// The loops of a fold that skip the points where padded reads fall outside their operands: the
/// coordinates, `(read, axis)`, that each step narrows its loop for, those that each step cuts
/// the register tile for at each of its points, and the reads that the loops keep in range at
/// every point, which read without their pad.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Clips {
    pub steps: HashMap<Step, Vec<(usize, usize)>>,
    pub cuts: HashMap<Step, Vec<(usize, usize)>>,
    pub kept: Vec<bool>,
}

impl Clips {
    /// Whether the loops keep read `read` in range.
    pub fn kept(&self, read: usize) -> bool {
        self.kept.get(read).copied().unwrap_or(false)
    }
}

/// Where the state of a fold into new results lives, as its holding says: one element per point
/// of the tiles that the loops inside its start walk.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StatePlace {
    /// The register tile: a fold that holds no state, whose state's tiles are those of the
    /// register tile.
    Reg,
    /// Private scratch, which each point folds into: one element for a fold that holds no state
    /// and has no register tile, or the scratch of a holding without one.
    Private,
    /// The scratch of a holding. From this step on, the loops walk only the register tile and
    /// reduced iterators. They fold the part of the scratch that the register tile covers into a
    /// copy in registers, which is copied in before them and back after.
    RegCopies(usize),
}

impl Block {
    /// The lanes of the vectors that compute the body of the block of `inst` along the iterator
    /// at `lane` on `target`: as many as the target's widest register holds of the widest dtype
    /// the body computes in ([`Block::vector_work`]), at least two, when the target computes that
    /// work in them ([`TargetDesc::computes_vectors`]). Masks, `i1`, are flags.
    pub fn body_lanes(
        module: &Module,
        inst: Inst,
        lane: usize,
        target: &TargetDesc,
    ) -> Option<usize> {
        let work = Block::vector_work(module, inst, lane).ok()?;
        let mut widest = 0;
        for &dtype in &work.dtypes {
            if dtype != DType::I1 {
                widest = widest.max(dtype.bits());
            }
        }
        if widest == 0 {
            return None;
        }
        let lanes = target.max_vector_bits() as usize / widest;
        if lanes < 2 {
            return None;
        }
        target
            .computes_vectors(&work.ops, &work.dtypes, lanes)
            .ok()?;
        Some(lanes)
    }

    /// The modes that compute the block in vectors of `lanes`, `(iterator, lanes)`: a loop over
    /// the vectors when the iterator holds more than one, then the lanes, with the loops of one
    /// point ([`Block::with_point_modes`]).
    pub fn lane_modes(&self, (lane, lanes): (usize, usize)) -> Vec<Mode> {
        let iter = IterRef::new(lane);
        let lanes_mode = Mode::new(iter, lanes, Some(ModeKind::Lanes));
        let modes = match self.iters[lane].extent.as_static() == Some(lanes) {
            true => vec![lanes_mode],
            false => vec![Mode::new(iter, lanes, None), lanes_mode],
        };
        self.with_point_modes(modes)
    }

    /// The lanes, `(iterator, lanes)`, of the vectors that compute the block of `inst` on
    /// `target`, a block without modes: along the iterator that is the last coordinate of every
    /// write ([`Block::lanes_iterator`]), as many as its body computes in there
    /// ([`Block::body_lanes`]), when the verifier accepts the block with their modes
    /// ([`Block::lane_modes`]).
    pub fn vector_lanes(
        module: &Module,
        inst: Inst,
        target: &TargetDesc,
    ) -> Option<(usize, usize)> {
        let block = module.op(inst).get::<Block>()?;
        if !block.loops.is_empty() {
            return None;
        }
        let lane = block.lanes_iterator()?;
        let lanes = Block::body_lanes(module, inst, lane, target)?;
        let mut tiled = block.clone();
        tiled.loops = block.lane_modes((lane, lanes));
        check_schedule(&tiled).ok()?;
        Some((lane, lanes))
    }

    /// The walk that the block's modes and the holdings of its reads give: its modes but those of
    /// its register tile are the nest, in order; its `reg` modes and its `lanes` mode the register
    /// tile, the lanes last, and the atom; the holdings of its reads its copies and prefetches, at
    /// the loop of the nest that their mode is. A block without modes walks
    /// [`Block::reference_modes`].
    pub fn walk(&self) -> Walk {
        let mut walk = Walk {
            clip: self.clip,
            ..Walk::default()
        };
        let modes = match self.loops.is_empty() {
            true => self.reference_modes(),
            false => self.loops.to_vec(),
        };
        let mut nest_of_mode = Vec::with_capacity(modes.len());
        let mut lanes = None;
        for mode in &modes {
            let parallelism = match &mode.kind {
                None => Parallelism::Seq,
                Some(ModeKind::Workers(workers)) => Parallelism::Par(workers.level.clone()),
                Some(ModeKind::Reg) => {
                    walk.reg.push(RegDim {
                        iter: mode.iter,
                        extent: mode.size,
                    });
                    nest_of_mode.push(None);
                    continue;
                }
                Some(ModeKind::Lanes) => {
                    lanes = Some((mode.iter, mode.size));
                    nest_of_mode.push(None);
                    continue;
                }
            };
            nest_of_mode.push(Some(walk.nest.len()));
            walk.nest.push(Loop {
                iter: mode.iter,
                extent: mode.size,
                parallelism,
            });
        }
        if let Some((iter, lane_count)) = lanes {
            // One vector of the lanes' iterator, when no `reg` mode holds several.
            if walk.reg.last().is_none_or(|last| last.iter != iter) {
                walk.reg.push(RegDim {
                    iter,
                    extent: lane_count,
                });
            }
            walk.atom = Some(AtomPlan {
                lanes: lane_count,
                iter,
            });
        }
        for read in &self.reads {
            let Some(holding) = &read.held else {
                continue;
            };
            let (Holding::Copy(held) | Holding::Prefetch(held)) = holding;
            let at = held
                .at
                .and_then(|mode| nest_of_mode.get(mode).copied().flatten());
            let operand = OperandRef(read.source.first());
            let level = held.level.clone();
            match holding {
                Holding::Copy(_) => walk.copies.push(CopyAt { operand, level, at }),
                Holding::Prefetch(_) => walk.prefetches.push(PrefetchAt { operand, level, at }),
            }
        }
        walk.nest_of_mode = nest_of_mode;
        walk
    }

    /// The modes of the reference walk: one point at a time of every iterator, in
    /// [`Block::walk_order`]; or in the order of the block, when a fold updates an operand where its
    /// points may collide, so that they write in their order.
    pub fn reference_modes(&self) -> Vec<Mode> {
        let in_order = self
            .folds
            .iter()
            .any(|fold| fold.updates_in_order(&self.iters));
        let positions = match in_order {
            true => (0..self.iters.len()).collect(),
            false => self.walk_order(),
        };
        let mut modes = Vec::with_capacity(positions.len());
        for position in positions {
            modes.push(point_mode(position));
        }
        modes
    }

    /// `modes` with a mode of one point, `d by 1`, for each iterator that the register tile does
    /// not hold and whose innermost mode in `modes`, if any, walks pieces of more points. The modes
    /// then say every loop of the block. Each goes inside the modes of its own iterator, and before
    /// the first mode after them that walks an iterator later in [`Block::walk_order`], other than
    /// a mode of workers; the rest go before the register tile, in that order.
    pub fn with_point_modes(&self, modes: Vec<Mode>) -> Vec<Mode> {
        let (innermost, held) = Mode::innermost_pieces(&modes);
        let first_held = modes
            .iter()
            .position(Mode::in_register_tile)
            .unwrap_or(modes.len());
        let order = self.walk_order();
        let mut points: ArrayVec<[usize; MAX_ITERS]> = ArrayVec::new();
        for &position in &order {
            if !held[position] && innermost[position] != Some(1) {
                points.push(position);
            }
        }
        let rank = |position: usize| order.iter().position(|&walked| walked == position);
        let (outside, inside) = modes.split_at(first_held);
        let mut left = [0_usize; MAX_ITERS];
        for mode in outside {
            left[mode.iter.position()] += 1;
        }
        let mut walked = Vec::with_capacity(modes.len() + points.len());
        for mode in outside {
            let position = mode.iter.position();
            if !matches!(mode.kind, Some(ModeKind::Workers(_))) {
                let before = |point: usize| left[point] == 0 && rank(point) < rank(position);
                while let Some(index) = points.iter().position(|&point| before(point)) {
                    walked.push(point_mode(points.remove(index)));
                }
            }
            walked.push(mode.clone());
            left[position] -= 1;
        }
        for point in points {
            walked.push(point_mode(point));
        }
        walked.extend_from_slice(inside);
        walked
    }

    /// The iterators of the block in the order its loops walk them by default, unless a fold
    /// updates an operand where its points may collide (see [`Block::reference_modes`]):
    /// 1. the iterators that are not reduced and that the coordinates of its first write take, in
    ///    the order they take them;
    /// 2. the other iterators that are not reduced, in the order of the block;
    /// 3. the reduced ones, in the order of the block.
    ///
    /// The loops then walk the elements of the first write in the order of its memory.
    pub fn walk_order(&self) -> ArrayVec<[usize; MAX_ITERS]> {
        let mut order = ArrayVec::new();
        let first_write = self.folds.first().and_then(|fold| fold.writes.first());
        for coord in first_write.map_or(&[][..], |write| &write.coords[..]) {
            for position in coord.iters() {
                if !self.iters[position].kind.is_reduced() && !order.contains(&position) {
                    order.push(position);
                }
            }
        }
        for reduced in [false, true] {
            for (position, iter) in self.iters.iter().enumerate() {
                if iter.kind.is_reduced() == reduced && !order.contains(&position) {
                    order.push(position);
                }
            }
        }
        order
    }
}

/// The mode of one point of the iterator at `position`.
fn point_mode(position: usize) -> Mode {
    Mode::new(IterRef::new(position), 1, None)
}

/// Whether each iterator of `block` has one point, which every loop holds fixed.
fn one_point_iters(block: &Block) -> [bool; MAX_ITERS] {
    let mut fixed = [false; MAX_ITERS];
    for (position, iter) in block.iters.iter().enumerate() {
        fixed[position] = iter.extent.as_static() == Some(1);
    }
    fixed
}

impl Mode {
    /// The mode of `iter` in pieces of `size` points, run as `kind` says, which names no memory.
    pub fn new(iter: IterRef, size: usize, kind: Option<ModeKind>) -> Mode {
        Mode {
            iter,
            size,
            kind,
            memory: None,
        }
    }

    /// This mode, walking the pieces that `memory` holds.
    pub fn in_memory(self, memory: Level) -> Mode {
        Mode {
            memory: Some(memory),
            ..self
        }
    }

    /// Whether the mode is one of the register tile: `reg` or `lanes`.
    pub fn in_register_tile(&self) -> bool {
        matches!(self.kind, Some(ModeKind::Reg | ModeKind::Lanes))
    }

    /// For each iterator of a block with the modes `modes`, the size of the pieces of its
    /// innermost mode outside the register tile, none where none walks it, and whether a mode of
    /// the register tile holds it.
    pub fn innermost_pieces(modes: &[Mode]) -> ([Option<usize>; MAX_ITERS], [bool; MAX_ITERS]) {
        let mut innermost = [None; MAX_ITERS];
        let mut held = [false; MAX_ITERS];
        for mode in modes {
            let position = mode.iter.position();
            match mode.in_register_tile() {
                true => held[position] = true,
                false => innermost[position] = Some(mode.size),
            }
        }
        (innermost, held)
    }
}

impl Walk {
    /// The loops of the block, outermost first: the nest's loops, then the register tile.
    pub fn steps(&self) -> Vec<Step> {
        let mut steps: Vec<Step> = (0..self.nest.len()).map(Step::Tiles).collect();
        if !self.reg.is_empty() {
            steps.push(Step::Reg);
        }
        steps
    }

    /// Whether the register tile holds the iterator at `position`.
    pub fn in_reg(&self, position: usize) -> bool {
        self.reg.iter().any(|dim| dim.iter.position() == position)
    }

    /// How many of the last of the tiles `tiles` of a fold's state, each `(position, extent)` of
    /// a parallel iterator, its register tile computes at once: those that are each a tile of the
    /// register tile, whose copies in and out of the state are one atom; the tiles before them
    /// are loops of scalars.
    pub fn state_atom_tiles(&self, tiles: &[(usize, usize)]) -> usize {
        let mut count = 0;
        for &(position, extent) in tiles.iter().rev() {
            let held = self
                .reg
                .iter()
                .any(|dim| dim.iter.position() == position && dim.extent == extent);
            if !held {
                break;
            }
            count += 1;
        }
        count
    }

    /// Whether `step` walks the iterator at `position`.
    pub fn walks(&self, step: Step, position: usize) -> bool {
        match step {
            Step::Tiles(index) => self.nest[index].iter.position() == position,
            Step::Reg => self.in_reg(position),
        }
    }

    /// The iterator a step walks, of a step that walks one.
    pub fn iter_of(&self, step: Step) -> usize {
        match step {
            Step::Tiles(index) => self.nest[index].iter.position(),
            Step::Reg => panic!("the register tile walks several iterators"),
        }
    }

    /// Whether each coordinate of `read`, a read of `block`, moves with the points of the register
    /// tile's iterator at `position` as a base plus a multiple of each, as the register tile's
    /// points need: affine in it, or `floordiv` and `mod` that the tiles of the iterator make
    /// affine ([`Walk::in_tile_terms`]).
    pub fn reads_in_steps(&self, block: &Block, read: &Read, position: usize) -> bool {
        if read.moves_by_multiples(position) {
            return true;
        }
        let Some((coords, within)) = self.in_tile_terms(block, read, position) else {
            return false;
        };
        coords.iter().all(|coord| {
            within
                .iter()
                .all(|&entry| coord.coefficient(entry).is_some())
        })
    }

    /// Whether `read`, a read of `block`, takes consecutive elements along the lanes, which run
    /// along the iterator at `lane`, or does not move with them: in its last coordinate by 1 and
    /// in no other, or so in the terms of its tiles ([`Walk::in_tile_terms`]), where a `floordiv`
    /// by a multiple of the lanes stays the same within a vector.
    pub fn reads_lanes_consecutively(&self, block: &Block, read: &Read, lane: usize) -> bool {
        if read.contiguous_along(lane) {
            return true;
        }
        let Some((coords, within)) = self.in_tile_terms(block, read, lane) else {
            return false;
        };
        let lane_entry = *within.last().expect("the lanes are an entry of the tile");
        let Some((last, outer)) = coords.split_last() else {
            return true;
        };
        !outer.iter().any(|coord| coord.uses_iter(lane_entry))
            && matches!(last.coefficient(lane_entry), Some(0 | 1))
    }

    /// The coordinates of `read`, a read of `block`, in the terms of the tiles of the register
    /// tile's iterator at `position`, and the positions of those terms that move within a tile:
    /// where no clip narrows them, each loop over the iterator moves its tiles on by its pieces,
    /// so the iterator is the sum of each mode's piece times that mode's index, plus a point of
    /// the register tile, and along the lanes that point is a vector of lanes plus a lane; each
    /// coordinate is then brought to its canonical form. None for a clipped iterator.
    pub fn in_tile_terms(
        &self,
        block: &Block,
        read: &Read,
        position: usize,
    ) -> Option<(Vec<AccessExpr>, Vec<usize>)> {
        let dim = self
            .reg
            .iter()
            .find(|dim| dim.iter.position() == position)?;
        if self.clip.contains(position) {
            return None;
        }
        // The tile's start, a term per mode over the iterator after the block's own iterators.
        let mut next = block.iters.len();
        let mut start = AccessExpr::constant(0);
        for entry in &self.nest {
            if entry.iter.position() == position {
                start = start + AccessExpr::iter(next) * AccessExpr::constant(entry.extent as i64);
                next += 1;
            }
        }
        let lanes_along = self
            .atom
            .as_ref()
            .filter(|atom| atom.iter.position() == position)
            .map(|atom| atom.lanes);
        let (within, form) = match lanes_along {
            Some(lanes) => (
                vec![(next, dim.extent / lanes), (next + 1, lanes)],
                start
                    + AccessExpr::iter(next) * AccessExpr::constant(lanes as i64)
                    + AccessExpr::iter(next + 1),
            ),
            None => (vec![(next, dim.extent)], start + AccessExpr::iter(next)),
        };
        let extent = |at: usize| {
            within
                .iter()
                .find(|&&(entry, _)| entry == at)
                .map(|&(_, points)| Dim::Static(points))
                .or_else(|| block.iters.get(at).map(|iter| iter.extent.clone()))
        };
        let mut coords = Vec::with_capacity(read.coords.len());
        for coord in &read.coords {
            let substituted = coord.substitute(&|at| (at == position).then(|| form.clone()));
            coords.push(substituted.canonical(&extent));
        }
        let entries = within.iter().map(|&(entry, _)| entry).collect();
        Some((coords, entries))
    }

    /// The innermost tile that the nest tiles each iterator to, `None` for one it leaves whole.
    pub fn innermost_tiles(&self) -> [Option<usize>; MAX_ITERS] {
        let mut tiles = [None; MAX_ITERS];
        for entry in &self.nest {
            tiles[entry.iter.position()] = Some(entry.extent);
        }
        tiles
    }

    /// How the loops over the tiles of the iterator at `position` of `block` build the tiles that
    /// do not hold a whole tile's points: an iterator whose tiles are all whole needs nothing; a
    /// loop over the tiles of an iterator of the register tile peels such a tile, so that the
    /// register tile keeps a shape known before the run; a loop over the tiles of any other
    /// iterator masks it.
    pub fn tail(&self, block: &Block, position: usize) -> Tail {
        match (self.whole_tiles(block, position), self.in_reg(position)) {
            (true, _) => Tail::Divisible,
            (false, true) => Tail::Peel,
            (false, false) => Tail::Mask,
        }
    }

    /// Whether every tile of every loop of the nest over the iterator at `position` of `block`
    /// holds a whole tile's points: its extent is static, each loop's tiles divide the tiles
    /// around them, and no clip narrows them.
    pub fn whole_tiles(&self, block: &Block, position: usize) -> bool {
        let Some(mut around) = block.iters[position].extent.as_static() else {
            return false;
        };
        if self.clip.contains(position) {
            return false;
        }
        for entry in &self.nest {
            if entry.iter.position() != position {
                continue;
            }
            if around % entry.extent != 0 {
                return false;
            }
            around = entry.extent;
        }
        true
    }

    /// The step of `steps` before which the state of `fold`, a fold of `block`, starts: the loop of
    /// the mode its holding names; without one, the first loop over a reduced iterator, or the
    /// register tile when no other loop walks one.
    pub fn fold_start(&self, block: &Block, steps: &[Step], fold: &Fold) -> usize {
        if let Some(StateHolding::Scratch(scratch)) = &fold.held {
            let nested = self.nest_of_mode[scratch.at].expect("a state is held at a loop");
            return nested;
        }
        let iters = &block.iters;
        steps
            .iter()
            .position(|&step| match step {
                Step::Reg => false,
                step => iters[self.iter_of(step)].kind.is_reduced(),
            })
            .or_else(|| steps.iter().position(|&step| step == Step::Reg))
            .unwrap_or(steps.len())
    }

    /// The tiles of a state of a fold of `block` with the steps `steps` that starts before step
    /// `start`: the parallel iterators that the steps from there on walk, with the extent of their
    /// tile there, those of the register tile last, in its order; a tile of the register tile's
    /// iterator that holds fewer points than the register tile keeps the register tile's. The
    /// state keeps one element per point of those tiles.
    pub fn state_tiles(&self, block: &Block, steps: &[Step], start: usize) -> Vec<(usize, Dim)> {
        let iters = &block.iters;
        let mut positions = Vec::new();
        for (position, iter) in iters.iter().enumerate() {
            if !iter.kind.is_reduced() && !self.in_reg(position) {
                positions.push(position);
            }
        }
        for dim in &self.reg {
            positions.push(dim.iter.position());
        }
        let mut tiles = Vec::new();
        for position in positions {
            if !steps[start..]
                .iter()
                .any(|&step| self.walks(step, position))
            {
                continue;
            }
            let outer = steps[..start].iter().rev().find_map(|&step| match step {
                Step::Tiles(index) if self.nest[index].iter.position() == position => {
                    Some(Dim::Static(self.nest[index].extent))
                }
                _ => None,
            });
            let walked = outer.unwrap_or_else(|| iters[position].extent.clone());
            let held = self.reg.iter().find(|dim| dim.iter.position() == position);
            let tile = match (held, walked.as_static()) {
                (Some(dim), Some(points)) if points < dim.extent => Dim::Static(dim.extent),
                _ => walked,
            };
            tiles.push((position, tile));
        }
        tiles
    }

    /// Whether registers hold a state of the tiles `tiles`: one tile of each iterator of the
    /// register tile, of its extent there.
    pub fn in_registers(&self, tiles: &[(usize, Dim)]) -> bool {
        !self.reg.is_empty()
            && tiles.len() == self.reg.len()
            && tiles
                .iter()
                .zip(&self.reg)
                .all(|((position, extent), dim)| {
                    *position == dim.iter.position() && *extent == Dim::Static(dim.extent)
                })
    }

    /// The step from which a part of a fold's state that starts before step `start`, and that
    /// registers do not hold, fits in registers: the first step after `start` from which on the
    /// steps walk reduced iterators and the register tile alone, when registers hold the state of
    /// the tiles they walk.
    pub fn register_run(&self, block: &Block, steps: &[Step], start: usize) -> Option<usize> {
        let iters = &block.iters;
        let walks_reduced_or_reg = |step: Step| match step {
            Step::Reg => true,
            step => iters[self.iter_of(step)].kind.is_reduced(),
        };
        let run = (start + 1..steps.len()).find(|&index| {
            steps[index..]
                .iter()
                .all(|&step| walks_reduced_or_reg(step))
        })?;
        self.in_registers(&self.state_tiles(block, steps, run))
            .then_some(run)
    }

    /// Where the state of `fold`, a fold of `block` with the steps `steps` that starts before step
    /// `start`, lives. Without a holding, it lives in registers when the block has a register
    /// tile, and in one element of private scratch otherwise. With one, it lives in its scratch.
    /// When loops over the register tile and reduced iterators alone follow, they fold the part of
    /// the scratch that the register tile covers into a copy in registers. Otherwise, or when a
    /// prefix folds each element in turn, each point folds into the scratch.
    pub fn state_place(
        &self,
        block: &Block,
        steps: &[Step],
        start: usize,
        fold: &Fold,
    ) -> StatePlace {
        if fold.held.is_none() {
            return match self.reg.is_empty() {
                true => StatePlace::Private,
                false => StatePlace::Reg,
            };
        }
        if block.iters.iter().any(|iter| iter.kind == IterKind::Prefix) {
            return StatePlace::Private;
        }
        match self.register_run(block, steps, start) {
            Some(run) => StatePlace::RegCopies(run),
            None => StatePlace::Private,
        }
    }

    /// The axes of a copy of what `read` of `block` reads inside loop `at` of the nest, or the
    /// whole nest, in the order that the loops inside walk its elements: one for each loop inside
    /// over an iterator the read takes, over its tiles, or its points for a loop of one point at a
    /// time, then one for the points of each iterator of the register tile that the read takes.
    pub fn copy_axes(&self, block: &Block, read: &Read, at: Option<usize>) -> Vec<CopyAxis> {
        let mut taken = BTreeSet::new();
        for coord in &read.coords {
            taken.extend(coord.iters());
        }
        let iters = &block.iters;
        let first = at.map_or(0, |index| index + 1);
        let mut around: Vec<Dim> = iters.iter().map(|iter| iter.extent.clone()).collect();
        for entry in &self.nest[..first] {
            around[entry.iter.position()] = Dim::Static(entry.extent);
        }
        let mut axes = Vec::new();
        for (index, entry) in self.nest.iter().enumerate().skip(first) {
            let position = entry.iter.position();
            if !taken.contains(&position) {
                continue;
            }
            if entry.extent == 1 {
                axes.push(CopyAxis::Points {
                    position,
                    extent: around[position].clone(),
                });
                continue;
            }
            let count = match &around[position] {
                Dim::Static(extent) => Dim::Static(extent.div_ceil(entry.extent)),
                dynamic => dynamic.clone(),
            };
            axes.push(CopyAxis::Tiles { index, count });
            around[position] = Dim::Static(entry.extent);
        }
        for dim in &self.reg {
            let position = dim.iter.position();
            if taken.contains(&position) {
                axes.push(CopyAxis::Points {
                    position,
                    extent: around[position].clone(),
                });
            }
        }
        axes
    }

    /// Whether a copy of what `read` reads, with the axes `axes`, moves whole vectors: its last
    /// axis holds the points of the atom's lanes, along which the read takes consecutive elements.
    pub fn copies_vectors(&self, read: &Read, axes: &[CopyAxis]) -> bool {
        let Some((atom, CopyAxis::Points { position, .. })) = self.atom.as_ref().zip(axes.last())
        else {
            return false;
        };
        atom.iter.position() == *position && read.contiguous_along(*position)
    }

    /// The outermost of the loops `steps` of a fold of `block` over one of the iterators `iters`
    /// that a coordinate takes, inside loops that fix all the others: a loop of the nest,
    /// sequential for an iterator of the register tile, which cuts each tile and keeps its shape.
    /// Narrowed there to where the coordinate stays in range, the loop skips every point where it
    /// does not. None when the register tile is reached first.
    pub fn clip_step(
        &self,
        block: &Block,
        steps: &[Step],
        iters: &BTreeSet<usize>,
    ) -> Option<Step> {
        let mut fixed = one_point_iters(block);
        for &step in steps {
            let Step::Tiles(nested) = step else {
                return None;
            };
            let entry = &self.nest[nested];
            let position = entry.iter.position();
            let others_fixed = iters.iter().all(|&other| other == position || fixed[other]);
            let keeps_reg = !self.in_reg(position) || entry.parallelism == Parallelism::Seq;
            if iters.contains(&position) && others_fixed && keeps_reg {
                return Some(step);
            }
            fixed[position] |= entry.extent == 1;
        }
        None
    }

    /// The register tile's iterator that a coordinate taking the iterators `iters` of `block`
    /// cuts the register tile along, and the loop of `steps` at whose points it does: a
    /// sequential loop over one point, inside every loop over that iterator's tiles, that fixes
    /// the last of the coordinate's other iterators. The iterator is the one of `iters` that the
    /// register tile holds other than along its lanes, and its extent is known before the run, so
    /// that each of its tiles has a count of points known too. At each point of that loop the
    /// coordinate moves with the cut iterator alone, and the register tile keeps the points of
    /// its tile where the coordinate stays in range. None when the register tile is reached
    /// first.
    pub fn cut_step(
        &self,
        block: &Block,
        steps: &[Step],
        iters: &BTreeSet<usize>,
    ) -> Option<(Step, usize)> {
        let lanes = self.atom.as_ref().map(|atom| atom.iter.position());
        let mut held: ArrayVec<[usize; MAX_ITERS]> = ArrayVec::new();
        for &position in iters {
            if self.in_reg(position) && Some(position) != lanes {
                held.push(position);
            }
        }
        let &[cut] = held.as_slice() else {
            return None;
        };
        block.iters[cut].extent.as_static()?;
        let last_tiles = self
            .nest
            .iter()
            .rposition(|entry| entry.iter.position() == cut);
        let mut fixed = one_point_iters(block);
        for &step in steps {
            let Step::Tiles(nested) = step else {
                return None;
            };
            let entry = &self.nest[nested];
            let position = entry.iter.position();
            fixed[position] |= entry.extent == 1;
            let inside_tiles = last_tiles.is_none_or(|last| nested > last);
            let one_point = entry.extent == 1 && entry.parallelism == Parallelism::Seq;
            let others_fixed = iters.iter().all(|&other| other == cut || fixed[other]);
            if one_point && iters.contains(&position) && inside_tiles && others_fixed {
                return Some((step, cut));
            }
        }
        None
    }

    /// The clips of the loops `steps` of a fold of the block of `inst`, whose numerics let a loop
    /// skip the products of the pad 0 summed into a float when `skip_zero_products` holds. For
    /// each padded read in place that contributes nothing where it falls outside its operand, each
    /// coordinate that can leave its axis is narrowed by the loop of `steps` that
    /// [`Walk::clip_step`] names, or else cuts the register tile at the loop that
    /// [`Walk::cut_step`] names, when the block clips the iterator it narrows or cuts and the
    /// coordinate moves by a constant nonzero step with it. A read whose coordinates all are
    /// narrowed or cut so, or stay in, reads without its pad; the others keep it.
    pub fn clips(
        &self,
        module: &Module,
        inst: Inst,
        steps: &[Step],
        skip_zero_products: bool,
    ) -> Clips {
        let block = module.op(inst).get::<Block>().expect("clips of a block");
        let mut clips = Clips {
            steps: HashMap::new(),
            cuts: HashMap::new(),
            kept: vec![false; block.reads.len()],
        };
        for (index, leaving) in block.skipped_pads(module, inst, skip_zero_products) {
            let read = &block.reads[index];
            if self.copied(read) {
                continue;
            }
            let mut kept = true;
            for axis in leaving {
                let coord = &read.coords[axis];
                let clip_step = match coord.reads().is_empty() {
                    true => self.clip_step(block, steps, &coord.iters()),
                    false => None,
                };
                let clips_iter = |iter: usize| {
                    let moves = coord
                        .coefficient(iter)
                        .is_some_and(|coefficient| coefficient != 0);
                    self.clip.contains(iter) && moves
                };
                let clipped = clip_step.filter(|&step| clips_iter(self.iter_of(step)));
                let cut = match (clipped, coord.reads().is_empty()) {
                    (None, true) => self
                        .cut_step(block, steps, &coord.iters())
                        .filter(|&(_, iter)| clips_iter(iter)),
                    _ => None,
                };
                match (clipped, cut) {
                    (Some(step), _) => clips.steps.entry(step).or_default().push((index, axis)),
                    (None, Some((step, _))) => {
                        clips.cuts.entry(step).or_default().push((index, axis))
                    }
                    (None, None) => kept = false,
                }
            }
            clips.kept[index] = kept;
        }
        clips
    }

    /// Whether `read` takes its elements from a copy that a holding makes, which holds its pad.
    fn copied(&self, read: &Read) -> bool {
        let Source::Operand(operand) = read.source else {
            return false;
        };
        self.copies
            .iter()
            .any(|copy| copy.operand == OperandRef(operand))
    }

    /// The piece of `block`'s loops that the blocks placed after it at `placement` read the
    /// elements of: the step of its loops before which the piece starts, and the tile of each
    /// iterator there. The piece is one tile of the loop of the placement's mode, or the register
    /// tile; for a fold, the elements finish with its state, so the piece starts no further in
    /// than its state does.
    pub fn piece(&self, block: &Block, placement: Placement) -> (usize, Vec<Dim>) {
        let steps = self.steps();
        let reg = steps
            .iter()
            .position(|&step| step == Step::Reg)
            .unwrap_or(steps.len());
        let mode_step = |mode: usize| {
            let nested = self.nest_of_mode.get(mode).copied().flatten();
            nested.map_or(reg, |index| index + 1)
        };
        let mut start = match placement {
            Placement::Reg => reg,
            Placement::Mode(mode) => mode_step(mode),
        };
        if let [fold] = block.folds.as_slice() {
            if fold.combine != CombineKind::None {
                start = start.min(self.fold_start(block, &steps, fold));
            }
        }
        (start, self.tiles_inside(block, &steps[..start]))
    }

    /// The tile of each iterator of `block` inside the loops `steps`: that of the innermost of
    /// them over it, or its whole extent.
    pub fn tiles_inside(&self, block: &Block, steps: &[Step]) -> Vec<Dim> {
        let mut tiles: Vec<Dim> = block.iters.iter().map(|iter| iter.extent.clone()).collect();
        for &step in steps {
            if let Step::Tiles(index) = step {
                let entry = &self.nest[index];
                tiles[entry.iter.position()] = Dim::Static(entry.extent);
            }
        }
        tiles
    }
}
