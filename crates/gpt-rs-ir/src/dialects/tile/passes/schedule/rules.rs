//! The rules that choose a block's schedule from its facts: its register tile, the order of its
//! loops, the workers' shares, the pieces of the caches and the copies into scratch. Each rule is a
//! constraint on the target's facts or a count of what the schedule moves. Issue and fill rates
//! enter only the constraints: that a step's loads or a cache's bytes keep up with the fold's
//! steps, and that the accumulators cover the latency of the step. No time is predicted for a
//! schedule, and none ranks schedules.

use std::collections::HashMap;
use std::ops::Range;

use tinyvec::ArrayVec;

use super::facts::{Facts, ReadFacts};
use crate::ir::{MAX_GROUP_RANK, MAX_ITERS};

/// The most iterators the rows of a register tile run along: the axes of a group of registers
/// ([`MAX_GROUP_RANK`]) but the one of the lanes' vectors.
const MAX_ROW_ITERATORS: usize = MAX_GROUP_RANK - 1;

/// The most register tiles that cover a share cut by the ends of its iterators: two along each
/// row's iterator and two along the lanes' ([`Facts::loaded`]).
const MAX_COVERS: usize = 1 << (MAX_ROW_ITERATORS + 1);

/// A register tile: the points of each iterator its rows run along, the outermost first, in the
/// order the vector code walks them, by `vectors` vectors of the lanes.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(super) struct RegisterTile {
    pub rows: ArrayVec<[(usize, usize); MAX_ROW_ITERATORS]>,
    pub vectors: usize,
}

impl RegisterTile {
    /// The tile of `vectors` vectors without rows.
    fn vectors(vectors: usize) -> RegisterTile {
        RegisterTile {
            rows: ArrayVec::new(),
            vectors,
        }
    }

    /// Its accumulators: one for each of its points along its rows and each of its vectors.
    pub fn accumulators(&self) -> usize {
        let mut count = self.vectors;
        for &(_, points) in &self.rows {
            count *= points;
        }
        count
    }
}

/// A schedule: its register tile, the iterators of its loops outside the register tile,
/// outermost first, the parallel ones and then the reduced ones, the points of each iterator in a
/// worker's share, the pieces of the caches that hold one ([`Facts::cache_pieces`]), and the
/// reads copied into scratch ([`Facts::scratch_copies`]).
#[derive(Debug, Clone)]
pub(super) struct Plan {
    pub tile: RegisterTile,
    pub parallel: ArrayVec<[usize; MAX_ITERS]>,
    pub reduced: ArrayVec<[usize; MAX_ITERS]>,
    pub share: [usize; MAX_ITERS],
    pub pieces: Vec<Piece>,
    pub copies: Vec<ScratchCopy>,
}

/// The piece of memory `memory`: the points of each iterator in one of its pieces.
#[derive(Debug, Clone)]
pub(super) struct Piece {
    pub memory: usize,
    pub points: [usize; MAX_ITERS],
}

/// A copy of read `read` into the scratch memory `memory`, made inside the first `depth` loops of
/// the nest that the plan's pieces and loops make ([`Facts::plan_nest`]).
#[derive(Debug, Clone, Copy)]
pub(super) struct ScratchCopy {
    pub read: usize,
    pub memory: usize,
    pub depth: usize,
}

/// What the plan rules compare of a candidate plan ([`Facts::counted`]), in order:
/// - whether its steps load no slower than they step the fold;
/// - the bytes its busiest worker moves into each memory that cannot take them in within the time
///   of its steps, then the vectors it loads, the registers a tile holds and its tiles;
/// - the same with the bytes of every memory;
/// - where it came among the candidates.
struct Counted {
    bound: bool,
    held_back: Vec<u64>,
    counts: Vec<u64>,
    came: usize,
    plan: Plan,
}

/// Puts `candidate` into `by_share` in the place of the plan of its share, where its loads keep up
/// with its steps and the plan's do not, or it holds the steps back by fewer bytes; or as the
/// first plan of its share.
fn keep_best_of_share(by_share: &mut Vec<Counted>, candidate: Counted) {
    let same_share = by_share
        .iter_mut()
        .find(|entry| entry.plan.share == candidate.plan.share);
    let Some(entry) = same_share else {
        by_share.push(candidate);
        return;
    };
    if (!candidate.bound, &candidate.held_back) < (!entry.bound, &entry.held_back) {
        *entry = candidate;
    }
}

/// What the workers of a split share ([`Facts::shared`]): the range of each iterator in each
/// worker's part ([`Facts::worker_ranges`]); the bytes each finds of each read in each cache, by
/// worker, cache and read; and, by cache, worker and read, the workers of its group that take the
/// same footprint of the read, itself among them.
struct Shared {
    ranges: Vec<[Range<usize>; MAX_ITERS]>,
    resident: Vec<Vec<Vec<u64>>>,
    takers: Vec<Vec<Vec<usize>>>,
}

/// The loops that the workers split ([`Facts::splits`]): the iterator of each and the points of
/// each part.
type Splits = ArrayVec<[(usize, usize); MAX_ITERS]>;

/// What the workers share, by split.
type SharedBySplit = HashMap<Splits, Shared>;

/// A loop of a nest: its iterator, the points of it that each of its iterations walks, and the
/// memory whose pieces it walks, if any.
#[derive(Debug, Clone, Copy)]
pub(super) struct Loop {
    pub position: usize,
    pub size: usize,
    pub memory: Option<usize>,
}

impl Facts {
    /// The points of each iterator that the register tile `tile` holds: its rows, its vectors of
    /// the lanes, which may hold more points than the iterator has, and one of each other.
    pub fn register_points(&self, tile: RegisterTile) -> [usize; MAX_ITERS] {
        let mut points = [1; MAX_ITERS];
        if let Some((lane, lanes)) = self.lanes {
            points[lane] = tile.vectors * lanes;
        }
        for &(row, count) in &tile.rows {
            points[row] = count;
        }
        points
    }

    /// The register tiles whose rows run along one of the iterators `rows_of`, alone or with one
    /// other of the iterators `candidates` outside or inside it, or along none, by up to as many
    /// vectors as the lanes' iterator fills: at least two points of each row iterator, and their
    /// registers ([`Facts::held`]) fit the target's. Whether the accumulators cover the latency of
    /// the fold's step depends on the rate the steps run at, which the plan counts
    /// ([`Facts::covers_latency`]).
    pub fn register_tiles(&self, rows_of: &[usize], candidates: &[usize]) -> Vec<RegisterTile> {
        let Some((lane, lanes)) = self.lanes else {
            return vec![RegisterTile::vectors(1)];
        };
        let most_vectors = self.extents[lane].div_ceil(lanes);
        let mut row_sets: Vec<ArrayVec<[usize; MAX_ROW_ITERATORS]>> = vec![ArrayVec::new()];
        for &row in rows_of {
            let mut alone = ArrayVec::new();
            alone.push(row);
            row_sets.push(alone);
            for &other in candidates {
                for pair in [[row, other], [other, row]] {
                    if other == row || pair.len() > MAX_ROW_ITERATORS {
                        continue;
                    }
                    let mut rows = ArrayVec::new();
                    rows.extend_from_slice(&pair);
                    if !row_sets.contains(&rows) {
                        row_sets.push(rows);
                    }
                }
            }
        }
        let mut tiles = Vec::new();
        for rows in &row_sets {
            self.grow_tiles(RegisterTile::vectors(1), rows, most_vectors, &mut tiles);
        }
        tiles
    }

    /// Pushes onto `tiles` the register tiles that grow `tile` by the row iterators `rest`,
    /// outermost first, at least two points of each and at most its extent, then by up to
    /// `most_vectors` vectors, whose registers fit the target's. The registers grow with each
    /// count, so each count stops at the first that does not fit.
    fn grow_tiles(
        &self,
        tile: RegisterTile,
        rest: &[usize],
        most_vectors: usize,
        tiles: &mut Vec<RegisterTile>,
    ) {
        let Some((&row, others)) = rest.split_first() else {
            for vectors in 1..=most_vectors {
                let grown = RegisterTile { vectors, ..tile };
                if self.held(grown) > self.registers {
                    break;
                }
                tiles.push(grown);
            }
            return;
        };
        for points in 2..=self.extents[row] {
            let mut grown = tile;
            grown.rows.push((row, points));
            if self.held(grown) > self.registers {
                break;
            }
            self.grow_tiles(grown, others, most_vectors, tiles);
        }
    }

    /// The registers that the register tile `tile` holds while it steps: its accumulators, the
    /// body's temporaries ([`Facts::temporaries_held`]) and the pads of its padded reads, plus the
    /// operands of one step ([`Facts::step_registers`]) or the blocks placed at it that run on each
    /// element as it finishes, whichever hold more. The blocks that run over the finished tile hold
    /// theirs after it, as the code keeps the tile in memory for them.
    fn held(&self, tile: RegisterTile) -> usize {
        let operands = self.step_registers(tile).max(self.placed_at_point);
        let stepping = tile.accumulators() + self.temporaries_held(tile) + self.pads + operands;
        stepping.max(self.placed_after)
    }

    /// The points of the register tile `tile` along the iterators of `along`: its points along
    /// each of its rows' iterators there, by its vectors where the lanes' iterator is there.
    fn points_along(&self, tile: RegisterTile, along: &[bool]) -> usize {
        let mut count = 1;
        if let Some((lane, _)) = self.lanes {
            if along[lane] {
                count *= tile.vectors;
            }
        }
        for &(row, points) in &tile.rows {
            if along[row] {
                count *= points;
            }
        }
        count
    }

    /// The most registers the body's temporaries hold at once in the register tile `tile`, as
    /// the vector code computes each at the tile's points along the iterators it varies along
    /// ([`Facts::points_along`]) and holds it from the op that computes it to its last use.
    fn temporaries_held(&self, tile: RegisterTile) -> usize {
        let mut end = 0;
        for temporary in &self.temporaries {
            end = end.max(temporary.held.end);
        }
        let mut most = 0;
        for position in 0..end {
            let mut held = 0;
            for temporary in &self.temporaries {
                if temporary.held.contains(&position) {
                    held += self.points_along(tile, &temporary.varies);
                }
            }
            most = most.max(held);
        }
        most
    }

    /// The schedule among the register tiles `tiles`. The candidates are the tiles, orders of the
    /// parallel loops and workers' shares of them ([`Facts::shares`]), each with the pieces of the
    /// caches it needs ([`Facts::cache_pieces`]). A candidate's accumulators cover the latency of
    /// the fold's step at the rate the steps run ([`Facts::step_rate`]), or are the most that a
    /// tile holds where none holds that many ([`Facts::covers_latency`]). The rules keep those
    /// whose steps load no slower than they step the fold, when any does, and then take:
    /// 1. for each share, the one whose busiest worker moves the fewest bytes into each memory that
    ///    cannot take them in within the time of the share's steps, outermost first
    ///    ([`Facts::busiest_moved`], [`Facts::keeps_up`]), as a memory that keeps up holds nothing
    ///    back;
    /// 2. of the shares, the one whose plan moves the fewest bytes into each memory, outermost
    ///    first;
    /// 3. of either, the one that loads the fewest vectors, then holds the fewest registers, as
    ///    registers that move no fewer bytes buy nothing, then runs the fewest tiles, then comes
    ///    first, as the orders and shares come in the order they are preferred where they tie.
    ///
    /// The reduced loops run inside ([`Facts::reduced_order`]).
    pub fn plan(&self, tiles: &[RegisterTile]) -> Option<Plan> {
        let reduced = self.reduced_order();
        let mut shared = HashMap::new();
        let mut most = 0;
        for tile in tiles {
            most = most.max(tile.accumulators());
        }
        let mut by_share: Vec<Counted> = Vec::new();
        let mut came = 0;
        for &tile in tiles {
            let register = self.register_points(tile);
            for parallel in self.parallel_orders(&register) {
                for share in self.shares(&register, &parallel) {
                    came += 1;
                    let mut plan = Plan {
                        tile,
                        parallel,
                        reduced,
                        share,
                        pieces: Vec::new(),
                        copies: Vec::new(),
                    };
                    plan.pieces = self.cache_pieces(&plan);
                    if let Some(candidate) = self.counted(plan, (most, came), &mut shared) {
                        keep_best_of_share(&mut by_share, candidate);
                    }
                }
            }
        }
        let mut best: Option<Counted> = None;
        for candidate in by_share {
            let better = best.as_ref().is_none_or(|least| {
                (!candidate.bound, &candidate.counts, candidate.came)
                    < (!least.bound, &least.counts, least.came)
            });
            if better {
                best = Some(candidate);
            }
        }
        let mut plan = best?.plan;
        plan.copies = self.scratch_copies(&plan);
        Some(plan)
    }

    /// What the plan rules compare of `plan`, the `came`-th candidate, whose register tile is
    /// among tiles that hold at most `most` accumulators ([`Facts::plan`]): none where its
    /// accumulators do not cover the latency of the fold's step ([`Facts::covers_latency`]).
    fn counted(
        &self,
        plan: Plan,
        (most, came): (usize, usize),
        shared: &mut SharedBySplit,
    ) -> Option<Counted> {
        let register = self.register_points(plan.tile);
        let nest = self.plan_nest(&plan);
        let counted = (nest.as_slice(), register.as_slice(), plan.share.as_slice());
        let moved = self.busiest_moved(counted, &plan.parallel, shared);
        let rate = self.step_rate(&plan.share, &moved);
        if !self.covers_latency(plan.tile, rate, most) {
            return None;
        }
        let loaded = self.loaded(plan.tile, &plan.share);
        let steps = self.steps(&plan.share);
        let mut held_back = Vec::with_capacity(moved.len() + loaded.len());
        for (&cache, &bytes) in self.caches.iter().zip(&moved) {
            held_back.push(if self.keeps_up(cache, bytes, steps) {
                0
            } else {
                bytes
            });
        }
        held_back.extend(loaded);
        let mut counts = moved;
        counts.extend(loaded);
        Some(Counted {
            bound: self.steps_bound(plan.tile),
            held_back,
            counts,
            came,
            plan,
        })
    }

    /// The loops of `plan` outside its register tile, outermost first: those over the pieces of
    /// each memory that holds one, the outermost memory first, then the plan's parallel loops at
    /// the register tile's points and its reduced loops at one point.
    pub fn plan_nest(&self, plan: &Plan) -> Vec<Loop> {
        let register = self.register_points(plan.tile);
        self.nest(&plan.pieces, &register, &plan.parallel, &plan.reduced)
    }

    /// The pieces of the caches, outermost first, whose bytes taken in over a worker's share
    /// cannot arrive within the time of the fold's steps at the rate the target measured, or that
    /// the target states no rate for. Each is the largest piece of the parallel iterators whose
    /// reads over every reduced point fit the cache, grown from the piece inside it along the
    /// innermost loops first. A cache gets one when it is more than the piece inside and less than
    /// the share, and the cache takes in fewer bytes with it. Those bytes count what the fold's
    /// state moves in and out where a piece cuts the reduction ([`Facts::state_into`]).
    ///
    /// Only the innermost cache cuts the reduction, so that the state it holds stays as small as
    /// that cache, and only where the cut lets the cache keep up with the steps: a cut that leaves
    /// the cache behind them moves the state besides and gains no steps. The checks compare with
    /// the time of the fold's steps, so a block without a fold gets none.
    fn cache_pieces(&self, plan: &Plan) -> Vec<Piece> {
        let Some(fold) = &self.fold else {
            return Vec::new();
        };
        let register = self.register_points(plan.tile);
        let mut step_ops = 1_f64;
        for &points in &plan.share {
            step_ops *= points as f64;
        }
        step_ops /= self.lanes.map_or(1, |(_, lanes)| lanes) as f64;
        let step_cycles = step_ops / fold.step_issue;
        // Outermost first, as the nest takes them; each memory's piece goes outside those inside.
        let mut pieces: Vec<Piece> = Vec::new();
        let mut inner = self.points_inside(
            &self.nest(&[], &register, &plan.parallel, &plan.reduced),
            plan.parallel.len() + plan.reduced.len(),
            &register,
            &plan.share,
        );
        let mut may_cut = true;
        for (index, store) in self.memories.iter().enumerate() {
            if store.copies_only {
                continue;
            }
            let cuts = std::mem::replace(&mut may_cut, false);
            let nest = self.nest(&pieces, &register, &plan.parallel, &plan.reduced);
            let moved = self.bytes_into(index, &nest, &register, &plan.share);
            let keeps_up = store
                .fill_rate
                .is_some_and(|rate| moved as f64 / rate <= step_cycles);
            if keeps_up {
                continue;
            }
            let Some(points) = self.largest_piece(index, &inner, plan, cuts) else {
                continue;
            };
            let piece = Piece {
                memory: index,
                points,
            };
            let with_piece = [std::slice::from_ref(&piece), &pieces[..]].concat();
            let nest_with_piece = self.nest(&with_piece, &register, &plan.parallel, &plan.reduced);
            let state = self.state_into(&nest_with_piece, &register, &plan.share);
            let with_state =
                self.bytes_into(index, &nest_with_piece, &register, &plan.share) + state;
            let catches_up = store
                .fill_rate
                .is_some_and(|rate| with_state as f64 / rate <= step_cycles);
            if with_state >= moved || (state > 0 && !catches_up) {
                continue;
            }
            inner = piece.points;
            pieces = with_piece;
        }
        pieces
    }

    /// The largest piece of the parallel iterators of `plan`, outermost first, grown from the
    /// piece `inner` in whole multiples of it along the innermost first, each to the share of
    /// `plan` before the next grows, whose reads over every reduced point take at most
    /// `capacity` bytes: none when it is `inner` or the share. Where the reads of `inner` over
    /// every reduced point take more and the piece `cuts`, it cuts the reduced iterators, the
    /// outermost first, each to the most points that fit, and holds the fold's state over its
    /// parallel points besides, which the loop over the cut keeps across its pieces.
    fn largest_piece(
        &self,
        memory: usize,
        inner: &[usize; MAX_ITERS],
        plan: &Plan,
        cuts: bool,
    ) -> Option<[usize; MAX_ITERS]> {
        let parallel = &plan.parallel;
        let mut piece = *inner;
        for (position, &reduced) in self.reduced.iter().enumerate() {
            if reduced {
                piece[position] = plan.share[position];
            }
        }
        let mut cut = false;
        let fits_with = |points: &[usize; MAX_ITERS], with_state: bool| {
            let state = match with_state {
                true => self.state_bytes(points),
                false => 0,
            };
            self.holds(memory, points, state)
        };
        for &position in &plan.reduced {
            if fits_with(&piece, cut) || !cuts {
                break;
            }
            cut = true;
            let mut points = plan.share[position];
            while points > 1 {
                let mut smaller = piece;
                smaller[position] = points;
                if fits_with(&smaller, cut) {
                    break;
                }
                points -= 1;
            }
            piece[position] = points;
        }
        if !fits_with(&piece, cut) {
            return None;
        }
        let fits = |points: &[usize; MAX_ITERS]| fits_with(points, cut);
        for &position in parallel.iter().rev() {
            let step = inner[position].max(1);
            while piece[position] + step <= plan.share[position] {
                let mut grown = piece;
                grown[position] += step;
                if !fits(&grown) {
                    break;
                }
                piece = grown;
            }
            if piece[position] < plan.share[position] {
                break;
            }
        }
        let mut grew = false;
        let mut whole = true;
        for &position in parallel {
            grew |= piece[position] > inner[position];
            whole &= piece[position] >= plan.share[position];
        }
        (cut || (grew && !whole)).then_some(piece)
    }

    /// The bytes that the fold's state moves in and out over `around` points of each iterator, by
    /// the loops `nest` around register tiles of `register` points. A loop over a reduced iterator
    /// that a piece cuts, with loops over parallel iterators inside it, moves the state over the
    /// points inside it in and out at each of its iterations and those of the loops outside it.
    /// Without such a loop, the state moves nothing.
    fn state_into(&self, nest: &[Loop], register: &[usize], around: &[usize]) -> u64 {
        let mut first_cut = None;
        for (index, walked) in nest.iter().enumerate() {
            if walked.memory.is_some() && self.reduced[walked.position] {
                first_cut = Some(index);
                break;
            }
        }
        let Some(cut) = first_cut else {
            return 0;
        };
        let points = self.points_inside(nest, cut + 1, register, around);
        let mut iterations = 1_u64;
        for outer in (0..=cut).rev() {
            let walked = nest[outer];
            let enclosing = self.points_inside(nest, outer, register, around)[walked.position];
            iterations *= enclosing.div_ceil(walked.size) as u64;
        }
        2 * self.state_bytes(&points) * iterations
    }

    /// The bytes of the fold's state over `points` of each iterator: an element of its dtype for
    /// each point of the parallel iterators, and none without a fold.
    fn state_bytes(&self, points: &[usize]) -> u64 {
        let Some(fold) = &self.fold else {
            return 0;
        };
        let mut elements = 1;
        for (position, &point) in points.iter().enumerate() {
            if !self.reduced[position] {
                elements *= point;
            }
        }
        let bytes = fold
            .state
            .bytes_for(elements)
            .expect("a state of countable bytes");
        bytes as u64
    }

    /// The reads that `plan` copies into each scratch memory, the largest read first. Each is
    /// copied at the outermost loop of the nest, or before the loops, inside which its elements fit
    /// the scratch beside the copies made before it. It is copied where the copy takes fewer bytes
    /// from the memory outside the scratch than the read's register tiles load from it in place, as
    /// no memory inside keeps them ([`Facts::read_in_place`]). A copy is made at a loop over an
    /// iterator the read takes, so that no loop over another remakes it.
    fn scratch_copies(&self, plan: &Plan) -> Vec<ScratchCopy> {
        let register = self.register_points(plan.tile);
        let nest = self.plan_nest(plan);
        let mut copies = Vec::new();
        let mut order: Vec<usize> = (0..self.reads.len()).collect();
        let whole = self.points_inside(&nest, 0, &register, &plan.share);
        order.sort_by_key(|&read| std::cmp::Reverse(self.reads[read].footprint(&whole)));
        for (index, store) in self.memories.iter().enumerate() {
            if !store.copies_only {
                continue;
            }
            // Copies at different loops are live together, so they share the scratch.
            let mut held = 0;
            for &read in &order {
                let facts = &self.reads[read];
                let in_place = self.read_in_place(read, &register, &plan.share);
                for depth in 0..=nest.len() {
                    let takes = depth == 0 || facts.depends[nest[depth - 1].position];
                    if !takes {
                        continue;
                    }
                    let points = self.points_inside(&nest, depth, &register, &plan.share);
                    let bytes = facts.footprint(&points);
                    if held + bytes > store.capacity {
                        continue;
                    }
                    let copied = bytes * self.fetches(read, &nest, depth, &register, &plan.share);
                    if copied < in_place {
                        held += bytes;
                        copies.push(ScratchCopy {
                            read,
                            memory: index,
                            depth,
                        });
                    }
                    break;
                }
            }
        }
        copies
    }

    /// Whether a step of the register tile `tile` issues its loads and broadcasts in no more
    /// cycles than the steps of its fold's state at the issue rate of the op that steps it:
    /// always, without a fold.
    fn steps_bound(&self, tile: RegisterTile) -> bool {
        let Some(fold) = &self.fold else {
            return true;
        };
        let (loads, broadcasts) = self.step_loads(tile);
        let load_cycles = loads as f64 / self.load_issue + broadcasts as f64 / self.broadcast_issue;
        load_cycles <= tile.accumulators() as f64 / fold.step_issue
    }

    /// Whether the accumulators of the register tile `tile` cover the latency of the fold's step
    /// at `rate` steps per cycle, so that a step never waits for the one before it on the same
    /// accumulator, or number `most`, the most that a register tile holds, where none holds that
    /// many: always, without a fold.
    fn covers_latency(&self, tile: RegisterTile, rate: f64, most: usize) -> bool {
        let Some(fold) = &self.fold else {
            return true;
        };
        let needed = (fold.latency * rate).ceil() as usize;
        tile.accumulators() >= needed.min(most)
    }

    /// The steps per cycle at which the busiest worker runs the vector steps of its share `share`
    /// when it moves `moved` bytes into each cache, outermost first ([`Facts::busiest_moved`]):
    /// the fold's step op's issue rate, or less where a cache filled from another cache takes
    /// those bytes in more slowly at the rate the target measured for it. A cache filled from main
    /// memory is left out: whether a value computed before the run is in it when the block runs
    /// depends on the runs before, which the counts do not see.
    fn step_rate(&self, share: &[usize], moved: &[u64]) -> f64 {
        let Some(fold) = &self.fold else {
            return 1.0;
        };
        let steps = self.steps(share);
        let mut rate = fold.step_issue;
        for (&cache, &bytes) in self.caches.iter().zip(moved) {
            let store = &self.memories[cache];
            if let (Some(fill), true, true) = (store.fill_rate, store.filled_from_cache, bytes > 0)
            {
                // The steps per byte first: one ratio gives one rate, whatever its terms.
                rate = rate.min(fill * (steps / bytes as f64));
            }
        }
        rate
    }

    /// Whether the cache `cache` takes in the `bytes` that the busiest worker moves into it over a
    /// share of `steps` vector steps ([`Facts::busiest_moved`], [`Facts::steps`]) within the time
    /// of those steps at the fold's step op's issue rate, at the rate the target measured for
    /// reading the memory outside it: not where it measured none, or without a fold.
    fn keeps_up(&self, cache: usize, bytes: u64, steps: f64) -> bool {
        let Some(fold) = &self.fold else {
            return false;
        };
        let fill = self.memories[cache].fill_rate;
        fill.is_some_and(|rate| bytes as f64 / rate <= steps / fold.step_issue)
    }

    /// The vector steps of a worker's share `share`: its points, those of the lanes' iterator in
    /// vectors.
    fn steps(&self, share: &[usize]) -> f64 {
        let mut steps = 1.0;
        for (position, &points) in share.iter().enumerate() {
            steps *= match self.lanes {
                Some((lane, lanes)) if lane == position => points.div_ceil(lanes) as f64,
                _ => points as f64,
            };
        }
        steps
    }

    /// What one step of the register tile `tile` takes of `read`: as many registers as the tile's
    /// points along the iterators it depends on ([`Facts::points_along`]), vectors where it
    /// depends on the lanes' iterator and broadcasts otherwise. It takes nothing where it depends
    /// on none of the iterators that a step walks, the reduced ones and the tile's, so that it is
    /// read before the loops.
    fn step_take(&self, read: &ReadFacts, tile: RegisterTile) -> Option<(usize, bool)> {
        let vector = self.lanes.is_some_and(|(lane, _)| read.depends[lane]);
        let mut steps = vector;
        for &(row, _) in &tile.rows {
            steps |= read.depends[row];
        }
        for (position, &depends) in read.depends.iter().enumerate() {
            steps |= depends && self.reduced[position];
        }
        steps.then(|| (self.points_along(tile, &read.depends), vector))
    }

    /// The vectors and the broadcasts one step of the register tile `tile` loads
    /// ([`Facts::step_take`]).
    fn step_loads(&self, tile: RegisterTile) -> (usize, usize) {
        let (mut loads, mut broadcasts) = (0, 0);
        for read in &self.reads {
            match self.step_take(read, tile) {
                Some((taken, true)) => loads += taken,
                Some((taken, false)) => broadcasts += taken,
                None => {}
            }
        }
        (loads, broadcasts)
    }

    /// The registers that the operands of one step of the register tile `tile` hold at once. The
    /// vector code walks the tile's points in the order of its rows, the outermost first, and
    /// loads each operand where a point first takes it ([`Facts::step_take`]). A read that the
    /// innermost loop leaves in place holds all the step takes, which the code keeps across that
    /// loop. A read along the outermost row's iterator holds what one of its points takes, as each
    /// point loads its own in the place of the one before. Any other read holds all the step
    /// takes, which every point of the outermost row takes.
    fn step_registers(&self, tile: RegisterTile) -> usize {
        let innermost = self.reduced_order().last().copied();
        let outermost = tile.rows.first().copied();
        let mut registers = 0;
        for read in &self.reads {
            let Some((taken, _)) = self.step_take(read, tile) else {
                continue;
            };
            let moves_inside = innermost.is_some_and(|position| read.depends[position]);
            registers += match outermost {
                Some((row, points)) if moves_inside && read.depends[row] => taken / points,
                _ => taken,
            };
        }
        registers
    }

    /// The vectors loaded into registers over `around` points of each iterator with register
    /// tiles `tile`, the tiles cut by the ends of their iterators counted as they are, the
    /// registers a tile holds, and the number of tiles.
    fn loaded(&self, tile: RegisterTile, around: &[usize]) -> [u64; 3] {
        // The tiles that cover `around`: each with the number of tiles of its points.
        let mut covers: ArrayVec<[(u64, RegisterTile); MAX_COVERS]> = ArrayVec::new();
        covers.push((1, RegisterTile::vectors(1)));
        for &(row, points) in &tile.rows {
            let mut grown: ArrayVec<[(u64, RegisterTile); MAX_COVERS]> = ArrayVec::new();
            for &(count, cover) in &covers {
                for (cut_count, cut_points) in cuts(around[row], points) {
                    let mut cut = cover;
                    cut.rows.push((row, cut_points));
                    grown.push((count * cut_count as u64, cut));
                }
            }
            covers = grown;
        }
        let mut lane = None;
        if let Some((position, lanes)) = self.lanes {
            lane = Some(position);
            let mut grown: ArrayVec<[(u64, RegisterTile); MAX_COVERS]> = ArrayVec::new();
            for &(count, cover) in &covers {
                for (cut_count, vectors) in cuts(around[position].div_ceil(lanes), tile.vectors) {
                    grown.push((count * cut_count as u64, RegisterTile { vectors, ..cover }));
                }
            }
            covers = grown;
        }
        let (mut repeats, mut parallel_repeats) = (1_u64, 1_u64);
        for (position, &extent) in around.iter().enumerate() {
            let in_tile =
                Some(position) == lane || tile.rows.iter().any(|&(row, _)| row == position);
            if in_tile {
                continue;
            }
            repeats *= extent as u64;
            if !self.reduced[position] {
                parallel_repeats *= extent as u64;
            }
        }
        let (mut loads, mut tiles) = (0_u64, 0_u64);
        for &(count, cover) in &covers {
            let (vector_loads, broadcasts) = self.step_loads(cover);
            loads += count * (vector_loads + broadcasts) as u64;
            tiles += count;
        }
        [
            loads * repeats,
            self.held(tile) as u64,
            tiles * parallel_repeats,
        ]
    }

    /// The orders of the loops over the parallel iterators of which the register tile holds
    /// `register` points and fewer than all: the orders of the classes of iterators that the
    /// same reads depend on, each class in the order of the block, as the order within a class
    /// moves nothing. Those whose outer loops are over iterators the largest read takes come
    /// first, so that among orders that move as much, the workers split the largest read and
    /// each reuses its part.
    fn parallel_orders(&self, register: &[usize]) -> Vec<ArrayVec<[usize; MAX_ITERS]>> {
        let mut classes: Vec<ArrayVec<[usize; MAX_ITERS]>> = Vec::new();
        let mut signatures: Vec<Vec<bool>> = Vec::new();
        for (position, &extent) in self.extents.iter().enumerate() {
            if self.reduced[position] || register[position] >= extent {
                continue;
            }
            let mut signature = Vec::with_capacity(self.reads.len());
            for read in &self.reads {
                signature.push(read.depends[position]);
            }
            match signatures.iter().position(|seen| *seen == signature) {
                Some(class) => classes[class].push(position),
                None => {
                    signatures.push(signature);
                    let mut class = ArrayVec::new();
                    class.push(position);
                    classes.push(class);
                }
            }
        }
        let mut orders = Vec::new();
        for permutation in permutations(classes.len()) {
            let mut order: ArrayVec<[usize; MAX_ITERS]> = ArrayVec::new();
            for class in permutation {
                order.extend_from_slice(&classes[class]);
            }
            orders.push(order);
        }
        let mut largest: Option<(u64, usize)> = None;
        for (index, read) in self.reads.iter().enumerate() {
            let bytes = read.footprint(&self.extents);
            if largest.is_none_or(|(most, _)| bytes > most) {
                largest = Some((bytes, index));
            }
        }
        if let Some((_, read)) = largest {
            let depends = &self.reads[read].depends;
            orders.sort_by_key(|order| {
                let mut untaken: ArrayVec<[bool; MAX_ITERS]> = ArrayVec::new();
                for &position in order {
                    untaken.push(!depends[position]);
                }
                untaken
            });
        }
        orders
    }

    /// The reduced iterators of more than one point, outermost first. The innermost loop moves the
    /// fewest reads, so that a read it leaves in place, such as the scale of a group of points,
    /// stays outside it. Of loops that move as many, the one along which the reads move least is
    /// innermost, so that it walks the smallest stride.
    fn reduced_order(&self) -> ArrayVec<[usize; MAX_ITERS]> {
        let mut order = ArrayVec::new();
        for (position, &extent) in self.extents.iter().enumerate() {
            if self.reduced[position] && extent > 1 {
                order.push(position);
            }
        }
        let moved = |position: usize| {
            let mut reads = 0;
            let mut least = i64::MAX;
            for read in &self.reads {
                if read.depends[position] {
                    reads += 1;
                    least = least.min(read.strides[position].abs());
                }
            }
            (reads, least)
        };
        order.sort_by_key(|&position| std::cmp::Reverse(moved(position)));
        order
    }

    /// The loops of a schedule whose register tile holds `register` points, outermost first: one
    /// over each iterator of each of `pieces`, outermost first, that holds fewer points than the
    /// piece around it, at the piece's points; then one over each iterator of `parallel` at the
    /// tile's points, then one over each of `reduced` at one point.
    fn nest(
        &self,
        pieces: &[Piece],
        register: &[usize],
        parallel: &[usize],
        reduced: &[usize],
    ) -> Vec<Loop> {
        let mut nest = Vec::with_capacity(pieces.len() * parallel.len() + reduced.len());
        let mut around = self.extents;
        let mut cut_by_pieces: ArrayVec<[usize; MAX_ITERS]> = ArrayVec::new();
        cut_by_pieces.extend_from_slice(parallel);
        cut_by_pieces.extend_from_slice(reduced);
        for piece in pieces {
            for &position in &cut_by_pieces {
                if piece.points[position] < around[position] {
                    nest.push(Loop {
                        position,
                        size: piece.points[position],
                        memory: Some(piece.memory),
                    });
                }
            }
            around = piece.points;
        }
        for &position in parallel {
            nest.push(Loop {
                position,
                size: register[position],
                memory: None,
            });
        }
        for &position in reduced {
            nest.push(Loop {
                position,
                size: 1,
                memory: None,
            });
        }
        nest
    }

    /// The points of each iterator that one iteration of the loop before loop `depth` of `nest`
    /// walks over `around` points of each: the points of its innermost loop outside `depth`, or
    /// all of them; past the last loop, those the register tile `register` holds.
    fn points_inside(
        &self,
        nest: &[Loop],
        depth: usize,
        register: &[usize],
        around: &[usize],
    ) -> [usize; MAX_ITERS] {
        let mut points = [1; MAX_ITERS];
        points[..around.len()].copy_from_slice(around);
        for nested in &nest[..depth] {
            points[nested.position] = nested.size;
        }
        if depth == nest.len() {
            for (position, point) in points.iter_mut().enumerate() {
                *point = (*point).min(register[position]).max(1);
            }
        }
        for (position, point) in points.iter_mut().enumerate() {
            *point = (*point).min(around[position]);
        }
        points
    }

    /// The bytes that the busiest worker moves into each cache, outermost first. The worker walks
    /// the loops `nest` around register tiles of `register` points over its part of the loops over
    /// `parallel`, of the size of `share` but where the iterators end. It moves the bytes of its
    /// reads over its part ([`Facts::read_into`]), less those that the workers that share the cache
    /// with it wrote before the block runs, which it finds there up to the cache's capacity for it
    /// ([`Facts::resident`]). Each read's bytes divide among the workers of its group that take the
    /// same footprint of it, as the cache takes that in once ([`Facts::shared`]). `shared` keeps
    /// what the workers share, by split. Scratch holds no read in place.
    fn busiest_moved(
        &self,
        (nest, register, share): (&[Loop], &[usize], &[usize]),
        parallel: &[usize],
        shared: &mut SharedBySplit,
    ) -> Vec<u64> {
        let caches = &self.caches;
        let splits = self.splits(share, parallel);
        let found = shared
            .entry(splits)
            .or_insert_with_key(|split| self.shared(caches, split));
        // What each worker's reads move into each cache over its own part, by the points of it.
        let mut into_by_points: HashMap<[usize; MAX_ITERS], Vec<Vec<u64>>> = HashMap::new();
        let mut parts = Vec::with_capacity(found.ranges.len());
        for ranges in &found.ranges {
            let mut points = [1; MAX_ITERS];
            for (position, range) in ranges.iter().enumerate() {
                points[position] = range.len();
            }
            into_by_points
                .entry(points)
                .or_insert_with_key(|part| self.moved_into(caches, nest, register, part));
            parts.push(points);
        }
        let mut busiest: Vec<u64> = Vec::new();
        let mut moved = Vec::with_capacity(caches.len());
        for (worker, points) in parts.iter().enumerate() {
            let worker_into = &into_by_points[points];
            moved.clear();
            for (cache, &memory) in caches.iter().enumerate() {
                let (mut taken_in, mut resident) = (0, 0);
                for (read, &bytes_in) in worker_into[cache].iter().enumerate() {
                    let takers = found.takers[cache][worker][read] as u64;
                    taken_in += bytes_in / takers;
                    resident += found.resident[worker][cache][read].min(bytes_in) / takers;
                }
                moved.push(taken_in - resident.min(self.memories[memory].capacity));
            }
            if moved > busiest {
                busiest.clone_from(&moved);
            }
        }
        busiest
    }

    /// The bytes that each read moves into each of the memories `caches`, by memory and read, by
    /// the loops `nest` around register tiles of `register` points over `points` of each
    /// iterator ([`Facts::read_into`]).
    fn moved_into(
        &self,
        caches: &[usize],
        nest: &[Loop],
        register: &[usize],
        points: &[usize],
    ) -> Vec<Vec<u64>> {
        let mut moved = Vec::with_capacity(caches.len());
        for &memory in caches {
            let depth = self.band_depth(memory, nest, register, points);
            let mut reads = Vec::with_capacity(self.reads.len());
            for read in 0..self.reads.len() {
                reads.push(self.read_into(read, nest, depth, register, points));
            }
            moved.push(reads);
        }
        moved
    }

    /// The loops over `parallel` that the workers split for the share `share`, outermost first:
    /// the iterator of each and the points of each part.
    fn splits(&self, share: &[usize], parallel: &[usize]) -> Splits {
        let mut splits = Splits::new();
        for &position in parallel {
            if share[position] < self.extents[position] {
                splits.push((position, share[position]));
            }
        }
        splits
    }

    /// What the workers of the split `splits` ([`Facts::splits`]) share in each of the memories
    /// `caches`: the bytes each finds of each read there ([`Facts::resident`]), and how many of the
    /// workers that share the memory with each, its group, take the same footprint of each read
    /// ([`ReadFacts::same_footprint`](super::facts::ReadFacts::same_footprint)).
    fn shared(&self, caches: &[usize], splits: &[(usize, usize)]) -> Shared {
        let ranges = self.worker_ranges(splits);
        let mut resident = Vec::with_capacity(ranges.len());
        for (worker, worker_ranges) in ranges.iter().enumerate() {
            let mut by_cache = Vec::with_capacity(caches.len());
            for &memory in caches {
                let mut by_read = Vec::with_capacity(self.reads.len());
                for read in 0..self.reads.len() {
                    by_read.push(self.resident(memory, read, worker, worker_ranges));
                }
                by_cache.push(by_read);
            }
            resident.push(by_cache);
        }
        let mut takers = Vec::with_capacity(caches.len());
        for &memory in caches {
            let sharing = self.memories[memory].sharing;
            let mut by_worker = Vec::with_capacity(ranges.len());
            for worker in 0..ranges.len() {
                let first = worker / sharing * sharing;
                let group = first..(first + sharing).min(ranges.len());
                let mut by_read = Vec::with_capacity(self.reads.len());
                for facts in &self.reads {
                    let mut same = 0;
                    for other in group.clone() {
                        same += usize::from(facts.same_footprint(&ranges[other], &ranges[worker]));
                    }
                    by_read.push(same);
                }
                by_worker.push(by_read);
            }
            takers.push(by_worker);
        }
        Shared {
            ranges,
            resident,
            takers,
        }
    }

    /// The range of each iterator in the share of each worker of the split `splits`, in the order
    /// of the workers: consecutive workers take consecutive iterations of the split loops, the
    /// innermost the fastest.
    fn worker_ranges(&self, splits: &[(usize, usize)]) -> Vec<[Range<usize>; MAX_ITERS]> {
        let mut workers = 1;
        for &(position, points) in splits {
            workers *= self.extents[position].div_ceil(points);
        }
        let mut all = Vec::with_capacity(workers);
        for worker in 0..workers {
            let mut ranges: [Range<usize>; MAX_ITERS] =
                std::array::from_fn(|position| 0..self.extents[position]);
            let mut rest = worker;
            for &(position, points) in splits.iter().rev() {
                let parts = self.extents[position].div_ceil(points);
                let part = rest % parts;
                rest /= parts;
                let end = ((part + 1) * points).min(self.extents[position]);
                ranges[position] = part * points..end;
            }
            all.push(ranges);
        }
        all
    }

    /// The bytes of read `read` over the ranges `ranges` of each iterator, the share of worker
    /// `worker`, that the workers that share memory `memory` with it wrote before the block runs:
    /// on average over the calls of the dispatch where it is known ([`ReadFacts::written`]).
    fn resident(&self, memory: usize, read: usize, worker: usize, ranges: &[Range<usize>]) -> u64 {
        let store = &self.memories[memory];
        let facts = &self.reads[read];
        let group = worker / store.sharing;
        let mut bytes = 0;
        for parts in &facts.written {
            for (writer, part) in parts.iter().enumerate() {
                if writer / store.sharing == group {
                    bytes += facts.bytes_within(ranges, part);
                }
            }
        }
        bytes / facts.written.len().max(1) as u64
    }

    /// The bytes that the reads move into memory `memory` over `around` points of each iterator
    /// by the loops `nest` around register tiles of `register` points ([`Facts::read_into`]).
    fn bytes_into(
        &self,
        memory: usize,
        nest: &[Loop],
        register: &[usize],
        around: &[usize],
    ) -> u64 {
        let depth = self.band_depth(memory, nest, register, around);
        let mut bytes = 0;
        for read in 0..self.reads.len() {
            bytes += self.read_into(read, nest, depth, register, around);
        }
        bytes
    }

    /// The bytes that read `read` moves over `around` points of each iterator by the loops `nest`
    /// around register tiles of `register` points into a memory that holds the reads of the band
    /// of the loops from `depth` on ([`Facts::band_depth`]): its bytes over that band once for
    /// each iteration of the loops outside it ([`Facts::fetches`]).
    fn read_into(
        &self,
        read: usize,
        nest: &[Loop],
        depth: usize,
        register: &[usize],
        around: &[usize],
    ) -> u64 {
        let points = self.points_inside(nest, depth, register, around);
        self.reads[read].footprint(&points) * self.fetches(read, nest, depth, register, around)
    }

    /// The depth in `nest` of the largest band of innermost loops around register tiles of
    /// `register` points, over `around` points of each iterator, whose reads fit memory `memory`.
    fn band_depth(
        &self,
        memory: usize,
        nest: &[Loop],
        register: &[usize],
        around: &[usize],
    ) -> usize {
        for depth in 0..=nest.len() {
            let points = self.points_inside(nest, depth, register, around);
            if self.holds(memory, &points, 0) {
                return depth;
            }
        }
        nest.len()
    }

    /// Whether memory `memory` holds the reads over `points` of each iterator and `more` bytes
    /// besides. Their bytes must fit its capacity. Where it places each line by its address, no set
    /// may take more of their lines than its ways but one. The lines of each read count in its most
    /// crowded set, as where the reads start relative to one another is not known.
    fn holds(&self, memory: usize, points: &[usize; MAX_ITERS], more: u64) -> bool {
        let store = &self.memories[memory];
        let mut bytes = more;
        for read in &self.reads {
            bytes += read.footprint(points);
        }
        if bytes > store.capacity {
            return false;
        }
        let Some(sets) = store.sets else {
            return true;
        };
        let key = (memory, *points);
        if let Some(&crowded) = self.crowded.borrow().get(&key) {
            return crowded < sets.ways;
        }
        let mut crowded = 0;
        for read in &self.reads {
            crowded += read.most_lines_in_a_set(points, sets);
        }
        self.crowded.borrow_mut().insert(key, crowded);
        crowded < sets.ways
    }

    /// The bytes that read `read` moves into the registers over `around` points of each iterator
    /// from a memory with none inside it to keep them: its bytes over each step of each register
    /// tile of `register` points, the tiles cut by the ends of the iterators counted as they are.
    fn read_in_place(&self, read: usize, register: &[usize], around: &[usize]) -> u64 {
        let mut shapes: Vec<(u64, [usize; MAX_ITERS])> = vec![(1, [1; MAX_ITERS])];
        for (position, &points) in around.iter().enumerate() {
            let size = register[position].min(points).max(1);
            let mut grown = Vec::with_capacity(shapes.len() * 2);
            for &(count, shape) in &shapes {
                for (tiles, tile_points) in cuts(points, size) {
                    let mut grown_shape = shape;
                    grown_shape[position] = tile_points;
                    grown.push((count * tiles as u64, grown_shape));
                }
            }
            shapes = grown;
        }
        let mut bytes = 0;
        for (count, shape) in &shapes {
            bytes += count * self.reads[read].footprint(shape);
        }
        bytes
    }

    /// How many times read `read` moves its bytes over the band of the loops of `nest` from
    /// `depth` on into a memory that holds them, over `around` points of each iterator: once for
    /// each iteration of the loops outside, but those of the innermost run of them over iterators
    /// it does not depend on, across which it stays.
    fn fetches(
        &self,
        read: usize,
        nest: &[Loop],
        depth: usize,
        register: &[usize],
        around: &[usize],
    ) -> u64 {
        let mut fetches = 1_u64;
        let mut staying = true;
        for outer in (0..depth).rev() {
            let walked = nest[outer];
            if staying && !self.reads[read].depends[walked.position] {
                continue;
            }
            staying = false;
            let enclosing = self.points_inside(nest, outer, register, around)[walked.position];
            fetches *= enclosing.div_ceil(walked.size) as u64;
        }
        fetches
    }

    /// The points of each iterator in a worker's share for each way the workers may split the loops
    /// over `parallel`, outermost first, at the register tile's `register` points. Each loop splits
    /// into equal parts of whole tiles, as many as its tiles allow, with each number of parts that
    /// gives another share, while the parts of all the loops are at most the workers. The most
    /// parts of the outer loops come first, so that among shares that count the same, the workers
    /// split the outermost loops. Consecutive workers take consecutive iterations of the split
    /// loops ([`Facts::splits`]).
    fn shares(&self, register: &[usize], parallel: &[usize]) -> Vec<[usize; MAX_ITERS]> {
        let mut shares = vec![(self.extents, 1)];
        for &position in parallel {
            let tiles = self.extents[position].div_ceil(register[position]);
            let mut grown = Vec::with_capacity(shares.len());
            for (share, workers) in &shares {
                let mut last_each = 0;
                for wanted in (1..=tiles.min(self.units / workers)).rev() {
                    let each = tiles.div_ceil(wanted);
                    if each == last_each {
                        continue;
                    }
                    last_each = each;
                    let mut split = *share;
                    split[position] = (each * register[position]).min(self.extents[position]);
                    grown.push((split, workers * tiles.div_ceil(each)));
                }
            }
            shares = grown;
        }
        let mut found = Vec::with_capacity(shares.len());
        for (share, _) in shares {
            found.push(share);
        }
        found
    }
}

/// The tiles of `size` that cover `extent`: how many whole ones, and the last one's points.
fn cuts(extent: usize, size: usize) -> ArrayVec<[(usize, usize); 2]> {
    let mut tiles = ArrayVec::new();
    if extent / size > 0 {
        tiles.push((extent / size, size));
    }
    if !extent.is_multiple_of(size) {
        tiles.push((1, extent % size));
    }
    tiles
}

/// Every permutation of `0..count`.
fn permutations(count: usize) -> Vec<ArrayVec<[usize; MAX_ITERS]>> {
    let mut all = vec![ArrayVec::new()];
    for next in 0..count {
        let mut grown = Vec::with_capacity(all.len() * (next + 1));
        for permutation in &all {
            for at in 0..=permutation.len() {
                let mut placed = *permutation;
                placed.insert(at, next);
                grown.push(placed);
            }
        }
        all = grown;
    }
    all
}
