//! Choosing the schedule of each dispatch.

mod facts;
mod rules;

use anyhow::Result;

use self::facts::Facts;
use self::rules::{Plan, RegisterTile};
use crate::dialects::tile::verify::{check_placements, check_schedule};
use crate::dialects::tile::{
    Block, Dispatch, DispatchOp, Held, Holding, IterList, Mode, ModeKind, ScratchAt, Source,
    StateHolding, Workers,
};
use crate::ir::{Inst, IterRef, Level, Module};
use crate::passes::{pass, Pass, Registered};
use crate::require;
use crate::rewrite::{
    pattern, Greedy, NoMatch, Pattern, PatternSet, Rewriter, Rewrites, Rewriting,
};
use crate::targets::{required, Target, TargetDesc};

/// Chooses the schedule of the first block of each dispatch on the target. The rules are
/// constraints on the target's facts, and counts of what the schedule moves, compared in order.
/// Issue and fill rates enter only the constraints: no time is predicted for a schedule, and none
/// ranks schedules. The schedule is the block's modes, outermost first:
/// 1. the workers' shares (`d by s : workers<level = ..>`);
/// 2. the loops over the pieces of the caches that need one (`d by t in m`);
/// 3. the loops over the parallel iterators that the register tile does not hold;
/// 4. the loops over the reduced iterators;
/// 5. the register tile (`d by r : reg`) and its lanes.
///
/// The register tile:
/// - It is `R` rows, along one or two parallel iterators, by `V` vectors of the lanes. Each row
///   iterator is one that the result's coordinates take and no `tile.index` of the body takes.
/// - The iterators of the rows are tried in groups. The group along which the fewest of the body's
///   ops vary comes first, as the rows share the other ops. Where the rows of a group share some
///   ops, the tile without rows is left out: it would do them at each point.
/// - Its registers fit the target's: the accumulators, the body's temporaries (one for each row
///   where they vary along the rows, one for each vector where the rows share them), and the
///   registers of one step's operands or of the blocks placed at the tile.
/// - With a fold, the accumulators cover the latency of the op that steps the state, at the rate
///   the steps run. That rate is the op's issue rate, or less where a cache that is filled from
///   another cache takes in the busiest worker's bytes more slowly, at the rate the target
///   measured. Where no tile holds that many accumulators, the tiles with the most do.
///
/// The choice of the tile and the order of the loops:
/// - The candidates are the tiles whose loads and broadcasts take no more cycles, at their issue
///   rates, than the fold's steps at its op's issue rate, when any tile does. The cycles of loads
///   and of broadcasts add, as a broadcast from memory is a load.
/// - Of these, the rules take the tile and the order of the parallel loops whose busiest worker
///   moves the fewest bytes into each cache, the outermost cache first. A tie goes to the one that
///   loads the fewest vectors, then holds the fewest registers, then runs the fewest tiles. Among
///   orders that still tie, the largest read's iterators run outermost.
/// - A cache holds the reads of the largest band of innermost loops whose reads fit all its ways
///   but one, shared among the workers that share the cache. It takes in once a footprint of a read
///   that several of those workers take. The count of a cache is the busiest worker's part of what
///   its group moves in. The counts take each tile of a loop as whole, and count bytes, not lines.
/// - The loops over the reduced iterators run inside the parallel loops, so the fold's state stays
///   in the register tile. The innermost of them moves the fewest reads, so that a read it leaves
///   in place, such as the scale of a group of points, stays outside it. Of those that move as
///   many, the one along which the reads move least is innermost.
///
/// Workers and caches:
/// - The workers split the outermost parallel loops in their order. Each loop splits into equal
///   parts of whole tiles, up to as many as it has tiles and the workers left allow, so that
///   consecutive workers take consecutive iterations.
/// - A value that a dispatch scheduled before, in module order, wrote in the run is in each cache
///   where the workers that share that cache with its writers wrote it, up to the cache's capacity.
///   The workers move into the cache none of the bytes of their reads that those workers wrote. The
///   counts take workers that share a cache to be consecutive, as the runtime pins worker t to unit
///   t.
/// - A cache gets the largest piece of the parallel iterators that fits it, where that piece moves
///   fewer bytes into it, when the bytes it takes in cannot arrive within the time of the fold's
///   steps at the rate the target measured, or the target measured none.
/// - Only the innermost cache may also cut the reduced iterators: where the reads of one register
///   tile over every reduced point do not fit it, its piece cuts them, but only where the cut lets
///   that cache keep up with the steps. The fold's state over the piece's parallel points is then
///   held in scratch across the loops over the cuts (`held scratch<at = ..>`), and its bytes count
///   in and out at each of their iterations.
/// - A read is copied into scratch, inside each worker's loops, at the outermost loop whose
///   elements fit beside the other copies, where that takes fewer bytes from outside than its
///   register tiles load in place.
///
/// Clips: the block clips each iterator along which a padded read's coordinate leaves its operand,
/// where the pad adds nothing to the fold, except for a read that a copy holds, as the copy holds
/// the pad. The register tile is cut at each tap to its points in range.
///
/// The pass fails where the target lacks a fact that a constraint needs, such as an op's cost or
/// latency, or the units of the parallel level, and names it.
#[pass(
    name = "schedule",
    stats(scheduled = "first blocks of dispatches given a schedule")
)]
#[derive(Default)]
pub struct Schedule {
    /// The target the schedules are chosen for.
    pub target: Option<Target>,
}

fn populate_schedule_patterns(set: &mut PatternSet, target: Option<Target>) {
    set.add(ScheduleFirstBlock { target });
}

impl Pass for Schedule {
    fn rewrites(&self) -> Rewrites {
        let mut once = PatternSet::new();
        populate_schedule_patterns(&mut once, self.target.clone());
        Rewrites {
            once,
            ..Rewrites::default()
        }
    }

    fn run(&self, module: &mut Rewriting) -> Result<bool> {
        let target = required(self.target.as_ref(), self.name())?;
        check_facts(module, target)?;
        Ok(Greedy::new(self.rewrites()).run(module)?.changed())
    }
}

/// Fails with the first fact that `target` lacks and the rules read for the first block of a
/// dispatch of `module`.
fn check_facts(module: &Module, target: &TargetDesc) -> Result<()> {
    for inst in module.insts(module.body()) {
        if module.op(inst).get::<Dispatch>().is_none() {
            continue;
        }
        let dispatch = module.op_as::<Dispatch>(inst).expect("a dispatch");
        let blocks = Dispatch::blocks(module, dispatch.body().id());
        let Some(&(first, _)) = blocks.first() else {
            continue;
        };
        Facts::of(
            module,
            inst,
            first,
            &placed(module, &blocks),
            target,
            dispatch.numerics.contract,
        )?;
    }
    Ok(())
}

/// The blocks after the first of `blocks` that run at its register tile: each takes the one
/// before it as an epilogue.
fn placed(module: &Module, blocks: &[(Inst, &Block)]) -> Vec<Inst> {
    let mut placed = Vec::new();
    for pair in blocks.windows(2) {
        let [(before, _), (after, _)] = pair else {
            continue;
        };
        if Block::epilogue_input(module, *before, *after).is_none() {
            break;
        }
        placed.push(*after);
    }
    placed
}

/// The first block of a dispatch, with no modes but its lanes: its schedule.
#[pattern]
struct ScheduleFirstBlock {
    target: Option<Target>,
}

impl Pattern for ScheduleFirstBlock {
    type Root = Dispatch;

    fn rewrite(&self, dispatch: DispatchOp, rewriter: &mut Rewriter) -> Result<(), NoMatch> {
        require!(let Some(target) = &self.target, "the pass has no target");
        let module = dispatch.module();
        let blocks = Dispatch::blocks(module, dispatch.body().id());
        require!(let Some(&(inst, block)) = blocks.first(), "the dispatch holds no block");
        require!(
            block
                .loops
                .iter()
                .all(|mode| mode.kind == Some(ModeKind::Lanes) || mode.kind.is_none()),
            "the block has a schedule"
        );
        require!(
            !block
                .iters
                .iter()
                .any(|iter| iter.extent.as_static() == Some(0)),
            "the block runs no point"
        );
        let contract = dispatch.numerics.contract;
        let placed_blocks = placed(module, &blocks);
        let found = Facts::of(
            module,
            dispatch.inst(),
            inst,
            &placed_blocks,
            target,
            contract,
        )
        .map_err(|err| NoMatch::because(format_args!("{err:#}")))?;
        require!(
            let Some(facts) = found,
            "the block's facts rule out a schedule: an extent is not known before the run, a read \
             picks from a list or is not affine, the block folds other than one value into one \
             state, folds in order or as prefixes, updates an operand in order, or its body has an \
             op without a lane-wise form"
        );
        let groups = row_iterators(block, &facts);
        let candidates = groups.concat();
        let mut tiles = Vec::new();
        for rows in &groups {
            for tile in facts.register_tiles(rows, &candidates) {
                let mut tiled = block.clone();
                tiled.loops = register_modes(block, &facts, tile);
                if check_schedule(&tiled).is_ok() {
                    tiles.push(tile);
                }
            }
            if tiles.iter().any(|tile| !tile.rows.is_empty()) {
                // Rows that share body work leave a tile without rows doing all of it at each
                // point.
                if rows
                    .iter()
                    .any(|&position| facts.ops_along[position] < facts.varying_ops)
                {
                    tiles.retain(|tile| !tile.rows.is_empty());
                }
                break;
            }
        }
        require!(let Some(plan) = facts.plan(&tiles), "no register tile fits the target");
        let skip_pad = dispatch.numerics.skip_pad;
        let scheduled = scheduled_block(module, inst, block, &facts, &plan, (target, skip_pad));
        check_schedule(&scheduled).map_err(|err| NoMatch::because(format_args!("{err:#}")))?;
        require!(
            scheduled != *block,
            "the block already walks as its schedule says"
        );
        let mut proposal = blocks.clone();
        proposal[0].1 = &scheduled;
        check_placements(module, &proposal)
            .map_err(|err| NoMatch::because(format_args!("{err:#}")))?;
        let current = module.op_as::<Block>(inst).expect("the first block");
        current.update(rewriter, |op| {
            op.loops = scheduled.loops;
            op.folds = scheduled.folds;
            op.reads = scheduled.reads;
            op.clip = scheduled.clip;
        });
        Schedule::SCHEDULED.add(1);
        Ok(())
    }
}

/// The iterators that the rows of the register tile of `block` may run along, in groups to try in
/// order. Each is a parallel iterator of more than one point, other than the lanes', that a
/// coordinate of the first write is, the last first. No `tile.index` of the body takes it
/// ([`Facts::indexed`]), as vector code computes the index of a point once for every row. The
/// groups are by the number of the body's ops that vary along them ([`Facts::ops_along`]), the
/// fewest first, as each row computes those and the rows share the others, such as a weight's
/// dequantization that rows along the activations' rows share.
fn row_iterators(block: &Block, facts: &Facts) -> Vec<Vec<usize>> {
    let lane = facts.lanes.map(|(position, _)| position);
    let mut rows = Vec::new();
    let Some(write) = block.folds.first().and_then(|fold| fold.writes.first()) else {
        return vec![rows];
    };
    for coord in write.coords.iter().rev() {
        let Some(position) = coord.as_iter() else {
            continue;
        };
        let parallel = !facts.reduced[position] && Some(position) != lane;
        let several = facts.extents[position] > 1;
        if parallel && several && !facts.indexed[position] && !rows.contains(&position) {
            rows.push(position);
        }
    }
    let mut groups: Vec<Vec<usize>> = Vec::new();
    let mut counts: Vec<usize> = rows
        .iter()
        .map(|&position| facts.ops_along[position])
        .collect();
    counts.sort_unstable();
    counts.dedup();
    for count in counts {
        let mut group = Vec::new();
        for &position in &rows {
            if facts.ops_along[position] == count {
                group.push(position);
            }
        }
        groups.push(group);
    }
    if groups.is_empty() {
        groups.push(Vec::new());
    }
    groups
}

/// The modes of `block` with the register tile `tile` alone: its rows and vectors, the lanes, and
/// the loops of one point.
fn register_modes(block: &Block, facts: &Facts, tile: RegisterTile) -> Vec<Mode> {
    let register = facts.register_points(tile);
    let mut modes = Vec::new();
    for (position, &points) in register.iter().enumerate() {
        if points < facts.extents[position] {
            modes.push(Mode::new(IterRef::new(position), points, None));
        }
    }
    modes.extend(tile_modes(facts, tile));
    block.with_point_modes(modes)
}

/// The modes of the register tile `tile`: its rows, the outermost first, its vectors when it
/// holds more than one, and its lanes.
fn tile_modes(facts: &Facts, tile: RegisterTile) -> Vec<Mode> {
    let mut modes = Vec::new();
    let Some((lane, lanes)) = facts.lanes else {
        return modes;
    };
    for &(row, points) in &tile.rows {
        modes.push(Mode::new(IterRef::new(row), points, Some(ModeKind::Reg)));
    }
    if tile.vectors > 1 {
        let points = tile.vectors * lanes;
        modes.push(Mode::new(IterRef::new(lane), points, Some(ModeKind::Reg)));
    }
    modes.push(Mode::new(IterRef::new(lane), lanes, Some(ModeKind::Lanes)));
    modes
}

/// `block` with the schedule of `plan`: the workers' shares, the loops over the parallel
/// iterators at the register tile's points and over the reduced ones at one point, the register
/// tile, its fold's state in registers, and its clips.
fn scheduled_block(
    module: &Module,
    inst: Inst,
    block: &Block,
    facts: &Facts,
    plan: &Plan,
    (target, skip_pad): (&TargetDesc, bool),
) -> Block {
    let mut modes = Vec::new();
    for &position in &plan.parallel {
        if plan.share[position] < facts.extents[position] {
            let workers = Workers {
                level: target.parallel_level(),
            };
            let kind = Some(ModeKind::Workers(workers));
            modes.push(Mode::new(
                IterRef::new(position),
                plan.share[position],
                kind,
            ));
        }
    }
    // Each worker makes its own copies, so a copy is made inside the innermost mode of workers.
    let workers_at = modes.len().checked_sub(1);
    let mut around = plan.share;
    // The mode of each loop of the nest, or of the mode around a loop of one iteration, which
    // has none.
    let mut mode_of_loop = Vec::new();
    for walked in facts.plan_nest(plan) {
        if walked.size >= around[walked.position] {
            mode_of_loop.push(modes.len().checked_sub(1));
            continue;
        }
        mode_of_loop.push(Some(modes.len()));
        let mode = Mode::new(IterRef::new(walked.position), walked.size, None);
        modes.push(match walked.memory {
            Some(memory) => mode.in_memory(Level::new(&facts.memories[memory].name)),
            None => mode,
        });
        around[walked.position] = walked.size;
    }
    modes.extend(tile_modes(facts, plan.tile));
    let loops = block.with_point_modes(modes.clone());
    let mut scheduled = block.clone();
    for copy in &plan.copies {
        let at_loop = match copy.depth {
            0 => None,
            depth => mode_of_loop[depth - 1],
        };
        let at = at_loop
            .max(workers_at)
            .map(|index| final_index(&modes, &loops, index));
        scheduled.reads[copy.read].held = Some(Holding::Copy(Held {
            level: Level::new(&facts.memories[copy.memory].name),
            at,
        }));
    }
    scheduled.clip = IterList::new(clippable(module, inst, &scheduled, skip_pad));
    scheduled.folds[0].held = state_holding(block, &loops);
    scheduled.loops = loops;
    scheduled
}

/// Where the fold of `block` keeps its state over the loops `loops`: in scratch from its first
/// loop over a reduced iterator, when a loop over a parallel iterator runs inside that one, as a
/// cache's piece that cuts the reduction makes; else nowhere, one element of each point of the
/// register tile at a time.
fn state_holding(block: &Block, loops: &[Mode]) -> Option<StateHolding> {
    let outside_tile = |mode: &&Mode| mode.kind.is_none();
    let reduced = |mode: &Mode| block.iters[mode.iter.position()].kind.is_reduced();
    let first = loops
        .iter()
        .position(|mode| outside_tile(&mode) && reduced(mode))?;
    let parallel_inside = loops[first + 1..]
        .iter()
        .filter(outside_tile)
        .any(|mode| !reduced(mode));
    parallel_inside.then_some(StateHolding::Scratch(ScratchAt { at: first }))
}

/// The index in `loops`, which [`Block::with_point_modes`] made from `modes`, of the mode at
/// `index` of `modes`, or of the next one it keeps.
fn final_index(modes: &[Mode], loops: &[Mode], index: usize) -> usize {
    let mut kept = 0;
    for (position, mode) in loops.iter().enumerate() {
        if kept < modes.len() && *mode == modes[kept] {
            if kept >= index {
                return position;
            }
            kept += 1;
        }
    }
    loops.len()
}

/// The iterators along which a coordinate of a padded read of the block of `inst`, as `block`,
/// moves by a constant other than 0, where the coordinate can leave its axis and the pad adds
/// nothing to the fold ([`Block::skipped_pads`]), but of reads of an operand that a copy holds,
/// as the copy holds the pad.
fn clippable(module: &Module, inst: Inst, block: &Block, skip_zero_products: bool) -> Vec<usize> {
    let mut copied = Vec::new();
    for read in &block.reads {
        if let (Source::Operand(operand), Some(Holding::Copy(_))) = (&read.source, &read.held) {
            copied.push(*operand);
        }
    }
    let mut clip = Vec::new();
    for (index, leaving) in block.skipped_pads(module, inst, skip_zero_products) {
        let read = &block.reads[index];
        let copy_holds =
            matches!(read.source, Source::Operand(operand) if copied.contains(&operand));
        if copy_holds {
            continue;
        }
        for axis in leaving {
            let coord = &read.coords[axis];
            if coord.has_reads() {
                continue;
            }
            for position in coord.iters() {
                let moves = coord.coefficient(position).is_some_and(|step| step != 0);
                if moves && !clip.contains(&position) {
                    clip.push(position);
                }
            }
        }
    }
    clip.sort_unstable();
    clip
}
