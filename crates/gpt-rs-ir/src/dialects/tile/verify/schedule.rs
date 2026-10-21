//! The rules of a block's schedule: its modes, its clips and the holdings of its reads.
//! `tile-to-loop` lowers every block that keeps them.

use std::collections::BTreeSet;

use anyhow::{bail, ensure, Result};

use crate::dialects::tile::ir::*;
use crate::dialects::tile::Walk;
use crate::ir::{Dim, OperandRef, MAX_GROUP_RANK, MAX_ITERS};

/// Fails with the rule that the schedule of `block` breaks.
pub(crate) fn check_schedule(block: &Block) -> Result<()> {
    check_in_order(block)?;
    check_modes(block)?;
    check_every_loop(block)?;
    check_holdings_at(block)?;
    let walk = block.walk();
    check_reg(block, &walk)?;
    check_nest_folds(block, &walk)?;
    check_copies(block, &walk)?;
    check_prefetches(block, &walk)
}

/// Fails with the rule that the modes of `block` break for what it walks in order: a block that
/// updates an operand where its points may collide has no modes, which would reorder its writes,
/// and a mode walks a prefix iterator or one folded in order one point at a time.
fn check_in_order(block: &Block) -> Result<()> {
    let in_order = block
        .folds
        .iter()
        .any(|fold| fold.updates_in_order(&block.iters));
    ensure!(
        block.loops.is_empty() || !in_order,
        "modes of a block that updates an operand where its points may collide would reorder its \
         writes"
    );
    for (index, mode) in block.loops.iter().enumerate() {
        let walked_in_order = match block.iters.get(mode.iter.position()).map(|iter| iter.kind) {
            Some(IterKind::Prefix) => "a prefix iterator",
            Some(IterKind::RedOrdered) => "a reduction in order",
            Some(IterKind::Par | IterKind::Red) | None => continue,
        };
        ensure!(
            mode.size == 1 && mode.kind.is_none(),
            "mode {index} walks {} in pieces of {} points, {walked_in_order}, which its loop walks \
             one point at a time",
            mode.iter,
            mode.size
        );
    }
    Ok(())
}

/// Fails with the rule that the modes and the clips of `block` break: each mode walks an iterator
/// of the block in pieces of at least one point, smaller than the pieces of the mode around it
/// over the same iterator; a mode of workers splits a parallel iterator; the modes of the register
/// tile hold parallel iterators, walk no memory's pieces and come after every other mode, with one
/// lanes mode, the last of all; and a clip names an iterator of the block.
fn check_modes(block: &Block) -> Result<()> {
    let iters = &block.iters;
    let (mut register_tile, mut lanes) = (false, false);
    let mut pieces: [Option<usize>; MAX_ITERS] = [None; MAX_ITERS];
    for (index, mode) in block.loops.iter().enumerate() {
        let Some(iter) = iters.get(mode.iter.position()) else {
            bail!(
                "mode {index} walks {}, which the block does not have",
                mode.iter
            );
        };
        ensure!(mode.size > 0, "mode {index} walks pieces of no points");
        ensure!(
            !lanes,
            "mode {index} comes after the lanes, the innermost mode"
        );
        let position = mode.iter.position();
        if let (None | Some(ModeKind::Workers(_)), Some(outer)) = (&mode.kind, pieces[position]) {
            ensure!(
                mode.size < outer,
                "mode {index} walks pieces of {} points of {}, inside pieces of {outer}",
                mode.size,
                mode.iter
            );
        }
        match &mode.kind {
            None | Some(ModeKind::Workers(_)) => {
                ensure!(
                    !register_tile,
                    "mode {index} walks pieces of {} inside the register tile, whose modes are \
                     the innermost",
                    mode.iter
                );
                pieces[position] = Some(mode.size);
            }
            Some(ModeKind::Reg | ModeKind::Lanes) => {
                ensure!(
                    iter.kind == IterKind::Par,
                    "mode {index} holds {} in the register tile, which is not a parallel iterator",
                    mode.iter
                );
                ensure!(
                    mode.memory.is_none(),
                    "mode {index} is a mode of the register tile, and walks the pieces of a memory"
                );
            }
        }
        match &mode.kind {
            Some(ModeKind::Workers(_)) => ensure!(
                iter.kind == IterKind::Par,
                "mode {index} runs pieces of {} on workers, which is not a parallel iterator: \
                 that splits a fold",
                mode.iter
            ),
            Some(ModeKind::Reg) => register_tile = true,
            Some(ModeKind::Lanes) => lanes = true,
            None => {}
        }
    }
    ensure!(
        !register_tile || lanes,
        "the register tile has no lanes mode"
    );
    for position in block.clip.positions() {
        ensure!(
            position < iters.len(),
            "the block clips d{position}, which it does not have"
        );
    }
    Ok(())
}

/// Fails with the iterator that the modes of `block`, when it has any, leave without a loop of one
/// point: every iterator but those of the register tile ends in a mode of one point, so that the
/// modes say every loop of the block.
fn check_every_loop(block: &Block) -> Result<()> {
    if block.loops.is_empty() {
        return Ok(());
    }
    let (innermost, held) = Mode::innermost_pieces(&block.loops);
    for position in 0..block.iters.len() {
        if held[position] {
            continue;
        }
        match innermost[position] {
            Some(1) => {}
            Some(size) => bail!(
                "the modes walk d{position} in pieces of {size} points, and none walks it one \
                 point at a time"
            ),
            None => bail!("no mode walks d{position}"),
        }
    }
    Ok(())
}

/// Fails with the rule that the holdings of `block` break for its modes: each is made at a mode of
/// the loops other than those of the register tile, and an operand has one holding.
fn check_holdings_at(block: &Block) -> Result<()> {
    let mut held = BTreeSet::new();
    for (index, read) in block.reads.iter().enumerate() {
        let Some(Holding::Copy(holding) | Holding::Prefetch(holding)) = &read.held else {
            continue;
        };
        let operand = read.source.first();
        ensure!(
            held.insert(operand),
            "read {index} holds ${operand}, which another read holds"
        );
        let Some(at) = holding.at else {
            continue;
        };
        let Some(mode) = block.loops.get(at) else {
            bail!(
                "read {index} is held at mode {at}, of loops of {} modes",
                block.loops.len()
            );
        };
        ensure!(
            matches!(mode.kind, None | Some(ModeKind::Workers(_))),
            "read {index} is held at mode {at}, a mode of the register tile, which runs its points \
             at once"
        );
    }
    Ok(())
}

/// Fails with the rule that the register tile of `block` and its atom break, under `walk`: at most
/// [`MAX_GROUP_RANK`] iterators, each held once, over the innermost piece of its iterator, which
/// holds at least one point; vectors of the lanes that fill the points of the last; no constraints
/// and no prefix iterator; reads that take each iterator of the tile by multiples of it, in the
/// terms of its tiles where they start at multiples of the tile's points
/// ([`Walk::reads_in_steps`]), or from a copy, and pick no operand of a list by one; and folds that
/// keep one element per point, whose plain writes name each iterator of the tile.
fn check_reg(block: &Block, walk: &Walk) -> Result<()> {
    let Some(atom) = &walk.atom else {
        return Ok(());
    };
    ensure!(
        walk.reg.len() <= MAX_GROUP_RANK,
        "a register tile holds at most {MAX_GROUP_RANK} iterators, not {}",
        walk.reg.len()
    );
    let tiles = walk.innermost_tiles();
    for (index, dim) in walk.reg.iter().enumerate() {
        let position = dim.iter.position();
        ensure!(
            walk.reg[..index].iter().all(|other| other.iter != dim.iter),
            "the register tile holds {} twice",
            dim.iter
        );
        let innermost = match tiles[position] {
            Some(extent) => Dim::Static(extent),
            None => block.iters[position].extent.clone(),
        };
        // When no loop cuts an iterator, its one piece may hold fewer points than the register
        // tile, like a last piece, and at least one: an atom runs at least one point.
        let fewer = tiles[position].is_none()
            && innermost
                .as_static()
                .is_some_and(|points| (1..dim.extent).contains(&points));
        ensure!(
            innermost == Dim::Static(dim.extent) || fewer,
            "the register tile holds {} points of {}, whose innermost piece has {innermost}",
            dim.extent,
            dim.iter
        );
    }
    // The walk puts the lanes' iterator last in the register tile.
    let last = walk.reg.last().expect("a register tile holds the lanes");
    ensure!(
        last.extent.is_multiple_of(atom.lanes),
        "the register tile holds {} points of {}, which vectors of {} lanes do not fill",
        last.extent,
        last.iter,
        atom.lanes
    );
    ensure!(
        block.constraints.is_empty(),
        "a register tile of a block with constraints would guard each point"
    );
    ensure!(
        block.iters.iter().all(|iter| iter.kind != IterKind::Prefix),
        "a block with a prefix iterator has no register tile"
    );
    check_reg_reads(block, walk)?;
    let copied: Vec<usize> = walk.copies.iter().map(|copy| copy.operand.0).collect();
    check_lanes(block, atom.iter.position(), &copied, Some(walk))?;
    check_reg_folds(block, walk)
}

/// Fails unless each read and each write of `block`, which computes in vectors along the iterator
/// at `lane`, takes consecutive elements along it or does not move with it, but for the reads of
/// the operands `not_loaded`, which come from copies that the walk lays out or from the block
/// before: vector code loads, broadcasts and stores whole registers. With the block's own walk,
/// `own_walk`, a read counts in the terms of its tiles ([`Walk::reads_lanes_consecutively`]).
pub(crate) fn check_lanes(
    block: &Block,
    lane: usize,
    not_loaded: &[usize],
    own_walk: Option<&Walk>,
) -> Result<()> {
    for read in &block.reads {
        let operand = read.source.first();
        let consecutive = match own_walk {
            Some(walk) => walk.reads_lanes_consecutively(block, read, lane),
            None => read.contiguous_along(lane),
        };
        ensure!(
            not_loaded.contains(&operand) || consecutive,
            "the lanes run along d{lane}, which the read of ${operand} takes other than in whole \
             vectors of consecutive elements"
        );
    }
    for (index, write) in block.writes().enumerate() {
        let Some((last, outer)) = write.coords.split_last() else {
            continue;
        };
        ensure!(
            last.coefficient(lane) == Some(1) && !outer.iter().any(|coord| coord.uses_iter(lane)),
            "the lanes run along d{lane}, which write {index} takes other than by 1 in its last \
             coordinate"
        );
    }
    Ok(())
}

/// Fails with the rule that the reads of `block` break for the register tile of `walk`.
fn check_reg_reads(block: &Block, walk: &Walk) -> Result<()> {
    for read in &block.reads {
        let (operand, copied) = match &read.source {
            Source::Operand(operand) => {
                let copied = walk
                    .copies
                    .iter()
                    .any(|copy| copy.operand == OperandRef(*operand));
                (*operand, copied)
            }
            Source::List { first, pick, .. } => {
                let picked_by = pick.iters();
                for dim in &walk.reg {
                    ensure!(
                        !picked_by.contains(&dim.iter.position()),
                        "the register tile holds {}, by which a read picks its operand of a list",
                        dim.iter
                    );
                }
                (*first, false)
            }
        };
        for dim in &walk.reg {
            let position = dim.iter.position();
            ensure!(
                copied || walk.reads_in_steps(block, read, position),
                "the register tile holds {}, which the read of ${operand} takes other than by a \
                 multiple of it",
                dim.iter
            );
        }
    }
    Ok(())
}

/// Fails with the rule that the folds of `block` break for the register tile of `walk`.
fn check_reg_folds(block: &Block, walk: &Walk) -> Result<()> {
    for fold in &block.folds {
        let one_element = fold.combine.keeps_one_element();
        ensure!(
            one_element,
            "a fold in a register tile keeps one element per point"
        );
        if fold.combine != CombineKind::None {
            continue;
        }
        for write in &fold.writes {
            for dim in &walk.reg {
                let position = dim.iter.position();
                ensure!(
                    write
                        .coords
                        .iter()
                        .any(|coord| coord.iters().contains(&position)),
                    "the register tile holds {}, which a write leaves out: its points would write \
                     one element",
                    dim.iter
                );
            }
        }
    }
    Ok(())
}

/// Fails with the rule that the folds of `block` break for its modes: `online_softmax` and
/// `logsumexp` fold every point of an element in one walk, a fold of `none` keeps no state, and
/// each other fold keeps its state where [`check_state`] says. Without modes, the block walks one
/// point at a time, and its folds keep one element each.
fn check_nest_folds(block: &Block, walk: &Walk) -> Result<()> {
    for (index, fold) in block.folds.iter().enumerate() {
        if block.loops.is_empty() {
            ensure!(
                fold.held.is_none(),
                "fold {index} holds its state at a mode, and the block has no modes"
            );
            check_reference_state(block, walk, index)?;
            continue;
        }
        ensure!(
            !matches!(
                fold.combine,
                CombineKind::OnlineSoftmax(_) | CombineKind::Logsumexp
            ),
            "a block that folds with `online_softmax` or `logsumexp` takes no modes: its fold \
             has no update of a piece"
        );
        if fold.combine == CombineKind::None {
            ensure!(
                fold.held.is_none(),
                "fold {index} holds a state, and a fold of `none` keeps none"
            );
            continue;
        }
        check_state(block, walk, index)?;
    }
    Ok(())
}

/// Fails with the rule that fold `index` of `block` breaks for where it keeps its state under
/// `walk`. A fold that holds none keeps the register tile's points, or one element without one,
/// so the loops from its first loop over a reduced iterator on walk no parallel iterator outside
/// the register tile. A fold that holds its state in scratch holds it at a mode of the nest, with
/// no loop over a reduced iterator outside, or it would finish an element more than once; its
/// state's tiles have static extents.
fn check_state(block: &Block, walk: &Walk, index: usize) -> Result<()> {
    let fold = &block.folds[index];
    if let Some(StateHolding::Scratch(scratch)) = &fold.held {
        let Some(nested) = walk.nest_of_mode.get(scratch.at) else {
            bail!(
                "fold {index} holds its state at mode {}, of loops of {} modes",
                scratch.at,
                block.loops.len()
            );
        };
        ensure!(
            nested.is_some(),
            "fold {index} holds its state at mode {}, a mode of the register tile, which runs \
             its points at once",
            scratch.at
        );
    }
    let steps = walk.steps();
    let start = walk.fold_start(block, &steps, fold);
    for &step in &steps[..start] {
        let position = walk.iter_of(step);
        ensure!(
            !block.iters[position].kind.is_reduced(),
            "fold {index} starts its state inside a loop over d{position}, a reduced iterator: \
             it would finish each element more than once"
        );
    }
    let prefix = block
        .iters
        .iter()
        .position(|iter| iter.kind == IterKind::Prefix);
    if let Some(position) = prefix {
        ensure!(
            steps[..start]
                .iter()
                .any(|&step| walk.walks(step, position)),
            "fold {index} starts its state outside the loop over d{position}, the prefix \
             iterator, whose point each element folds up to"
        );
    }
    let tiles = walk.state_tiles(block, &steps, start);
    if fold.held.is_none() {
        let held_by_reg = match walk.reg.is_empty() {
            true => tiles.is_empty(),
            false => walk.in_registers(&tiles),
        };
        ensure!(
            held_by_reg,
            "fold {index} holds no state, and its loops walk the elements of more than the \
             register tile inside its first loop over a reduced iterator"
        );
        return Ok(());
    }
    for (position, extent) in tiles {
        ensure!(
            extent.as_static().is_some(),
            "the state of fold {index} keeps the piece of d{position}, whose extent {extent} is \
             not static"
        );
    }
    Ok(())
}

/// Fails when the state of fold `index` of `block`, a block without modes, under `walk`, its
/// reference walk, keeps the piece of an iterator whose extent is not static: one element for each
/// point of the parallel iterators that the walk takes inside its first loop over a reduced
/// iterator, as when the block updates an operand in order and walks its iterators in theirs.
fn check_reference_state(block: &Block, walk: &Walk, index: usize) -> Result<()> {
    let fold = &block.folds[index];
    if fold.combine == CombineKind::None || fold.writes[0].into.is_some() {
        return Ok(());
    }
    let steps = walk.steps();
    let start = walk.fold_start(block, &steps, fold);
    for (position, extent) in walk.state_tiles(block, &steps, start) {
        ensure!(
            extent.as_static().is_some(),
            "the state of fold {index} keeps the piece of d{position}, whose extent {extent} is \
             not static"
        );
    }
    Ok(())
}

/// Fails with the rule that a copy of an operand breaks for `block` under `walk`: the operand is
/// what one read takes alone, a read of a list picks one operand wherever the copy is made, and
/// the copy's axes have static extents.
fn check_copies(block: &Block, walk: &Walk) -> Result<()> {
    for copy in &walk.copies {
        let OperandRef(operand) = copy.operand;
        let Some(read) = block.only_read_of(operand) else {
            bail!(
                "{} is held in a copy, and the block uses it other than in one read of it alone",
                copy.operand
            );
        };
        if let Source::List { pick, .. } = &read.source {
            for position in pick.iters() {
                let fixed = copy.at.is_some_and(|at| {
                    walk.nest[..=at]
                        .iter()
                        .any(|entry| entry.iter.position() == position && entry.extent == 1)
                });
                ensure!(
                    fixed,
                    "{} is copied where d{position}, by which its read picks the operand, does \
                     not stand at one point",
                    copy.operand
                );
            }
        }
        for axis in walk.copy_axes(block, read, copy.at) {
            ensure!(
                axis.extent().as_static().is_some(),
                "the copy of {} has an axis of {} elements, which is not static",
                copy.operand,
                axis.extent()
            );
        }
    }
    Ok(())
}

/// Fails with the rule that a prefetch breaks for `block` under `walk`: the operand is what one
/// read takes alone, in place, and the prefetch asks at a loop over an iterator that the read
/// takes.
fn check_prefetches(block: &Block, walk: &Walk) -> Result<()> {
    for prefetch in &walk.prefetches {
        let OperandRef(operand) = prefetch.operand;
        let name = prefetch.operand;
        let Some(read) = block.only_read_of(operand) else {
            bail!("{name} is prefetched, and the block uses it other than in one read of it alone");
        };
        ensure!(
            matches!(read.source, Source::Operand(_)) && read.pad.is_none(),
            "{name} is prefetched, and its read pads or picks it from a list"
        );
        let Some(at) = prefetch.at else {
            continue;
        };
        let mut taken = BTreeSet::new();
        for coord in &read.coords {
            taken.extend(coord.iters());
        }
        let walked = walk.nest.get(at);
        ensure!(
            walked.is_some_and(|entry| taken.contains(&entry.iter.position())),
            "{name} is prefetched at a mode over an iterator its read does not take"
        );
    }
    Ok(())
}
