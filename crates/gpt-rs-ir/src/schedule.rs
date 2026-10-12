//! The schedule record of a dispatch: how it runs. `tile-to-loop` realizes each field and makes no
//! decision of its own, and a field left out is the reference: one sequential loop per iterator,
//! in the order of the block.
//!
//! Fields name the iterators of the dispatch's block by position, as `d0`.

use anyhow::{bail, ensure, Result};

use crate::dialects::tile::{Block, CombineKind};
use crate::index::{IterKind, IterRef};
use crate::module::{Module, Region};
use crate::types::Dim;

/// How a dispatch runs: `#sched<nest = [...]>`.
#[derive(crate::Attr, Debug, Clone, Default, PartialEq, Eq, Hash)]
#[attr(format = "`#sched` `<` struct($nest) `>`")]
pub struct Schedule {
    /// Loops over tiles of the iterators, outermost first. Inside them, a loop walks the points of
    /// each iterator's innermost tile: those of the parallel iterators first, then those of the
    /// reduced ones, each in the order of the block.
    #[attr(default)]
    pub nest: Vec<Loop>,
}

/// One loop of a nest: over the tiles of `extent` points of iterator `iter`, within its tile of
/// the loop around it, or within its whole extent. The last tile of a range may be smaller:
/// `d1 by 64 : par<level = core>`.
#[derive(crate::Attr, Debug, Clone, PartialEq, Eq, Hash)]
#[attr(format = "$iter `by` $extent `:` $parallelism")]
pub struct Loop {
    pub iter: IterRef,
    pub extent: usize,
    pub parallelism: Parallelism,
}

/// How a loop of a nest runs its tiles.
#[derive(crate::Attr, Debug, Clone, PartialEq, Eq, Hash)]
pub enum Parallelism {
    /// One after another, in increasing order.
    Seq,
    /// As units of work at an execution level, which may run at the same time.
    Par(ParLoop),
}

/// Where a parallel loop runs its tiles: on units of `level`, `units` of them, each running a
/// contiguous run of tiles, or one unit per tile.
#[derive(crate::AttrGroup, Debug, Clone, PartialEq, Eq, Hash)]
pub struct ParLoop {
    pub level: Level,
    #[attr(default)]
    pub units: Option<usize>,
}

/// An execution level of a CPU, from the outermost.
#[derive(crate::Attr, Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Level {
    /// The cores that share one last-level cache.
    L3,
    /// One core.
    Core,
}

/// One loop that `tile-to-loop` builds for a fold of a block under a nest.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Step {
    /// Loop `index` of the nest, over tiles.
    Tiles(usize),
    /// Over the points of the innermost tile of the iterator at this position.
    Points(usize),
}

impl Schedule {
    /// The loops of a fold of `block`, outermost first: the nest's loops, then one over the points
    /// of each iterator. A fold into new results walks the parallel iterators first and the
    /// reduced ones after them; a fold of `none` walks every iterator in the order of the block.
    pub fn steps(&self, block: &Block, folds: bool) -> Vec<Step> {
        let mut steps: Vec<Step> = (0..self.nest.len()).map(Step::Tiles).collect();
        let iters = &block.iters;
        let mut points: Vec<usize> = (0..iters.len()).collect();
        if folds {
            points.sort_by_key(|&position| iters[position].kind.is_reduced());
        }
        steps.extend(points.into_iter().map(Step::Points));
        steps
    }

    /// The iterator a step walks.
    pub fn iter_of(&self, step: Step) -> usize {
        match step {
            Step::Tiles(index) => self.nest[index].iter.position(),
            Step::Points(iter) => iter,
        }
    }

    /// Where the fold's state lives, for the steps `steps` of a fold of `block`: the first step
    /// over a reduced iterator, before which the state of each element starts, and the parallel
    /// iterators that steps from there on walk, with the extent of their tile there. The state
    /// keeps one element per point of those tiles.
    pub fn state_tiles(&self, block: &Block, steps: &[Step]) -> (usize, Vec<(usize, Dim)>) {
        let iters = &block.iters;
        let start = steps
            .iter()
            .position(|&step| iters[self.iter_of(step)].kind.is_reduced())
            .unwrap_or(steps.len());
        let mut tiles = Vec::new();
        for (position, iter) in iters.iter().enumerate() {
            if iter.kind.is_reduced() {
                continue;
            }
            if !steps[start..]
                .iter()
                .any(|&step| self.iter_of(step) == position)
            {
                continue;
            }
            let outer = steps[..start].iter().rev().find_map(|&step| match step {
                Step::Tiles(index) if self.nest[index].iter.position() == position => {
                    Some(Dim::Static(self.nest[index].extent))
                }
                _ => None,
            });
            tiles.push((position, outer.unwrap_or_else(|| iter.extent.clone())));
        }
        (start, tiles)
    }

    /// Fails with the rule the schedule breaks for the dispatch whose body is `body`.
    pub fn legal(&self, module: &Module, body: Region) -> Result<()> {
        if self.nest.is_empty() {
            return Ok(());
        }
        let blocks: Vec<&Block> = module
            .insts(body)
            .filter_map(|inst| module.op(inst).get::<Block>())
            .collect();
        let &[block] = blocks.as_slice() else {
            bail!(
                "a nest schedules the one block of a dispatch, and this one holds {}",
                blocks.len()
            );
        };
        let iters = &block.iters;
        let mut tiles: Vec<Option<usize>> = vec![None; iters.len()];
        let mut innermost = None;
        for (index, entry) in self.nest.iter().enumerate() {
            let position = entry.iter.position();
            let Some(iter) = iters.get(position) else {
                bail!(
                    "loop {index} walks {}, which the block does not have",
                    entry.iter
                );
            };
            ensure!(entry.extent > 0, "loop {index} walks empty tiles");
            if let Some(outer) = tiles[position] {
                ensure!(
                    entry.extent < outer,
                    "loop {index} walks tiles of {} points of {}, inside tiles of {outer}",
                    entry.extent,
                    entry.iter
                );
            }
            tiles[position] = Some(entry.extent);
            ensure!(
                iter.kind != IterKind::Prefix,
                "loop {index} walks {}, a prefix iterator, which a nest leaves whole",
                entry.iter
            );
            let Parallelism::Par(par) = &entry.parallelism else {
                continue;
            };
            ensure!(
                !iter.kind.is_reduced(),
                "loop {index} runs tiles of {}, a reduction iterator, in parallel: that splits \
                 the reduction",
                entry.iter
            );
            ensure!(par.units != Some(0), "loop {index} runs on no units");
            if let Some(outer) = innermost {
                ensure!(
                    par.level >= outer,
                    "loop {index} runs at level {}, inside a loop at level {}",
                    level_name(par.level),
                    level_name(outer)
                );
            }
            innermost = Some(par.level);
        }
        for fold in &block.folds {
            ensure!(
                fold.writes.iter().all(|write| write.into.is_none()),
                "a nest of a block that updates an operand would reorder its writes"
            );
            ensure!(
                !matches!(
                    fold.combine,
                    CombineKind::OnlineSoftmax(_) | CombineKind::Logsumexp
                ),
                "a nest of `online_softmax` or `logsumexp` needs their update of a tile"
            );
            if fold.combine == CombineKind::None {
                continue;
            }
            let steps = self.steps(block, true);
            for (position, extent) in self.state_tiles(block, &steps).1 {
                ensure!(
                    extent.as_static().is_some(),
                    "the state of a fold keeps the tile of d{position}, whose extent {extent} is \
                     not static"
                );
            }
        }
        Ok(())
    }
}

fn level_name(level: Level) -> &'static str {
    crate::attr::Mnemonic::mnemonic(level)
}
