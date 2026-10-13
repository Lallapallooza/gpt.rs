//! The loops of a `tile.dispatch` under its schedule record.

use crate::dialects::tile::ir::*;
use crate::ir::Dim;

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
}
