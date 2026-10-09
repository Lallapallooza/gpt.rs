//! The greedy driver: removes dead ops, folds, and applies patterns until nothing changes.

use std::collections::{HashSet, VecDeque};

use anyhow::{ensure, Result};

use crate::doc::{fired, Fired};
use crate::fold::fold_inst;
use crate::module::{Inst, Module};
use crate::rewrite::{PatternSet, Rewriter};

/// The folds and patterns a greedy rewrite applies.
#[derive(Default)]
pub struct Rewrites {
    /// Whether ops fold: by their `#[fold]` folders, and by their traits.
    pub folds: bool,
    /// The patterns to apply.
    pub patterns: PatternSet,
}

/// The driver of the passes made of folds and patterns. It erases each op it visits that is dead,
/// folds one that folds, and otherwise rewrites it with the first pattern that applies. Only the
/// ops a step touched are visited again.
pub struct Greedy {
    rewrites: Rewrites,
    erase_dead: bool,
    fuel: u64,
}

/// What a run of the driver did.
#[derive(Debug, Default, Clone, Copy)]
pub struct GreedyReport {
    /// Ops replaced by an existing value or a constant.
    pub folded: usize,
    /// Dead ops erased.
    pub erased: usize,
    /// Ops rewritten by a pattern.
    pub rewritten: usize,
}

impl GreedyReport {
    /// Whether the run changed the module.
    pub fn changed(&self) -> bool {
        self.folded + self.erased + self.rewritten > 0
    }
}

/// The step the driver took on one op.
enum Step {
    Erased,
    Folded,
    Rewritten,
}

impl Greedy {
    /// The driver that applies `rewrites`.
    pub fn new(rewrites: Rewrites) -> Self {
        Self {
            rewrites,
            erase_dead: false,
            fuel: 0,
        }
    }

    pub fn erase_dead(mut self) -> Self {
        self.erase_dead = true;
        self
    }

    /// Stops after `fuel` steps. 0 runs to a fixed point.
    pub fn fuel(mut self, fuel: u64) -> Self {
        self.fuel = fuel;
        self
    }

    /// Runs the driver to a fixed point, or until its fuel runs out.
    pub fn run(&self, module: &mut Module) -> Result<GreedyReport> {
        let mut queue: VecDeque<Inst> = module.walk(module.body()).into();
        let mut queued: HashSet<Inst> = queue.iter().copied().collect();
        // A converging rewrite takes a few steps per op and pattern. Rewrites that undo each other
        // run past this bound.
        let bound = (queue.len() + 1) * (self.rewrites.patterns.len() + 3);
        let mut report = GreedyReport::default();
        let mut steps = 0usize;
        while let Some(inst) = queue.pop_front() {
            if self.fuel != 0 && steps as u64 == self.fuel {
                break;
            }
            queued.remove(&inst);
            if module.is_erased(inst) {
                continue;
            }
            let mut rw = Rewriter::new(module);
            let Some(step) = self.step(inst, &mut rw)? else {
                continue;
            };
            match step {
                Step::Erased => report.erased += 1,
                Step::Folded => report.folded += 1,
                Step::Rewritten => report.rewritten += 1,
            }
            steps += 1;
            ensure!(steps <= bound, "did not converge within {bound} steps");
            for touched in rw.touched {
                if queued.insert(touched) {
                    queue.push_back(touched);
                }
            }
        }
        Ok(report)
    }

    /// Applies the first step that changes `inst`, and returns it.
    fn step(&self, inst: Inst, rw: &mut Rewriter<'_>) -> Result<Option<Step>> {
        if self.erase_dead && rw.module.is_trivially_dead(inst) {
            rw.erase(inst)?;
            return Ok(Some(Step::Erased));
        }
        let fold = self.rewrites.folds.then(|| fold_inst(rw.module, inst));
        if let Some(fold) = fold.flatten() {
            rw.apply_fold_result(inst, fold)?;
            return Ok(Some(Step::Folded));
        }
        for pattern in self.rewrites.patterns.iter() {
            if pattern.try_rewrite(inst, rw)? {
                fired(Fired::Pattern(pattern.name()));
                return Ok(Some(Step::Rewritten));
            }
        }
        Ok(None)
    }
}
