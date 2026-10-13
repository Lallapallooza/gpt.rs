//! The greedy driver: removes dead ops, folds, and applies patterns until nothing changes.

use std::collections::{HashMap, HashSet, VecDeque};

use anyhow::{ensure, Result};

use crate::ir::{Inst, Module, OpKey, Region};
use crate::rewrite::{fold_inst, Applied, Editor, PatternSet};
use crate::traits::IsolatedFromAbove;

/// The folds and patterns a greedy rewrite applies.
#[derive(Default)]
pub struct Rewrites {
    /// Whether ops fold: by their `#[fold]` folders, and by their traits. The constants of a
    /// function then go to its start, and equal ones become one.
    pub folds: bool,
    /// The patterns to apply.
    pub patterns: PatternSet,
}

impl Rewrites {
    /// The patterns that `populate` adds, without folds: the rewrites of a pass that applies
    /// patterns.
    pub fn patterns(populate: fn(&mut PatternSet)) -> Rewrites {
        let mut patterns = PatternSet::new();
        populate(&mut patterns);
        Rewrites {
            folds: false,
            patterns,
        }
    }
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
        let mut constants = Constants::default();
        let mut steps = 0usize;
        while let Some(inst) = queue.pop_front() {
            if self.fuel != 0 && steps as u64 == self.fuel {
                break;
            }
            queued.remove(&inst);
            if module.is_erased(inst) {
                continue;
            }
            let Some((step, touched)) = self.step(module, inst, &mut constants)? else {
                continue;
            };
            match step {
                Step::Erased => report.erased += 1,
                Step::Folded => report.folded += 1,
                Step::Rewritten => report.rewritten += 1,
            }
            steps += 1;
            ensure!(steps <= bound, "did not converge within {bound} steps");
            for touched in touched {
                if queued.insert(touched) {
                    queue.push_back(touched);
                }
            }
        }
        Ok(report)
    }

    /// Applies the first step that changes `inst`, and returns it with the instructions it
    /// touched.
    fn step(
        &self,
        module: &mut Module,
        inst: Inst,
        constants: &mut Constants,
    ) -> Result<Option<(Step, Vec<Inst>)>> {
        if self.erase_dead && module.is_trivially_dead(inst) {
            let mut editor = Editor::new(module);
            editor.erase(inst)?;
            return Ok(Some((Step::Erased, editor.touched)));
        }
        if self.rewrites.folds {
            if let Some(touched) = constants.place(module, inst)? {
                return Ok(Some((Step::Folded, touched)));
            }
        }
        let fold = self.rewrites.folds.then(|| fold_inst(module, inst));
        if let Some(fold) = fold.flatten() {
            let mut editor = Editor::new(module);
            editor.apply_fold(inst, fold)?;
            return Ok(Some((Step::Folded, editor.touched)));
        }
        Ok(match self.rewrites.patterns.apply(module, inst, false)? {
            Applied::Rewritten { touched, .. } => Some((Step::Rewritten, touched)),
            Applied::NoMatch(_) => None,
        })
    }
}

/// The constants the driver has seen, one for each value in each home. A constant's home is the
/// body of the nearest op isolated from above that holds it.
#[derive(Default)]
struct Constants(HashMap<(Region, OpKey), Inst>);

impl Constants {
    /// Puts the constant `inst` with the constants at the start of its home, and merges it with
    /// an equal constant there. Every use of a constant is an op that is not a constant. So a
    /// constant placed with the first instructions of its home is defined before every use of it,
    /// in the home and in the regions the home holds. Gives the instructions the step touched, or
    /// `None` when `inst` is not a constant or is in place already.
    fn place(&mut self, module: &mut Module, inst: Inst) -> Result<Option<Vec<Inst>>> {
        if module.op(inst).as_constant().is_none() {
            return Ok(None);
        }
        let (Some(home), Some(key)) = (home(module, inst), module.cse_key(inst)) else {
            return Ok(None);
        };
        if let Some(&kept) = self.0.get(&(home, key.clone())) {
            if kept != inst && !module.is_erased(kept) {
                if !leads(module, kept, home) {
                    lead(module, kept, home);
                }
                let mut editor = Editor::new(module);
                let kept_value = editor.module.results(kept)[0];
                editor.replace(inst, &[kept_value])?;
                return Ok(Some(editor.touched));
            }
        }
        self.0.insert((home, key), inst);
        if leads(module, inst, home) {
            return Ok(None);
        }
        lead(module, inst, home);
        Ok(Some(Vec::new()))
    }
}

/// The body of the nearest op isolated from above that holds `inst`, such as its function's.
fn home(module: &Module, inst: Inst) -> Option<Region> {
    let mut region = module.parent_region(inst)?;
    loop {
        let op = module.parent_op(region)?;
        if module.op(op).has_trait::<IsolatedFromAbove>() {
            return Some(region);
        }
        region = module.parent_region(op)?;
    }
}

/// Whether `inst` is in `home` with only constants before it.
fn leads(module: &Module, inst: Inst, home: Region) -> bool {
    if module.parent_region(inst) != Some(home) {
        return false;
    }
    for placed in module.insts(home) {
        if placed == inst {
            return true;
        }
        if module.op(placed).as_constant().is_none() {
            return false;
        }
    }
    false
}

/// Moves the constant `inst` after the constants at the start of `home`, so they keep the order
/// the driver placed them in. It reads nothing, so it may go anywhere before its uses.
fn lead(module: &mut Module, inst: Inst, home: Region) {
    let mut after_constants = None;
    for placed in module.insts(home) {
        if placed != inst && module.op(placed).as_constant().is_none() {
            after_constants = Some(placed);
            break;
        }
    }
    let anchor = after_constants.expect("a region ends with its terminator");
    module.move_before(inst, anchor);
}
