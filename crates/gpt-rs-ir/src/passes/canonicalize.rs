use anyhow::Result;

use super::{pass, Greedy, Pass, Rewrites};
use crate::module::Module;
use crate::rewrite::canonical_patterns;

/// Simplifies the module without changing what it computes: folds ops, erases dead ones and
/// applies every canonicalization pattern until nothing changes.
///
/// The folds are the ops' `#[fold]` folders and the folds of their traits (`Commutative`,
/// `Involution`, ...). The patterns are those that `#[pattern(canonical)]` registers. Each keeps
/// the results bit for bit, apart from the sign and payload of a NaN, which IEEE 754 leaves
/// unspecified.
#[pass(
    name = "canonicalize",
    stats(
        folded = "ops replaced by an existing value or a constant",
        erased = "dead ops erased",
        rewritten = "ops rewritten by a pattern",
    )
)]
#[derive(Default)]
pub struct Canonicalize {
    /// Stops after this many folds, erasures and rewrites. 0 runs to a fixed point. Use it to
    /// bisect a pattern that loops or a rewrite that breaks a program.
    pub fuel: u64,
}

impl Pass for Canonicalize {
    fn rewrites(&self) -> Rewrites {
        Rewrites {
            folds: true,
            patterns: canonical_patterns(),
        }
    }

    fn run(&self, module: &mut Module) -> Result<bool> {
        let report = Greedy::new(self.rewrites())
            .erase_dead()
            .fuel(self.fuel)
            .run(module)?;
        Self::FOLDED.add(report.folded);
        Self::ERASED.add(report.erased);
        Self::REWRITTEN.add(report.rewritten);
        Ok(report.changed())
    }
}
