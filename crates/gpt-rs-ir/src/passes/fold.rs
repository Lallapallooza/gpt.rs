use anyhow::Result;

use super::{pass, Greedy, Pass, Rewrites};
use crate::module::Module;

/// Replaces each op that folds by an existing value or a constant, until nothing changes. The
/// folds are those of `canonicalize`, without its patterns.
#[pass(
    name = "fold",
    stats(folded = "ops replaced by an existing value or a constant")
)]
pub struct Fold;

impl Pass for Fold {
    fn rewrites(&self) -> Rewrites {
        Rewrites {
            folds: true,
            ..Rewrites::default()
        }
    }

    fn run(&self, module: &mut Module) -> Result<bool> {
        let report = Greedy::new(self.rewrites()).run(module)?;
        Fold::FOLDED.add(report.folded);
        Ok(report.changed())
    }
}
