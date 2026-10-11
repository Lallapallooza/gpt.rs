use anyhow::Result;

use super::{pass, Greedy, Pass, Rewrites};
use crate::module::Module;

/// Erases the ops that have no effect and whose results nothing reads, until nothing changes.
#[pass(name = "dce", stats(erased = "dead ops erased"))]
pub struct Dce;

impl Pass for Dce {
    fn run(&self, module: &mut Module) -> Result<bool> {
        let report = Greedy::new(Rewrites::default()).erase_dead().run(module)?;
        Dce::ERASED.add(report.erased);
        Ok(report.changed())
    }
}
