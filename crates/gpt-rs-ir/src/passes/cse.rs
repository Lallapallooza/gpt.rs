//! `cse`: replaces an op by an earlier op that computes the same values.

use anyhow::Result;

use super::{pass, Pass};
use crate::rewrite::{cse, Rewriting};

/// Replaces each op by an earlier op that computes the same values: the same op with the same
/// attributes and operands. Ops with effects or regions, and terminators, are never replaced. An
/// op in a region also matches an op of an enclosing region, unless a region isolated from above
/// lies between them.
#[pass(name = "cse", stats(replaced = "ops replaced by an earlier op"))]
pub struct Cse;

impl Pass for Cse {
    fn run(&self, module: &mut Rewriting) -> Result<bool> {
        let replaced = cse(module)?;
        Cse::REPLACED.add(replaced);
        Ok(replaced > 0)
    }
}
