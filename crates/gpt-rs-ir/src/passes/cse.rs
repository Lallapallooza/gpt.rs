//! `cse`: replaces an op by an earlier op that computes the same values.

use anyhow::Result;

use super::{pass, Pass};
use crate::module::{Inst, Module, OpKey, Region};
use crate::rewrite::Editor;
use crate::scope::ScopedMap;
use crate::traits::IsolatedFromAbove;

/// Replaces each op by an earlier op that computes the same values: the same op with the same
/// attributes and operands. Ops with effects or regions, and terminators, are never replaced. An
/// op in a region also matches an op of an enclosing region, unless a region isolated from above
/// lies between them.
#[pass(name = "cse", stats(replaced = "ops replaced by an earlier op"))]
pub struct Cse;

impl Pass for Cse {
    fn run(&self, module: &mut Module) -> Result<bool> {
        let body = module.body();
        let replaced = cse(module, body, &mut ScopedMap::new())?;
        Cse::REPLACED.add(replaced);
        Ok(replaced > 0)
    }
}

/// The number of ops of `region` and the regions inside it that it replaces.
fn cse(module: &mut Module, region: Region, seen: &mut ScopedMap<OpKey, Inst>) -> Result<usize> {
    let mut replaced = 0;
    for inst in module.insts(region).collect::<Vec<_>>() {
        let isolated = module.op(inst).has_trait::<IsolatedFromAbove>();
        for inner in module.regions(inst).to_vec() {
            seen.push(isolated);
            replaced += cse(module, inner, seen)?;
            seen.pop();
        }
        let Some(key) = module.cse_key(inst) else {
            continue;
        };
        match seen.get(&key) {
            Some(earlier) => {
                let values = module.results(earlier).to_vec();
                Editor::new(module).replace(inst, &values)?;
                replaced += 1;
            }
            None => {
                seen.insert(key, inst);
            }
        }
    }
    Ok(replaced)
}
