//! The CSE driver: replaces an op by an earlier op that computes the same values.

use anyhow::Result;

use crate::ir::{Inst, Module, OpKey, Region, ScopedMap};
use crate::rewrite::{Editor, Rewriting};
use crate::traits::IsolatedFromAbove;

/// Replaces each op of the module that `rewriting` transforms by an earlier op that computes the
/// same values, and gives how many it replaced: the driver of `cse`.
pub fn cse(rewriting: &mut Rewriting) -> Result<usize> {
    let module = &mut *rewriting.module;
    let body = module.body();
    cse_region(module, body, &mut ScopedMap::new())
}

/// The number of ops of `region` and the regions inside it that it replaces.
fn cse_region(
    module: &mut Module,
    region: Region,
    seen: &mut ScopedMap<OpKey, Inst>,
) -> Result<usize> {
    let mut replaced = 0;
    for inst in module.insts(region).collect::<Vec<_>>() {
        let isolated = module.op(inst).has_trait::<IsolatedFromAbove>();
        for inner in module.regions(inst).to_vec() {
            seen.push(isolated);
            replaced += cse_region(module, inner, seen)?;
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
