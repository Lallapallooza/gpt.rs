//! `prune-params`: removes the parameter arguments that nothing reads.

use anyhow::Result;

use super::{pass, Pass};
use crate::dialects::func::Func;
use crate::ir::Module;

/// Removes from every function the arguments bound to a parameter that nothing reads. A run-time
/// argument stays even when nothing reads it, so the run-time arguments keep their positions.
#[pass(name = "prune-params", stats(pruned = "parameter arguments removed"))]
pub struct PruneParams;

impl Pass for PruneParams {
    fn run(&self, module: &mut Module) -> Result<bool> {
        let mut pruned = 0;
        for func in module.funcs() {
            let attrs = module.op(func).get::<Func>().expect("a function");
            let args = module.args(module.regions(func)[0]);
            let unread: Vec<usize> = (0..args.len())
                .filter(|&i| attrs.param(i).is_some() && module.uses(args[i]).is_empty())
                .collect();
            // From the last, so that the indices still to remove keep their places.
            for &index in unread.iter().rev() {
                module.remove_arg(func, index)?;
            }
            pruned += unread.len();
        }
        PruneParams::PRUNED.add(pruned);
        Ok(pruned > 0)
    }
}
