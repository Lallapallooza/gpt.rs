//! `prune-params`: removes the parameter arguments that nothing reads.

use anyhow::Result;

use super::{pass, Pass};
use crate::dialects::func::{Func, FuncOp};
use crate::require;
use crate::rewrite::{
    pattern, Greedy, NoMatch, Pattern, PatternSet, Rewriter, Rewrites, Rewriting,
};

/// Removes from every function the arguments bound to a parameter that nothing reads. A run-time
/// argument stays even when nothing reads it, so the run-time arguments keep their positions.
#[pass(name = "prune-params", stats(pruned = "parameter arguments removed"))]
pub struct PruneParams;

fn populate_prune_params_patterns(set: &mut PatternSet) {
    set.add(RemoveUnreadParams);
}

impl Pass for PruneParams {
    fn rewrites(&self) -> Rewrites {
        Rewrites::once(populate_prune_params_patterns)
    }

    fn run(&self, module: &mut Rewriting) -> Result<bool> {
        Ok(Greedy::new(self.rewrites()).run(module)?.changed())
    }
}

/// A function loses the arguments bound to a parameter that nothing reads.
#[pattern]
struct RemoveUnreadParams;

impl Pattern for RemoveUnreadParams {
    type Root = Func;

    fn rewrite(&self, func: FuncOp, rewriter: &mut Rewriter) -> Result<(), NoMatch> {
        let body = func.body();
        let mut unread = Vec::new();
        for (index, arg) in body.args().iter().enumerate() {
            if func.param(index).is_some() && arg.uses().is_empty() {
                unread.push(index);
            }
        }
        require!(!unread.is_empty(), "every parameter argument is read");
        // From the last, so that the indices still to remove keep their places.
        for &index in unread.iter().rev() {
            rewriter.remove_arg(body.id(), index);
        }
        PruneParams::PRUNED.add(unread.len());
        Ok(())
    }
}
