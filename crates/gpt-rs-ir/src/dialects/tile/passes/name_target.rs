//! Naming the target of each dispatch.

use anyhow::Result;

use crate::dialects::tile::{Dispatch, DispatchOp, Numerics};
use crate::passes::{pass, run_patterns, Pass, Registered};
use crate::require;
use crate::rewrite::{pattern, NoMatch, Pattern, PatternSet, Rewriter, Rewrites, Rewriting};
use crate::targets::{required, Target};

/// Names on each dispatch the target it is compiled for, with the rounding changes that the pass
/// allows there ([`Numerics::on_target`]): `contract`, a product and the sum it enters rounded
/// once, only where the target computes `isa.fma`. The passes after it choose for that target,
/// and the dispatch's verifier checks it against that target.
#[pass(
    name = "name-target",
    stats(named = "dispatches that name their target")
)]
#[derive(Default)]
pub struct NameTarget {
    /// The target the dispatches are compiled for.
    pub target: Option<Target>,
    /// The rounding changes the dispatches may make.
    pub numerics: Numerics,
}

fn populate_name_target_patterns(set: &mut PatternSet, target: Option<Target>, numerics: Numerics) {
    set.add(NameDispatchTarget { target, numerics });
}

impl Pass for NameTarget {
    fn rewrites(&self) -> Rewrites {
        let mut patterns = PatternSet::new();
        populate_name_target_patterns(&mut patterns, self.target.clone(), self.numerics);
        Rewrites {
            patterns,
            ..Rewrites::default()
        }
    }

    fn run(&self, module: &mut Rewriting) -> Result<bool> {
        required(self.target.as_ref(), self.name())?;
        run_patterns(module, self.rewrites(), &Self::NAMED)
    }
}

/// A dispatch that names no target names the pass's, with the numerics it allows there.
#[pattern]
struct NameDispatchTarget {
    target: Option<Target>,
    numerics: Numerics,
}

impl Pattern for NameDispatchTarget {
    type Root = Dispatch;

    fn rewrite(&self, dispatch: DispatchOp, rewriter: &mut Rewriter) -> Result<(), NoMatch> {
        require!(let Some(target) = &self.target, "the pass has no target");
        require!(dispatch.target.is_none(), "the dispatch names its target");
        let numerics = self.numerics.on_target(target.has_fma());
        dispatch.update(rewriter, |named| {
            named.target = Some(target.reference().clone());
            named.numerics = numerics;
        });
        Ok(())
    }
}
