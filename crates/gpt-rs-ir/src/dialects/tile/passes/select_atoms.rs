//! Choosing the atom of the first block of each dispatch.

use anyhow::Result;

use crate::dialects::tile::{Block, Dispatch, DispatchOp};
use crate::passes::{pass, run_patterns, Pass, Registered};
use crate::require;
use crate::rewrite::{pattern, NoMatch, Pattern, PatternSet, Rewriter, Rewrites, Rewriting};
use crate::targets::{required, Target};

/// Chooses how the first block of each dispatch computes its innermost step on the target that
/// the dispatch names.
///
/// - Vectors: the first block of a dispatch computes on vectors whose lanes run along the
///   iterator that is the last coordinate of every write, when its body computes in `arith` ops
///   that `isa` has a lane-wise form of, the verifier accepts it computing in them, and each block
///   after it is its epilogue and computes in the same lanes ([`Dispatch::vector_lanes`]). A
///   vector holds as many lanes as the target's widest register holds of the block's widest
///   dtype. The block's loops walk the vectors of that iterator (`d by lanes`), unless its extent
///   is one vector, and run each in the lanes (`d by lanes : lanes`).
/// - Any other dispatch computes on scalars.
#[pass(
    name = "select-atoms",
    stats(vectorized = "dispatches whose first block computes on vectors")
)]
#[derive(Default)]
pub struct SelectAtoms {
    /// The target the atoms are for.
    pub target: Option<Target>,
}

fn populate_select_atoms_patterns(set: &mut PatternSet, target: Option<Target>) {
    set.add(SelectVectors { target });
}

impl Pass for SelectAtoms {
    fn rewrites(&self) -> Rewrites {
        let mut patterns = PatternSet::new();
        populate_select_atoms_patterns(&mut patterns, self.target.clone());
        Rewrites {
            patterns,
            ..Rewrites::default()
        }
    }

    fn run(&self, module: &mut Rewriting) -> Result<bool> {
        required(self.target.as_ref(), self.name())?;
        run_patterns(module, self.rewrites(), &Self::VECTORIZED)
    }
}

/// A dispatch that names the pass's target, whose first block can compute on vectors: the block
/// walks the vectors and runs each in the lanes.
#[pattern]
struct SelectVectors {
    target: Option<Target>,
}

impl Pattern for SelectVectors {
    type Root = Dispatch;

    fn rewrite(&self, dispatch: DispatchOp, rewriter: &mut Rewriter) -> Result<(), NoMatch> {
        require!(let Some(target) = &self.target, "the pass has no target");
        require!(
            dispatch.target.as_ref() == Some(target.reference()),
            "the dispatch names another target, or none"
        );
        let module = dispatch.module();
        let blocks = Dispatch::blocks(module, dispatch.body().id());
        require!(
            let Some(lanes) = Dispatch::vector_lanes(module, &blocks, target),
            "vectors do not compute the first block"
        );
        let (inst, block) = blocks[0];
        let first = module
            .op_as::<Block>(inst)
            .expect("the first block of the dispatch");
        first.set_loops(rewriter, block.lane_modes(lanes));
        Ok(())
    }
}
