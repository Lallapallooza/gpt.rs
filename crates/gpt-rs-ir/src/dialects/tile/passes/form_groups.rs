//! Grouping blocks with their epilogues.

use anyhow::Result;

use crate::dialects::tile::utils::in_function_outside_groups;
use crate::dialects::tile::{Block, BlockOp, Group, Yield};
use crate::ir::Use;
use crate::passes::{pass, Pass};
use crate::require;
use crate::rewrite::{
    pattern, Greedy, NoMatch, Pattern, PatternSet, Rewriter, Rewrites, Rewriting,
};

/// Groups each block with the chain of blocks that run on its result element by element, its
/// epilogue, into a `tile.group`, which `seal-dispatches` outlines into one dispatch. Grouping
/// moves blocks and changes none: each keeps its body, reads, writes and folds, so it is exact.
///
/// - A block joins the chain when it is the one use of the result of the block before it and
///   takes it as an epilogue (`Block::takes_epilogue`): it runs on each element of that result,
///   where that block writes it, over the same leading parallel iterators.
/// - A block that runs at each call joins no chain of a block known before the run: what runs
///   once, when the parameters load, stays apart from what runs at each call.
/// - The group yields the result of the chain's last block, and stands where that block stood.
#[pass(
    name = "form-groups",
    stats(grouped = "blocks grouped with the block before them")
)]
pub struct FormGroups;

fn populate_form_groups_patterns(set: &mut PatternSet) {
    set.add(GroupChain);
}

impl Pass for FormGroups {
    fn rewrites(&self) -> Rewrites {
        Rewrites::once(populate_form_groups_patterns)
    }

    fn run(&self, module: &mut Rewriting) -> Result<bool> {
        Ok(Greedy::new(self.rewrites()).run(module)?.changed())
    }
}

/// A block of a function, outside a group, and the blocks after it that each take the one before
/// as an epilogue move into a `tile.group`, which stands where the last of them stood and yields
/// its result. The blocks go in the order of the function, so a block that reads the results of
/// two blocks joins the chain of the first one.
#[pattern]
struct GroupChain;

impl Pattern for GroupChain {
    type Root = Block;

    fn rewrite(&self, first: BlockOp, rewriter: &mut Rewriter) -> Result<(), NoMatch> {
        require!(
            in_function_outside_groups(first.module(), first.inst()),
            "the block is in a group, or outside a function"
        );
        let chain = chain_from(first);
        require!(
            chain.len() > 1,
            "no block takes the block's result as an epilogue"
        );
        let last = &chain[chain.len() - 1];
        let result = last.outs()[0];
        let users: Vec<Use> = result.uses().to_vec();
        rewriter.set_insertion_point(last);
        rewriter.set_origin(last);
        let given = rewriter.create_with_regions(Group, (), |_, builder, _| {
            let mut yielded = Vec::new();
            for block in &chain {
                yielded = builder.move_here(block);
            }
            builder.create(Yield, yielded);
        });
        for user in users {
            rewriter.set_operand(user.inst, user.index, given[0]);
        }
        FormGroups::GROUPED.add(chain.len() - 1);
        Ok(())
    }
}

/// `first` and each block after it that takes the one before it as an epilogue: the one use of
/// its result, in the same region, known before the run when the block before it is.
fn chain_from(first: BlockOp<'_>) -> Vec<BlockOp<'_>> {
    let module = first.module();
    let mut chain = vec![first];
    loop {
        let last = &chain[chain.len() - 1];
        let &[result] = last.outs() else {
            break;
        };
        let &[Use { inst: user, index }] = result.uses() else {
            break;
        };
        let Some(next) = module.op_as::<Block>(user) else {
            break;
        };
        let same_region = module.parent_region(user) == module.parent_region(last.inst());
        let same_phase = result.known_before_run() == next.outs()[0].known_before_run();
        if !same_region || !same_phase || !last.takes_epilogue(&next, index) {
            break;
        }
        chain.push(next);
    }
    chain
}
