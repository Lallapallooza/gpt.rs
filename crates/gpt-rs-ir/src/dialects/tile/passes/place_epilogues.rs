//! Placing the blocks after the first of a dispatch in its register tile.

use anyhow::Result;

use crate::dialects::tile::verify::check_placements;
use crate::dialects::tile::{Block, Dispatch, DispatchOp, Placement, Source};
use crate::passes::{pass, run_patterns, Pass};
use crate::require;
use crate::rewrite::{pattern, NoMatch, Pattern, PatternSet, Rewriter, Rewrites, Rewriting};

/// Places each block of a dispatch after the first at the register tile of the first block (`at
/// reg`), when the blocks so placed keep the verifier's placement rules ([`check_placements`]).
/// Each then runs where the block before it finishes its elements: on each element as it is
/// finished, such as an epilogue, or once the tile is finished, such as the block that adds the
/// lanes of the first block's sums. A dispatch whose blocks cannot all run there keeps them apart.
#[pass(
    name = "place-epilogues",
    stats(placed = "blocks placed in the loops of the block before them")
)]
pub struct PlaceEpilogues;

fn populate_place_epilogues_patterns(set: &mut PatternSet) {
    set.add(PlaceAtRegisterTile);
}

impl Pass for PlaceEpilogues {
    fn rewrites(&self) -> Rewrites {
        Rewrites::patterns(populate_place_epilogues_patterns)
    }

    fn run(&self, module: &mut Rewriting) -> Result<bool> {
        run_patterns(module, self.rewrites(), &Self::PLACED)
    }
}

/// The blocks after the first of a dispatch, none placed, each reading the result of the block
/// before it: each placed at the register tile of the first, where the verifier accepts them all
/// so placed.
#[pattern]
struct PlaceAtRegisterTile;

impl Pattern for PlaceAtRegisterTile {
    type Root = Dispatch;

    fn rewrite(&self, dispatch: DispatchOp, rewriter: &mut Rewriter) -> Result<(), NoMatch> {
        let module = dispatch.module();
        let blocks = Dispatch::blocks(module, dispatch.body().id());
        require!(blocks.len() > 1, "the dispatch holds one block");
        let mut placed: Vec<Block> = Vec::with_capacity(blocks.len() - 1);
        for pair in blocks.windows(2) {
            let [(before, _), (after, later)] = pair else {
                continue;
            };
            require!(
                later.reads.iter().all(|read| read.at.is_none()),
                "a block is placed"
            );
            require!(
                let Some(input) = Block::epilogue_input(module, *before, *after),
                "a block does not read the result of the block before it"
            );
            let placing = later
                .reads
                .iter()
                .position(
                    |read| matches!(read.source, Source::Operand(operand) if operand == input),
                )
                .expect("a block reads the result of the block before it");
            let mut proposed = (*later).clone();
            proposed.reads[placing].at = Some(Placement::Reg);
            placed.push(proposed);
        }
        let mut proposal = vec![blocks[0]];
        for (&(inst, _), block) in blocks[1..].iter().zip(&placed) {
            proposal.push((inst, block));
        }
        check_placements(module, &proposal)
            .map_err(|err| NoMatch::because(format_args!("{err:#}")))?;
        for (&(inst, _), block) in blocks[1..].iter().zip(placed) {
            let current = module.op_as::<Block>(inst).expect("a block");
            current.set_reads(rewriter, block.reads);
        }
        Ok(())
    }
}
