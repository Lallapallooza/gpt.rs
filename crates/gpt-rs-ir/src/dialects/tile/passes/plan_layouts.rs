//! Choosing the layout of each value known before the run.

use anyhow::Result;

use crate::dialects::tile::{par_iters, Block, BlockOp, Coords, Fold, Read, Source, Write};
use crate::ir::IndexExpr;
use crate::passes::{pass, run_patterns, Pass};
use crate::require;
use crate::rewrite::{pattern, NoMatch, Pattern, PatternSet, Rewriter, Rewrites, Rewriting};

/// Lays out each value known before the run that one block computes and another alone reads, so
/// that the reader takes the lanes of its vectors from consecutive elements. When the read takes
/// the iterator that the reader's lanes run along alone, in a coordinate other than the last, the
/// value is read from a copy that moves that axis last and keeps the others in their order.
/// `canonicalize` folds the copy into the block that computes the value, which then writes it in
/// the copy's order. A value of the checkpoint keeps its layout: a copy of it would hold the
/// weights in memory twice.
#[pass(
    name = "plan-layouts",
    stats(laid_out = "values written with the axis of their reader's lanes last")
)]
pub struct PlanLayouts;

fn populate_plan_layouts_patterns(set: &mut PatternSet) {
    set.add(LanesLast);
}

impl Pass for PlanLayouts {
    fn rewrites(&self) -> Rewrites {
        Rewrites::patterns(populate_plan_layouts_patterns)
    }

    fn run(&self, module: &mut Rewriting) -> Result<bool> {
        run_patterns(module, self.rewrites(), &Self::LAID_OUT)
    }
}

/// A block that reads a value known before the run, which another block computes for it alone,
/// with its lanes iterator alone in a coordinate other than the last: it reads a copy of the value
/// with that axis last.
#[pattern]
struct LanesLast;

impl Pattern for LanesLast {
    type Root = Block;

    fn rewrite(&self, block: BlockOp, rewriter: &mut Rewriter) -> Result<(), NoMatch> {
        require!(
            !block.is_copy(),
            "a copy moves elements in the order it writes them"
        );
        require!(let Some(lane) = block.lanes_iterator(), "the block has no lanes iterator");
        let mut found = None;
        for (index, read) in block.reads.iter().enumerate() {
            let Source::Operand(operand) = read.source else {
                continue;
            };
            let value = block.ins()[operand];
            let produced = value
                .defining_op::<Block>()
                .is_some_and(|producer| !producer.is_copy());
            if !value.known_before_run() || !value.has_one_use() || !produced {
                continue;
            }
            if let Some(axis) = lanes_axis(read, lane) {
                found = Some((index, operand, axis));
                break;
            }
        }
        require!(
            let Some((index, operand, axis)) = found,
            "no read of a computed value known before the run takes the lanes in another \
             coordinate than the last"
        );
        let value = block.ins()[operand];
        let ty = value.tensor_type();
        let order = lanes_last_order(ty.rank(), axis);
        let mut dims = Vec::with_capacity(order.len());
        for &source_axis in &order {
            dims.push(ty.dims[source_axis].clone());
        }
        let (iters, coords) = par_iters(&dims);
        let mut copy_coords = vec![IndexExpr::constant(0); order.len()];
        for (position, &source_axis) in order.iter().enumerate() {
            copy_coords[source_axis] = coords[position].clone();
        }
        let copy = Block::new(
            iters,
            vec![Read::new(0, copy_coords)],
            vec![Fold::plain(Write::new(coords))],
        );
        let copied = copy.build(rewriter, vec![value.id()], |_, args| vec![args[0]]);
        let read = &block.reads[index];
        let mut reordered = Coords::new();
        for &source_axis in &order {
            reordered.push(read.coords[source_axis].clone());
        }
        rewriter.set_operand(&block, operand, copied[0]);
        block.update(rewriter, |attrs| attrs.reads[index].coords = reordered);
        Ok(())
    }
}

/// The axis at which `read` takes the iterator `lane` alone, when it is not its last coordinate,
/// no other coordinate uses `lane`, and the read has no pad.
fn lanes_axis(read: &Read, lane: usize) -> Option<usize> {
    if read.pad.is_some() {
        return None;
    }
    let axis = read
        .coords
        .iter()
        .position(|coord| coord.as_iter() == Some(lane))?;
    let others_use_lane = read
        .coords
        .iter()
        .enumerate()
        .any(|(other, coord)| other != axis && coord.uses_iter(lane));
    (axis + 1 < read.coords.len() && !others_use_lane).then_some(axis)
}

/// The axes of a value of `rank` axes in the order with `axis` last, the others in their order.
fn lanes_last_order(rank: usize, axis: usize) -> Vec<usize> {
    let mut order = Vec::with_capacity(rank);
    for other in 0..rank {
        if other != axis {
            order.push(other);
        }
    }
    order.push(axis);
    order
}
