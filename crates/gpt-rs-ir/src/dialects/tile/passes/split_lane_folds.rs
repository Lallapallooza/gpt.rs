//! Splitting a fold into the folds of the lanes of vectors.

use std::collections::HashMap;

use anyhow::Result;

use crate::dialects::tile::verify::check_schedule;
use crate::dialects::tile::{
    copy_body_ops, Block, BlockOp, CombineKind, Dispatch, DispatchOp, Fold, Index, Iter, IterKind,
    Iters, Mode, ModeKind, PadSkip, Read, Source, Write,
};
use crate::ir::{AccessExpr, DType, Dim, IndexExpr, IterRef, Value, F64};
use crate::passes::{pass, run_patterns, Pass, Registered};
use crate::require;
use crate::rewrite::{pattern, Build, NoMatch, Pattern, PatternSet, Rewriter, Rewrites, Rewriting};
use crate::targets::{required, Target, TargetDesc};
use crate::traits::{Associative, Commutative};

/// Splits the first block of a dispatch into the folds of the lanes of vectors and the block that
/// folds them. It applies where vectors along the block's result cannot compute it
/// ([`Dispatch::vector_lanes`]) and the step of its fold is commutative and associative. The step
/// is associative exactly, as `maximum` and `minimum` are, or on floats where the dispatch's
/// numerics allow reassociation (`reassoc`), as `add` is.
///
/// - The block has no modes and folds one new result by `sum`, `max` or `min`, with its parallel
///   iterators first. Its last iterator `k` is a `red` one of a static extent, along which its
///   body computes in lanes ([`Block::body_lanes`]). The verifier accepts the lane folds computing
///   in their lanes.
/// - Where the lanes do not fill the extent, each read that takes `k` takes it as a whole
///   coordinate on an axis of that many elements. Each of those reads without a pad pads past the
///   axis with the identity of the fold's step ([`CombineKind::identity`]), -0 for a float sum.
///   At the pad of one of those reads, the body yields what changes nothing in the fold
///   ([`Block::pad_skip`]). It does so exactly, or as products of the pad 0 where the dispatch's
///   numerics skip them (`skip_pad`).
/// - The lane folds replace `k` by `j = k / lanes` in its place and by `l = k mod lanes`, a
///   parallel iterator after the others. Their vectors run along `l`. Their reads and `tile.index`
///   values take `lanes * j + l` for `k`, and they give the folds with the lanes as a last axis.
/// - The folding block folds the lanes of each element in halves, the upper half onto the lower,
///   until one is left. The middle lane of an odd count stays for the next step, and the IR holds
///   the order of the steps.
#[pass(
    name = "split-lane-folds",
    stats(split = "dispatches whose first fold folds its points in the lanes of vectors")
)]
#[derive(Default)]
pub struct SplitLaneFolds {
    /// The target whose vectors the lane folds compute in.
    pub target: Option<Target>,
}

fn populate_split_lane_folds_patterns(set: &mut PatternSet, target: Option<Target>) {
    set.add(SplitIntoLanes { target });
}

impl Pass for SplitLaneFolds {
    fn rewrites(&self) -> Rewrites {
        let mut patterns = PatternSet::new();
        populate_split_lane_folds_patterns(&mut patterns, self.target.clone());
        Rewrites {
            patterns,
            ..Rewrites::default()
        }
    }

    fn run(&self, module: &mut Rewriting) -> Result<bool> {
        required(self.target.as_ref(), self.name())?;
        run_patterns(module, self.rewrites(), &Self::SPLIT)
    }
}

/// A dispatch whose first block cannot compute on vectors along its result, and whose numerics
/// let that block's fold regroup: the fold splits into the folds of its lanes and the block that
/// folds them.
#[pattern]
struct SplitIntoLanes {
    target: Option<Target>,
}

impl Pattern for SplitIntoLanes {
    type Root = Dispatch;

    fn rewrite(&self, dispatch: DispatchOp, rewriter: &mut Rewriter) -> Result<(), NoMatch> {
        require!(let Some(target) = &self.target, "the pass has no target");
        let module = dispatch.module();
        let blocks = Dispatch::blocks(module, dispatch.body().id());
        require!(
            Dispatch::vector_lanes(module, &blocks, target).is_none(),
            "vectors along the result compute the first block"
        );
        require!(
            let Some(first) = dispatch.body().insts().find_map(|inst| module.op_as::<Block>(inst)),
            "the dispatch holds no block"
        );
        require!(
            regroups(&first, dispatch.numerics.reassoc),
            "the first block's fold is no sum, max or min whose steps the numerics let regroup"
        );
        let skip_zero_products = dispatch.numerics.skip_pad;
        require!(
            let Some(folds) = lane_folds(&first, (target, skip_zero_products)),
            "the first block is no fold whose last iterator lanes can split"
        );
        split_lanes(&first, folds, rewriter);
        Ok(())
    }
}

/// Whether the steps of the one fold of `block` may run in another order and grouping: a sum, max
/// or min whose step is commutative and associative on the state's dtype, or associative on
/// integers alone where `reassoc` lets floats regroup.
fn regroups(block: &BlockOp, reassoc: bool) -> bool {
    let [fold] = &block.folds[..] else {
        return false;
    };
    if !matches!(
        fold.combine,
        CombineKind::Sum | CombineKind::Max | CombineKind::Min
    ) {
        return false;
    }
    let Some(step) = fold.combine.rule().step() else {
        return false;
    };
    let Some(dtype) = block
        .outs()
        .first()
        .and_then(|result| result.value_type().dtype())
    else {
        return false;
    };
    let associative = step.get_trait::<Associative>();
    let exact = associative.is_some_and(|law| law.covers(dtype));
    let regrouped = associative.is_some() && reassoc;
    step.get_trait::<Commutative>().is_some() && (exact || regrouped)
}

/// The value a read pads with past its axis in the lane folds of the fold `combine` on `dtype`:
/// the identity of its step on `dtype` ([`CombineKind::identity`]), -0 for a float sum. `None`
/// when a pad, an f64, does not hold it exactly, as for a `min` on si64, whose identity is the
/// largest si64.
fn lane_pad(combine: &CombineKind, dtype: DType) -> Option<F64> {
    combine.identity(dtype)?.exact_f64().map(F64)
}

/// The folds of the lanes of a block: the reduced iterator they split, their lanes, and the block
/// of the lane folds.
struct LaneFolds {
    reduced: usize,
    lanes: usize,
    split: Block,
}

/// The lane folds of `block` on `target`, when it meets the conditions of [`SplitLaneFolds`].
fn lane_folds(
    block: &BlockOp,
    (target, skip_zero_products): (&TargetDesc, bool),
) -> Option<LaneFolds> {
    let [fold] = &block.folds[..] else {
        return None;
    };
    let [write] = &fold.writes[..] else {
        return None;
    };
    if !block.loops.is_empty() || write.into.is_some() {
        return None;
    }
    let parallel = block
        .iters
        .iter()
        .take_while(|iter| iter.kind == IterKind::Par)
        .count();
    if block.iters[parallel..]
        .iter()
        .any(|iter| iter.kind == IterKind::Par)
    {
        return None;
    }
    let reduced = block
        .iters
        .len()
        .checked_sub(1)
        .filter(|&last| last >= parallel && block.iters[last].kind == IterKind::Red)?;
    let extent = block.iters[reduced].extent.as_static()?;
    let module = block.module();
    let lanes = Block::body_lanes(module, block.inst(), reduced, target)?;
    let mut split = lane_fold_block(block, reduced, lanes);
    if !extent.is_multiple_of(lanes) {
        if !leaves_operands_past(block, (reduced, extent)) {
            return None;
        }
        let state = block.outs().first()?.value_type().dtype()?;
        let pad = lane_pad(&fold.combine, state)?;
        for read in split.reads.iter_mut() {
            if read.uses_iter(reduced) && read.pad.is_none() {
                read.pad = Some(pad);
            }
        }
        let body = block.body().id();
        let mut adds_nothing = false;
        for (index, read) in split.reads.iter().enumerate() {
            if !read.uses_iter(reduced) {
                continue;
            }
            adds_nothing |= match split.pad_skip(module, body, index) {
                Some(PadSkip::Exact) => true,
                Some(PadSkip::ZeroProducts) => skip_zero_products,
                None => false,
            };
        }
        if !adds_nothing {
            return None;
        }
    }
    check_schedule(&split).ok()?;
    Some(LaneFolds {
        reduced,
        lanes,
        split,
    })
}

/// Whether every read of `block` that takes its iterator at `reduced`, of `extent` points, takes
/// it as a whole coordinate on an axis of its operand of `extent` elements, so that a point past
/// the extent reads past the axis, where a pad stands in for the operand.
fn leaves_operands_past(block: &BlockOp, (reduced, extent): (usize, usize)) -> bool {
    let operands = block.ins();
    for read in &block.reads {
        let Source::Operand(operand) = read.source else {
            return false;
        };
        let Some(dims) = operands[operand].tensor_type().static_dims() else {
            return false;
        };
        for (axis, coord) in read.coords.iter().enumerate() {
            let whole = coord.as_iter() == Some(reduced) && dims[axis] == extent;
            if coord.uses_iter(reduced) && !whole {
                return false;
            }
        }
    }
    true
}

/// The point of the reduced iterator at `reduced` of `block` that the lane folds of `lanes` lanes
/// reach: `lanes * j + l`, of `j` in its place and the lane `l` after the iterators of `block`.
fn walked(block: &Block, reduced: usize, lanes: usize) -> IndexExpr {
    let lane = block.iters.len();
    IndexExpr::iter(reduced) * IndexExpr::constant(lanes as i64) + IndexExpr::iter(lane)
}

/// The block of the lane folds of `block` along its iterator at `reduced` ([`SplitLaneFolds`]).
fn lane_fold_block(block: &Block, reduced: usize, lanes: usize) -> Block {
    let lane = block.iters.len();
    let walked_access = AccessExpr::from(walked(block, reduced, lanes));
    let mut split = block.map_exprs(&|expr| {
        expr.substitute(&|iter| (iter == reduced).then(|| walked_access.clone()))
    });
    let extent = block.iters[reduced]
        .extent
        .as_static()
        .expect("lane folds have a static reduced extent");
    split.iters[reduced].extent = Dim::Static(extent.div_ceil(lanes));
    split.iters.push(Iter::par(Dim::Static(lanes)));
    split.folds[0].writes[0].coords.push(AccessExpr::iter(lane));
    split.loops = split.with_point_modes(vec![Mode::new(
        IterRef::new(lane),
        lanes,
        Some(ModeKind::Lanes),
    )]);
    split
}

/// Replaces `block` by its lane folds ([`lane_fold_block`]) and the block that folds their lanes
/// ([`fold_lanes`]).
fn split_lanes(block: &BlockOp, folds: LaneFolds, rewriter: &mut Rewriter) {
    let LaneFolds {
        reduced,
        lanes,
        split,
    } = folds;
    let module = block.module();
    let reduced_point = walked(block, reduced, lanes);
    rewriter.set_insertion_point(block);
    let body = block.body();
    let operands: Vec<Value> = block.ins().iter().map(|input| input.id()).collect();
    let partial = split.build(rewriter, operands, |body_builder, args| {
        let mut values = HashMap::new();
        for (old, &new) in body.args().into_iter().zip(args) {
            values.insert(old.id(), new);
        }
        for body_inst in body.insts() {
            let Some(index) = module.op_as::<Index>(body_inst) else {
                continue;
            };
            let expr = index
                .expr
                .substitute(&|iter| (iter == reduced).then(|| reduced_point.clone()));
            let dtype = index.dtype;
            let value = body_builder.create(Index { expr, dtype }, ());
            values.insert(index.output().id(), value);
        }
        copy_body_ops(module, body.id(), body_builder.builder, &mut values)
    });
    let folded = fold_lanes(block, lanes, partial, rewriter);
    rewriter.replace_op(block, folded);
}

/// The block that folds the `lanes` lanes of each element by the step of the fold of `block`:
/// over the parallel iterators of `block`, which lead its list, reading the lane folds `partial`
/// where `block` writes, at each lane.
fn fold_lanes(
    block: &Block,
    lanes: usize,
    partial: Vec<Value>,
    rewriter: &mut Rewriter,
) -> Vec<Value> {
    let write = &block.folds[0].writes[0];
    let mut iters = Iters::new();
    for iter in &block.iters {
        if iter.kind != IterKind::Par {
            break;
        }
        iters.push(iter.clone());
    }
    let mut reads = Vec::with_capacity(lanes);
    for at in 0..lanes {
        let lane_coord = AccessExpr::constant(at as i64);
        reads.push(Read::new(
            0,
            [write.coords.to_vec(), vec![lane_coord]].concat(),
        ));
    }
    let folding = Block::new(
        iters,
        reads,
        vec![Fold::plain(Write::new(write.coords.to_vec()))],
    );
    let step = block.folds[0]
        .combine
        .rule()
        .step()
        .expect("a fold that regroups steps by an op");
    folding.build(rewriter, partial, |body, lane_folds| {
        let mut halves = lane_folds.to_vec();
        while halves.len() > 1 {
            // Of an odd count, the middle lane stays in the lower half, for the next step.
            let upper_half = halves.split_off(halves.len().div_ceil(2));
            for (lower, upper) in halves.iter_mut().zip(upper_half) {
                let (folded, _) =
                    body.builder
                        .create_or_fold_op(step.clone(), vec![*lower, upper], None);
                *lower = folded[0];
            }
        }
        halves
    })
}
