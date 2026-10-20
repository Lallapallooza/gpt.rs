//! Padding a read with the identity of the fold it feeds.

use anyhow::Result;

use crate::dialects::arith;
use crate::dialects::tile::{Block, BlockOp, CombineKind, Source};
use crate::ir::{AccessExpr, Constraint, DType, IndexExpr, Relation, ValueRef, F64};
use crate::passes::{pass, run_patterns, Pass};
use crate::require;
use crate::rewrite::{pattern, NoMatch, Pattern, PatternSet, Rewriter, Rewrites, Rewriting};

/// Pads the read of a block with the identity of the fold it feeds, in place of the constraints
/// that keep the read inside its operand: the points those constraints skipped read the identity,
/// which leaves the fold as it is. `tensor-to-tile` writes a padded window as such constraints,
/// which hold for every dtype; a block whose read is padded can hold a register tile, which a
/// block with constraints cannot. The points it adds compute the body at the border of the
/// operand, so it is a pass rather than a canonical form.
///
/// - The block reads one operand, of static extents, in one read without a pad, and folds what
///   its body yields, the read or casts of it, in one fold into new results whose combine has a
///   right identity.
/// - The read's dtype and the dtype of each cast hold the identity exactly, as the same f64, so
///   that each cast carries it bit for bit: a cast through a dtype without it, as `f32` through
///   `si8` under `max`, would saturate it.
/// - Each constraint it drops is `0 <= c` or `c <= n - 1` for the coordinate `c` of the read on an
///   axis of `n` elements; the other constraints stay.
#[pass(
    name = "pad-with-identity",
    stats(padded = "reads padded with the identity of their fold")
)]
pub struct PadWithIdentity;

fn populate_pad_with_identity_patterns(set: &mut PatternSet) {
    set.add(BoundsToPad);
}

impl Pass for PadWithIdentity {
    fn rewrites(&self) -> Rewrites {
        Rewrites::patterns(populate_pad_with_identity_patterns)
    }

    fn run(&self, module: &mut Rewriting) -> Result<bool> {
        run_patterns(module, self.rewrites(), &Self::PADDED)
    }
}

/// `0 <= c` and `c <= n - 1` for the coordinate `c` of a block's one read on an axis of `n`
/// elements are a pad of the identity of the block's one fold, which takes the read.
#[pattern]
struct BoundsToPad;

impl Pattern for BoundsToPad {
    type Root = Block;

    fn rewrite(&self, block: BlockOp, rewriter: &mut Rewriter) -> Result<(), NoMatch> {
        require!(
            !block.constraints.is_empty(),
            "the block has no constraints"
        );
        require!(let [read] = block.reads.as_slice(), "the block has more than one read");
        require!(read.pad.is_none(), "the read has a pad");
        require!(let Source::Operand(operand) = read.source, "the read picks from a list");
        require!(let [fold] = block.folds.as_slice(), "the block has more than one fold");
        require!(
            fold.writes.iter().all(|write| write.into.is_none()),
            "the fold writes into an operand"
        );
        let body = block.body();
        require!(let [yielded] = body.yielded()[..], "the body yields more than one value");
        let (source, cast_dtypes) = uncast(yielded);
        require!(
            source == body.args()[0],
            "the body yields other than the read or casts of it"
        );
        require!(
            let Some(value) = exact_identity(&fold.combine, scalar_dtype(source)),
            "the fold's combine has no right identity that the read's dtype, and a pad, an f64, \
             hold exactly"
        );
        for dtype in cast_dtypes {
            let carried = exact_identity(&fold.combine, dtype);
            require!(
                carried.map(f64::to_bits) == Some(value.to_bits()),
                "a cast to {dtype} does not carry the identity to the identity there"
            );
        }
        let input = block.ins()[operand];
        require!(
            let Some(extents) = input.tensor_type().static_dims(),
            "the operand has extents known only during the run"
        );
        let mut kept = Vec::with_capacity(block.constraints.len());
        for constraint in &block.constraints {
            if !bounds_read(constraint, &read.coords, &extents) {
                kept.push(constraint.clone());
            }
        }
        require!(
            kept.len() < block.constraints.len(),
            "no constraint keeps the read inside its operand"
        );
        block.update(rewriter, |padded| {
            padded.reads[0].pad = Some(F64(value));
            padded.constraints = kept.into();
        });
        Ok(())
    }
}

/// The value that `value` is a cast of, through each `arith.cast` that gives it, and the dtype of
/// each cast on the way, the one of `value` first.
fn uncast(value: ValueRef<'_>) -> (ValueRef<'_>, Vec<DType>) {
    let mut source = value;
    let mut cast_dtypes = Vec::new();
    while let Some(cast) = source.defining_op::<arith::Cast>() {
        cast_dtypes.push(scalar_dtype(source));
        source = cast.input();
    }
    (source, cast_dtypes)
}

/// The right identity of `combine` on `dtype` as an f64, when `dtype` and an f64 hold it exactly.
fn exact_identity(combine: &CombineKind, dtype: DType) -> Option<f64> {
    combine.identity(dtype)?.exact_f64()
}

/// The dtype of the scalar `value`.
fn scalar_dtype(value: ValueRef<'_>) -> DType {
    value
        .value_type()
        .as_scalar()
        .expect("a block's body computes scalars")
}

/// Whether `constraint` is `0 <= c` or `c <= n - 1` for the coordinate `c` of a read at `coords`
/// on an axis of `n` of `extents` elements.
fn bounds_read(constraint: &Constraint, coords: &[AccessExpr], extents: &[usize]) -> bool {
    let [cmp] = constraint.0.as_slice() else {
        return false;
    };
    if cmp.op != Relation::LessEqual {
        return false;
    }
    let zero = AccessExpr::from(IndexExpr::constant(0));
    for (coord, &extent) in coords.iter().zip(extents) {
        let last = AccessExpr::from(IndexExpr::constant(extent as i64 - 1));
        let low = cmp.lhs == zero && cmp.rhs == *coord;
        let high = cmp.lhs == *coord && cmp.rhs == last;
        if low || high {
            return true;
        }
    }
    false
}
