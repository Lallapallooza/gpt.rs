//! The canonical patterns of `tile.block`: the normal form of one block, from its own attributes,
//! its body and the constant values of its operands, and index math in the index language. They
//! never make a block read another tensor or compute another block's values.

use std::collections::BTreeSet;
use std::marker::PhantomData;

use crate::dialects::arith::{self, Literal};
use crate::dialects::tile::{Block, Index, IterKind, IterList, Source};
use crate::host::HostScalar;
use crate::ir::{
    AccessExpr, BinaryOp, Dim, InOp, IndexExpr, Inst, IntFormat, Matched, Module, RegionRef, Value,
    ValueRef,
};
use crate::require;
use crate::rewrite::{pattern, NoMatch, Pattern, Rewriter};

/// Rewrites the expression of every `tile.index` in `body` with `rewrite_expr`, and returns
/// whether any changed.
fn map_body_indices(
    rewriter: &mut Rewriter<'_>,
    body: RegionRef<'_>,
    rewrite_expr: &dyn Fn(&IndexExpr) -> IndexExpr,
) -> bool {
    let mut changed = false;
    for inst in body.insts() {
        let Some(index) = body.module().op_as::<Index>(inst) else {
            continue;
        };
        let expr = rewrite_expr(&index.expr);
        if expr != index.expr {
            let dtype = index.dtype;
            rewriter.set_op(inst, Index { expr, dtype });
            changed = true;
        }
    }
    changed
}

/// Stages giving the block `block` the attributes `attrs` over the operands `ins`, compacted, and
/// returns whether that changes the block.
fn update(
    rewriter: &mut Rewriter<'_>,
    block: &Matched<'_, Block>,
    attrs: Block,
    ins: Vec<Value>,
) -> bool {
    let (attrs, ins) = attrs.compact(ins);
    let current: Vec<Value> = block.ins().iter().map(|input| input.id()).collect();
    if attrs == *block.attrs() && ins == current {
        return false;
    }
    rewriter.set_op(block, attrs);
    rewriter.set_operands(block, ins);
    true
}

/// A read whose value the body does not use is removed, with its body argument. Operands that
/// hold the same value merge into the first, and operands the block no longer names are removed.
#[pattern(canonical)]
struct CompactInputs;

impl Pattern for CompactInputs {
    type Root = Block;

    fn rewrite(
        &self,
        block: Matched<'_, Block>,
        rewriter: &mut Rewriter<'_>,
    ) -> Result<(), NoMatch> {
        let args = block.body().args();
        let unused: Vec<usize> = (0..args.len())
            .filter(|&index| args[index].uses().is_empty())
            .collect();
        let mut attrs = block.attrs().clone();
        for &index in unused.iter().rev() {
            attrs.reads.remove(index);
            rewriter.remove_arg(block.body(), index);
        }
        let ins = block.ins().iter().map(|input| input.id()).collect();
        let updated = update(rewriter, &block, attrs, ins);
        require!(
            !unused.is_empty() || updated,
            "every read is used and every operand distinct"
        );
        Ok(())
    }
}

/// A read without `pad` of a constant whose elements are all equal is that element, an
/// `arith.constant` in the body.
#[pattern(canonical)]
struct SplatReads;

impl Pattern for SplatReads {
    type Root = Block;

    fn rewrite(
        &self,
        block: Matched<'_, Block>,
        rewriter: &mut Rewriter<'_>,
    ) -> Result<(), NoMatch> {
        let mut splats: Vec<(usize, HostScalar)> = Vec::new();
        for (index, read) in block.reads.iter().enumerate() {
            let (Source::Operand(operand), None) = (&read.source, read.pad) else {
                continue;
            };
            if let Some(element) = block.ins()[*operand].splat() {
                splats.push((index, element));
            }
        }
        require!(!splats.is_empty(), "no read is of a splat constant");
        let body = block.body();
        let first = body.insts().next().expect("a body ends with tile.yield");
        rewriter.set_insertion_point(first);
        let args = body.args();
        let mut attrs = block.attrs().clone();
        for &(index, element) in splats.iter().rev() {
            let value = Literal::new(element).expect("a scalar is a literal");
            let constant = rewriter.create(arith::Constant { value }, ());
            rewriter.replace_all_uses(args[index], constant);
            rewriter.remove_arg(body, index);
            attrs.reads.remove(index);
        }
        rewriter.set_op(&block, attrs);
        Ok(())
    }
}

/// The pattern removes an iterator of extent 1, which is always 0. It writes 0 for it in every
/// expression and write, and moves the later iterators down one position.
#[pattern(canonical)]
struct DropUnitIters;

impl Pattern for DropUnitIters {
    type Root = Block;

    fn rewrite(
        &self,
        block: Matched<'_, Block>,
        rewriter: &mut Rewriter<'_>,
    ) -> Result<(), NoMatch> {
        let mut lanes = BTreeSet::new();
        for fold in &block.folds {
            lanes.extend(fold.combine.rule().lanes().positions());
        }
        let mut unit = BTreeSet::new();
        for (position, iter) in block.iters.iter().enumerate() {
            if iter.extent.is_one() && iter.kind != IterKind::Prefix && !lanes.contains(&position) {
                unit.insert(position);
            }
        }
        require!(!unit.is_empty(), "no iterator has extent 1");
        // What each iterator becomes: 0 for one that is removed, its new position for one that
        // stays.
        let mut kept_position = Vec::with_capacity(block.iters.len());
        let mut renumbered = Vec::with_capacity(block.iters.len());
        for position in 0..block.iters.len() {
            if unit.contains(&position) {
                kept_position.push(None);
                renumbered.push(IndexExpr::constant(0));
            } else {
                let kept = position - unit.range(..position).count();
                kept_position.push(Some(kept));
                renumbered.push(IndexExpr::iter(kept));
            }
        }
        let subst = |position: usize| Some(renumbered[position].clone());
        let mut attrs = block.map_exprs(&|expr: &AccessExpr| expr.substitute(&subst).simplify());
        let mut position = 0;
        attrs.iters.retain(|_| {
            let keep = !unit.contains(&position);
            position += 1;
            keep
        });
        for fold in &mut attrs.folds {
            if let Some(lanes) = fold.combine.lanes_mut() {
                *lanes = IterList::new(lanes.positions().filter_map(|lane| kept_position[lane]));
            }
        }
        map_body_indices(rewriter, block.body(), &|expr| {
            expr.substitute(&subst).simplify()
        });
        rewriter.set_op(&block, attrs);
        Ok(())
    }
}

/// Every index expression is in canonical form: sums with merged coefficients, `floordiv` and
/// `mod` reduced where the iterators' extents allow, and `min` and `max` decided where the
/// extents decide them.
#[pattern(canonical)]
struct CanonicalIndices;

impl Pattern for CanonicalIndices {
    type Root = Block;

    fn rewrite(
        &self,
        block: Matched<'_, Block>,
        rewriter: &mut Rewriter<'_>,
    ) -> Result<(), NoMatch> {
        let attrs = block.attrs();
        let extent = |position: usize| attrs.extent(position);
        let canonical_attrs = attrs.map_exprs(&|expr| expr.canonical(&extent));
        let body_changed =
            map_body_indices(rewriter, block.body(), &|expr| expr.canonical(&extent));
        let attrs_changed = canonical_attrs != *attrs;
        require!(
            attrs_changed || body_changed,
            "every index expression is canonical"
        );
        if attrs_changed {
            rewriter.set_op(&block, canonical_attrs);
        }
        Ok(())
    }
}

/// A block that reads its operand at its own iterators, in the order it writes them, into a
/// result of the operand's type, and yields what it reads, is that operand.
#[pattern(canonical)]
struct EraseIdentityBlock;

impl Pattern for EraseIdentityBlock {
    type Root = Block;

    fn rewrite(
        &self,
        block: Matched<'_, Block>,
        rewriter: &mut Rewriter<'_>,
    ) -> Result<(), NoMatch> {
        let body = block.body();
        require!(block.is_map(), "the block folds, filters or updates");
        require!(body.insts().count() == 1, "the body computes");
        require!(
            body.yielded() == body.args(),
            "the body yields something other than its reads"
        );
        let read = &block.reads[0];
        require!(
            let (Source::Operand(operand), None) = (&read.source, read.pad),
            "the read is padded or picks an operand"
        );
        let input = block.ins()[*operand];
        require!(
            read.coords == block.folds[0].writes[0].coords,
            "the read and the write differ"
        );
        let by_iterators = read.coords.iter().all(|coord| coord.as_iter().is_some());
        require!(by_iterators, "a coordinate is not an iterator");
        require!(
            input.value_type() == block.outs()[0].value_type(),
            "the result has another type"
        );
        rewriter.replace_op(&block, input);
        Ok(())
    }
}

/// An integer `arith` op as an operation of the index language.
trait IndexArith: InOp {
    /// The expression of the op on the expressions of its operands, in order.
    fn apply(operands: Vec<IndexExpr>) -> IndexExpr;
}

/// `op` on the two expressions `operands`.
fn binary(op: BinaryOp, operands: Vec<IndexExpr>) -> IndexExpr {
    let [lhs, rhs]: [IndexExpr; 2] = operands.try_into().expect("two operands");
    IndexExpr::binary(op, lhs, rhs)
}

impl IndexArith for arith::Add {
    fn apply(operands: Vec<IndexExpr>) -> IndexExpr {
        binary(BinaryOp::Add, operands)
    }
}

impl IndexArith for arith::Sub {
    fn apply(operands: Vec<IndexExpr>) -> IndexExpr {
        binary(BinaryOp::Sub, operands)
    }
}

impl IndexArith for arith::Mul {
    fn apply(operands: Vec<IndexExpr>) -> IndexExpr {
        binary(BinaryOp::Mul, operands)
    }
}

impl IndexArith for arith::Minimum {
    fn apply(operands: Vec<IndexExpr>) -> IndexExpr {
        binary(BinaryOp::Min, operands)
    }
}

impl IndexArith for arith::Maximum {
    fn apply(operands: Vec<IndexExpr>) -> IndexExpr {
        binary(BinaryOp::Max, operands)
    }
}

impl IndexArith for arith::Neg {
    fn apply(mut operands: Vec<IndexExpr>) -> IndexExpr {
        -operands.pop().expect("one operand")
    }
}

/// The expression of the integer scalar `value`: the expression of its `tile.index`, or its
/// constant.
fn index_value(value: ValueRef<'_>) -> Option<IndexExpr> {
    if let Some(index) = value.defining_op::<Index>() {
        return Some(index.expr.clone());
    }
    let constant = value.splat()?.as_int()?;
    Some(IndexExpr::constant(i64::try_from(constant).ok()?))
}

/// Whether every value of each of `exprs` is in the range of `format` at every point of the block
/// that holds `inst`: then `arith` neither wraps nor a `tile.index` saturates. An si64 index wraps
/// as `arith` does and never saturates.
fn in_range(module: &Module, inst: Inst, format: IntFormat, exprs: &[&IndexExpr]) -> bool {
    if format.signed && format.bits == 64 {
        return true;
    }
    let block = module
        .parent_region(inst)
        .and_then(|body| module.parent_op(body))
        .and_then(|block| module.op_as::<Block>(block));
    let Some(block) = block else {
        return false;
    };
    let iter_range = |position: usize| match block.iters.get(position)?.extent {
        Dim::Static(extent) if extent > 0 => Some((0, extent as i64 - 1)),
        _ => None,
    };
    exprs.iter().all(|expr| {
        expr.bounds(&iter_range).is_some_and(|(low, high)| {
            i128::from(low) >= format.min() && i128::from(high) <= format.max()
        })
    })
}

/// Integer `arith` on `tile.index` values and constants is one `tile.index` of the combined
/// expression, so that index math is in the index language. It folds where no value leaves the
/// range of the dtype: always for si64, and for another dtype where the block's iterator extents
/// bound the values.
#[pattern(canonical(
    arith::Add,
    arith::Sub,
    arith::Mul,
    arith::Minimum,
    arith::Maximum,
    arith::Neg
))]
struct FoldIndexArith<T>(PhantomData<T>);

impl<T: IndexArith> Pattern for FoldIndexArith<T> {
    type Root = T;

    fn rewrite(&self, op: Matched<'_, T>, rewriter: &mut Rewriter<'_>) -> Result<(), NoMatch> {
        let module = op.module();
        let mut operands = Vec::new();
        let mut reads_index = false;
        for position in 0..module.operands(op.inst()).len() {
            let value = op.operand(position);
            reads_index |= value.defining_op::<Index>().is_some();
            require!(
                let Some(expr) = index_value(value),
                "an operand is neither a tile.index nor an integer constant"
            );
            operands.push(expr);
        }
        require!(reads_index, "no operand is a tile.index");
        let dtype = op
            .result(0)
            .value_type()
            .as_scalar()
            .expect("arith gives a scalar");
        require!(let Some(format) = dtype.int_format(), "the op computes on floats");
        let expr = T::apply(operands.clone()).simplify();
        let mut exprs: Vec<&IndexExpr> = operands.iter().collect();
        exprs.push(&expr);
        require!(
            in_range(module, op.inst(), format, &exprs),
            "a value may leave the range of the dtype"
        );
        let index = rewriter.create(Index { expr, dtype }, ());
        rewriter.replace_op(&op, index);
        Ok(())
    }
}
