//! The canonical patterns of `tile.block`: the normal form of one block, from its own attributes,
//! its body and the constant values of its operands; index math in the index language; copy
//! blocks composed into the blocks that read them and that they read; and maps that write
//! straight into the operand that a copy of their result updates. They never make a block compute
//! another block's values.

use std::cmp::Ordering;
use std::collections::{BTreeSet, HashMap};
use std::marker::PhantomData;

use tinyvec::ArrayVec;

use crate::dialects::arith::{self, Literal};
use crate::dialects::tile::{
    copy_body_ops, Block, BlockOp, CombineKind, Coords, Index, Iter, IterKind, IterList, Read,
    Source, UnscheduledBlock, Write,
};
use crate::host::HostScalar;
use crate::ir::{
    AccessExpr, BinaryOp, Dim, Expr, InOp, IndexExpr, Inst, IntFormat, Module, OpRef, OperandRef,
    ReadKind, RegionRef, Value, ValueRef, MAX_ITERS,
};
use crate::require;
use crate::rewrite::{pattern, NoMatch, Pattern, Rewriter};

/// Rewrites the expression of every `tile.index` in `body` with `rewrite_expr`, and returns
/// whether any changed.
fn map_body_indices(
    rewriter: &mut Rewriter,
    body: RegionRef,
    rewrite_expr: &dyn Fn(&IndexExpr) -> IndexExpr,
) -> bool {
    let mut changed = false;
    for inst in body.insts() {
        let Some(index) = body.module().op_as::<Index>(inst) else {
            continue;
        };
        let expr = rewrite_expr(&index.expr);
        if expr != index.expr {
            index.set_expr(rewriter, expr);
            changed = true;
        }
    }
    changed
}

/// Stages giving the block `block` the attributes `attrs` over the operands `ins`, compacted, and
/// returns whether that changes the block.
fn update(rewriter: &mut Rewriter, block: &BlockOp, attrs: Block, ins: Vec<Value>) -> bool {
    let (compact_attrs, compact_ins) = attrs.compact(ins);
    let current: Vec<Value> = block.ins().iter().map(|input| input.id()).collect();
    if compact_attrs == *block.attrs() && compact_ins == current {
        return false;
    }
    block.update(rewriter, |op| *op = compact_attrs);
    rewriter.set_operands(block, compact_ins);
    true
}

/// A read whose value the body does not use is removed, with its body argument. Operands that
/// hold the same value merge into the first, and operands the block no longer names are removed.
#[pattern(canonical)]
struct CompactInputs;

impl Pattern for CompactInputs {
    type Root = UnscheduledBlock;

    fn rewrite(&self, block: BlockOp, rewriter: &mut Rewriter) -> Result<(), NoMatch> {
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

/// Two reads of the same operand at the same coordinates, with the same pad, are one: the body
/// takes the value of the first for both.
#[pattern(canonical)]
struct MergeEqualReads;

impl Pattern for MergeEqualReads {
    type Root = Block;

    fn rewrite(&self, block: BlockOp, rewriter: &mut Rewriter) -> Result<(), NoMatch> {
        let mut repeated = Vec::new();
        for (index, read) in block.reads.iter().enumerate() {
            if let Some(first) = block.reads[..index]
                .iter()
                .position(|earlier| earlier == read)
            {
                repeated.push((index, first));
            }
        }
        require!(!repeated.is_empty(), "no two reads are equal");
        let body = block.body();
        let args = body.args();
        let mut attrs = block.attrs().clone();
        for &(index, first) in repeated.iter().rev() {
            rewriter.replace_all_uses(args[index], args[first]);
            rewriter.remove_arg(body, index);
            attrs.reads.remove(index);
        }
        block.update(rewriter, |op| *op = attrs);
        Ok(())
    }
}

/// The element at every point of `value`: of a constant whose elements are all equal, or of a
/// block that reads nothing and yields a constant.
fn splat_value(value: ValueRef) -> Option<HostScalar> {
    if let Some(element) = value.splat() {
        return Some(element);
    }
    let block = value.defining_op::<Block>()?;
    if !block.is_map() || !block.reads.is_empty() {
        return None;
    }
    let [yielded] = block.body().yielded()[..] else {
        return None;
    };
    yielded.splat()
}

/// A read without `pad` of a value whose elements are all equal, a constant or a block that fills
/// one, is that element, an `arith.constant` in the body.
#[pattern(canonical)]
struct SplatReads;

impl Pattern for SplatReads {
    type Root = UnscheduledBlock;

    fn rewrite(&self, block: BlockOp, rewriter: &mut Rewriter) -> Result<(), NoMatch> {
        let mut splats: Vec<(usize, HostScalar)> = Vec::new();
        for (index, read) in block.reads.iter().enumerate() {
            let (Source::Operand(operand), None) = (&read.source, read.pad) else {
                continue;
            };
            if let Some(element) = splat_value(block.ins()[*operand]) {
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
        block.update(rewriter, |op| *op = attrs);
        Ok(())
    }
}

/// Stages renumbering the iterators of `block`: the iterator at each position moves to
/// `placed[position]`, or, where that is `None`, is removed and reads as 0.
fn renumber_iters(rewriter: &mut Rewriter, block: &BlockOp, placed: &[Option<usize>; MAX_ITERS]) {
    let renumbered: [IndexExpr; MAX_ITERS] =
        std::array::from_fn(|position| match placed[position] {
            Some(new_position) => IndexExpr::iter(new_position),
            None => IndexExpr::constant(0),
        });
    let access = |position: usize| Some(AccessExpr::from_pure(&renumbered[position]));
    let mut attrs = block.map_exprs(&|expr: &AccessExpr| expr.substitute(&access).simplify());
    let kept = placed.iter().flatten().count();
    attrs.iters.clear();
    attrs.iters.resize(kept, Iter::default());
    for (position, iter) in block.iters.iter().enumerate() {
        if let Some(new_position) = placed[position] {
            attrs.iters[new_position] = iter.clone();
        }
    }
    for fold in &mut attrs.folds {
        if let Some(value_iters) = fold.combine.value_iters_mut() {
            *value_iters = IterList::new(
                value_iters
                    .positions()
                    .filter_map(|value_iter| placed[value_iter]),
            );
        }
    }
    let index = |position: usize| Some(renumbered[position].clone());
    map_body_indices(rewriter, block.body(), &|expr| {
        expr.substitute(&index).simplify()
    });
    block.update(rewriter, |op| *op = attrs);
}

/// The pattern removes an iterator of extent 1, which is always 0. It writes 0 for it in every
/// expression and write, and moves the later iterators down one position.
#[pattern(canonical)]
struct DropUnitIters;

impl Pattern for DropUnitIters {
    type Root = UnscheduledBlock;

    fn rewrite(&self, block: BlockOp, rewriter: &mut Rewriter) -> Result<(), NoMatch> {
        let mut value_iters = BTreeSet::new();
        for fold in &block.folds {
            value_iters.extend(fold.combine.rule().value_iters().positions());
        }
        let mut unit = BTreeSet::new();
        for (position, iter) in block.iters.iter().enumerate() {
            let of_value = value_iters.contains(&position);
            if iter.extent.is_one() && iter.kind != IterKind::Prefix && !of_value {
                unit.insert(position);
            }
        }
        require!(!unit.is_empty(), "no iterator has extent 1");
        let mut placed = [None; MAX_ITERS];
        for (position, new_position) in placed.iter_mut().enumerate().take(block.iters.len()) {
            if !unit.contains(&position) {
                *new_position = Some(position - unit.range(..position).count());
            }
        }
        renumber_iters(rewriter, &block, &placed);
        Ok(())
    }
}

/// The parallel iterators that the block writes at come first, in the order its writes first take
/// them, and its other iterators follow in their order. The iterators only name the block's
/// points, so this renames them: a block and the blocks that run on its result element by element
/// then name each element's point alike.
#[pattern(canonical)]
struct OrderItersByWrite;

impl Pattern for OrderItersByWrite {
    type Root = UnscheduledBlock;

    fn rewrite(&self, block: BlockOp, rewriter: &mut Rewriter) -> Result<(), NoMatch> {
        let mut order = ArrayVec::<[usize; MAX_ITERS]>::new();
        for write in block.writes() {
            for coord in &write.coords {
                let Some(position) = coord.as_iter() else {
                    continue;
                };
                if block.iters[position].kind == IterKind::Par && !order.contains(&position) {
                    order.push(position);
                }
            }
        }
        for position in 0..block.iters.len() {
            if !order.contains(&position) {
                order.push(position);
            }
        }
        require!(
            order
                .iter()
                .enumerate()
                .any(|(new_position, position)| new_position != *position),
            "the iterators are in the order of the writes"
        );
        let mut placed = [None; MAX_ITERS];
        for (new_position, position) in order.into_iter().enumerate() {
            placed[position] = Some(new_position);
        }
        renumber_iters(rewriter, &block, &placed);
        Ok(())
    }
}

/// Every index expression is in canonical form: sums with merged coefficients, `floordiv` and
/// `mod` reduced where the iterators' extents allow, and `min` and `max` decided where the
/// extents decide them.
#[pattern(canonical)]
struct CanonicalIndices;

impl Pattern for CanonicalIndices {
    type Root = UnscheduledBlock;

    fn rewrite(&self, block: BlockOp, rewriter: &mut Rewriter) -> Result<(), NoMatch> {
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
            block.update(rewriter, |op| *op = canonical_attrs);
        }
        Ok(())
    }
}

/// A block that reads its operand at its own iterators, in the order it writes them, into a
/// result of the operand's type, and yields what it reads, is that operand.
#[pattern(canonical)]
struct EraseIdentityBlock;

impl Pattern for EraseIdentityBlock {
    type Root = UnscheduledBlock;

    fn rewrite(&self, block: BlockOp, rewriter: &mut Rewriter) -> Result<(), NoMatch> {
        require!(block.is_copy(), "the block does more than move elements");
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

/// The read `read` of `block`, of the result of `copy`, as a read of the copy's operand: the
/// copy's coordinates, with each of its iterators replaced by the coordinate of `read` that the
/// copy writes it at, and the copy's operands renumbered from `offset` on. `None` when the read is
/// padded and the composed read would read the pad elsewhere, or when it places the block or holds
/// what it takes, which name the copy's result.
fn compose(block: &Block, read: &Read, copy: &BlockOp, offset: usize) -> Option<Read> {
    if read.at.is_some() || read.held.is_some() {
        return None;
    }
    let mut composed = copy.read_through(0, read, offset);
    composed.pad = match read.pad {
        None => composed.pad,
        Some(pad) if pads_alike(block, read, copy) => Some(pad),
        Some(_) => return None,
    };
    Some(composed)
}

/// Whether the padded `read` of `block`, of the result of `copy`, reads its pad from the copy's
/// operand exactly where it reads it from the result. The copy pads with nothing or the same
/// value and reads no index from an operand, and each coordinate of `read` that may leave the
/// result is the coordinate, over a whole axis of the operand, of the copy's iterator it writes.
fn pads_alike(block: &Block, read: &Read, copy: &BlockOp) -> bool {
    let copy_read = &copy.reads[0];
    if copy_read
        .pad
        .is_some_and(|copy_pad| Some(copy_pad) != read.pad)
    {
        return false;
    }
    if copy_read.coords.iter().any(|coord| coord.has_reads()) {
        return false;
    }
    let Source::Operand(operand) = copy_read.source else {
        return false;
    };
    let operand_dims = &copy.ins()[operand].tensor_type().dims;
    let result_dims = &copy.outs()[0].tensor_type().dims;
    let write = &copy.folds[0].writes[0].coords;
    let iter_range = |position: usize| block.iter_range(position);
    for ((coord, expr), dim) in write.iter().zip(&read.coords).zip(result_dims.iter()) {
        let inside = match (dim, expr.bounds(&iter_range)) {
            (Dim::Static(extent), Some((low, high))) => low >= 0 && high < *extent as i64,
            _ => false,
        };
        if inside {
            continue;
        }
        let Some(position) = coord.as_iter() else {
            return false;
        };
        let covered =
            copy_read
                .coords
                .iter()
                .zip(operand_dims.iter())
                .any(|(operand_coord, operand_dim)| {
                    operand_coord.as_iter() == Some(position)
                        && *operand_dim == copy.iters[position].extent
                });
        if !covered {
            return false;
        }
    }
    true
}

/// The axis of its operand that each iterator of `copy` reads, when the copy only reorders the
/// axes of the operand and adds or removes axes of extent 1: it reads without pad, at distinct
/// iterators over whole axes, or at 0 on axes of extent 1, and each iterator it does not read has
/// extent 1.
fn reordered_axes(copy: &BlockOp) -> Option<[Option<usize>; MAX_ITERS]> {
    if !copy.is_copy() {
        return None;
    }
    let read = &copy.reads[0];
    let Source::Operand(operand) = read.source else {
        return None;
    };
    if read.pad.is_some() {
        return None;
    }
    let dims = &copy.ins()[operand].tensor_type().dims;
    let mut axis_of = [None; MAX_ITERS];
    for (axis, (coord, dim)) in read.coords.iter().zip(dims.iter()).enumerate() {
        if coord.as_const() == Some(0) && dim.is_one() {
            continue;
        }
        let position = coord.as_iter()?;
        if copy.iters[position].extent != *dim || axis_of[position].is_some() {
            return None;
        }
        axis_of[position] = Some(axis);
    }
    for (position, iter) in copy.iters.iter().enumerate() {
        if axis_of[position].is_none() && !iter.extent.is_one() {
            return None;
        }
    }
    Some(axis_of)
}

/// The block that computes the operand of `copy`, and which of its results that is, when the
/// copy alone reads it and the block writes it as a new tensor, one coordinate per axis.
fn sole_producer<'m>(copy: &BlockOp<'m>) -> Option<(BlockOp<'m>, usize)> {
    let Source::Operand(operand) = copy.reads[0].source else {
        return None;
    };
    let input = copy.ins()[operand];
    let producer = input.defining_op::<Block>()?;
    let result = producer.outs().iter().position(|out| *out == input)?;
    let write = producer.writes().nth(result)?;
    let fresh = write.into.is_none() && write.coords.len() == input.tensor_type().rank();
    (input.has_one_use() && fresh).then_some((producer, result))
}

/// The write of result `result` of `block` in the order of `copy`, its only reader, which
/// reorders its axes. `None` while the block writes an axis that the copy removes at an
/// iterator rather than at 0.
fn reordered_write(block: &BlockOp, result: usize, copy: &BlockOp) -> Option<Write> {
    let (producer, produced) = sole_producer(copy)?;
    if producer.inst() != block.inst() || produced != result {
        return None;
    }
    let axis_of = reordered_axes(copy)?;
    let write = block.writes().nth(result)?;
    for (axis, coord) in write.coords.iter().enumerate() {
        if !axis_of.contains(&Some(axis)) && coord.as_const() != Some(0) {
            return None;
        }
    }
    let mut coords = Coords::new();
    for coord in &copy.folds[0].writes[0].coords {
        let operand_axis = coord.as_iter().and_then(|position| axis_of[position]);
        match operand_axis {
            Some(axis) => coords.push(write.coords[axis].clone()),
            None => coords.push(AccessExpr::constant(0)),
        }
    }
    Some(Write { into: None, coords })
}

/// Whether the block that computes the operand of `copy` takes the copy's order instead: the
/// copy alone reads it and only reorders its axes.
fn producer_reorders(copy: &BlockOp) -> bool {
    sole_producer(copy).is_some() && reordered_axes(copy).is_some()
}

/// A read of the result of a copy block reads the copy's operand, at the copy's coordinates of
/// the point it reads. A read that stays inside the result takes the copy's `pad`. A padded read
/// composes where each of its coordinates that may leave the result leaves the operand with it.
#[pattern(canonical)]
struct ComposeCopyReads;

impl Pattern for ComposeCopyReads {
    type Root = UnscheduledBlock;

    fn rewrite(&self, block: BlockOp, rewriter: &mut Rewriter) -> Result<(), NoMatch> {
        let mut attrs = block.attrs().clone();
        let mut ins: Vec<Value> = block.ins().iter().map(|input| input.id()).collect();
        let mut composed = false;
        for (index, read) in block.reads.iter().enumerate() {
            let Source::Operand(operand) = read.source else {
                continue;
            };
            let Some(copy) = block.ins()[operand].defining_op::<Block>() else {
                continue;
            };
            if !copy.is_copy() || producer_reorders(&copy) {
                continue;
            }
            let Some(composed_read) = compose(&block, read, &copy, ins.len()) else {
                continue;
            };
            attrs.reads[index] = composed_read;
            for input in copy.ins() {
                ins.push(input.id());
            }
            composed = true;
        }
        require!(composed, "no read of a copy composes");
        update(rewriter, &block, attrs, ins);
        Ok(())
    }
}

/// When the only reader of a block's result is a copy that reorders its axes, the block writes
/// that result in the copy's order. The block's new result replaces the copy.
#[pattern(canonical)]
struct WriteInCopyOrder;

impl Pattern for WriteInCopyOrder {
    type Root = UnscheduledBlock;

    fn rewrite(&self, block: BlockOp, rewriter: &mut Rewriter) -> Result<(), NoMatch> {
        let module = block.module();
        let mut found = None;
        for (result, out) in block.outs().iter().enumerate() {
            let [only] = out.uses() else {
                continue;
            };
            let Some(copy) = module.op_as::<Block>(only.inst) else {
                continue;
            };
            if let Some(write) = reordered_write(&block, result, &copy) {
                found = Some((result, copy, write));
                break;
            }
        }
        require!(
            let Some((result, copy, write)) = found,
            "no result is only reordered by a copy"
        );
        let mut attrs = block.attrs().clone();
        *attrs.writes_mut().nth(result).expect("a write per result") = write;
        let ins: Vec<Value> = block.ins().iter().map(|input| input.id()).collect();
        let outs = rebuild(rewriter, attrs, ins, block.body());
        rewriter.replace_op(&copy, outs[result]);
        for (index, out) in block.outs().iter().enumerate() {
            if index != result {
                rewriter.replace_all_uses(*out, outs[index]);
            }
        }
        rewriter.erase(&block);
        Ok(())
    }
}

/// The block of `attrs` on `ins`, with a copy of `body`, before the root; gives its results.
fn rebuild(rewriter: &mut Rewriter, attrs: Block, ins: Vec<Value>, body: RegionRef) -> Vec<Value> {
    let module = body.module();
    attrs.build(rewriter, ins, |body_builder, args| {
        let mut values: HashMap<Value, Value> = HashMap::new();
        for (arg, &new_arg) in body.args().iter().zip(args) {
            values.insert(arg.id(), new_arg);
        }
        copy_body_ops(module, body.id(), body_builder.builder, &mut values)
    })
}

/// Whether `block` only copies the elements of one operand into another operand: one read,
/// without a pad, whose value the body yields at each of its parallel points, and a write into
/// the other operand.
fn copies_into(block: &BlockOp) -> bool {
    let ([read], [fold]) = (&block.reads[..], &block.folds[..]) else {
        return false;
    };
    let [write] = &fold.writes[..] else {
        return false;
    };
    let Source::Operand(source) = read.source else {
        return false;
    };
    let body = block.body();
    fold.combine == CombineKind::None
        && block.constraints.is_empty()
        && write.into.is_some_and(|base| base.0 != source)
        && read.pad.is_none()
        && block.iters.iter().all(|iter| iter.kind == IterKind::Par)
        && body.insts().count() == 1
        && body.yielded() == body.args()
}

/// When the only reader of a map's result is a copy into another operand, the map writes into
/// that operand itself, so its values go straight where the copy put them. This holds when the
/// copy reads at each point the element the map writes at that point, and the map writes each
/// point at an element of its own.
#[pattern(canonical)]
struct WriteIntoUpdate;

impl Pattern for WriteIntoUpdate {
    type Root = UnscheduledBlock;

    fn rewrite(&self, update: BlockOp, rewriter: &mut Rewriter) -> Result<(), NoMatch> {
        require!(copies_into(&update), "the block is no copy into an operand");
        let Source::Operand(source) = update.reads[0].source else {
            unreachable!("a copy into an operand reads one operand");
        };
        let copied = update.ins()[source];
        require!(copied.has_one_use(), "another op reads the copied tensor");
        require!(
            let Some(map) = copied.defining_op::<Block>(),
            "no block computes the copied tensor"
        );
        require!(map.is_map(), "no map computes the copied tensor");
        let produced = &map.folds[0].writes[0];
        require!(
            produced.translates(&map.iters),
            "the map writes some element at more than one point"
        );
        require!(
            map.iters == update.iters && produced.coords == update.reads[0].coords,
            "the copy reads at a point another element than the map writes there"
        );
        let mut ins: Vec<Value> = map.ins().iter().map(|input| input.id()).collect();
        let offset = ins.len();
        ins.extend(update.ins().iter().map(|input| input.id()));
        let write = &update.folds[0].writes[0];
        let shifted = |operand: usize| operand + offset;
        let mut attrs = map.attrs().clone();
        attrs.folds[0].writes[0] = Write {
            into: write.into.map(|base| OperandRef(shifted(base.0))),
            coords: write
                .coords
                .iter()
                .map(|coord| coord.rename_operands(&shifted))
                .collect(),
        };
        let (compact_attrs, compact_ins) = attrs.compact(ins);
        let outs = rebuild(rewriter, compact_attrs, compact_ins, map.body());
        rewriter.replace_op(&update, outs[0]);
        rewriter.erase(&map);
        Ok(())
    }
}

/// The expressions of the `tile.index` ops of `body`.
fn body_indices(body: RegionRef) -> Vec<IndexExpr> {
    let mut exprs = Vec::new();
    for inst in body.insts() {
        if let Some(index) = body.module().op_as::<Index>(inst) {
            exprs.push(index.expr.clone());
        }
    }
    exprs
}

/// The expression `expr` with the iterator at `position` split into two, at `position` and the
/// next, of which the second has extent `inner`, and the later iterators one place on.
fn split_iter<R: ReadKind>(expr: &Expr<R>, position: usize, inner: usize) -> Expr<R> {
    expr.substitute(&|iter: usize| match iter.cmp(&position) {
        Ordering::Less => None,
        Ordering::Equal => {
            Some(Expr::iter(position) * Expr::constant(inner as i64) + Expr::iter(position + 1))
        }
        Ordering::Greater => Some(Expr::iter(iter + 1)),
    })
}

/// `block` with the iterator at `position` split into two in its place, of which the second has
/// extent `inner`, and its expressions in canonical form.
fn split_block(block: &Block, position: usize, outer: usize, inner: usize) -> Block {
    let mut split = block.map_exprs(&|expr| split_iter(expr, position, inner));
    let kind = split.iters[position].kind;
    split.iters[position].extent = Dim::Static(outer);
    split
        .iters
        .insert(position + 1, Iter::new(kind, Dim::Static(inner)));
    for fold in &mut split.folds {
        let Some(value_iters) = fold.combine.value_iters_mut() else {
            continue;
        };
        let shifted = |value_iter: usize| match value_iter > position {
            true => value_iter + 1,
            false => value_iter,
        };
        *value_iters = IterList::new(value_iters.positions().map(shifted));
    }
    split.map_exprs(&|expr| expr.canonical(&|at| split.extent(at)))
}

/// An iterator of static extent `outer * inner` that no fresh write and no value's iterators name
/// splits into two in its place, of extents `outer` and `inner`, when that removes a `floordiv`
/// or `mod` from the block's expressions. The points keep their order, so every fold keeps its
/// bits.
#[pattern(canonical)]
struct DelinearizeIters;

impl Pattern for DelinearizeIters {
    type Root = UnscheduledBlock;

    fn rewrite(&self, block: BlockOp, rewriter: &mut Rewriter) -> Result<(), NoMatch> {
        require!(
            block.iters.len() < MAX_ITERS,
            "the block has as many iterators as it may"
        );
        let mut named = BTreeSet::new();
        for write in block.writes() {
            if write.into.is_none() {
                named.extend(write.coords.iter().filter_map(|coord| coord.as_iter()));
            }
        }
        for fold in &block.folds {
            named.extend(fold.combine.rule().value_iters().positions());
        }
        let indices = body_indices(block.body());
        let mut before = 0;
        for expr in block.exprs() {
            before += expr.divisions();
        }
        for expr in &indices {
            before += expr.divisions();
        }
        // The split that leaves the fewest divisions: its count, the block, the iterator and the
        // extent of its second part.
        let mut best: Option<(usize, Block, usize, usize)> = None;
        for (position, iter) in block.iters.iter().enumerate() {
            let Dim::Static(extent) = iter.extent else {
                continue;
            };
            if named.contains(&position) || iter.kind == IterKind::Prefix {
                continue;
            }
            let mut factors = Vec::new();
            for expr in block.exprs() {
                factors.extend(expr.divisors_of(position));
            }
            for expr in &indices {
                factors.extend(expr.divisors_of(position));
            }
            for divisor in factors {
                let Ok(inner) = usize::try_from(divisor) else {
                    continue;
                };
                if inner <= 1 || inner >= extent || extent % inner != 0 {
                    continue;
                }
                let split = split_block(&block, position, extent / inner, inner);
                let mut after = 0;
                for expr in split.exprs() {
                    after += expr.divisions();
                }
                for expr in &indices {
                    let index = split_iter(expr, position, inner);
                    after += index.canonical(&|at| split.extent(at)).divisions();
                }
                if after < best.as_ref().map_or(before, |(fewest, ..)| *fewest) {
                    best = Some((after, split, position, inner));
                }
            }
        }
        require!(
            let Some((_, split, position, inner)) = best,
            "no iterator splits to remove a floordiv or mod"
        );
        map_body_indices(rewriter, block.body(), &|expr| {
            split_iter(expr, position, inner).canonical(&|at| split.extent(at))
        });
        block.update(rewriter, |op| *op = split);
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
fn index_value(value: ValueRef) -> Option<IndexExpr> {
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
    let enclosing = module
        .parent_region(inst)
        .and_then(|body| module.parent_op(body))
        .and_then(|parent| module.op_as::<Block>(parent));
    let Some(block) = enclosing else {
        return false;
    };
    let iter_range = |position: usize| block.iter_range(position);
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

    fn rewrite(&self, op: OpRef<T>, rewriter: &mut Rewriter) -> Result<(), NoMatch> {
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
