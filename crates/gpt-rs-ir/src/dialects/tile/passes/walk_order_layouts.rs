//! Laying out values computed before the run in the order their reader's schedule walks them.

use std::collections::HashMap;

use anyhow::Result;

use crate::dialects::func::Return;
use crate::dialects::host;
use crate::dialects::tile::verify::check_schedule;
use crate::dialects::tile::{
    Block, CombineKind, Coords, CopyAxis, Dispatch, Fold, Index, Iter, IterKind, IterList, Read,
    Source, Write,
};
use crate::ir::{gcd, AccessExpr, Dim, Inst, Module, Region, SymbolRef, TensorType, Type, Value};
use crate::passes::{pass, run_patterns, Pass};
use crate::require;
use crate::rewrite::{
    pattern, Builder, NoMatch, Pattern, PatternSet, Rewriter, Rewrites, Rewriting,
};

/// Lays out each value computed before the run, which a dispatch reads in its first block alone, in
/// the order that block's schedule walks it. The layout's axes are those of a copy of the read over
/// the whole nest ([`Walk::copy_axes`](crate::dialects::tile::Walk::copy_axes)): one for the tiles
/// of each loop over an iterator the read takes and one for the points of its innermost tile,
/// consecutive axes of one iterator merged. Each axis of the value whose coordinate is an iterator
/// splits into those axes, at sizes that divide the points around them. The dispatch that computes
/// the value computes it in that layout, and the reader takes it there, at `d floordiv t` and `d
/// mod t`. A dispatch that no call names any more is erased. A value of the checkpoint keeps its
/// layout: a copy of it would hold the weights in memory twice.
#[pass(
    name = "walk-order-layouts",
    stats(laid_out = "values laid out in the order their reader walks them")
)]
pub struct WalkOrderLayouts;

fn populate_walk_order_layouts_patterns(set: &mut PatternSet) {
    set.add(LayOutInWalkOrder);
}

impl Pass for WalkOrderLayouts {
    fn rewrites(&self) -> Rewrites {
        Rewrites::patterns(populate_walk_order_layouts_patterns)
    }

    fn run(&self, module: &mut Rewriting) -> Result<bool> {
        run_patterns(module, self.rewrites(), &Self::LAID_OUT)
    }
}

/// An axis of a layout in walk order: of the value's axis `axis` of `whole` elements, whose
/// coordinate is the iterator `position`, the coordinate `(x mod within) floordiv divisor`, of
/// `within / divisor` elements.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Digit {
    axis: usize,
    position: usize,
    whole: usize,
    within: usize,
    divisor: usize,
}

impl Digit {
    fn extent(&self) -> usize {
        self.within / self.divisor
    }
}

/// A call of a dispatch whose first block reads a value computed before the run, which another
/// dispatch computes for it alone in one block that writes each element once, in a layout other
/// than the order the reader walks it: the value is computed in that order, and read there.
#[pattern]
struct LayOutInWalkOrder;

impl Pattern for LayOutInWalkOrder {
    type Root = host::Dispatch;

    fn rewrite(&self, call: host::DispatchOp, rewriter: &mut Rewriter) -> Result<(), NoMatch> {
        let module = call.module();
        require!(
            let Some(reader) = Dispatch::named(module, &call.callee),
            "the callee is not a dispatch of the module"
        );
        let reader_body = module.regions(reader)[0];
        let blocks = Dispatch::blocks(module, reader_body);
        require!(let Some(&(first, block)) = blocks.first(), "the dispatch holds no block");
        let call_args = module.operands(call.inst()).to_vec();
        require!(
            let Some(found) = laid_out_arg(module, &call_args, reader_body, (first, block)),
            "no value computed before the run that one block alone reads is laid out other than \
             in its reader's walk order"
        );
        let LaidOutArg {
            arg,
            producer,
            producer_dispatch,
            writers,
            read_index,
            digits,
        } = found;
        let value_type = module
            .ty(call_args[arg])
            .as_tensor()
            .expect("a tensor")
            .clone();
        let mut dims = Vec::with_capacity(digits.len());
        for digit in &digits {
            dims.push(Dim::Static(digit.extent()));
        }
        let laid_out = Type::Tensor(TensorType::new(value_type.dtype, dims));
        let producer_name = Dispatch::add_named_by_text(rewriter, |scratch, name| {
            let layout = (digits.as_slice(), writers.as_slice(), &laid_out);
            build_producer(module, producer_dispatch, layout, scratch, name)
        });
        let reader_name = Dispatch::add_named_by_text(rewriter, |scratch, name| {
            let changed = (arg, read_index, digits.as_slice(), &laid_out);
            build_reader(module, reader, first, changed, scratch, name)
        });
        let producer_args = module.operands(producer).to_vec();
        rewriter.set_insertion_point(producer);
        let computed = rewriter.create(
            host::Dispatch {
                callee: SymbolRef(producer_name),
            },
            producer_args,
        );
        let mut reader_args = call_args;
        reader_args[arg] = computed[0];
        rewriter.set_insertion_point(call.inst());
        let replaced = rewriter.create(
            host::Dispatch {
                callee: SymbolRef(reader_name),
            },
            reader_args,
        );
        rewriter.replace_op(&call, replaced);
        rewriter.erase(producer);
        for (dispatch, replaced_call) in [(reader, call.inst()), (producer_dispatch, producer)] {
            if Dispatch::callers(module, dispatch) == [replaced_call] {
                rewriter.erase(dispatch);
            }
        }
        WalkOrderLayouts::LAID_OUT.add(1);
        Ok(())
    }
}

/// An argument of a call that its reader takes in another layout than its walk order: the
/// argument, the `host.dispatch` that computes it and its dispatch, the iterator of that
/// dispatch's block that writes each axis of it ([`producer_dispatch`]), the read of the reader's
/// first block that takes it, and the layout in walk order.
struct LaidOutArg {
    arg: usize,
    producer: Inst,
    producer_dispatch: Inst,
    writers: Vec<Option<usize>>,
    read_index: usize,
    digits: Vec<Digit>,
}

/// The first of `args`, the arguments of a call of the dispatch whose body is `reader_body` and
/// whose first block is `first`, as `(inst, block)`, that is laid out anew. Such an argument is
/// computed before the run by a dispatch for it alone, whose one block writes each element once.
/// The first block alone reads it, in a layout other than its walk order, and the verifier accepts
/// the new layout.
fn laid_out_arg(
    module: &Module,
    args: &[Value],
    reader_body: Region,
    (first, block): (Inst, &Block),
) -> Option<LaidOutArg> {
    for (arg, &value) in args.iter().enumerate() {
        let value_ref = module.value_ref(value);
        if !value_ref.known_before_run() || !value_ref.has_one_use() {
            continue;
        }
        let Some(producer) = value_ref.defining_inst() else {
            continue;
        };
        if module.op(producer).get::<host::Dispatch>().is_none() {
            continue;
        }
        let param = module.args(reader_body)[arg];
        let Some(read_index) = only_read(module, first, block, param) else {
            continue;
        };
        let Some(digits) = walk_digits(module, first, block, &block.reads[read_index]) else {
            continue;
        };
        let mut changed = block.clone();
        changed.reads[read_index].coords = digit_coords(&block.reads[read_index], &digits);
        if check_schedule(&changed).is_err() {
            continue;
        }
        let Some((producer_dispatch, writers)) = producer_dispatch(module, producer) else {
            continue;
        };
        return Some(LaidOutArg {
            arg,
            producer,
            producer_dispatch,
            writers,
            read_index,
            digits,
        });
    }
    None
}

/// The dispatch that `producer`, a `host.dispatch`, calls, and the iterator of its block that
/// writes each axis of its result, none for an axis of one point that it writes at 0. It is found
/// when the dispatch has one block alone, whose one result it returns, with these rules:
/// - no fold and no constraint, and parallel iterators alone, each the coordinate of one axis of
///   that result of its extent, so that it writes each element once, into no operand;
/// - it reads operands alone, none from a holding;
/// - no `tile.index` in its body, so that the block takes its iterators through the coordinates of
///   its reads alone.
fn producer_dispatch(module: &Module, producer: Inst) -> Option<(Inst, Vec<Option<usize>>)> {
    let callee = &module.op(producer).get::<host::Dispatch>()?.callee;
    let dispatch = Dispatch::named(module, callee)?;
    let body = module.regions(dispatch)[0];
    let blocks = Dispatch::blocks(module, body);
    let [(block_inst, block)] = blocks.as_slice() else {
        return None;
    };
    let [fold] = block.folds.as_slice() else {
        return None;
    };
    let [write] = fold.writes.as_slice() else {
        return None;
    };
    let result_dims = module
        .ty(*module.results(*block_inst).first()?)
        .as_tensor()?
        .static_dims()?;
    let [block_result] = module.results(*block_inst) else {
        return None;
    };
    let returned = module.insts(body).last()?;
    let simple = fold.combine == CombineKind::None
        && write.into.is_none()
        && block.constraints.is_empty()
        && module.insts(body).count() == 2
        && module.operands(returned) == [*block_result];
    if !simple {
        return None;
    }
    for read in &block.reads {
        if !matches!(read.source, Source::Operand(_)) || read.held.is_some() {
            return None;
        }
    }
    // A `tile.index` of the body names the old iterators, which the layout's digits replace.
    for body_inst in module.insts(module.regions(*block_inst)[0]) {
        if module.op(body_inst).get::<Index>().is_some() {
            return None;
        }
    }
    let mut writers = Vec::with_capacity(write.coords.len());
    let mut written = vec![false; block.iters.len()];
    for (axis, coord) in write.coords.iter().enumerate() {
        if coord.as_const() == Some(0) && result_dims[axis] == 1 {
            writers.push(None);
            continue;
        }
        let position = coord.as_iter()?;
        let iter = &block.iters[position];
        let whole = iter.extent.as_static() == Some(result_dims[axis]);
        if written[position] || iter.kind.is_reduced() || !whole {
            return None;
        }
        written[position] = true;
        writers.push(Some(position));
    }
    if written.contains(&false) {
        return None;
    }
    Some((dispatch, writers))
}

/// The index of the read of `block`, the block of `inst`, that takes `param`, when it takes it
/// alone from an operand that no other op of the dispatch takes, and nothing else of the block
/// names it ([`Block::only_read_of`]).
fn only_read(module: &Module, inst: Inst, block: &Block, param: Value) -> Option<usize> {
    let [only_use] = module.uses(param) else {
        return None;
    };
    if only_use.inst != inst {
        return None;
    }
    let operand = only_use.index;
    block.only_read_of(operand)?;
    block
        .reads
        .iter()
        .position(|read| read.source == Source::Operand(operand))
}

/// The axes of the layout in which `block`, the block of `inst`, walks what `read` reads. It holds
/// when every coordinate of the read is an iterator of its own, or 0 along an axis of one element,
/// and the layout differs from the value's. For each iterator the read takes, there is a digit for
/// each loop over its tiles and one for the points of its innermost tile. The digits come in the
/// order of the copy axes of the read over the whole nest, at sizes that divide the points around
/// them.
fn walk_digits(module: &Module, inst: Inst, block: &Block, read: &Read) -> Option<Vec<Digit>> {
    if read.pad.is_some() {
        return None;
    }
    let Source::Operand(operand) = read.source else {
        return None;
    };
    let dims = module
        .ty(module.operands(inst)[operand])
        .as_tensor()?
        .static_dims()?;
    let mut axis_of = HashMap::new();
    for (axis, coord) in read.coords.iter().enumerate() {
        match coord.as_iter() {
            Some(position) => {
                if axis_of.insert(position, axis).is_some() {
                    return None;
                }
            }
            None if dims[axis] == 1 => {}
            None => return None,
        }
    }
    let walk = block.walk();
    let mut within: HashMap<usize, usize> = HashMap::new();
    let mut digits: Vec<Digit> = Vec::new();
    for copy_axis in walk.copy_axes(block, read, None) {
        let (position, size) = match copy_axis {
            CopyAxis::Tiles { index, .. } => {
                let entry = &walk.nest[index];
                (entry.iter.position(), entry.extent)
            }
            CopyAxis::Points { position, .. } => (position, 1),
        };
        let axis = *axis_of.get(&position)?;
        let around = *within.entry(position).or_insert(dims[axis]);
        let divisor = gcd(around as u64, size.max(1) as u64) as usize;
        within.insert(position, divisor);
        let digit = Digit {
            axis,
            position,
            whole: dims[axis],
            within: around,
            divisor,
        };
        if digit.extent() == 1 {
            continue;
        }
        match digits.last_mut() {
            Some(last) if last.position == position && last.divisor == around => {
                last.divisor = divisor;
            }
            _ => digits.push(digit),
        }
    }
    for (&position, &axis) in &axis_of {
        if within.get(&position) != Some(&1) && dims[axis] > 1 {
            return None;
        }
    }
    if digits.len() > crate::ir::MAX_RANK {
        return None;
    }
    let mut in_order = true;
    for (index, digit) in digits.iter().enumerate() {
        in_order &= digit.axis == index && digit.divisor == 1 && digit.within == dims[digit.axis];
    }
    let unit_axes = dims.iter().filter(|&&extent| extent == 1).count();
    (!(in_order && digits.len() + unit_axes == dims.len())).then_some(digits)
}

/// The coordinates at which a read whose coordinates are iterators takes a value laid out in
/// `digits`: `(x mod within) floordiv divisor` of the iterator of each.
fn digit_coords(read: &Read, digits: &[Digit]) -> Coords {
    let mut coords = Coords::new();
    for digit in digits {
        let mut coord = read.coords[digit.axis].clone();
        if digit.within < digit.whole {
            coord = AccessExpr::modulo(coord, AccessExpr::constant(digit.within as i64));
        }
        if digit.divisor > 1 {
            coord = AccessExpr::floordiv(coord, AccessExpr::constant(digit.divisor as i64));
        }
        coords.push(coord);
    }
    coords
}

/// Builds into `scratch`, named `name`, the dispatch of `dispatch` with its result laid out in
/// `digits`, of type `laid_out`. `dispatch` is a `tile.dispatch` of `module` whose one block writes
/// each element of its result once, each axis by the iterator that `writers` names. The new block
/// runs over the digits: each iterator of the old block is the sum of the iterators of its axis's
/// digits times their divisors.
fn build_producer(
    module: &Module,
    dispatch: Inst,
    (digits, writers, laid_out): (&[Digit], &[Option<usize>], &Type),
    scratch: &mut Module,
    name: &str,
) -> Inst {
    let op = module.op(dispatch).get::<Dispatch>().expect("a dispatch");
    let body = module.regions(dispatch)[0];
    let (block_inst, block) = Dispatch::blocks(module, body)[0];
    let mut old_iters = vec![AccessExpr::constant(0); block.iters.len()];
    for (index, digit) in digits.iter().enumerate() {
        let position = writers[digit.axis].expect("a digit runs along an axis of several points");
        let scaled = AccessExpr::iter(index) * AccessExpr::constant(digit.divisor as i64);
        old_iters[position] = (old_iters[position].clone() + scaled).simplify();
    }
    let mut reindexed = block.clone();
    reindexed.iters.clear();
    for digit in digits {
        reindexed
            .iters
            .push(Iter::new(IterKind::Par, Dim::Static(digit.extent())));
    }
    for read in &mut reindexed.reads {
        let mut coords = Coords::new();
        for coord in &read.coords {
            let substituted = coord.substitute(&|position| Some(old_iters[position].clone()));
            coords.push(substituted.simplify());
        }
        read.coords = coords;
    }
    let mut identity = Vec::with_capacity(digits.len());
    for index in 0..digits.len() {
        identity.push(AccessExpr::iter(index));
    }
    reindexed.folds = vec![Fold::plain(Write::new(identity))].into();
    reindexed.loops = Default::default();
    reindexed.clip = IterList::default();
    let region = scratch.new_region(op.args.clone());
    let mut map: HashMap<Value, Value> = HashMap::new();
    for (&old, &new) in module.args(body).iter().zip(scratch.args(region)) {
        map.insert(old, new);
    }
    let mut builder = Builder::at_end(scratch, region);
    let computed = copy_block(&mut builder, module, block_inst, reindexed, &map);
    builder.create(Return, vec![computed[0]]);
    let reindexed_dispatch = Dispatch {
        name: name.to_string(),
        args: op.args.clone(),
        results: vec![laid_out.clone()],
        facts: op.facts.clone(),
        target: op.target.clone(),
        numerics: op.numerics,
    };
    let top = scratch.body();
    Builder::at_end(scratch, top).create_op(
        reindexed_dispatch.into(),
        Vec::new(),
        None,
        vec![region],
    )
}

/// Builds into `scratch`, named `name`, the dispatch `reader` of `module` taking its argument
/// `arg` laid out in `digits`, of type `laid_out`: its first block, of `first`, reads it at the
/// read `read_index` at the digits' coordinates.
fn build_reader(
    module: &Module,
    reader: Inst,
    first: Inst,
    (arg, read_index, digits, laid_out): (usize, usize, &[Digit], &Type),
    scratch: &mut Module,
    name: &str,
) -> Inst {
    let op = module.op(reader).get::<Dispatch>().expect("a dispatch");
    let body = module.regions(reader)[0];
    let mut args = op.args.clone();
    args[arg] = laid_out.clone();
    let region = scratch.new_region(args.clone());
    let mut map: HashMap<Value, Value> = HashMap::new();
    for (&old, &new) in module.args(body).iter().zip(scratch.args(region)) {
        map.insert(old, new);
    }
    let mut builder = Builder::at_end(scratch, region);
    for inst in module.insts(body) {
        if inst != first {
            builder
                .create_copy(module, inst, &mut map)
                .expect("the ops of a dispatch read its own values");
            continue;
        }
        let block = module.op(inst).get::<Block>().expect("the first block");
        let mut changed = block.clone();
        changed.reads[read_index].coords = digit_coords(&block.reads[read_index], digits);
        let created = copy_block(&mut builder, module, inst, changed, &map);
        for (&old, &new) in module.results(inst).iter().zip(created.iter()) {
            map.insert(old, new);
        }
    }
    let laid_out_dispatch = Dispatch {
        name: name.to_string(),
        args,
        results: op.results.clone(),
        facts: op.facts.clone(),
        target: op.target.clone(),
        numerics: op.numerics,
    };
    let top = scratch.body();
    Builder::at_end(scratch, top).create_op(
        laid_out_dispatch.into(),
        Vec::new(),
        None,
        vec![region],
    )
}

/// Creates with `builder` the block `block` on the values that `map` gives for the operands of
/// `inst`, a block of `module`, with a copy of its body.
fn copy_block(
    builder: &mut Builder,
    module: &Module,
    inst: Inst,
    block: Block,
    map: &HashMap<Value, Value>,
) -> Vec<Value> {
    let mut operands = Vec::with_capacity(module.operands(inst).len());
    for operand in module.operands(inst) {
        operands.push(map[operand]);
    }
    let block_body = module.regions(inst)[0];
    let created = builder.create_with_regions(block, operands, |_, body_builder, block_args| {
        let mut body_map: HashMap<Value, Value> = HashMap::new();
        for (&old, &new) in module.args(block_body).iter().zip(block_args) {
            body_map.insert(old, new);
        }
        for body_inst in module.insts(block_body) {
            body_builder
                .create_copy(module, body_inst, &mut body_map)
                .expect("a block's body reads its own values");
        }
    });
    created.to_vec()
}
