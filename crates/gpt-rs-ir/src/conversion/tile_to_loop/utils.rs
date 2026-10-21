//! Building `loop` IR: index values, loops, conditionals, and scalar cells of private scratch.

use anyhow::Result;
use smallvec::SmallVec;

use crate::dialects::arith::ScalarBuilder;
use crate::dialects::loops::{
    Alloc, Atom, For, Forall, If, Index, Load, LoadOperands, Pick, PickOperands, Predicate,
    Prefetch, PrefetchOperands, Store, StoreOperands, Yield,
};
use crate::host::HostScalar;
use crate::ir::{
    Constraint, DType, Dim, Expr, IndexExpr, Level, Ranked, ReadKind, Space, Value, Values,
    MAX_GROUP_RANK,
};
use crate::rewrite::{Build, Builder};

/// The positions of the operands that `exprs` name, in increasing order.
fn used_operands<'e, R: ReadKind + 'e>(exprs: impl IntoIterator<Item = &'e Expr<R>>) -> Vec<usize> {
    let mut used = Vec::new();
    for expr in exprs {
        for position in expr.iters() {
            if !used.contains(&position) {
                used.push(position);
            }
        }
    }
    used.sort_unstable();
    used
}

/// `expr` with each operand position replaced by its place in `used`, simplified.
fn renumber<R: ReadKind>(expr: &Expr<R>, used: &[usize]) -> Expr<R> {
    let slot = |position: usize| {
        used.iter()
            .position(|&used_position| used_position == position)
            .map(Expr::iter)
    };
    expr.substitute(&slot).simplify()
}

/// The operands at `used`.
fn pick_operands(operands: &[Value], used: &[usize]) -> Vec<Value> {
    let mut picked = Vec::with_capacity(used.len());
    for &position in used {
        picked.push(operands[position]);
    }
    picked
}

/// The value of `expr`, whose `di` is `operands[i]`: the operand itself when `expr` is one.
pub(super) fn index(builder: &mut Builder, expr: IndexExpr, operands: &[Value]) -> Value {
    let simplified = expr.simplify();
    if let Some(position) = simplified.as_iter() {
        return operands[position];
    }
    let used = used_operands([&simplified]);
    let renumbered = renumber(&simplified, &used);
    builder.create(Index { expr: renumbered }, pick_operands(operands, &used))
}

/// The points from `start` up to `end`: `end - start`.
pub(super) fn points_between(builder: &mut Builder, start: Value, end: Value) -> Value {
    index(
        builder,
        IndexExpr::iter(1) - IndexExpr::iter(0),
        &[start, end],
    )
}

/// The offset of `value` from `start`: `value - start`.
pub(super) fn offset_from(builder: &mut Builder, value: Value, start: Value) -> Value {
    index(
        builder,
        IndexExpr::iter(0) - IndexExpr::iter(1),
        &[value, start],
    )
}

/// The index `value`.
pub(super) fn constant(builder: &mut Builder, value: i64) -> Value {
    index(builder, IndexExpr::constant(value), &[])
}

/// The extent `dim`, which a launch binds when it is a symbol.
pub(super) fn extent(builder: &mut Builder, dim: &Dim) -> Value {
    index(builder, IndexExpr::dim(dim), &[])
}

/// Whether every one of `constraints` holds, as an `i1`. Their `di` is `operands[i]`, and they
/// read no operand element.
pub(super) fn predicate(
    builder: &mut Builder,
    constraints: &[Constraint],
    operands: &[Value],
) -> Value {
    let used = used_operands(constraints.iter().flat_map(Constraint::exprs));
    let mut renumbered = SmallVec::new();
    for constraint in constraints {
        renumbered.push(constraint.map_exprs(&|expr| renumber(expr, &used)));
    }
    builder.create(
        Predicate {
            constraints: renumbered,
        },
        pick_operands(operands, &used),
    )
}

/// A `loop.for` from `start` up to `end`, whose body `body` builds from the index.
pub(super) fn for_range(
    builder: &mut Builder,
    start: Value,
    end: Value,
    body: impl FnOnce(&mut Builder, Value) -> Result<()>,
) -> Result<()> {
    builder.create_with_body(For, (start, end), |inner, args| {
        body(inner, args[0])?;
        inner.create(Yield, ());
        Ok(())
    })?;
    Ok(())
}

/// A `loop.forall` at `level` from `start` up to `end`, whose body `body` builds from the index of
/// the unit.
pub(super) fn forall_range(
    builder: &mut Builder,
    level: Level,
    start: Value,
    end: Value,
    body: impl FnOnce(&mut Builder, Value) -> Result<()>,
) -> Result<()> {
    builder.create_with_body(Forall { level }, (start, end), |inner, args| {
        body(inner, args[0])?;
        inner.create(Yield, ());
        Ok(())
    })?;
    Ok(())
}

/// The axes of a box of points that one `loop.atom` walks: the tile it holds of each, in vectors
/// of `lanes` lanes along the last, and how many points of each it computes, at most the tile's,
/// none past the tile's axes.
pub(super) struct AtomBox {
    pub(super) tile: Ranked<usize>,
    pub(super) lanes: usize,
    pub(super) counts: [Option<Value>; MAX_GROUP_RANK],
}

/// What [`each_point`] builds at each point of a box: from the builder there and the point's
/// offset along each axis.
pub(super) type PointBody<'a> = dyn FnMut(&mut Builder<'_>, &[Value]) -> Result<()> + 'a;

/// Builds `body` at each point of a box, from the point's offset along each of its axes: a
/// `loop.for` from 0 over each of the counts `loops`, the first outermost, then, with `atom`, one
/// `loop.atom` over the axes it gives, inside them.
pub(super) fn each_point(
    builder: &mut Builder,
    loops: &[Value],
    atom: Option<&AtomBox>,
    body: &mut PointBody,
) -> Result<()> {
    each_offset(builder, loops, atom, &mut Vec::new(), body)
}

fn each_offset(
    builder: &mut Builder,
    loops: &[Value],
    atom: Option<&AtomBox>,
    offsets: &mut Vec<Value>,
    body: &mut PointBody,
) -> Result<()> {
    let Some((&count, rest)) = loops.split_first() else {
        let Some(atom_box) = atom else {
            return body(builder, offsets);
        };
        let op = Atom {
            tile: atom_box.tile,
            lanes: atom_box.lanes,
        };
        let counts: Vec<Value> = atom_box.counts.iter().flatten().copied().collect();
        builder.create_with_body(op, counts, |inner, args| {
            body(inner, &[&offsets[..], args].concat())?;
            inner.create(Yield, ());
            Ok(())
        })?;
        return Ok(());
    };
    let zero = constant(builder, 0);
    for_range(builder, zero, count, |loop_builder, offset| {
        offsets.push(offset);
        let built = each_offset(loop_builder, rest, atom, offsets, body);
        offsets.pop();
        built
    })
}

/// A `loop.if` on `condition`, whose body `body` builds.
pub(super) fn if_then(
    builder: &mut Builder,
    condition: Value,
    body: impl FnOnce(&mut Builder) -> Result<()>,
) -> Result<()> {
    builder.create_with_body(If, condition, |inner, _| {
        body(inner)?;
        inner.create(Yield, ());
        Ok(())
    })?;
    Ok(())
}

/// The element of `buffer` at `indices`, or `pad` where they fall outside it.
pub(super) fn load(
    builder: &mut Builder,
    buffer: Value,
    indices: Vec<Value>,
    pad: Option<crate::ir::F64>,
) -> Value {
    builder.create(Load { pad }, LoadOperands::<Values> { buffer, indices })
}

/// The buffer of `buffers` at position `index`.
pub(super) fn pick(builder: &mut Builder, index: Value, buffers: Vec<Value>) -> Value {
    builder.create(Pick, PickOperands::<Values> { index, buffers })
}

/// Stores `value` into `buffer` at `indices`.
pub(super) fn store(builder: &mut Builder, value: Value, buffer: Value, indices: Vec<Value>) {
    builder.create(
        Store,
        StoreOperands::<Values> {
            value,
            buffer,
            indices,
        },
    );
}

/// Asks the cache `level` for the element of `buffer` at `indices`.
pub(super) fn prefetch(builder: &mut Builder, buffer: Value, indices: Vec<Value>, level: Level) {
    builder.create(
        Prefetch { level },
        PrefetchOperands::<Values> { buffer, indices },
    );
}

/// A buffer of `space` and `dtype` with the extents `extents`.
pub(super) fn alloc(builder: &mut Builder, extents: &[usize], dtype: DType, space: Space) -> Value {
    let shape: Ranked<usize> = extents.iter().copied().collect();
    builder.create(
        Alloc {
            shape,
            dtype,
            space,
        },
        (),
    )
}

/// A cell of private scratch that holds one `dtype` value.
pub(super) fn cell(builder: &mut Builder, dtype: DType) -> Value {
    alloc(builder, &[], dtype, Space::Private)
}

/// The value of `cell`.
pub(super) fn get(builder: &mut Builder, cell: Value) -> Value {
    load(builder, cell, Vec::new(), None)
}

/// Writes `value` into `cell`.
pub(super) fn set(builder: &mut Builder, cell: Value, value: Value) {
    store(builder, value, cell, Vec::new());
}

/// The constant `value`.
pub(super) fn lit(builder: &mut Builder, value: HostScalar) -> Value {
    ScalarBuilder { builder }.lit(value)
}
