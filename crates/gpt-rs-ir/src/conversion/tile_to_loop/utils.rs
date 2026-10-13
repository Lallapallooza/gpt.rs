//! Building `loop` IR: index values, loops, conditionals, and scalar cells of private scratch.

use anyhow::Result;
use smallvec::SmallVec;

use crate::dialects::r#loop::{
    Alloc, For, Forall, If, Index, Level, Load, LoadOperands, Predicate, Store, StoreOperands,
    Yield,
};
use crate::dialects::tile::Body;
use crate::host::HostScalar;
use crate::ir::{
    Constraint, DType, Dim, Expr, IndexExpr, Ranked, ReadKind, Relation, Value, Values,
};
use crate::rewrite::Builder;

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
            .map(IndexExpr::iter)
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
pub(super) fn index(builder: &mut Builder<'_>, expr: IndexExpr, operands: &[Value]) -> Value {
    let simplified = expr.simplify();
    if let Some(position) = simplified.as_iter() {
        return operands[position];
    }
    let used = used_operands([&simplified]);
    let renumbered = renumber(&simplified, &used);
    builder.create(Index { expr: renumbered }, pick_operands(operands, &used))
}

/// The index `value`.
pub(super) fn constant(builder: &mut Builder<'_>, value: i64) -> Value {
    index(builder, IndexExpr::constant(value), &[])
}

/// The extent `dim`, which a launch binds when it is a symbol.
pub(super) fn extent(builder: &mut Builder<'_>, dim: &Dim) -> Value {
    index(builder, IndexExpr::dim(dim), &[])
}

/// Whether every one of `constraints` holds, as an `i1`. Their `di` is `operands[i]`, and they
/// read no operand element.
pub(super) fn predicate(
    builder: &mut Builder<'_>,
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

/// The constraint `lhs < rhs`.
pub(super) fn less(lhs: IndexExpr, rhs: IndexExpr) -> Constraint {
    Constraint::cmp(lhs, Relation::Less, rhs)
}

/// A `loop.for` from `start` up to `end`, whose body `body` builds from the index.
pub(super) fn for_range(
    builder: &mut Builder<'_>,
    start: Value,
    end: Value,
    body: impl FnOnce(&mut Builder<'_>, Value) -> Result<()>,
) -> Result<()> {
    let mut pending = Some(body);
    let mut outcome = Ok(());
    builder.create_with_regions(For, (start, end), |_, inner, args| {
        let build = pending.take().expect("a loop has one body");
        outcome = build(inner, args[0]);
        inner.create(Yield, ());
    });
    outcome
}

/// A `loop.forall` at `level` from `start` up to `end`, whose body `body` builds from the index of
/// the unit.
pub(super) fn forall_range(
    builder: &mut Builder<'_>,
    level: Level,
    start: Value,
    end: Value,
    body: impl FnOnce(&mut Builder<'_>, Value) -> Result<()>,
) -> Result<()> {
    let mut pending = Some(body);
    let mut outcome = Ok(());
    builder.create_with_regions(Forall { level }, (start, end), |_, inner, args| {
        let build = pending.take().expect("a forall has one body");
        outcome = build(inner, args[0]);
        inner.create(Yield, ());
    });
    outcome
}

/// A `loop.for` over the extent `dim`, from 0.
pub(super) fn for_extent(
    builder: &mut Builder<'_>,
    dim: &Dim,
    body: impl FnOnce(&mut Builder<'_>, Value) -> Result<()>,
) -> Result<()> {
    let start = constant(builder, 0);
    let end = extent(builder, dim);
    for_range(builder, start, end, body)
}

/// A `loop.if` on `condition`, whose body `body` builds.
pub(super) fn if_then(
    builder: &mut Builder<'_>,
    condition: Value,
    body: impl FnOnce(&mut Builder<'_>) -> Result<()>,
) -> Result<()> {
    let mut pending = Some(body);
    let mut outcome = Ok(());
    builder.create_with_regions(If, condition, |_, inner, _| {
        let build = pending.take().expect("an if has one body");
        outcome = build(inner);
        inner.create(Yield, ());
    });
    outcome
}

/// The element of `buffer` at `indices`, or `pad` where they fall outside it.
pub(super) fn load(
    builder: &mut Builder<'_>,
    buffer: Value,
    indices: Vec<Value>,
    pad: Option<crate::ir::F64>,
) -> Value {
    builder.create(Load { pad }, LoadOperands::<Values> { buffer, indices })
}

/// Stores `value` into `buffer` at `indices`.
pub(super) fn store(builder: &mut Builder<'_>, value: Value, buffer: Value, indices: Vec<Value>) {
    builder.create(
        Store,
        StoreOperands::<Values> {
            value,
            buffer,
            indices,
        },
    );
}

/// A private buffer of `dtype` with the extents `shape`.
pub(super) fn alloc(builder: &mut Builder<'_>, shape: &[usize], dtype: DType) -> Value {
    let shape: Ranked<usize> = shape.iter().copied().collect();
    builder.create(Alloc { shape, dtype }, ())
}

/// A cell of private scratch that holds one `dtype` value.
pub(super) fn cell(builder: &mut Builder<'_>, dtype: DType) -> Value {
    alloc(builder, &[], dtype)
}

/// The value of `cell`.
pub(super) fn get(builder: &mut Builder<'_>, cell: Value) -> Value {
    load(builder, cell, Vec::new(), None)
}

/// Writes `value` into `cell`.
pub(super) fn set(builder: &mut Builder<'_>, cell: Value, value: Value) {
    store(builder, value, cell, Vec::new());
}

/// The constant `value`.
pub(super) fn lit(builder: &mut Builder<'_>, value: HostScalar) -> Value {
    Body { builder }.lit(value)
}
