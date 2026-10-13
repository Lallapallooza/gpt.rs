//! The ops of the `loop` dialect and their attributes.

use smallvec::SmallVec;

use crate::ir::{Constraint, DType, IndexExpr, Ranked, Type, F64};
use crate::traits::{IsolatedFromAbove, Terminator};

// ------------------------------------------------------------------------------------------------
// `loop.func` and `loop.return`.

/// `loop.func`: one launch of a dispatch. A top-level symbol, called by `host.dispatch`.
///
/// - Its body's block arguments are its parameters, `global` buffers, and `access` says how it
///   uses each.
/// - It gives no results: it writes them into its `out` parameters. A call with tensors puts each
///   argument into an `in` parameter, in order, and gives the elements of each `out` parameter as a
///   result, in order.
#[crate::op(
    "loop.func",
    regions(body(Block(Return))),
    traits(IsolatedFromAbove),
    symbol
)]
pub struct Func {
    pub name: String,
    pub args: Vec<Type>,
    pub access: Vec<Access>,
}

/// How a `loop.func` uses a parameter.
#[derive(crate::Attr, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Access {
    /// It reads the parameter and never writes it.
    In,
    /// It writes every element of the parameter, which holds nothing it reads before.
    Out,
}

/// `loop.return`: ends the body of a `loop.func`.
#[crate::op("loop.return", traits(Terminator))]
pub struct Return;

// ------------------------------------------------------------------------------------------------
// Control flow: `loop.for`, `loop.if` and `loop.yield`.

/// `loop.for`: runs its body once for each index from `start` up to, not including, `end`, in
/// increasing order. The body's block argument is the index. It runs no time when `end` is not
/// above `start`.
#[crate::op(
    "loop.for",
    operands(start(Index), end(Index)),
    regions(body(Block(Yield)))
)]
pub struct For;

/// `loop.forall`: runs its body once for each index from `start` up to, not including, `end`, as
/// units of work at `level`, which may run at the same time and in any order. The body's block
/// argument is the index. Two units MUST NOT access one element of a buffer when one of them
/// writes it, so every order gives the same values.
#[crate::op(
    "loop.forall",
    operands(start(Index), end(Index)),
    regions(body(Block(Yield)))
)]
pub struct Forall {
    pub level: Level,
}

/// `loop.if`: runs its body when `condition`, an `i1`, holds.
#[crate::op("loop.if", operands(condition(Scalar)), regions(then(Block(Yield))))]
pub struct If;

/// `loop.yield`: ends the body of a `loop.for`, a `loop.forall` or a `loop.if`.
#[crate::op("loop.yield", traits(Terminator))]
pub struct Yield;

/// An execution level of a CPU, from the outermost.
#[derive(crate::Attr, Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Level {
    /// The cores that share one last-level cache.
    L3,
    /// One core.
    Core,
}

// ------------------------------------------------------------------------------------------------
// Index values: `loop.index` and `loop.predicate`.

/// `loop.index`: the value of `expr`, an expression of the index language in which `di` is
/// operand `i` and `{T}` the extent the launch binds to the symbol `T`.
#[crate::op("loop.index", operands(ins(Variadic(Index))), results(output(Index)))]
pub struct Index {
    pub expr: IndexExpr,
}

/// `loop.predicate`: whether every one of `constraints` holds, as an `i1`. The constraints are
/// those of a `tile.block`, in which `di` is operand `i`, and read no operand element.
#[crate::op(
    "loop.predicate",
    operands(ins(Variadic(Index))),
    results(output(Scalar))
)]
pub struct Predicate {
    pub constraints: SmallVec<[Constraint; 1]>,
}

// ------------------------------------------------------------------------------------------------
// Buffers: `loop.alloc`, `loop.load` and `loop.store`.

/// `loop.alloc`: a `private` buffer of `dtype` with the static extents `shape`, which lives as long
/// as the region that allocates it. No element holds a value until a store writes it.
#[crate::op("loop.alloc", results(output(Buffer)), effects(alloc))]
pub struct Alloc {
    pub shape: Ranked<usize>,
    pub dtype: DType,
}

/// `loop.load`: the element of `buffer` at `indices`, one per axis. Without `pad`, the position
/// MUST be inside the buffer, and a store MUST have written the element. With `pad v`, a position
/// outside the buffer gives `v`, a value of the buffer's dtype.
#[crate::op(
    "loop.load",
    operands(buffer(Buffer), indices(Variadic(Index))),
    results(output(Scalar)),
    effects(read(buffer))
)]
pub struct Load {
    #[attr(default)]
    pub pad: Option<F64>,
}

/// `loop.store`: writes `value` into `buffer` at `indices`, one per axis, which MUST be a position
/// inside the buffer.
#[crate::op(
    "loop.store",
    operands(value(Scalar), buffer(Buffer), indices(Variadic(Index))),
    effects(write(buffer))
)]
pub struct Store;

// ------------------------------------------------------------------------------------------------
// The dialect

#[derive(crate::Dialect, Debug, Clone, PartialEq, Eq, Hash)]
#[dialect(name = "loop", op = Loop)]
pub enum LoopOp {
    Func(Func),
    Return(Return),
    For(For),
    Forall(Forall),
    If(If),
    Yield(Yield),
    Index(Index),
    Predicate(Predicate),
    Alloc(Alloc),
    Load(Load),
    Store(Store),
}
