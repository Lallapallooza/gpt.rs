//! The ops of the `loop` dialect and their attributes.

use smallvec::SmallVec;

use crate::ir::{Constraint, DType, IndexExpr, Level, Ranked, Space, Type, F64};
use crate::targets::TargetRef;
use crate::traits::{IsolatedFromAbove, Terminator};

// ------------------------------------------------------------------------------------------------
// `loop.func` and `loop.return`.

/// `loop.func`: one launch of a dispatch. A top-level symbol, called by `host.dispatch` with
/// tensors and by `host.launch` with buffers.
///
/// - Its body's block arguments are its parameters, `global` buffers, and `access` says how it
///   uses each.
/// - It gives no results: it writes them into its `out` and `inout` parameters. A call with tensors
///   puts each argument into an `in` or `inout` parameter, in order, and gives the elements of each
///   `inout` or `out` parameter as a result, in order.
/// - When it names a `target`, each `loop.forall` of its body runs at an execution level of the
///   target, and each `loop.prefetch` asks a memory that the target prefetches into for a line.
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
    /// The target the dispatch's schedule was chosen for, which `loop-to-isa` writes vectors for.
    #[attr(default)]
    pub target: Option<TargetRef>,
    /// Whether a product and the sum it enters may round once, as one `arith.fma`: the dispatch's
    /// `numerics.contract`, which the `contract` pass realizes.
    #[attr(default)]
    pub contract: bool,
}

/// How a `loop.func` uses a parameter.
#[derive(crate::Attr, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Access {
    /// It reads the parameter and never writes it.
    In,
    /// It writes every element of the parameter, which holds nothing it reads before. With
    /// `over`, the parameter may share the buffer of the `in` parameter at that position. No other
    /// parameter may share that buffer, and the launch MUST read its elements only at positions
    /// that it has not written yet.
    Out {
        #[attr(default)]
        over: Option<usize>,
    },
    /// It updates the parameter in place: the parameter holds the elements of the argument when
    /// the launch starts, and those of the result when it ends.
    Inout,
}

/// `loop.return`: ends the body of a `loop.func`.
#[crate::op("loop.return", traits(Terminator))]
pub struct Return;

// ------------------------------------------------------------------------------------------------
// Control flow: `loop.for`, `loop.forall`, `loop.if`, `loop.atom` and `loop.yield`.

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

/// `loop.atom`: runs its body once for each point of the box `tile` whose coordinates are below
/// `extents`, one per axis, in row-major order. The points are those of a register tile, which
/// `loop-to-isa` computes at once in vectors of `lanes` elements along the last axis. The body's
/// block arguments are the point's coordinates. Each extent MUST be at least 1 and MUST NOT exceed
/// its axis of `tile`: an atom runs at least one point. Each point is a unit of work: two points
/// MUST NOT access one element of a buffer when one of them writes it.
#[crate::op(
    "loop.atom",
    operands(extents(Variadic(Index))),
    regions(body(Block(Yield)))
)]
pub struct Atom {
    pub tile: Ranked<usize>,
    pub lanes: usize,
}

/// `loop.yield`: ends the body of a `loop.for`, a `loop.forall`, a `loop.if` or a `loop.atom`.
#[crate::op("loop.yield", traits(Terminator))]
pub struct Yield;

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
// Buffers: `loop.alloc`, `loop.pick`, `loop.load`, `loop.store` and `loop.prefetch`.

/// `loop.alloc`: a buffer of `dtype` with the static extents `shape`, in `private` scratch or in
/// `reg` vector registers, which lives as long as the region that allocates it. No element holds a
/// value until a store writes it.
#[crate::op("loop.alloc", results(output(Buffer)), effects(alloc))]
pub struct Alloc {
    pub shape: Ranked<usize>,
    pub dtype: DType,
    #[attr(default)]
    pub space: Space,
}

/// `loop.pick`: the buffer of `buffers` at position `index`, which MUST be one of them. The
/// buffers are of one type, and the result is the picked buffer itself: a store into it writes
/// that buffer.
#[crate::op(
    "loop.pick",
    operands(index(Index), buffers(Variadic(Buffer))),
    results(output(Buffer))
)]
pub struct Pick;

/// `loop.load`: the element of `buffer` at `indices`, one per axis. Without `pad`, the position
/// MUST be inside the buffer, and the element MUST hold a value: the caller's, in an `in` or
/// `inout` parameter, or one a store wrote. With `pad v`, a position outside the buffer gives `v`,
/// a value of the buffer's dtype.
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

/// `loop.prefetch`: asks the memory `level`, one that the target prefetches into, for the line that
/// holds the element of `buffer` at `indices`, one per axis, which a later load reads. A hint: it
/// computes and changes nothing, and a position outside the buffer reads nothing.
#[crate::op(
    "loop.prefetch",
    operands(buffer(Buffer), indices(Variadic(Index))),
    effects(read(buffer))
)]
pub struct Prefetch {
    pub level: Level,
}

// ------------------------------------------------------------------------------------------------
// The dialect

#[derive(crate::Dialect, Debug, Clone, PartialEq, Eq, Hash)]
#[dialect(name = "loop", op = Loop)]
pub enum LoopOps {
    Func(Func),
    Return(Return),
    For(For),
    Forall(Forall),
    If(If),
    Atom(Atom),
    Yield(Yield),
    Index(Index),
    Predicate(Predicate),
    Alloc(Alloc),
    Pick(Pick),
    Load(Load),
    Store(Store),
    Prefetch(Prefetch),
}
