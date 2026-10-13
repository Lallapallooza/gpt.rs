//! The ops of the `tile` dialect and their attributes.

use std::collections::BTreeMap;

use anyhow::Result;
use smallvec::SmallVec;
use tinyvec::ArrayVec;

use crate::dialects::func::Return;
use crate::dialects::r#loop::Level;
use crate::ir::{
    AccessExpr, Constraint, DType, Dim, IndexExpr, IterRef, OperandRef, Type, F64, MAX_ITERS,
};
use crate::traits::{IsolatedFromAbove, Terminator};
use crate::Attr;

// ------------------------------------------------------------------------------------------------
// `tile.block`, `tile.index` and `tile.yield`, and the attributes of a block.

/// `tile.block`: a generalized reduction over the iteration domain `iters`, restricted by
/// `constraints`.
///
/// - Its iterators are named by their position: `d0` is the first of `iters`.
/// - At each point of the domain, every read gives the body one scalar: an element of an operand
///   at index expressions of the iterators. Read `i` is the body's argument `i`. A read with
///   `pad v` gives `v` where its coordinates fall outside the operand; a read without it MUST stay
///   inside.
/// - The body computes scalars in `arith` and yields them. `tile.index` gives it the value of an
///   index expression at the point.
/// - `folds` splits the yielded values into groups, in yield order: each fold takes the values its
///   combine kind folds over the reduction iterators, and gives the results its writes place, in
///   result order. A point where one of the `constraints` fails contributes nothing.
/// - Each write is `[c, ...]`, the coordinates of a new tensor, each a parallel or prefix iterator
///   or `0`; or `$k[e, ...]`, an updated copy of operand `k` at those index expressions. Reads see
///   every operand as it was before the block.
/// - The reference order folds the points in lexicographic order of `iters`, the first one
///   outermost, rounding each step to the result dtype.
#[crate::op(
    "tile.block",
    operands(ins(Variadic(Tensor))),
    results(outs(Variadic(Tensor))),
    regions(body(Block(Yield))),
    traits(IsolatedFromAbove)
)]
pub struct Block {
    pub iters: Iters,
    pub reads: SmallVec<[Read; 4]>,
    pub folds: SmallVec<[Fold; 2]>,
    #[attr(default)]
    pub constraints: SmallVec<[Constraint; 1]>,
}

/// One group of the values that the body of a `tile.block` yields: the combine kind that folds
/// them over the reduction iterators, the write of each of its results, and the operands that
/// hold the state the fold starts from, one per state component, or none. Its text form is
/// `kind -> write` or `kind -> (write, ...)`, then `init $k` or `init ($k, ...)` when it has a
/// start state.
#[derive(crate::Attr, Debug, Clone, PartialEq, Eq, Hash)]
#[attr(format = "$combine `->` tuple($writes) (`init` tuple($init)^)?")]
pub struct Fold {
    pub combine: CombineKind,
    pub writes: SmallVec<[Write; 1]>,
    #[attr(default)]
    pub init: SmallVec<[OperandRef; 1]>,
}

/// `tile.index`: the value of `expr` at the current point of the enclosing block, converted to
/// `dtype` as `tensor.iota` converts its indices. The expression reads no operand: integer `arith`
/// on index values folds into it.
#[crate::op("tile.index", results(output(Scalar)))]
pub struct Index {
    pub expr: IndexExpr,
    pub dtype: DType,
}

/// `tile.yield`: ends the body of a `tile.block` or `tile.group`. Its operands are
/// the values the region gives.
#[crate::op("tile.yield", operands(values(Variadic(Any))), traits(Terminator))]
pub struct Yield;

/// Which operand a read takes its elements from.
#[derive(crate::Attr, Debug, Clone, PartialEq, Eq, Hash)]
#[attr(via = super::utils::SourceText)]
pub enum Source {
    /// Operand `i`.
    Operand(usize),
    /// One of the operands `first..first + count`, all of one type, picked at each point by the
    /// value of `pick`: per-expert weights that stay separate tensors.
    List {
        first: usize,
        count: usize,
        pick: Box<AccessExpr>,
    },
}

/// The coordinates of a read or a write, one per axis.
pub type Coords = SmallVec<[AccessExpr; 4]>;

/// A read of a `tile.block`: an operand element at index expressions, one per axis. With `pad`, a
/// point whose coordinates fall outside the operand reads that value; without it, every point
/// MUST be inside.
#[derive(crate::Attr, Debug, Clone, PartialEq, Eq, Hash)]
#[attr(format = "$source `` $coords (`pad` $pad^)?")]
pub struct Read {
    pub source: Source,
    pub coords: Coords,
    pub pad: Option<F64>,
}

/// Where a `tile.block` writes one of its results: the coordinates of a new tensor, or of an
/// operand that the result updates (`into`).
#[derive(crate::Attr, Debug, Clone, PartialEq, Eq, Hash)]
#[attr(format = "($into^)? `` $coords")]
pub struct Write {
    /// The operand that the result is an updated copy of.
    pub into: Option<OperandRef>,
    pub coords: Coords,
}

/// Iterators by position, written by name: `[d0, d2]`.
#[derive(crate::Attr, Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
#[attr(format = "$0")]
pub struct IterList(pub(super) ArrayVec<[IterRef; MAX_ITERS]>);

/// How a `tile.block` treats an iterator.
#[derive(crate::Attr, Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum IterKind {
    /// The results are indexed by it.
    #[default]
    Par,
    /// The combine folds it.
    Red,
    /// The combine folds it in order: no schedule may split or reorder it.
    RedOrdered,
    /// The combine folds it, and every prefix of the fold is a result.
    Prefix,
}

/// An iterator of a `tile.block`: a kind and an extent, `red 8`. Its position in the block's list
/// names it: `d0` is the first.
#[derive(crate::Attr, Debug, Clone, Default, PartialEq, Eq, Hash)]
#[attr(format = "$kind $extent")]
pub struct Iter {
    pub kind: IterKind,
    pub extent: Dim,
}

/// The iterators of a `tile.block`, at most [`MAX_ITERS`].
pub type Iters = ArrayVec<[Iter; MAX_ITERS]>;

// ------------------------------------------------------------------------------------------------
// Combine kinds: the closed set of folds over the reduction iterators of a `tile.block`.
//
// Each kind says how many values the block body yields for it, how many results it gives and of
// which dtypes, the dimensions it appends, and whether reassociating its fold keeps every bit: its
// `CombineRule`. The interpreter's `Acc` is its fold.

/// How a fold of `tile.block` combines the values the body yields over its reduction iterators.
/// [`CombineKind::rule`] gives what the kind takes and gives.
#[derive(Attr, Debug, Clone, PartialEq, Eq, Hash)]
pub enum CombineKind {
    /// No fold: the result takes the value of each point, the last one where points collide.
    None,
    /// The fold of `tensor.reduce_sum`: `add`, from 0.
    Sum,
    /// The fold of `tensor.reduce_max`: `maximum`, from `-inf` or the smallest integer.
    Max,
    /// The fold of `tensor.reduce_min`: `minimum`, from `+inf` or the largest integer.
    Min,
    /// Bitwise and, from all ones.
    And,
    /// Bitwise or, from 0.
    Or,
    /// The body yields `(value, index)`. The results are the largest value and its index: ties
    /// go to the smaller index, and NaN is larger than every number, as in `tensor.argmax`.
    Argmax,
    /// As `argmax`, with the smallest value.
    Argmin,
    /// The body yields `(value, index)`. The results are the `k` largest values, largest first,
    /// and their indices, along a new last dimension of extent `k`. Of equal values, the smaller
    /// index comes first, as in `tensor.top_k`.
    TopK(KBest<true>),
    /// As `top_k`, with the `k` smallest values, smallest first, as in `tensor.bottom_k`.
    BottomK(KBest<false>),
    /// The body yields `(score, value)`. The results are the state of the softmax-weighted sum:
    /// `m = max_j s_j`, `l = sum_j exp(s_j - m)` and `o = sum_j exp(s_j - m) v_j`, the weighted sum
    /// being `o / l`.
    OnlineSoftmax(Softmax<false>),
    /// The body yields states `(m, l, o)` of `online_softmax`, which merge in point order into one.
    OnlineSoftmaxMerge(Softmax<true>),
    /// The body yields a score. The results are the state `m = max_j s_j` and
    /// `l = sum_j exp(s_j - m)`, the log-sum-exp being `m + log(l)`.
    Logsumexp,
    /// The body yields states `(m, l)` of `logsumexp`, which merge in point order into one.
    LogsumexpMerge,
}

/// The attribute of `top_k` and `bottom_k`: how many values they keep. `LARGEST` is whether they
/// keep the largest.
#[derive(crate::AttrGroup, Debug, Clone, PartialEq, Eq, Hash)]
pub struct KBest<const LARGEST: bool> {
    pub k: usize,
}

/// The attribute of `online_softmax` and its merge: the parallel iterators that only the value
/// depends on, which the `m` and `l` states leave out. `MERGE` is whether the body yields states.
#[derive(crate::AttrGroup, Debug, Clone, Default, PartialEq, Eq, Hash)]
pub struct Softmax<const MERGE: bool> {
    #[attr(default)]
    pub lanes: IterList,
}

/// What a combine kind of `tile.block` takes and gives. Each kind has a type that implements it,
/// which [`CombineKind::rule`] gives.
pub trait CombineRule {
    /// How many values the body yields for it.
    fn yields(&self) -> usize;
    /// How many results it gives.
    fn results(&self) -> usize;
    /// The dtypes of the results, given the dtypes the body yields for it.
    fn result_dtypes(&self, yields: &[DType]) -> Result<Vec<DType>>;
    /// The dimensions it appends to result `result`: `k` for `top_k`.
    fn extra_dims(&self, result: usize) -> Vec<Dim>;
    /// The parallel iterators that only the value depends on.
    fn lanes(&self) -> IterList;
    /// Whether result `result` leaves the lanes out of its write.
    fn drops_lanes(&self, result: usize) -> bool;
    /// Whether folding points in any grouping and order gives the same bits for elements of
    /// `dtype`.
    fn exact(&self, dtype: DType) -> bool;
    /// How many init values it takes: one per state component, and one per result.
    fn init_count(&self) -> usize;
    /// Whether it folds into the elements of an operand that a write updates.
    fn folds_into_operand(&self) -> bool;
}

// ------------------------------------------------------------------------------------------------
// `tile.group` and `tile.dispatch`.

/// `tile.group`: blocks that become one generated function, while dispatches are formed. Its
/// results are the tensors its body yields.
///
/// - The body holds `tile.block`s, which read values of the enclosing function, and ends with
///   `tile.yield`.
/// - `grid` lists the parallel iterators, by position, that every block of the group shares and
///   that a schedule may split across workers.
#[crate::op(
    "tile.group",
    results(outs(Variadic(Tensor))),
    regions(body(Block(Yield)))
)]
pub struct Group {
    #[attr(default)]
    pub grid: IterList,
}

/// `tile.dispatch`: one fused group of blocks, which becomes one generated function. A top-level
/// symbol, called by `host.dispatch`.
///
/// - Its body's block arguments are its arguments, and its `func.return` gives its results.
/// - `grid` lists the parallel iterators, by position, that its blocks share. `facts` gives a
///   range and a divisor of each dimension symbol of its arguments.
/// - `schedule` says how it runs, which changes no value it computes.
#[crate::op(
    "tile.dispatch",
    regions(body(Block(Return))),
    traits(IsolatedFromAbove),
    symbol
)]
pub struct Dispatch {
    pub name: String,
    pub args: Vec<Type>,
    pub results: Vec<Type>,
    #[attr(default)]
    pub grid: IterList,
    #[attr(default)]
    pub facts: BTreeMap<String, SymFact>,
    #[attr(default)]
    pub schedule: Schedule,
}

/// What a dispatch knows of a dimension symbol: a range of extents, and a divisor of every one,
/// `[1, 64] % 8`.
#[derive(crate::Attr, Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[attr(format = "`[` $min `,` $max `]` (`%` $divisor^)?")]
pub struct SymFact {
    pub min: u64,
    pub max: u64,
    #[attr(default = 1)]
    pub divisor: u64,
}

/// The schedule record of a dispatch, `#sched<nest = [...]>`: how it runs. `tile-to-loop` realizes
/// each field and makes no decision of its own, and a field left out is the reference: one
/// sequential loop per iterator, in the order of the block. Fields name the iterators of the
/// dispatch's block by position, as `d0`.
#[derive(crate::Attr, Debug, Clone, Default, PartialEq, Eq, Hash)]
#[attr(format = "`#sched` `<` struct($nest) `>`")]
pub struct Schedule {
    /// Loops over tiles of the iterators, outermost first. Inside them, a loop walks the points of
    /// each iterator's innermost tile: those of the parallel iterators first, then those of the
    /// reduced ones, each in the order of the block.
    #[attr(default)]
    pub nest: Vec<Loop>,
}

/// One loop of a nest: over the tiles of `extent` points of iterator `iter`, within its tile of
/// the loop around it, or within its whole extent. The last tile of a range may be smaller:
/// `d1 by 64 : par<level = core>`.
#[derive(crate::Attr, Debug, Clone, PartialEq, Eq, Hash)]
#[attr(format = "$iter `by` $extent `:` $parallelism")]
pub struct Loop {
    pub iter: IterRef,
    pub extent: usize,
    pub parallelism: Parallelism,
}

/// How a loop of a nest runs its tiles.
#[derive(crate::Attr, Debug, Clone, PartialEq, Eq, Hash)]
pub enum Parallelism {
    /// One after another, in increasing order.
    Seq,
    /// As units of work at an execution level, which may run at the same time.
    Par(ParLoop),
}

/// Where a parallel loop runs its tiles: on units of `level`, `units` of them, each running a
/// contiguous run of tiles, or one unit per tile.
#[derive(crate::AttrGroup, Debug, Clone, PartialEq, Eq, Hash)]
pub struct ParLoop {
    pub level: Level,
    #[attr(default)]
    pub units: Option<usize>,
}

// ------------------------------------------------------------------------------------------------
// The dialect

#[derive(crate::Dialect, Debug, Clone, PartialEq, Eq, Hash)]
#[dialect(name = "tile", op = Tile)]
pub enum TileOp {
    // A block's iterators, reads and folds make it several times the size of any other op. Boxed,
    // it leaves the other ops their size.
    Block(Box<Block>),
    Index(Index),
    Yield(Yield),
    Group(Group),
    Dispatch(Dispatch),
}
