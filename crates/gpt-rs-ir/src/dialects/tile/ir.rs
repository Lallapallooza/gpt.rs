//! The ops of the `tile` dialect and their attributes.

use std::collections::BTreeMap;

use anyhow::Result;
use smallvec::SmallVec;
use tinyvec::ArrayVec;

use crate::dialects::func::Return;
use crate::ir::{
    AccessExpr, Constraint, DType, Dim, IndexExpr, IterRef, Level, OperandRef, Type, F64, MAX_ITERS,
};
use crate::targets::TargetRef;
use crate::traits::{IsolatedFromAbove, Terminator};
use crate::{Attr, Op};

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
/// - `loops` says how the block runs, which changes no value it computes: its modes, outermost
///   first, each cutting one iterator into pieces (see [`Mode`]). Every iterator outside the
///   register tile ends in a mode of one point, so the modes say every loop. Left out, the block
///   walks each iterator one point at a time, in this order:
///   1. the iterators that are not reduced and that the coordinates of its first write take, in
///      the order they take them;
///   2. the other iterators that are not reduced, in the order of `iters`;
///   3. the reduced ones, in the order of `iters`.
///
///   When a fold updates an operand where its points may collide, all come in the order of
///   `iters`.
/// - `clip` names the iterators whose loops skip the points where a padded read falls outside its
///   operand, when such a point contributes nothing the fold keeps.
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
    #[attr(default)]
    pub loops: Vec<Mode>,
    #[attr(default)]
    pub clip: IterList,
}

/// One group of the values that the body of a `tile.block` yields: the combine kind that folds
/// them over the reduction iterators, the write of each of its results, the operands that hold the
/// state the fold starts from, one per state component, or none, and where the block keeps the
/// state while its loops walk the points (see [`StateHolding`]). Its text form is `kind -> write`
/// or `kind -> (write, ...)`, then `init $k` or `init ($k, ...)` when it has a start state, then
/// `held scratch<at = 2>` when it holds its state.
#[derive(crate::Attr, Debug, Clone, PartialEq, Eq, Hash)]
#[attr(format = "$combine `->` tuple($writes) (`init` tuple($init)^)? (`held` $held^)?")]
pub struct Fold {
    pub combine: CombineKind,
    pub writes: SmallVec<[Write; 1]>,
    #[attr(default)]
    pub init: SmallVec<[OperandRef; 1]>,
    pub held: Option<StateHolding>,
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
/// MUST be inside. With `at`, the operand is the result of the block before it in a dispatch,
/// and this block runs in the loops of the first block of its run where the placement says (see
/// [`Placement`]). With `held`, the block keeps what the read takes where the holding says.
#[derive(crate::Attr, Debug, Clone, PartialEq, Eq, Hash)]
#[attr(format = "$source `` $coords (`pad` $pad^)? (`at` $at^)? (`held` $held^)?")]
pub struct Read {
    pub source: Source,
    pub coords: Coords,
    pub pad: Option<F64>,
    pub at: Option<Placement>,
    pub held: Option<Holding>,
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
// The schedule of a block: the modes of its loops, and the placements and the holdings of its reads.

/// One mode of a block's loops: iterator `iter` in pieces of `size` points, within its piece of
/// the mode around it over the same iterator, or its whole extent; the last piece may hold fewer.
/// The pieces run one after another, or as `kind` says: `d2 by 64`,
/// `d0 by 8 : workers<level = core>`, `d1 by 16 : lanes`. A mode that walks the pieces that a
/// memory after the registers holds names it (`d5 by 128 in l1`): the schedule pass sizes its
/// pieces so that one of them, and what the loops inside it read there, fits that memory. The
/// verifier checks only that the memory is a cache or scratch of the dispatch's target.
#[derive(crate::Attr, Debug, Clone, PartialEq, Eq, Hash)]
#[attr(format = "$iter `by` $size (`:` $kind^)? (`in` $memory^)?")]
pub struct Mode {
    pub iter: IterRef,
    pub size: usize,
    pub kind: Option<ModeKind>,
    pub memory: Option<Level>,
}

/// How a mode runs its pieces, other than one after another.
#[derive(crate::Attr, Debug, Clone, PartialEq, Eq, Hash)]
pub enum ModeKind {
    /// Each on a worker of the execution level `level`, at the same time.
    Workers(Workers),
    /// All the points of one piece at once, in registers: the register tile holds `size` points of
    /// the iterator. The `reg` and `lanes` modes are the innermost.
    Reg,
    /// The lanes of a vector, `size` of them, along the iterator: the register tile's last
    /// iterator, which a `reg` mode of it holds several vectors of, or else one.
    Lanes,
}

/// The workers a mode runs its pieces on: those of the execution level `level`.
#[derive(crate::AttrGroup, Debug, Clone, PartialEq, Eq, Hash)]
pub struct Workers {
    pub level: Level,
}

/// Where a placed block runs: in a piece of the loops of the first block of its run, its register
/// tile (`reg`), the points it computes at once, one for a block that computes on scalars; or a
/// piece of a mode of those loops by its position (`at 2`). A run is a block and the placed
/// blocks after it, each reading the result of the block before it, all placed at one piece.
///
/// - The block runs once the block before it has finished the elements of its result in each
///   such piece, on each of its own points whose reads of that result take only those elements:
///   at the point where the block before finishes an element, when it reads that element alone,
///   and over the piece once it is finished otherwise.
/// - Its iterators are the first iterators of the run's first block, parallel in both and of the
///   same extents; a point of it is the point of the run's loops there.
#[derive(crate::Attr, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Placement {
    Reg,
    #[attr(format = "$0")]
    Mode(usize),
}

/// Where a block keeps what a read takes from its operand while its loops walk it: a copy in its
/// scratch, sized for the memory `level`, that each piece of mode `at` of its loops makes, or the
/// block before its loops (`copy<level = l1, at = 2>`); or, for an operand it reads in place, a
/// request to `level` for the elements of the next piece of mode `at`
/// (`prefetch<level = l2, at = 3>`).
#[derive(crate::Attr, Debug, Clone, PartialEq, Eq, Hash)]
pub enum Holding {
    Copy(Held),
    Prefetch(Held),
}

/// The memory a holding is for, and the mode of the block's loops where it is made.
#[derive(crate::AttrGroup, Debug, Clone, PartialEq, Eq, Hash)]
pub struct Held {
    pub level: Level,
    #[attr(default)]
    pub at: Option<usize>,
}

/// Where a fold keeps its state while the block's loops walk the points: in a scratch started
/// before the loop of mode `at`, in each piece of the modes around it, and finished after that
/// loop (`scratch<at = 2>`). It holds one element for each point of the parallel iterators that
/// the modes from `at` on walk. The register tile folds into a copy in registers of its part of the
/// scratch while the loops inside walk only the register tile and reduced iterators. A fold that
/// holds none keeps one element at a time, of each point of the register tile where the block has
/// one, from its first loop over a reduced iterator on.
#[derive(crate::Attr, Debug, Clone, PartialEq, Eq, Hash)]
pub enum StateHolding {
    Scratch(ScratchAt),
}

/// The mode of the block's loops where a fold's scratch is made.
#[derive(crate::AttrGroup, Debug, Clone, PartialEq, Eq, Hash)]
pub struct ScratchAt {
    pub at: usize,
}

// ------------------------------------------------------------------------------------------------
// Combine kinds: the closed set of folds over the reduction iterators of a `tile.block`.
//
// Each kind says how many values the block body yields for it, how many results it gives and of
// which dtypes, the dimensions it appends, and whether reassociating its fold keeps every bit: its
// `CombineRule`.

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
#[expect(
    clippy::min_ident_chars,
    reason = "the text form prints the field `k` by name, as in `top_k<k = 3>`"
)]
pub struct KBest<const LARGEST: bool> {
    pub k: usize,
}

/// The attribute of `online_softmax` and its merge: the value's iterators, the parallel
/// iterators that only the value depends on, which the `m` and `l` states leave out. `MERGE` is
/// whether the body yields states.
#[derive(crate::AttrGroup, Debug, Clone, Default, PartialEq, Eq, Hash)]
pub struct Softmax<const MERGE: bool> {
    #[attr(default)]
    pub value_iters: IterList,
}

/// What a combine kind of `tile.block` takes and gives. Each kind has a type that implements it,
/// which [`CombineKind::rule`] gives. A kind states what differs from the defaults: one yielded
/// value and one result, of no appended dimension and no value's iterators, folded exactly in one
/// sweep, from a start state of one value per result, into new results only.
pub trait CombineRule {
    /// How many values the body yields for it.
    fn yields(&self) -> usize {
        1
    }
    /// How many results it gives.
    fn results(&self) -> usize {
        1
    }
    /// The dtypes of the results, given the dtypes the body yields for it. `kind`, the kind's
    /// name, names it in errors.
    fn result_dtypes(&self, kind: &str, yields: &[DType]) -> Result<Vec<DType>>;
    /// The dimensions it appends to result `result`: `k` for `top_k`.
    fn extra_dims(&self, _result: usize) -> Vec<Dim> {
        Vec::new()
    }
    /// The value's iterators: the parallel iterators that only the value depends on.
    fn value_iters(&self) -> IterList {
        IterList::default()
    }
    /// Whether result `result` leaves the value's iterators out of its write.
    fn drops_value_iters(&self, _result: usize) -> bool {
        false
    }
    /// Whether the yielded value at `yielded` must not depend on the value's iterators: a value
    /// that the results without them come from.
    fn independent_of_value_iters(&self, _yielded: usize) -> bool {
        false
    }
    /// Whether folding points in any grouping and order gives the same bits for elements of
    /// `dtype`.
    fn exact(&self, _dtype: DType) -> bool {
        true
    }
    /// How many init values it takes: one per state component, and one per result.
    fn init_count(&self) -> usize {
        self.results()
    }
    /// Whether it folds into the elements of an operand that a write updates.
    fn folds_into_operand(&self) -> bool {
        false
    }
    /// How many sweeps over the points of an element its fold takes: one, or two for
    /// `online_softmax` and `logsumexp`, whose first finds the maximum.
    fn sweeps(&self) -> usize {
        1
    }
    /// The components of the state of an element for yielded values of `dtypes`: the dtype of
    /// each, and the extents of its values, none for one value, `[k]` for `k` of them.
    fn components(&self, dtypes: &[DType]) -> Vec<(DType, Vec<usize>)> {
        vec![(dtypes[0], Vec::new())]
    }
    /// The `arith` op that folds a yielded value into the state, for a kind whose state is one
    /// value that one op updates: `arith.add` for `sum`.
    fn step(&self) -> Option<Op> {
        None
    }
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
/// - `target` names the target its blocks' loops are for, and `numerics` the rounding changes the
///   lowerings below `tile` may make: how it runs changes no value it computes, except by those.
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
    pub target: Option<TargetRef>,
    #[attr(default)]
    pub numerics: Numerics,
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

/// Rounding changes against the reference, each named, so that none is silent:
/// `<contract = true, skip_pad = true>`; left out, none. An entry point allows a compilation a
/// set of them, which its passes may make. The pass that names a dispatch's target writes on it
/// those that its values may hold there ([`Numerics::on_target`]), for the passes before it that
/// made one and the lowerings after it that make one.
#[derive(crate::Attr, Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
#[attr(
    format = "`<` struct($contract, $skip_pad, $reassoc, $winograd, $scale_fold, $expand_math) `>`"
)]
pub struct Numerics {
    /// A product and the sum it enters round once, as one `arith.fma`, where they are f32 or f64.
    #[attr(default)]
    pub contract: bool,
    /// A point where a padded read falls outside its operand is skipped when the body yields to a
    /// float `sum` products of the pad 0 with its other values: nothing where those are finite,
    /// but for the sign of a sum that is zero. A point that yields the fold's identity is skipped
    /// without it.
    #[attr(default)]
    pub skip_pad: bool,
    /// A sum adds its points in the lanes of a vector, each lane its own, and the lanes after.
    #[attr(default)]
    pub reassoc: bool,
    /// A 3x3 convolution runs as Winograd's transforms, which round on their own.
    #[attr(default)]
    pub winograd: bool,
    /// A static scale of a dot's result rounds into each weight before the sum
    /// (`dot-scale-fold`), instead of into the sum once.
    #[attr(default)]
    pub scale_fold: bool,
    /// `exp`, `log`, `tanh`, `sin`, `cos` and `erf` of f32 are polynomials (`expand-math`): on
    /// every f32 input within 1 ulp of the interpreter's function, with its infinities and NaNs.
    #[attr(default)]
    pub expand_math: bool,
}

// ------------------------------------------------------------------------------------------------
// The dialect

#[derive(crate::Dialect, Debug, Clone, PartialEq, Eq, Hash)]
#[dialect(name = "tile", op = Tile)]
pub enum TileOps {
    // A block's iterators, reads, folds and loops, and a dispatch's facts, make them several times
    // the size of any other op. Boxed, they leave the other ops their size.
    Block(Box<Block>),
    Index(Index),
    Yield(Yield),
    Group(Group),
    Dispatch(Box<Dispatch>),
}
