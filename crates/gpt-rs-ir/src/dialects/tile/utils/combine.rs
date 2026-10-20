//! What each combine kind of `tile.block` takes and gives, its [`CombineRule`], and how
//! `tile-to-loop` builds its fold of the points of an element, its [`CombineFold`], written over
//! [`Scalars`].

use std::marker::PhantomData;

use anyhow::{ensure, Result};

use crate::dialects::arith::math::{self, Combiner};
use crate::dialects::tensor::utils::class::{ensure_dtype_class, ensure_index_dtype, ARITHMETIC};
use crate::dialects::tile::ir::*;
use crate::dialects::tile::IterList;
use crate::dialects::{arith, tensor};
use crate::host::HostScalar;
use crate::ir::{Constraint, DType, Dim, InOp, IndexExpr, Relation};
use crate::traits::RightIdentity;
use crate::Op;

/// How a combine kind folds the points of one element, over scalar operations whose values,
/// slots and cells are `V`, `S` and `C`: the state starts, takes each point's yielded values in
/// each sweep, and gives the results. [`CombineKind::fold`] gives the fold of each kind. It gives
/// the bits that the interpreter's fold of the kind gives ([`crate::dialects::tile::Acc`]).
pub trait CombineFold<V: Copy, S: Copy, C: Copy>: CombineRule {
    /// Starts the state from `init`, one value per component, or from no point when it is empty.
    fn start(&self, ops: &mut Ops<V, S, C>, dtypes: &[DType], init: &[V]) -> Result<()>;
    /// Prepares the state for sweep `sweep`, after the sweep before it.
    fn next_sweep(&self, _ops: &mut Ops<V, S, C>, _sweep: usize) -> Result<()> {
        Ok(())
    }
    /// Folds the values that one point yields, in sweep `sweep`.
    fn push(&self, ops: &mut Ops<V, S, C>, sweep: usize, yielded: &[V]) -> Result<()>;
    /// Finishes the state: the values of each result, one, or `k` along the dimension the kind
    /// appends.
    fn finish(&self, ops: &mut Ops<V, S, C>) -> Result<Vec<Vec<V>>>;
}

/// The scalar operations that a [`CombineFold`] is written in: `tile-to-loop` builds the `arith`
/// and `loop` ops that compute them. The state of an element has components, each one value or
/// several at slots: positions, which index expressions compute.
pub trait Scalars {
    type Value: Copy;
    type Slot: Copy;
    /// A scalar that a fold keeps beside its state while it takes a point.
    type Cell: Copy;

    /// The one result of the `arith` op `op` on `operands`.
    fn apply(&mut self, op: Op, operands: &[Self::Value]) -> Result<Self::Value>;
    /// The constant `value`.
    fn constant(&mut self, value: HostScalar) -> Self::Value;
    fn dtype(&self, value: Self::Value) -> DType;
    /// Component `component` of the state, at `slot` when it holds several values.
    fn get(&mut self, component: usize, slot: Option<Self::Slot>) -> Self::Value;
    fn set(&mut self, component: usize, slot: Option<Self::Slot>, value: Self::Value);
    /// A new cell of `dtype`.
    fn cell(&mut self, dtype: DType) -> Self::Cell;
    fn load(&mut self, cell: Self::Cell) -> Self::Value;
    fn store(&mut self, cell: Self::Cell, value: Self::Value);
    /// The slot that the integer `value` names.
    fn slot_of(&mut self, value: Self::Value) -> Self::Slot;
    /// The slot `expr`, whose `di` is `operands[i]`.
    fn slot(&mut self, expr: IndexExpr, operands: &[Self::Slot]) -> Self::Slot;
    /// Whether every one of `constraints` holds, their `di` being `operands[i]`, as an `i1`.
    fn holds(&mut self, constraints: &[Constraint], operands: &[Self::Slot]) -> Self::Value;
    /// Runs `body` at each slot from `start` up to `end`, in order.
    fn each_slot(
        &mut self,
        start: Self::Slot,
        end: Self::Slot,
        body: &mut SlotBody<Self::Value, Self::Slot, Self::Cell>,
    ) -> Result<()>;
    /// Runs `body` where `condition`, an `i1`, holds.
    fn when(
        &mut self,
        condition: Self::Value,
        body: &mut WhenBody<Self::Value, Self::Slot, Self::Cell>,
    ) -> Result<()>;
}

/// The scalar operations of a fold whose values, slots and cells are `V`, `S` and `C`.
pub type Ops<'o, V, S, C> = dyn Scalars<Value = V, Slot = S, Cell = C> + 'o;

/// What a loop over slots builds at each slot: from the scalar operations there.
pub type SlotBody<'b, V, S, C> = dyn FnMut(&mut Ops<'_, V, S, C>, S) -> Result<()> + 'b;

/// What a conditional builds where its condition holds.
pub type WhenBody<'b, V, S, C> = dyn FnMut(&mut Ops<'_, V, S, C>) -> Result<()> + 'b;

/// `none`: one value, written per point.
struct NoFold;

/// `sum`, `max`, `min`, `and` and `or`: a fold with the `tensor` op `C`, whose `arith` op folds
/// each value in.
struct Monoid<C>(PhantomData<C>);

/// `argmax` and `argmin`. `LARGEST` is whether they keep the largest value.
struct Arg<const LARGEST: bool>;

/// `logsumexp` and its merge. `MERGE` is whether the body yields states.
struct LogSumExp<const MERGE: bool>;

impl CombineKind {
    /// Whether the kind keeps one element per point of its result: `none`, or a fold of one value
    /// whose state is that element, `sum`, `max`, `min`, `and` and `or`.
    pub fn keeps_one_element(&self) -> bool {
        matches!(
            self,
            CombineKind::None
                | CombineKind::Sum
                | CombineKind::Max
                | CombineKind::Min
                | CombineKind::And
                | CombineKind::Or
        )
    }

    /// The value's iterators of a softmax kind.
    pub fn value_iters_mut(&mut self) -> Option<&mut IterList> {
        match self {
            CombineKind::OnlineSoftmax(softmax) => Some(&mut softmax.value_iters),
            CombineKind::OnlineSoftmaxMerge(softmax) => Some(&mut softmax.value_iters),
            _ => None,
        }
    }

    /// The fold of the kind, over scalar operations whose values, slots and cells are `V`, `S`
    /// and `C`: what it takes and gives, and how it folds.
    pub fn fold<V: Copy, S: Copy, C: Copy>(&self) -> &dyn CombineFold<V, S, C> {
        match self {
            CombineKind::None => &NoFold,
            CombineKind::Sum => &Monoid::<tensor::Add>(PhantomData),
            CombineKind::Max => &Monoid::<tensor::Maximum>(PhantomData),
            CombineKind::Min => &Monoid::<tensor::Minimum>(PhantomData),
            CombineKind::And => &Monoid::<tensor::And>(PhantomData),
            CombineKind::Or => &Monoid::<tensor::Or>(PhantomData),
            CombineKind::Argmax => &Arg::<true>,
            CombineKind::Argmin => &Arg::<false>,
            CombineKind::TopK(best) => best,
            CombineKind::BottomK(best) => best,
            CombineKind::OnlineSoftmax(softmax) => softmax,
            CombineKind::OnlineSoftmaxMerge(softmax) => softmax,
            CombineKind::Logsumexp => &LogSumExp::<false>,
            CombineKind::LogsumexpMerge => &LogSumExp::<true>,
        }
    }

    /// What the kind takes and gives.
    pub fn rule(&self) -> &dyn CombineRule {
        self.fold::<(), (), ()>()
    }

    /// The right identity of the kind's step on `dtype`, which leaves every state of the fold as
    /// it is: -0.0 for a float `sum`, -inf for a float `max`. `None` for a kind without a step, or
    /// a step without an identity on `dtype`.
    pub fn identity(&self, dtype: DType) -> Option<HostScalar> {
        let step = self.rule().step()?;
        step.get_trait::<RightIdentity>()?.0.value(dtype)
    }
}

/// Checks that `yields` holds `count` values for `kind`.
fn ensure_yields(kind: &str, yields: &[DType], count: usize) -> Result<()> {
    ensure!(
        yields.len() == count,
        "{kind} takes {count} yielded values, got {}",
        yields.len()
    );
    Ok(())
}

/// Checks that the body yields `(value, index)`, a value `kind` orders and an index.
fn ensure_ranked(kind: &str, yields: &[DType]) -> Result<()> {
    ensure_yields(kind, yields, 2)?;
    ensure_index_dtype(&format!("the indices of {kind}"), yields[1])?;
    ensure_dtype_class(
        &format!("the value that {kind} orders"),
        yields[0],
        ARITHMETIC,
    )
}

/// Checks that the body yields `count` f32 values for `kind`, and gives the f32 dtypes of its
/// `count` state components.
fn f32_state(kind: &str, yields: &[DType], count: usize) -> Result<Vec<DType>> {
    ensure!(
        yields.iter().all(|&dtype| dtype == DType::F32),
        "{kind} folds f32 values, got {yields:?}"
    );
    Ok(vec![DType::F32; count])
}

impl CombineRule for NoFold {
    fn result_dtypes(&self, kind: &str, yields: &[DType]) -> Result<Vec<DType>> {
        ensure_yields(kind, yields, 1)?;
        Ok(yields.to_vec())
    }

    fn init_count(&self) -> usize {
        0
    }

    fn folds_into_operand(&self) -> bool {
        true
    }
}

impl<C: Combiner> CombineRule for Monoid<C>
where
    <C as tensor::Binary>::Scalar: InOp,
{
    fn result_dtypes(&self, kind: &str, yields: &[DType]) -> Result<Vec<DType>> {
        ensure_yields(kind, yields, 1)?;
        let classes = <C::Scalar as arith::Binary>::CLASSES;
        ensure_dtype_class(&format!("the value that {kind} folds"), yields[0], classes)?;
        Ok(yields.to_vec())
    }

    /// Only a float sum rounds differently when its points regroup.
    fn exact(&self, dtype: DType) -> bool {
        C::COMBINE != CombineKind::Sum || dtype.is_integer()
    }

    fn folds_into_operand(&self) -> bool {
        true
    }

    fn step(&self) -> Option<Op> {
        Some(<C as tensor::Binary>::Scalar::default().into())
    }
}

impl<const LARGEST: bool> CombineRule for Arg<LARGEST> {
    fn yields(&self) -> usize {
        2
    }

    fn results(&self) -> usize {
        2
    }

    fn result_dtypes(&self, kind: &str, yields: &[DType]) -> Result<Vec<DType>> {
        ensure_ranked(kind, yields)?;
        Ok(yields.to_vec())
    }

    fn components(&self, dtypes: &[DType]) -> Vec<(DType, Vec<usize>)> {
        best_components(1, dtypes)
    }
}

impl<const LARGEST: bool> CombineRule for KBest<LARGEST> {
    fn yields(&self) -> usize {
        2
    }

    fn results(&self) -> usize {
        2
    }

    fn result_dtypes(&self, kind: &str, yields: &[DType]) -> Result<Vec<DType>> {
        ensure_ranked(kind, yields)?;
        Ok(yields.to_vec())
    }

    fn extra_dims(&self, _result: usize) -> Vec<Dim> {
        vec![Dim::Static(self.k)]
    }

    fn init_count(&self) -> usize {
        0
    }

    fn components(&self, dtypes: &[DType]) -> Vec<(DType, Vec<usize>)> {
        best_components(self.k, dtypes)
    }
}

impl<const MERGE: bool> CombineRule for Softmax<MERGE> {
    /// A score and a value, or a state of three components.
    fn yields(&self) -> usize {
        if MERGE {
            3
        } else {
            2
        }
    }

    fn results(&self) -> usize {
        3
    }

    fn result_dtypes(&self, kind: &str, yields: &[DType]) -> Result<Vec<DType>> {
        ensure_yields(kind, yields, self.yields())?;
        f32_state(kind, yields, 3)
    }

    fn value_iters(&self) -> IterList {
        self.value_iters
    }

    /// The `m` and `l` states.
    fn drops_value_iters(&self, result: usize) -> bool {
        result < 2
    }

    /// The score, which `m` and `l` come from, or those two states themselves.
    fn independent_of_value_iters(&self, yielded: usize) -> bool {
        match MERGE {
            true => yielded < 2,
            false => yielded == 0,
        }
    }

    fn exact(&self, _dtype: DType) -> bool {
        false
    }

    fn sweeps(&self) -> usize {
        if MERGE {
            1
        } else {
            2
        }
    }

    fn components(&self, _dtypes: &[DType]) -> Vec<(DType, Vec<usize>)> {
        softmax_components(true, MERGE)
    }
}

impl<const MERGE: bool> CombineRule for LogSumExp<MERGE> {
    /// A score, or a state of two components.
    fn yields(&self) -> usize {
        if MERGE {
            2
        } else {
            1
        }
    }

    fn results(&self) -> usize {
        2
    }

    fn result_dtypes(&self, kind: &str, yields: &[DType]) -> Result<Vec<DType>> {
        ensure_yields(kind, yields, self.yields())?;
        f32_state(kind, yields, 2)
    }

    fn exact(&self, _dtype: DType) -> bool {
        false
    }

    fn sweeps(&self) -> usize {
        if MERGE {
            1
        } else {
            2
        }
    }

    fn components(&self, _dtypes: &[DType]) -> Vec<(DType, Vec<usize>)> {
        softmax_components(false, MERGE)
    }
}

// ------------------------------------------------------------------------------------------------
// How each kind folds, over `Scalars`.

impl<'o, V: Copy, S: Copy, C: Copy> dyn Scalars<Value = V, Slot = S, Cell = C> + 'o {
    /// The one result of the `arith` op `op` on `operands`.
    fn op(&mut self, op: impl Into<Op>, operands: &[V]) -> Result<V> {
        self.apply(op.into(), operands)
    }

    /// The slot `position`.
    fn slot_at(&mut self, position: i64) -> S {
        self.slot(IndexExpr::constant(position), &[])
    }
}

/// `lhs < rhs`, one constraint.
fn less(lhs: IndexExpr, rhs: IndexExpr) -> Constraint {
    Constraint::cmp(lhs, Relation::Less, rhs)
}

impl<V: Copy, S: Copy, C: Copy> CombineFold<V, S, C> for NoFold {
    fn start(&self, _ops: &mut Ops<V, S, C>, _dtypes: &[DType], _init: &[V]) -> Result<()> {
        Ok(())
    }

    /// The value of the last point.
    fn push(&self, ops: &mut Ops<V, S, C>, _sweep: usize, yielded: &[V]) -> Result<()> {
        ops.set(0, None, yielded[0]);
        Ok(())
    }

    fn finish(&self, ops: &mut Ops<V, S, C>) -> Result<Vec<Vec<V>>> {
        Ok(vec![vec![ops.get(0, None)]])
    }
}

/// The running value, from `init` or the initial value of `C`, each point's value folded in with
/// the `arith` op of `C`.
impl<V: Copy, S: Copy, C: Copy, X: Combiner> CombineFold<V, S, C> for Monoid<X>
where
    <X as tensor::Binary>::Scalar: InOp,
{
    fn start(&self, ops: &mut Ops<V, S, C>, dtypes: &[DType], init: &[V]) -> Result<()> {
        let first = match init {
            [value] => *value,
            _ => ops.constant(math::fold_init::<X>(dtypes[0])?),
        };
        ops.set(0, None, first);
        Ok(())
    }

    fn push(&self, ops: &mut Ops<V, S, C>, _sweep: usize, yielded: &[V]) -> Result<()> {
        let current = ops.get(0, None);
        let step = <X as tensor::Binary>::Scalar::default();
        let folded = ops.op(step, &[current, yielded[0]])?;
        ops.set(0, None, folded);
        Ok(())
    }

    fn finish(&self, ops: &mut Ops<V, S, C>) -> Result<Vec<Vec<V>>> {
        Ok(vec![vec![ops.get(0, None)]])
    }
}

/// The state of a `k`-best fold over `places` places: the values and the indices of the best
/// candidates, best first, and how many of the places hold one, an si64.
fn best_components(places: usize, dtypes: &[DType]) -> Vec<(DType, Vec<usize>)> {
    vec![
        (dtypes[0], vec![places]),
        (dtypes[1], vec![places]),
        (DType::Si64, Vec::new()),
    ]
}

/// The value of an empty place of a `k`-best state: the identity of the order.
fn best_fill(largest: bool, dtype: DType) -> Result<HostScalar> {
    match largest {
        true => math::fold_init::<tensor::Maximum>(dtype),
        false => math::fold_init::<tensor::Minimum>(dtype),
    }
}

/// Whether the candidate `(value, index)` ranks before `(held, held_index)`: its value is better,
/// with NaN above every number, or the values are equal and its index is smaller.
fn ranks_before<V: Copy, S: Copy, C: Copy>(
    ops: &mut Ops<V, S, C>,
    largest: bool,
    (value, index): (V, V),
    (held, held_index): (V, V),
) -> Result<V> {
    let dtype = ops.dtype(value);
    let (above, below) = match largest {
        true => (value, held),
        false => (held, value),
    };
    let (better, equal) = match dtype.is_float() {
        true => {
            let above_nan = ops.op(arith::NotEqual, &[above, above])?;
            let below_nan = ops.op(arith::NotEqual, &[below, below])?;
            let below_number = ops.op(arith::Not, &[below_nan])?;
            let nan_over_number = ops.op(arith::And, &[above_nan, below_number])?;
            let greater = ops.op(arith::Greater, &[above, below])?;
            let better = ops.op(arith::Or, &[nan_over_number, greater])?;
            let both_nan = ops.op(arith::And, &[above_nan, below_nan])?;
            let same = ops.op(arith::Equal, &[above, below])?;
            (better, ops.op(arith::Or, &[both_nan, same])?)
        }
        false => (
            ops.op(arith::Greater, &[above, below])?,
            ops.op(arith::Equal, &[above, below])?,
        ),
    };
    let earlier = ops.op(arith::Less, &[index, held_index])?;
    let tie_won = ops.op(arith::And, &[equal, earlier])?;
    ops.op(arith::Or, &[better, tie_won])
}

/// Starts a `k`-best state with every place empty, then puts the candidate of `init` in.
fn start_best<V: Copy, S: Copy, C: Copy>(
    ops: &mut Ops<V, S, C>,
    (places, largest): (usize, bool),
    dtypes: &[DType],
    init: &[V],
) -> Result<()> {
    let fill = ops.constant(best_fill(largest, dtypes[0])?);
    let missing = ops.constant(HostScalar::int(dtypes[1], -1)?);
    let start = ops.slot_at(0);
    let end = ops.slot_at(places as i64);
    ops.each_slot(start, end, &mut |inner, slot| {
        inner.set(0, Some(slot), fill);
        inner.set(1, Some(slot), missing);
        Ok(())
    })?;
    let none = ops.constant(HostScalar::int(DType::Si64, 0)?);
    ops.set(2, None, none);
    if let &[value, index] = init {
        insert(ops, (places, largest), value, index)?;
    }
    Ok(())
}

/// Puts the candidate `(value, index)` among the held ones of a `k`-best state. It goes after
/// every held one that ranks before or with it, so equal values keep the order of their indices
/// and then the order they came in, as a sort by index and then by value gives.
fn insert<V: Copy, S: Copy, C: Copy>(
    ops: &mut Ops<V, S, C>,
    (places, largest): (usize, bool),
    value: V,
    index: V,
) -> Result<()> {
    let capacity = places as i64;
    let zero = ops.constant(HostScalar::int(DType::Si64, 0)?);
    let one = ops.constant(HostScalar::int(DType::Si64, 1)?);
    // Its place: how many held candidates rank before it or with it.
    let place_count = ops.cell(DType::Si64);
    ops.store(place_count, zero);
    let held_count = ops.get(2, None);
    let held = ops.slot_of(held_count);
    let start = ops.slot_at(0);
    let end = ops.slot_at(capacity);
    ops.each_slot(start, end, &mut |inner, slot| {
        let filled = inner.holds(
            &[less(IndexExpr::iter(0), IndexExpr::iter(1))],
            &[slot, held],
        );
        let slot_value = inner.get(0, Some(slot));
        let slot_index = inner.get(1, Some(slot));
        let after = ranks_before(inner, largest, (value, index), (slot_value, slot_index))?;
        let not_after = inner.op(arith::Not, &[after])?;
        let before = inner.op(arith::And, &[filled, not_after])?;
        let step = inner.op(arith::Select, &[before, one, zero])?;
        let so_far = inner.load(place_count);
        let counted = inner.op(arith::Add, &[so_far, step])?;
        inner.store(place_count, counted);
        Ok(())
    })?;
    let ranked_before = ops.load(place_count);
    let place = ops.slot_of(ranked_before);
    // Move the candidates from its place on one down, the last one out.
    let shifts = ops.slot_at(capacity - 1);
    ops.each_slot(start, shifts, &mut |inner, step| {
        let from_end = IndexExpr::constant(capacity - 1) - IndexExpr::iter(0);
        let slot = inner.slot(from_end, &[step]);
        let moves = inner.holds(
            &[less(IndexExpr::iter(0), IndexExpr::iter(1))],
            &[place, slot],
        );
        inner.when(moves, &mut |moving| {
            let before_it = IndexExpr::iter(0) - IndexExpr::constant(1);
            let previous = moving.slot(before_it, &[slot]);
            let moved_value = moving.get(0, Some(previous));
            let moved_index = moving.get(1, Some(previous));
            moving.set(0, Some(slot), moved_value);
            moving.set(1, Some(slot), moved_index);
            Ok(())
        })
    })?;
    let room = less(IndexExpr::iter(0), IndexExpr::constant(capacity));
    let kept = ops.holds(&[room], &[place]);
    ops.when(kept, &mut |inner| {
        inner.set(0, Some(place), value);
        inner.set(1, Some(place), index);
        Ok(())
    })?;
    let held_before = ops.get(2, None);
    let one_more = ops.op(arith::Add, &[held_before, one])?;
    let most = ops.constant(HostScalar::int(DType::Si64, i128::from(capacity))?);
    let held_after = ops.op(arith::Minimum, &[one_more, most])?;
    ops.set(2, None, held_after);
    Ok(())
}

/// The `places` best values, then their indices. A fold of fewer than `places` points keeps the
/// identity of its order and the index -1 in the places left.
fn best_results<V: Copy, S: Copy, C: Copy>(ops: &mut Ops<V, S, C>, places: usize) -> Vec<Vec<V>> {
    let mut values = Vec::with_capacity(places);
    let mut indices = Vec::with_capacity(places);
    for rank in 0..places {
        let at = ops.slot_at(rank as i64);
        values.push(ops.get(0, Some(at)));
        indices.push(ops.get(1, Some(at)));
    }
    vec![values, indices]
}

impl<V: Copy, S: Copy, C: Copy, const LARGEST: bool> CombineFold<V, S, C> for Arg<LARGEST> {
    fn start(&self, ops: &mut Ops<V, S, C>, dtypes: &[DType], init: &[V]) -> Result<()> {
        start_best(ops, (1, LARGEST), dtypes, init)
    }

    fn push(&self, ops: &mut Ops<V, S, C>, _sweep: usize, yielded: &[V]) -> Result<()> {
        insert(ops, (1, LARGEST), yielded[0], yielded[1])
    }

    fn finish(&self, ops: &mut Ops<V, S, C>) -> Result<Vec<Vec<V>>> {
        Ok(best_results(ops, 1))
    }
}

impl<V: Copy, S: Copy, C: Copy, const LARGEST: bool> CombineFold<V, S, C> for KBest<LARGEST> {
    fn start(&self, ops: &mut Ops<V, S, C>, dtypes: &[DType], init: &[V]) -> Result<()> {
        start_best(ops, (self.k, LARGEST), dtypes, init)
    }

    fn push(&self, ops: &mut Ops<V, S, C>, _sweep: usize, yielded: &[V]) -> Result<()> {
        insert(ops, (self.k, LARGEST), yielded[0], yielded[1])
    }

    fn finish(&self, ops: &mut Ops<V, S, C>) -> Result<Vec<Vec<V>>> {
        Ok(best_results(ops, self.k))
    }
}

/// The f32 state of a softmax fold: `(m, l, o)` with values, `(m, l)` without, and after them the
/// maximum the state started from, while the fold sweeps twice.
fn softmax_components(with_value: bool, merge: bool) -> Vec<(DType, Vec<usize>)> {
    let components = match with_value {
        true => 3,
        false => 2,
    };
    let kept_start = usize::from(!merge);
    vec![(DType::F32, Vec::new()); components + kept_start]
}

/// `e^(from - to)`, the rescale factor of a state whose maximum is `from` to the maximum `to`, or 0
/// when `from` is `-inf`: a state of no points contributes nothing.
fn rescale<V: Copy, S: Copy, C: Copy>(ops: &mut Ops<V, S, C>, from: V, to: V) -> Result<V> {
    let difference = ops.op(arith::Sub, &[from, to])?;
    let factor = ops.op(arith::Exp, &[difference])?;
    let none = ops.constant(HostScalar::from_f64(DType::F32, f64::NEG_INFINITY)?);
    let empty = ops.op(arith::Equal, &[from, none])?;
    let zero = ops.constant(HostScalar::from_f64(DType::F32, 0.0)?);
    ops.op(arith::Select, &[empty, zero, factor])
}

/// Starts the softmax state `(m, l, o)`, or `(m, l)` without values: the maximum, the sum of
/// `e^(s - m)`, and the sum of `e^(s - m) v`, from `init`, or from no point.
fn start_softmax<V: Copy, S: Copy, C: Copy>(
    ops: &mut Ops<V, S, C>,
    with_value: bool,
    init: &[V],
) -> Result<()> {
    let empty = [f64::NEG_INFINITY, 0.0, 0.0];
    let components = match with_value {
        true => 3,
        false => 2,
    };
    for (component, &start) in empty.iter().enumerate().take(components) {
        let value = match init.get(component) {
            Some(&value) => value,
            None => ops.constant(HostScalar::from_f64(DType::F32, start)?),
        };
        ops.set(component, None, value);
    }
    Ok(())
}

/// `online_softmax` (`with_value`) and `logsumexp`: the maximum in the first sweep, then the sums
/// of `e^(s - m)` and of `e^(s - m) v`, the start state rescaled to the maximum. The state keeps
/// the start maximum after its components.
struct TwoPass {
    with_value: bool,
}

impl TwoPass {
    /// The component that keeps the maximum the state started from.
    fn kept(&self) -> usize {
        match self.with_value {
            true => 3,
            false => 2,
        }
    }

    fn start<V: Copy, S: Copy, C: Copy>(&self, ops: &mut Ops<V, S, C>, init: &[V]) -> Result<()> {
        start_softmax(ops, self.with_value, init)?;
        let max = ops.get(0, None);
        ops.set(self.kept(), None, max);
        Ok(())
    }

    fn next_sweep<V: Copy, S: Copy, C: Copy>(&self, ops: &mut Ops<V, S, C>) -> Result<()> {
        let kept = self.kept();
        let start_max = ops.get(kept, None);
        let max = ops.get(0, None);
        let factor = rescale(ops, start_max, max)?;
        for component in 1..kept {
            let value = ops.get(component, None);
            let scaled = ops.op(arith::Mul, &[value, factor])?;
            ops.set(component, None, scaled);
        }
        Ok(())
    }

    fn push<V: Copy, S: Copy, C: Copy>(
        &self,
        ops: &mut Ops<V, S, C>,
        sweep: usize,
        yielded: &[V],
    ) -> Result<()> {
        let score = yielded[0];
        let max = ops.get(0, None);
        if sweep == 0 {
            let larger = ops.op(arith::Maximum, &[max, score])?;
            ops.set(0, None, larger);
            return Ok(());
        }
        let difference = ops.op(arith::Sub, &[score, max])?;
        let weight = ops.op(arith::Exp, &[difference])?;
        let sum_so_far = ops.get(1, None);
        let sum = ops.op(arith::Add, &[sum_so_far, weight])?;
        ops.set(1, None, sum);
        if self.with_value {
            let weighted_so_far = ops.get(2, None);
            let term = ops.op(arith::Mul, &[weight, yielded[1]])?;
            let weighted = ops.op(arith::Add, &[weighted_so_far, term])?;
            ops.set(2, None, weighted);
        }
        Ok(())
    }

    fn results<V: Copy, S: Copy, C: Copy>(&self, ops: &mut Ops<V, S, C>) -> Vec<Vec<V>> {
        let mut results = Vec::with_capacity(self.kept());
        for component in 0..self.kept() {
            results.push(vec![ops.get(component, None)]);
        }
        results
    }
}

/// `online_softmax_merge` (`with_value`) and `logsumexp_merge`: each point's state merged into the
/// running one.
struct Merge {
    with_value: bool,
}

impl Merge {
    fn components(&self) -> usize {
        match self.with_value {
            true => 3,
            false => 2,
        }
    }

    fn push<V: Copy, S: Copy, C: Copy>(&self, ops: &mut Ops<V, S, C>, yielded: &[V]) -> Result<()> {
        let max = ops.get(0, None);
        let merged_max = ops.op(arith::Maximum, &[max, yielded[0]])?;
        let own = rescale(ops, max, merged_max)?;
        let point = rescale(ops, yielded[0], merged_max)?;
        for (component, &part) in yielded.iter().enumerate().skip(1) {
            let held = ops.get(component, None);
            let kept = ops.op(arith::Mul, &[held, own])?;
            let added = ops.op(arith::Mul, &[part, point])?;
            let sum = ops.op(arith::Add, &[kept, added])?;
            ops.set(component, None, sum);
        }
        ops.set(0, None, merged_max);
        Ok(())
    }

    fn results<V: Copy, S: Copy, C: Copy>(&self, ops: &mut Ops<V, S, C>) -> Vec<Vec<V>> {
        let mut results = Vec::with_capacity(self.components());
        for component in 0..self.components() {
            results.push(vec![ops.get(component, None)]);
        }
        results
    }
}

/// A softmax fold of `with_value`, merging states when `merge` holds: [`TwoPass`] or [`Merge`].
fn softmax_start<V: Copy, S: Copy, C: Copy>(
    ops: &mut Ops<V, S, C>,
    (with_value, merge): (bool, bool),
    init: &[V],
) -> Result<()> {
    match merge {
        true => start_softmax(ops, with_value, init),
        false => TwoPass { with_value }.start(ops, init),
    }
}

fn softmax_push<V: Copy, S: Copy, C: Copy>(
    ops: &mut Ops<V, S, C>,
    (with_value, merge): (bool, bool),
    sweep: usize,
    yielded: &[V],
) -> Result<()> {
    match merge {
        true => Merge { with_value }.push(ops, yielded),
        false => TwoPass { with_value }.push(ops, sweep, yielded),
    }
}

fn softmax_results<V: Copy, S: Copy, C: Copy>(
    ops: &mut Ops<V, S, C>,
    (with_value, merge): (bool, bool),
) -> Vec<Vec<V>> {
    match merge {
        true => Merge { with_value }.results(ops),
        false => TwoPass { with_value }.results(ops),
    }
}

impl<V: Copy, S: Copy, C: Copy, const MERGE: bool> CombineFold<V, S, C> for Softmax<MERGE> {
    fn start(&self, ops: &mut Ops<V, S, C>, _dtypes: &[DType], init: &[V]) -> Result<()> {
        softmax_start(ops, (true, MERGE), init)
    }

    fn next_sweep(&self, ops: &mut Ops<V, S, C>, _sweep: usize) -> Result<()> {
        TwoPass { with_value: true }.next_sweep(ops)
    }

    fn push(&self, ops: &mut Ops<V, S, C>, sweep: usize, yielded: &[V]) -> Result<()> {
        softmax_push(ops, (true, MERGE), sweep, yielded)
    }

    fn finish(&self, ops: &mut Ops<V, S, C>) -> Result<Vec<Vec<V>>> {
        Ok(softmax_results(ops, (true, MERGE)))
    }
}

impl<V: Copy, S: Copy, C: Copy, const MERGE: bool> CombineFold<V, S, C> for LogSumExp<MERGE> {
    fn start(&self, ops: &mut Ops<V, S, C>, _dtypes: &[DType], init: &[V]) -> Result<()> {
        softmax_start(ops, (false, MERGE), init)
    }

    fn next_sweep(&self, ops: &mut Ops<V, S, C>, _sweep: usize) -> Result<()> {
        TwoPass { with_value: false }.next_sweep(ops)
    }

    fn push(&self, ops: &mut Ops<V, S, C>, sweep: usize, yielded: &[V]) -> Result<()> {
        softmax_push(ops, (false, MERGE), sweep, yielded)
    }

    fn finish(&self, ops: &mut Ops<V, S, C>) -> Result<Vec<Vec<V>>> {
        Ok(softmax_results(ops, (false, MERGE)))
    }
}
