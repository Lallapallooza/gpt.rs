//! The combine kinds of `tile.block` as `loop` IR: the state each keeps in private scratch while it
//! folds the points of an element, and the `arith` ops that update it. Each folds as the kind's
//! [`Acc`](crate::dialects::tile::Acc) does, bit for bit.

use std::marker::PhantomData;

use anyhow::{bail, Result};

use super::utils::{
    cell, constant, for_range, get, if_then, index, less, lit, load, predicate, set, store,
};
use crate::dialects::arith::math::{self, Combiner};
use crate::dialects::tile::{Body, CombineKind};
use crate::dialects::{arith, tensor};
use crate::host::HostScalar;
use crate::ir::{DType, InOp, IndexExpr, Value};
use crate::rewrite::Builder;

/// The state of a fold for one element: a buffer per state component, which holds the state of
/// every element of a tile, and the element's position in them.
pub(super) struct State {
    pub(super) buffers: Vec<Value>,
    pub(super) at: Vec<Value>,
}

impl State {
    /// The value of component `component`.
    fn get(&self, builder: &mut Builder<'_>, component: usize) -> Value {
        load(builder, self.buffers[component], self.at.clone(), None)
    }

    fn set(&self, builder: &mut Builder<'_>, component: usize, value: Value) {
        store(builder, value, self.buffers[component], self.at.clone());
    }

    /// The value at `slot` of component `component`, which holds several.
    fn get_slot(&self, builder: &mut Builder<'_>, component: usize, slot: Value) -> Value {
        let at = [self.at.clone(), vec![slot]].concat();
        load(builder, self.buffers[component], at, None)
    }

    fn set_slot(&self, builder: &mut Builder<'_>, component: usize, slot: Value, value: Value) {
        let at = [self.at.clone(), vec![slot]].concat();
        store(builder, value, self.buffers[component], at);
    }
}

/// How one combine kind folds the points of an element.
pub(super) trait FoldLowering {
    /// How many sweeps over the points of the element the fold takes: one, or two for the softmax
    /// kinds, whose first finds the maximum.
    fn sweeps(&self) -> usize {
        1
    }

    /// The dtype of each state component for yielded values of `dtypes`, and the extents one
    /// element's component has: none for one value, `[k]` for `k` of them.
    fn components(&self, dtypes: &[DType]) -> Vec<(DType, Vec<usize>)>;

    /// Starts `state` from `init`, one value per state component, or from no point when it is
    /// empty.
    fn start(
        &self,
        builder: &mut Builder<'_>,
        state: &State,
        dtypes: &[DType],
        init: &[Value],
    ) -> Result<()>;

    /// Prepares `state` for sweep `sweep`, after the sweep before it.
    fn next_sweep(&self, _b: &mut Builder<'_>, _state: &State, _sweep: usize) -> Result<()> {
        Ok(())
    }

    /// Folds the values one point yields into `state`, in sweep `sweep`.
    fn push(
        &self,
        builder: &mut Builder<'_>,
        state: &State,
        sweep: usize,
        yielded: &[Value],
    ) -> Result<()>;

    /// The values of each result: one, or `k` along the dimension the kind appends.
    fn results(&self, builder: &mut Builder<'_>, state: &State) -> Vec<Vec<Value>>;
}

/// The fold of `kind`, which folds values: every kind but `none`.
pub(super) fn fold_of(kind: &CombineKind) -> Result<Box<dyn FoldLowering>> {
    Ok(match kind {
        CombineKind::None => bail!("`none` writes each point and folds nothing"),
        CombineKind::Sum => Box::new(Monoid::<tensor::Add>(PhantomData)),
        CombineKind::Max => Box::new(Monoid::<tensor::Maximum>(PhantomData)),
        CombineKind::Min => Box::new(Monoid::<tensor::Minimum>(PhantomData)),
        CombineKind::And => Box::new(Monoid::<tensor::And>(PhantomData)),
        CombineKind::Or => Box::new(Monoid::<tensor::Or>(PhantomData)),
        CombineKind::Argmax => Box::new(Best::<true> { k: 1 }),
        CombineKind::Argmin => Box::new(Best::<false> { k: 1 }),
        CombineKind::TopK(best) => Box::new(Best::<true> { k: best.k }),
        CombineKind::BottomK(best) => Box::new(Best::<false> { k: best.k }),
        CombineKind::OnlineSoftmax(_) => Box::new(TwoPass::<true>),
        CombineKind::Logsumexp => Box::new(TwoPass::<false>),
        CombineKind::OnlineSoftmaxMerge(_) => Box::new(Merge::<true>),
        CombineKind::LogsumexpMerge => Box::new(Merge::<false>),
    })
}

// ------------------------------------------------------------------------------------------------
// `sum`, `max`, `min`, `and` and `or`.

/// A fold with the `tensor` op `C`, whose `arith` op folds each value into the running one.
struct Monoid<C>(PhantomData<C>);

impl<C: Combiner> FoldLowering for Monoid<C>
where
    <C as tensor::Binary>::Scalar: InOp,
{
    fn components(&self, dtypes: &[DType]) -> Vec<(DType, Vec<usize>)> {
        vec![(dtypes[0], Vec::new())]
    }

    fn start(
        &self,
        builder: &mut Builder<'_>,
        state: &State,
        dtypes: &[DType],
        init: &[Value],
    ) -> Result<()> {
        let first = match init {
            [value] => *value,
            _ => lit(builder, math::fold_init::<C>(dtypes[0])?),
        };
        state.set(builder, 0, first);
        Ok(())
    }

    fn push(
        &self,
        builder: &mut Builder<'_>,
        state: &State,
        _sweep: usize,
        yielded: &[Value],
    ) -> Result<()> {
        let current = state.get(builder, 0);
        let folded = builder.create(
            <C as tensor::Binary>::Scalar::default(),
            (current, yielded[0]),
        );
        state.set(builder, 0, folded);
        Ok(())
    }

    fn results(&self, builder: &mut Builder<'_>, state: &State) -> Vec<Vec<Value>> {
        vec![vec![state.get(builder, 0)]]
    }
}

// ------------------------------------------------------------------------------------------------
// `argmax`, `argmin`, `top_k` and `bottom_k`.

/// The `k` best `(value, index)` candidates, best first, in two components of `k` values, and a
/// component with how many of them hold a candidate. A candidate goes after every held one that
/// ranks before or with it, so equal values keep the order of their indices and then the order
/// they came in, as the reference's sort by index and then by value gives. `LARGEST` is whether
/// the best is the largest.
struct Best<const LARGEST: bool> {
    k: usize,
}

/// Whether the candidate `(value, index)` ranks before `(held, held_index)`: its value is better,
/// with NaN above every number, or the values are equal and its index is smaller.
fn ranks_before(
    builder: &mut Builder<'_>,
    largest: bool,
    (value, index): (Value, Value),
    (held, held_index): (Value, Value),
) -> Value {
    let dtype = Body { builder }.dtype(value);
    let (above, below) = if largest {
        (value, held)
    } else {
        (held, value)
    };
    let (better, equal) = if dtype.is_float() {
        let mut body = Body { builder };
        let above_nan = body.is_nan(above);
        let below_nan = body.is_nan(below);
        let below_number = body.create(arith::Not, below_nan);
        let nan_over_number = body.create(arith::And, (above_nan, below_number));
        let greater = body.create(arith::Greater, (above, below));
        let better = body.create(arith::Or, (nan_over_number, greater));
        let both_nan = body.create(arith::And, (above_nan, below_nan));
        let same = body.create(arith::Equal, (above, below));
        (better, body.create(arith::Or, (both_nan, same)))
    } else {
        (
            builder.create(arith::Greater, (above, below)),
            builder.create(arith::Equal, (above, below)),
        )
    };
    let earlier = builder.create(arith::Less, (index, held_index));
    let tie_won = builder.create(arith::And, (equal, earlier));
    builder.create(arith::Or, (better, tie_won))
}

impl<const LARGEST: bool> Best<LARGEST> {
    /// The value of an empty place: the identity of the order.
    fn fill(dtype: DType) -> Result<HostScalar> {
        if LARGEST {
            math::fold_init::<tensor::Maximum>(dtype)
        } else {
            math::fold_init::<tensor::Minimum>(dtype)
        }
    }

    /// Puts the candidate `(value, index_value)` among the held ones of `state`.
    fn insert(
        &self,
        builder: &mut Builder<'_>,
        state: &State,
        value: Value,
        index_value: Value,
    ) -> Result<()> {
        let capacity = self.k as i64;
        let zero = lit(builder, HostScalar::int(DType::Si64, 0)?);
        let one = lit(builder, HostScalar::int(DType::Si64, 1)?);
        // Its place: how many held candidates rank before it or with it.
        let place_count = cell(builder, DType::Si64);
        set(builder, place_count, zero);
        let held_count = state.get(builder, 2);
        let held = builder.create(arith::ToIndex, held_count);
        let start = constant(builder, 0);
        let end = constant(builder, capacity);
        for_range(builder, start, end, |builder, slot| {
            let filled = predicate(
                builder,
                &[less(IndexExpr::iter(0), IndexExpr::iter(1))],
                &[slot, held],
            );
            let slot_value = state.get_slot(builder, 0, slot);
            let slot_index = state.get_slot(builder, 1, slot);
            let after = ranks_before(
                builder,
                LARGEST,
                (value, index_value),
                (slot_value, slot_index),
            );
            let not_after = builder.create(arith::Not, after);
            let before = builder.create(arith::And, (filled, not_after));
            let step = builder.create(arith::Select, (before, one, zero));
            let so_far = get(builder, place_count);
            let counted = builder.create(arith::Add, (so_far, step));
            set(builder, place_count, counted);
            Ok(())
        })?;
        let ranked_before = get(builder, place_count);
        let place = builder.create(arith::ToIndex, ranked_before);
        // Move the candidates from its place on one down, the last one out.
        let shifts = constant(builder, capacity - 1);
        for_range(builder, start, shifts, |builder, step| {
            let slot = index(
                builder,
                IndexExpr::constant(capacity - 1) - IndexExpr::iter(0),
                &[step],
            );
            let moves = predicate(
                builder,
                &[less(IndexExpr::iter(0), IndexExpr::iter(1))],
                &[place, slot],
            );
            if_then(builder, moves, |builder| {
                let previous = index(
                    builder,
                    IndexExpr::iter(0) - IndexExpr::constant(1),
                    &[slot],
                );
                let moved_value = state.get_slot(builder, 0, previous);
                let moved_index = state.get_slot(builder, 1, previous);
                state.set_slot(builder, 0, slot, moved_value);
                state.set_slot(builder, 1, slot, moved_index);
                Ok(())
            })
        })?;
        let kept = predicate(
            builder,
            &[less(IndexExpr::iter(0), IndexExpr::constant(capacity))],
            &[place],
        );
        if_then(builder, kept, |builder| {
            state.set_slot(builder, 0, place, value);
            state.set_slot(builder, 1, place, index_value);
            Ok(())
        })?;
        let held_before = state.get(builder, 2);
        let one_more = builder.create(arith::Add, (held_before, one));
        let most = lit(builder, HostScalar::int(DType::Si64, i128::from(capacity))?);
        let held_after = builder.create(arith::Minimum, (one_more, most));
        state.set(builder, 2, held_after);
        Ok(())
    }
}

impl<const LARGEST: bool> FoldLowering for Best<LARGEST> {
    fn components(&self, dtypes: &[DType]) -> Vec<(DType, Vec<usize>)> {
        vec![
            (dtypes[0], vec![self.k]),
            (dtypes[1], vec![self.k]),
            (DType::Si64, Vec::new()),
        ]
    }

    fn start(
        &self,
        builder: &mut Builder<'_>,
        state: &State,
        dtypes: &[DType],
        init: &[Value],
    ) -> Result<()> {
        let fill = lit(builder, Self::fill(dtypes[0])?);
        let missing = lit(builder, HostScalar::int(dtypes[1], -1)?);
        let start = constant(builder, 0);
        let end = constant(builder, self.k as i64);
        for_range(builder, start, end, |builder, slot| {
            state.set_slot(builder, 0, slot, fill);
            state.set_slot(builder, 1, slot, missing);
            Ok(())
        })?;
        let none = lit(builder, HostScalar::int(DType::Si64, 0)?);
        state.set(builder, 2, none);
        if let [value, index_value] = init {
            self.insert(builder, state, *value, *index_value)?;
        }
        Ok(())
    }

    fn push(
        &self,
        builder: &mut Builder<'_>,
        state: &State,
        _sweep: usize,
        yielded: &[Value],
    ) -> Result<()> {
        self.insert(builder, state, yielded[0], yielded[1])
    }

    fn results(&self, builder: &mut Builder<'_>, state: &State) -> Vec<Vec<Value>> {
        let mut values = Vec::with_capacity(self.k);
        let mut indices = Vec::with_capacity(self.k);
        for rank in 0..self.k {
            let at = constant(builder, rank as i64);
            values.push(state.get_slot(builder, 0, at));
            indices.push(state.get_slot(builder, 1, at));
        }
        vec![values, indices]
    }
}

// ------------------------------------------------------------------------------------------------
// `online_softmax`, `logsumexp` and their merges.

/// `e^(from - to)`, the rescale factor of a state whose maximum is `from` to the maximum `to`, or 0
/// when `from` is `-inf`: a state of no points contributes nothing.
fn rescale(builder: &mut Builder<'_>, from: Value, to: Value) -> Result<Value> {
    let difference = builder.create(arith::Sub, (from, to));
    let factor = builder.create(arith::Exp, difference);
    let none = lit(
        builder,
        HostScalar::from_f64(DType::F32, f64::NEG_INFINITY)?,
    );
    let empty = builder.create(arith::Equal, (from, none));
    let zero = lit(builder, HostScalar::from_f64(DType::F32, 0.0)?);
    Ok(builder.create(arith::Select, (empty, zero, factor)))
}

/// Starts the softmax state `(m, l, o)`, or `(m, l)` without values: the maximum, the sum of
/// `e^(s - m)`, and the sum of `e^(s - m) v`, from `init`, or from no point.
fn start_softmax<const WITH_VALUE: bool>(
    builder: &mut Builder<'_>,
    state: &State,
    init: &[Value],
) -> Result<()> {
    let empty = [f64::NEG_INFINITY, 0.0, 0.0];
    let components = if WITH_VALUE { 3 } else { 2 };
    for (component, &start) in empty.iter().enumerate().take(components) {
        let value = match init.get(component) {
            Some(&value) => value,
            None => lit(builder, HostScalar::from_f64(DType::F32, start)?),
        };
        state.set(builder, component, value);
    }
    Ok(())
}

/// `online_softmax` (`WITH_VALUE`) and `logsumexp`: the maximum in the first sweep, then the sums
/// of `e^(s - m)` and of `e^(s - m) v`, the start state rescaled to the maximum. The state keeps
/// the start maximum after its components.
struct TwoPass<const WITH_VALUE: bool>;

impl<const WITH_VALUE: bool> FoldLowering for TwoPass<WITH_VALUE> {
    fn sweeps(&self) -> usize {
        2
    }

    fn components(&self, _dtypes: &[DType]) -> Vec<(DType, Vec<usize>)> {
        let components = if WITH_VALUE { 4 } else { 3 };
        vec![(DType::F32, Vec::new()); components]
    }

    fn start(
        &self,
        builder: &mut Builder<'_>,
        state: &State,
        _dtypes: &[DType],
        init: &[Value],
    ) -> Result<()> {
        start_softmax::<WITH_VALUE>(builder, state, init)?;
        let max = state.get(builder, 0);
        state.set(builder, state.buffers.len() - 1, max);
        Ok(())
    }

    fn next_sweep(&self, builder: &mut Builder<'_>, state: &State, _sweep: usize) -> Result<()> {
        let kept = state.buffers.len() - 1;
        let start_max = state.get(builder, kept);
        let max = state.get(builder, 0);
        let factor = rescale(builder, start_max, max)?;
        for component in 1..kept {
            let value = state.get(builder, component);
            let scaled = builder.create(arith::Mul, (value, factor));
            state.set(builder, component, scaled);
        }
        Ok(())
    }

    fn push(
        &self,
        builder: &mut Builder<'_>,
        state: &State,
        sweep: usize,
        yielded: &[Value],
    ) -> Result<()> {
        let score = yielded[0];
        let max = state.get(builder, 0);
        if sweep == 0 {
            let larger = builder.create(arith::Maximum, (max, score));
            state.set(builder, 0, larger);
            return Ok(());
        }
        let difference = builder.create(arith::Sub, (score, max));
        let weight = builder.create(arith::Exp, difference);
        let sum_so_far = state.get(builder, 1);
        let sum = builder.create(arith::Add, (sum_so_far, weight));
        state.set(builder, 1, sum);
        if WITH_VALUE {
            let weighted_so_far = state.get(builder, 2);
            let term = builder.create(arith::Mul, (weight, yielded[1]));
            let weighted = builder.create(arith::Add, (weighted_so_far, term));
            state.set(builder, 2, weighted);
        }
        Ok(())
    }

    fn results(&self, builder: &mut Builder<'_>, state: &State) -> Vec<Vec<Value>> {
        let components = state.buffers.len() - 1;
        let mut results = Vec::with_capacity(components);
        for component in 0..components {
            results.push(vec![state.get(builder, component)]);
        }
        results
    }
}

/// `online_softmax_merge` (`WITH_VALUE`) and `logsumexp_merge`: each point's state merged into
/// the running one.
struct Merge<const WITH_VALUE: bool>;

impl<const WITH_VALUE: bool> FoldLowering for Merge<WITH_VALUE> {
    fn components(&self, _dtypes: &[DType]) -> Vec<(DType, Vec<usize>)> {
        let components = if WITH_VALUE { 3 } else { 2 };
        vec![(DType::F32, Vec::new()); components]
    }

    fn start(
        &self,
        builder: &mut Builder<'_>,
        state: &State,
        _dtypes: &[DType],
        init: &[Value],
    ) -> Result<()> {
        start_softmax::<WITH_VALUE>(builder, state, init)
    }

    fn push(
        &self,
        builder: &mut Builder<'_>,
        state: &State,
        _sweep: usize,
        yielded: &[Value],
    ) -> Result<()> {
        let max = state.get(builder, 0);
        let merged_max = builder.create(arith::Maximum, (max, yielded[0]));
        let own = rescale(builder, max, merged_max)?;
        let point = rescale(builder, yielded[0], merged_max)?;
        for (component, &part) in yielded.iter().enumerate().skip(1) {
            let held = state.get(builder, component);
            let kept = builder.create(arith::Mul, (held, own));
            let added = builder.create(arith::Mul, (part, point));
            let sum = builder.create(arith::Add, (kept, added));
            state.set(builder, component, sum);
        }
        state.set(builder, 0, merged_max);
        Ok(())
    }

    fn results(&self, builder: &mut Builder<'_>, state: &State) -> Vec<Vec<Value>> {
        let mut results = Vec::with_capacity(state.buffers.len());
        for component in 0..state.buffers.len() {
            results.push(vec![state.get(builder, component)]);
        }
        results
    }
}
