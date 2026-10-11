//! [`Acc`], the fold of a combine kind as the interpreter runs it: one accumulator per result
//! element, folding the values the body yields at each point of the reduction iterators.

use anyhow::{anyhow, bail, Result};

use crate::dialects::tensor::interp::kernels::best_k;
use crate::dialects::tile::CombineKind;
use crate::dialects::{arith, tensor};
use crate::host::{HostScalar, ScalarValue};
use crate::scalar::{self, Combiner};
use crate::types::DType;

/// The fold of one combine kind for one result element.
pub struct Acc(Box<dyn Accumulator>);

/// What a fold does with each point's yielded values, and what it gives at the end.
trait Accumulator {
    fn push(&mut self, values: &[HostScalar]) -> Result<()>;
    /// The results so far, in result order.
    fn finish(&self) -> Result<Vec<HostScalar>>;
}

impl Acc {
    /// The accumulator of `kind` before any point, for yielded values of `dtypes`, starting from
    /// the state `init` when there is one.
    pub fn new(kind: &CombineKind, dtypes: &[DType], init: &[HostScalar]) -> Result<Acc> {
        let accumulator: Box<dyn Accumulator> = match kind {
            CombineKind::None => Box::new(Last(None)),
            CombineKind::Sum => Box::new(Monoid::<tensor::Add>::new(dtypes, init)?),
            CombineKind::Max => Box::new(Monoid::<tensor::Maximum>::new(dtypes, init)?),
            CombineKind::Min => Box::new(Monoid::<tensor::Minimum>::new(dtypes, init)?),
            CombineKind::And => Box::new(Monoid::<tensor::And>::new(dtypes, init)?),
            CombineKind::Or => Box::new(Monoid::<tensor::Or>::new(dtypes, init)?),
            CombineKind::Argmax => Box::new(Ranked::new(1, true, dtypes, init)),
            CombineKind::Argmin => Box::new(Ranked::new(1, false, dtypes, init)),
            CombineKind::TopK(best) => Box::new(Ranked::new(best.k, true, dtypes, init)),
            CombineKind::BottomK(best) => Box::new(Ranked::new(best.k, false, dtypes, init)),
            CombineKind::OnlineSoftmax(_) => Box::new(TwoPass::<true>::new(init)?),
            CombineKind::OnlineSoftmaxMerge(_) => Box::new(Merge::<true>::new(init)?),
            CombineKind::Logsumexp => Box::new(TwoPass::<false>::new(init)?),
            CombineKind::LogsumexpMerge => Box::new(Merge::<false>::new(init)?),
        };
        Ok(Acc(accumulator))
    }

    /// Folds one point's yielded values in.
    pub fn push(&mut self, values: &[HostScalar]) -> Result<()> {
        self.0.push(values)
    }

    /// The results of the fold so far, in result order.
    pub fn finish(&self) -> Result<Vec<HostScalar>> {
        self.0.finish()
    }
}

/// `none`: the last value written.
struct Last(Option<HostScalar>);

impl Accumulator for Last {
    fn push(&mut self, values: &[HostScalar]) -> Result<()> {
        self.0 = Some(values[0]);
        Ok(())
    }

    fn finish(&self) -> Result<Vec<HostScalar>> {
        let last = self
            .0
            .ok_or_else(|| anyhow!("no point wrote this element"))?;
        Ok(vec![last])
    }
}

/// `sum`, `max`, `min`, `and` and `or`: the running value of a fold with the `tensor` op `C`.
struct Monoid<C> {
    value: HostScalar,
    combiner: std::marker::PhantomData<C>,
}

impl<C: Combiner> Monoid<C> {
    /// The fold from `init`, or from the initial value of `C` for the dtype `dtypes[0]`.
    fn new(dtypes: &[DType], init: &[HostScalar]) -> Result<Self> {
        let value = match init {
            [value] => *value,
            _ => scalar::fold_init::<C>(dtypes[0])?,
        };
        Ok(Monoid {
            value,
            combiner: std::marker::PhantomData,
        })
    }
}

impl<C: Combiner> Accumulator for Monoid<C> {
    fn push(&mut self, values: &[HostScalar]) -> Result<()> {
        self.value = scalar::binary::<C::Scalar>(self.value, values[0])?;
        Ok(())
    }

    fn finish(&self) -> Result<Vec<HostScalar>> {
        Ok(vec![self.value])
    }
}

/// `argmax`, `argmin`, `top_k` and `bottom_k`: every candidate `(value, index)`, the start state
/// first, ranked when the fold finishes.
struct Ranked {
    k: usize,
    largest: bool,
    items: Vec<(HostScalar, HostScalar)>,
    value_dtype: DType,
    index_dtype: DType,
}

impl Ranked {
    fn new(k: usize, largest: bool, dtypes: &[DType], init: &[HostScalar]) -> Self {
        let items = match init {
            [value, index] => vec![(*value, *index)],
            _ => Vec::new(),
        };
        Ranked {
            k,
            largest,
            items,
            value_dtype: dtypes[0],
            index_dtype: dtypes[1],
        }
    }
}

impl Accumulator for Ranked {
    fn push(&mut self, values: &[HostScalar]) -> Result<()> {
        self.items.push((values[0], values[1]));
        Ok(())
    }

    /// The `k` best values, then their indices. A fold of fewer than `k` points fills the rest
    /// with the identity of its order and the index -1.
    fn finish(&self) -> Result<Vec<HostScalar>> {
        // Equal values rank by index: the candidates in index order, then ranked by value.
        let mut by_index = self.items.clone();
        by_index.sort_by_key(|&(_, index)| index.as_int());
        let ranking = best_k(by_index.len(), self.k, self.largest, |first, second| {
            scalar::order(by_index[first].0, by_index[second].0)
        });
        let fill = if self.largest {
            scalar::fold_init::<tensor::Maximum>(self.value_dtype)?
        } else {
            scalar::fold_init::<tensor::Minimum>(self.value_dtype)?
        };
        let mut values = Vec::with_capacity(self.k);
        let mut indices = Vec::with_capacity(self.k);
        for rank in 0..self.k {
            match ranking.get(rank) {
                Some(&position) => {
                    values.push(by_index[position].0);
                    indices.push(by_index[position].1);
                }
                None => {
                    values.push(fill);
                    indices.push(HostScalar::int(self.index_dtype, -1)?);
                }
            }
        }
        Ok([values, indices].concat())
    }
}

fn f32_of(value: HostScalar) -> Result<f32> {
    match value.value {
        ScalarValue::F32(x) if value.dtype == DType::F32 => Ok(x),
        _ => bail!("expected an f32 value, got a {}", value.dtype),
    }
}

fn f32_scalar(x: f32) -> HostScalar {
    HostScalar {
        dtype: DType::F32,
        value: ScalarValue::F32(x),
    }
}

/// `e^(from - to)`, the rescale factor of a state whose maximum is `from` to a new maximum `to`.
/// A state with no points, whose maximum is `-inf`, contributes nothing: its factor is 0.
fn rescale(from: f32, to: f32) -> Result<f32> {
    if from == f32::NEG_INFINITY {
        return Ok(0.0);
    }
    let difference = scalar::binary::<arith::Sub>(f32_scalar(from), f32_scalar(to))?;
    f32_of(scalar::unary::<arith::Exp>(difference)?)
}

fn add(lhs: f32, rhs: f32) -> Result<f32> {
    f32_of(scalar::binary::<arith::Add>(
        f32_scalar(lhs),
        f32_scalar(rhs),
    )?)
}

fn mul(lhs: f32, rhs: f32) -> Result<f32> {
    f32_of(scalar::binary::<arith::Mul>(
        f32_scalar(lhs),
        f32_scalar(rhs),
    )?)
}

fn maximum(lhs: f32, rhs: f32) -> Result<f32> {
    f32_of(scalar::binary::<arith::Maximum>(
        f32_scalar(lhs),
        f32_scalar(rhs),
    )?)
}

/// A softmax state `(m, l, o)`: the maximum score, the sum of `e^(s - m)`, and the sum of
/// `e^(s - m) v`.
#[derive(Clone, Copy)]
struct State {
    max: f32,
    sum: f32,
    weighted: f32,
}

impl State {
    /// The state `init`, or the state of no points.
    fn start(init: &[HostScalar]) -> Result<Self> {
        let component = |index: usize, default: f32| -> Result<f32> {
            init.get(index).map_or(Ok(default), |&value| f32_of(value))
        };
        Ok(State {
            max: component(0, f32::NEG_INFINITY)?,
            sum: component(1, 0.0)?,
            weighted: component(2, 0.0)?,
        })
    }

    /// The results: `(m, l, o)` with values, `(m, l)` without.
    fn results<const WITH_VALUE: bool>(self) -> Vec<HostScalar> {
        let mut results = vec![f32_scalar(self.max), f32_scalar(self.sum)];
        if WITH_VALUE {
            results.push(f32_scalar(self.weighted));
        }
        results
    }
}

/// `online_softmax` (`WITH_VALUE`) and `logsumexp` on scores: the start state, then every point's
/// score and value, folded in two passes when the fold finishes.
struct TwoPass<const WITH_VALUE: bool> {
    start: State,
    points: Vec<(f32, f32)>,
}

impl<const WITH_VALUE: bool> TwoPass<WITH_VALUE> {
    fn new(init: &[HostScalar]) -> Result<Self> {
        Ok(TwoPass {
            start: State::start(init)?,
            points: Vec::new(),
        })
    }
}

impl<const WITH_VALUE: bool> Accumulator for TwoPass<WITH_VALUE> {
    fn push(&mut self, values: &[HostScalar]) -> Result<()> {
        let value = if WITH_VALUE { f32_of(values[1])? } else { 0.0 };
        self.points.push((f32_of(values[0])?, value));
        Ok(())
    }

    /// Two passes in point order: the maximum, then the sums of `e^(s - m)` and of `e^(s - m) v`,
    /// each step rounded to f32.
    fn finish(&self) -> Result<Vec<HostScalar>> {
        let mut max = self.start.max;
        for &(score, _) in &self.points {
            max = maximum(max, score)?;
        }
        let factor = rescale(self.start.max, max)?;
        let mut sum = mul(self.start.sum, factor)?;
        let mut weighted = mul(self.start.weighted, factor)?;
        for &(score, value) in &self.points {
            let difference = scalar::binary::<arith::Sub>(f32_scalar(score), f32_scalar(max))?;
            let weight = f32_of(scalar::unary::<arith::Exp>(difference)?)?;
            sum = add(sum, weight)?;
            if WITH_VALUE {
                weighted = add(weighted, mul(weight, value)?)?;
            }
        }
        let state = State { max, sum, weighted };
        Ok(state.results::<WITH_VALUE>())
    }
}

/// `online_softmax_merge` (`WITH_VALUE`) and `logsumexp_merge`: the running state, each point's
/// state merged in.
struct Merge<const WITH_VALUE: bool> {
    state: State,
}

impl<const WITH_VALUE: bool> Merge<WITH_VALUE> {
    fn new(init: &[HostScalar]) -> Result<Self> {
        Ok(Merge {
            state: State::start(init)?,
        })
    }
}

impl<const WITH_VALUE: bool> Accumulator for Merge<WITH_VALUE> {
    fn push(&mut self, values: &[HostScalar]) -> Result<()> {
        let point_max = f32_of(values[0])?;
        let point_sum = f32_of(values[1])?;
        let point_weighted = if WITH_VALUE { f32_of(values[2])? } else { 0.0 };
        let max = maximum(self.state.max, point_max)?;
        let (own, point) = (rescale(self.state.max, max)?, rescale(point_max, max)?);
        self.state = State {
            max,
            sum: add(mul(self.state.sum, own)?, mul(point_sum, point)?)?,
            weighted: add(mul(self.state.weighted, own)?, mul(point_weighted, point)?)?,
        };
        Ok(())
    }

    fn finish(&self) -> Result<Vec<HostScalar>> {
        Ok(self.state.results::<WITH_VALUE>())
    }
}
