//! What each combine kind of `tile.block` takes and gives: its [`CombineRule`].

use std::marker::PhantomData;

use anyhow::{ensure, Result};

use crate::dialects::arith::math::Combiner;
use crate::dialects::tensor::utils::class::{ensure_dtype_class, ensure_index_dtype, ARITHMETIC};
use crate::dialects::tile::ir::*;
use crate::dialects::tile::IterList;
use crate::dialects::{arith, tensor};
use crate::ir::{DType, Dim};

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
    /// The lanes of a softmax kind, which name iterators.
    pub fn lanes_mut(&mut self) -> Option<&mut IterList> {
        match self {
            CombineKind::OnlineSoftmax(softmax) => Some(&mut softmax.lanes),
            CombineKind::OnlineSoftmaxMerge(softmax) => Some(&mut softmax.lanes),
            _ => None,
        }
    }

    /// What the kind takes and gives.
    pub fn rule(&self) -> &dyn CombineRule {
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
    fn yields(&self) -> usize {
        1
    }

    fn results(&self) -> usize {
        1
    }

    fn result_dtypes(&self, yields: &[DType]) -> Result<Vec<DType>> {
        ensure_yields("none", yields, 1)?;
        Ok(yields.to_vec())
    }

    fn extra_dims(&self, _result: usize) -> Vec<Dim> {
        Vec::new()
    }

    fn lanes(&self) -> IterList {
        IterList::default()
    }

    fn drops_lanes(&self, _result: usize) -> bool {
        false
    }

    fn exact(&self, _dtype: DType) -> bool {
        true
    }

    fn init_count(&self) -> usize {
        0
    }

    fn folds_into_operand(&self) -> bool {
        true
    }
}

impl<C: Combiner> CombineRule for Monoid<C> {
    fn yields(&self) -> usize {
        1
    }

    fn results(&self) -> usize {
        1
    }

    fn result_dtypes(&self, yields: &[DType]) -> Result<Vec<DType>> {
        let kind = C::COMBINE.name();
        ensure_yields(kind, yields, 1)?;
        let classes = <C::Scalar as arith::Binary>::CLASSES;
        ensure_dtype_class(&format!("the value that {kind} folds"), yields[0], classes)?;
        Ok(yields.to_vec())
    }

    fn extra_dims(&self, _result: usize) -> Vec<Dim> {
        Vec::new()
    }

    fn lanes(&self) -> IterList {
        IterList::default()
    }

    fn drops_lanes(&self, _result: usize) -> bool {
        false
    }

    /// Only a float sum rounds differently when its points regroup.
    fn exact(&self, dtype: DType) -> bool {
        C::COMBINE != CombineKind::Sum || dtype.is_integer()
    }

    fn init_count(&self) -> usize {
        1
    }

    fn folds_into_operand(&self) -> bool {
        true
    }
}

impl<const LARGEST: bool> CombineRule for Arg<LARGEST> {
    fn yields(&self) -> usize {
        2
    }

    fn results(&self) -> usize {
        2
    }

    fn result_dtypes(&self, yields: &[DType]) -> Result<Vec<DType>> {
        ensure_ranked(if LARGEST { "argmax" } else { "argmin" }, yields)?;
        Ok(yields.to_vec())
    }

    fn extra_dims(&self, _result: usize) -> Vec<Dim> {
        Vec::new()
    }

    fn lanes(&self) -> IterList {
        IterList::default()
    }

    fn drops_lanes(&self, _result: usize) -> bool {
        false
    }

    fn exact(&self, _dtype: DType) -> bool {
        true
    }

    fn init_count(&self) -> usize {
        2
    }

    fn folds_into_operand(&self) -> bool {
        false
    }
}

impl<const LARGEST: bool> CombineRule for KBest<LARGEST> {
    fn yields(&self) -> usize {
        2
    }

    fn results(&self) -> usize {
        2
    }

    fn result_dtypes(&self, yields: &[DType]) -> Result<Vec<DType>> {
        ensure_ranked(if LARGEST { "top_k" } else { "bottom_k" }, yields)?;
        Ok(yields.to_vec())
    }

    fn extra_dims(&self, _result: usize) -> Vec<Dim> {
        vec![Dim::Static(self.k)]
    }

    fn lanes(&self) -> IterList {
        IterList::default()
    }

    fn drops_lanes(&self, _result: usize) -> bool {
        false
    }

    fn exact(&self, _dtype: DType) -> bool {
        true
    }

    fn init_count(&self) -> usize {
        0
    }

    fn folds_into_operand(&self) -> bool {
        false
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

    fn result_dtypes(&self, yields: &[DType]) -> Result<Vec<DType>> {
        let kind = if MERGE {
            "online_softmax_merge"
        } else {
            "online_softmax"
        };
        ensure_yields(kind, yields, self.yields())?;
        f32_state(kind, yields, 3)
    }

    fn extra_dims(&self, _result: usize) -> Vec<Dim> {
        Vec::new()
    }

    fn lanes(&self) -> IterList {
        self.lanes
    }

    /// The `m` and `l` states.
    fn drops_lanes(&self, result: usize) -> bool {
        result < 2
    }

    fn exact(&self, _dtype: DType) -> bool {
        false
    }

    fn init_count(&self) -> usize {
        3
    }

    fn folds_into_operand(&self) -> bool {
        false
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

    fn result_dtypes(&self, yields: &[DType]) -> Result<Vec<DType>> {
        let kind = if MERGE {
            "logsumexp_merge"
        } else {
            "logsumexp"
        };
        ensure_yields(kind, yields, self.yields())?;
        f32_state(kind, yields, 2)
    }

    fn extra_dims(&self, _result: usize) -> Vec<Dim> {
        Vec::new()
    }

    fn lanes(&self) -> IterList {
        IterList::default()
    }

    fn drops_lanes(&self, _result: usize) -> bool {
        false
    }

    fn exact(&self, _dtype: DType) -> bool {
        false
    }

    fn init_count(&self) -> usize {
        2
    }

    fn folds_into_operand(&self) -> bool {
        false
    }
}
