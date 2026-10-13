//! What the contraction ops give their readers: constructors, the axes of a contraction,
//! accessors, and the initial values of the ops that folds combine with.

use std::ops::DerefMut;

use anyhow::{anyhow, ensure, Result};

use crate::dialects::arith::math::{Combiner, Float};
use crate::dialects::tensor::ir::*;
use crate::dialects::tensor::utils::shape::axis_index;
use crate::dialects::tile::CombineKind;
use crate::ir::{DType, IntFormat, Matched, Perm, Ranked, MAX_RANK};

impl DotGeneral {
    /// The contraction of the `lhs` axes with the `rhs` axes, in pairs, without batch axes.
    pub fn contract(lhs: impl AsRef<[i64]>, rhs: impl AsRef<[i64]>) -> Self {
        Self {
            batch_lhs: Ranked::new(),
            batch_rhs: Ranked::new(),
            contract_lhs: lhs.as_ref().iter().copied().collect(),
            contract_rhs: rhs.as_ref().iter().copied().collect(),
            accum: None,
            out: None,
        }
    }

    /// This contraction batched over the `lhs` axes paired with the `rhs` axes.
    pub fn batch(self, lhs: impl AsRef<[i64]>, rhs: impl AsRef<[i64]>) -> Self {
        Self {
            batch_lhs: lhs.as_ref().iter().copied().collect(),
            batch_rhs: rhs.as_ref().iter().copied().collect(),
            ..self
        }
    }

    /// The batch and contract axes for operands of ranks `lhs_rank` and `rhs_rank`, negative ones
    /// counted from the end. Fails when an axis is out of range or an operand's axis is named
    /// twice.
    pub fn axes(&self, lhs_rank: usize, rhs_rank: usize) -> Result<DotAxes> {
        let [batch_lhs, contract_lhs] =
            operand_axes("lhs", lhs_rank, &self.batch_lhs, &self.contract_lhs)?;
        let [batch_rhs, contract_rhs] =
            operand_axes("rhs", rhs_rank, &self.batch_rhs, &self.contract_rhs)?;
        Ok(DotAxes {
            batch_lhs,
            batch_rhs,
            contract_lhs,
            contract_rhs,
            lhs_rank,
            rhs_rank,
        })
    }

    /// This dot with the batch and contracting axes of `axes`.
    pub fn with_axes(self, axes: &DotAxes) -> Self {
        let listed = |axes: &[usize]| axes.iter().map(|&axis| axis as i64).collect();
        Self {
            batch_lhs: listed(&axes.batch_lhs),
            batch_rhs: listed(&axes.batch_rhs),
            contract_lhs: listed(&axes.contract_lhs),
            contract_rhs: listed(&axes.contract_rhs),
            ..self
        }
    }

    /// This contraction accumulating in `accum` and giving `out`.
    pub fn with_dtypes(self, accum: DType, out: DType) -> Self {
        Self {
            accum: Some(accum),
            out: Some(out),
            ..self
        }
    }
}

/// An operand of a `tensor.dot_general`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Side {
    Lhs,
    Rhs,
}

/// The axes of a `tensor.dot_general`, each in `0..rank` of its operand, for the ranks of its
/// operands. The result holds the batch axes, then the free axes of `lhs`, then those of `rhs`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DotAxes {
    pub batch_lhs: Ranked<usize>,
    pub batch_rhs: Ranked<usize>,
    pub contract_lhs: Ranked<usize>,
    pub contract_rhs: Ranked<usize>,
    lhs_rank: usize,
    rhs_rank: usize,
}

impl DotAxes {
    /// The batch axes of the operand on `side`.
    pub fn batch(&self, side: Side) -> &[usize] {
        match side {
            Side::Lhs => &self.batch_lhs,
            Side::Rhs => &self.batch_rhs,
        }
    }

    /// The contracting axes of the operand on `side`.
    pub fn contract(&self, side: Side) -> &[usize] {
        match side {
            Side::Lhs => &self.contract_lhs,
            Side::Rhs => &self.contract_rhs,
        }
    }

    /// The rank of the operand on `side`.
    pub fn rank(&self, side: Side) -> usize {
        match side {
            Side::Lhs => self.lhs_rank,
            Side::Rhs => self.rhs_rank,
        }
    }

    /// The free axes of the operand on `side`, neither batch nor contracting, in axis order.
    pub fn free(&self, side: Side) -> Ranked<usize> {
        let (batch, contract) = (self.batch(side), self.contract(side));
        let mut free = Ranked::new();
        for axis in 0..self.rank(side) {
            if !batch.contains(&axis) && !contract.contains(&axis) {
                free.push(axis);
            }
        }
        free
    }

    /// The result axis that each axis of the operand on `side` lands on, or `None` for a
    /// contracting axis.
    pub fn result_axes(&self, side: Side) -> Ranked<Option<usize>> {
        let mut landing = Ranked::new();
        landing.resize(self.rank(side), None);
        for (result_axis, &axis) in self.batch(side).iter().enumerate() {
            landing[axis] = Some(result_axis);
        }
        let first_free = match side {
            Side::Lhs => self.batch_lhs.len(),
            Side::Rhs => self.batch_lhs.len() + self.free(Side::Lhs).len(),
        };
        for (offset, &axis) in self.free(side).iter().enumerate() {
            landing[axis] = Some(first_free + offset);
        }
        landing
    }

    /// These axes with the operand on `side` read through a transpose by `perm`: its batch and
    /// contracting axes name the axes of the transpose's input they read. The free axes land on
    /// the same result axes only when `perm` keeps their order.
    pub fn through(&self, side: Side, perm: &Perm) -> DotAxes {
        let batch = perm.read_axes(self.batch(side));
        let contract = perm.read_axes(self.contract(side));
        let mut axes = self.clone();
        match side {
            Side::Lhs => (axes.batch_lhs, axes.contract_lhs) = (batch, contract),
            Side::Rhs => (axes.batch_rhs, axes.contract_rhs) = (batch, contract),
        }
        axes
    }
}

impl ReduceAttrs {
    /// The axes of an input of rank `rank` that the reduction keeps, in increasing order.
    pub fn kept_axes(&self, rank: usize) -> Result<Ranked<usize>> {
        let reduced = self.reduced_axes(rank)?;
        Ok((0..rank).filter(|axis| !reduced.contains(axis)).collect())
    }

    /// The reduced axes of an input of rank `rank`, negative ones counted from the end, in
    /// increasing order. Fails when an axis is out of range or named twice.
    pub fn reduced_axes(&self, rank: usize) -> Result<Ranked<usize>> {
        let mut axes = Ranked::new();
        for &axis in &self.axes {
            let index = axis_index(axis, rank)?;
            ensure!(!axes.contains(&index), "axis {axis} is named twice");
            axes.push(index);
        }
        axes.sort_unstable();
        Ok(axes)
    }

    /// `dims` with each of the reduced `axes` set to `one`, or dropped without `keepdims`.
    pub(crate) fn reduced_dims<D: Clone + Default>(
        &self,
        axes: &[usize],
        dims: &[D],
        one: D,
    ) -> Ranked<D> {
        let mut out = Ranked::new();
        for (axis, dim) in dims.iter().enumerate() {
            if !axes.contains(&axis) {
                out.push(dim.clone());
            } else if self.keepdims {
                out.push(one.clone());
            }
        }
        out
    }
}

/// The `batch` and `contract` axes of the `side` operand, of rank `rank`, each in range and named
/// once.
fn operand_axes(
    side: &str,
    rank: usize,
    batch: &[i64],
    contract: &[i64],
) -> Result<[Ranked<usize>; 2]> {
    let mut named = [false; MAX_RANK];
    let mut out = [Ranked::new(), Ranked::new()];
    for (indices, (group, axes)) in out
        .iter_mut()
        .zip([("batch", batch), ("contract", contract)])
    {
        for &axis in axes {
            let index = axis_index(axis, rank).map_err(|_| {
                anyhow!("{group}_{side} axis {axis} is out of range for rank {rank}")
            })?;
            ensure!(
                !std::mem::replace(&mut named[index], true),
                "{side} axis {axis} is named twice"
            );
            indices.push(index);
        }
    }
    Ok(out)
}

impl Matched<'_, DotGeneral> {
    /// The dot's axes, non-negative, for the ranks of its operands.
    pub fn axes(&self) -> DotAxes {
        let (lhs_rank, rhs_rank) = (
            self.lhs().tensor_type().rank(),
            self.rhs().tensor_type().rank(),
        );
        self.attrs()
            .axes(lhs_rank, rhs_rank)
            .expect("a verified dot has valid axes")
    }
}

impl<T: Reduce> Matched<'_, T> {
    /// The reduced axes, non-negative and sorted, for the rank of the input.
    pub fn reduced_axes(&self) -> Ranked<usize> {
        let rank = self.input().tensor_type().rank();
        self.attrs()
            .reduced_axes(rank)
            .expect("a verified reduction has valid axes")
    }

    /// The input axis that result axis `axis` holds: the same axis with `keepdims`, otherwise the
    /// kept axis of that index.
    pub fn input_axis(&self, axis: usize) -> usize {
        if self.keepdims {
            return axis;
        }
        let rank = self.input().tensor_type().rank();
        let kept = self
            .attrs()
            .kept_axes(rank)
            .expect("a verified reduction has valid axes");
        kept[axis]
    }
}

impl Combiner for Add {
    const COMBINE: CombineKind = CombineKind::Sum;
    fn float_init<F: Float>() -> Option<F> {
        Some(F::ZERO)
    }
    fn int_init(_format: IntFormat) -> i128 {
        0
    }
}

impl Combiner for Maximum {
    const COMBINE: CombineKind = CombineKind::Max;
    fn float_init<F: Float>() -> Option<F> {
        Some(F::NEG_INFINITY)
    }
    fn int_init(format: IntFormat) -> i128 {
        format.min()
    }
}

impl Combiner for Minimum {
    const COMBINE: CombineKind = CombineKind::Min;
    fn float_init<F: Float>() -> Option<F> {
        Some(F::INFINITY)
    }
    fn int_init(format: IntFormat) -> i128 {
        format.max()
    }
}

/// Every bit set: true for `i1`, so on `i1` a fold with `and` is "all".
impl Combiner for And {
    const COMBINE: CombineKind = CombineKind::And;
    fn float_init<F: Float>() -> Option<F> {
        None
    }
    fn int_init(_format: IntFormat) -> i128 {
        -1
    }
}

/// 0, so on `i1` a fold with `or` is "any".
impl Combiner for Or {
    const COMBINE: CombineKind = CombineKind::Or;
    fn float_init<F: Float>() -> Option<F> {
        None
    }
    fn int_init(_format: IntFormat) -> i128 {
        0
    }
}

/// The dtypes of an op that accumulates its input: `accum`, by default the default accumulator
/// of the input dtype, and `out`, by default the input dtype.
pub trait Accumulates {
    /// The attributes `accum` and `out`.
    fn dtypes(&self) -> (Option<DType>, Option<DType>);

    fn dtypes_mut(&mut self) -> (&mut Option<DType>, &mut Option<DType>);

    /// The accumulator dtype for an input of dtype `input`.
    fn accum_dtype(&self, input: DType) -> DType {
        self.dtypes().0.unwrap_or(input.default_accumulator())
    }

    /// The result dtype for an input of dtype `input`.
    fn out_dtype(&self, input: DType) -> DType {
        self.dtypes().1.unwrap_or(input)
    }
}

impl Accumulates for DotGeneral {
    fn dtypes(&self) -> (Option<DType>, Option<DType>) {
        (self.accum, self.out)
    }

    fn dtypes_mut(&mut self) -> (&mut Option<DType>, &mut Option<DType>) {
        (&mut self.accum, &mut self.out)
    }
}

impl Accumulates for ReduceAttrs {
    fn dtypes(&self) -> (Option<DType>, Option<DType>) {
        (self.accum, self.out)
    }

    fn dtypes_mut(&mut self) -> (&mut Option<DType>, &mut Option<DType>) {
        (&mut self.accum, &mut self.out)
    }
}

impl Accumulates for ReduceWindowAttrs {
    fn dtypes(&self) -> (Option<DType>, Option<DType>) {
        (self.accum, self.out)
    }

    fn dtypes_mut(&mut self) -> (&mut Option<DType>, &mut Option<DType>) {
        (&mut self.accum, &mut self.out)
    }
}

/// An op of a family accumulates as its attribute group does.
impl<T: DerefMut<Target = A>, A: Accumulates + 'static> Accumulates for T {
    fn dtypes(&self) -> (Option<DType>, Option<DType>) {
        (**self).dtypes()
    }

    fn dtypes_mut(&mut self) -> (&mut Option<DType>, &mut Option<DType>) {
        (**self).dtypes_mut()
    }
}
