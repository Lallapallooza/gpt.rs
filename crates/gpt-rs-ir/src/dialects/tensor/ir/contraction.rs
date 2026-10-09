//! Ops that combine many elements into one.

use anyhow::{anyhow, ensure, Result};
use smallvec::SmallVec;

use super::class::{ensure_class, ensure_dtype_class, Class, ARITHMETIC};
use super::elementwise::BinaryKind;
use super::shape::{axis_index, free_axes};
use crate::op::{NoRegions, OpDef, OwnedTypes, TypeCx, Types};
use crate::types::{DType, Dim, TensorType};
use crate::Attr;

/// `tensor.dot_general`: a contraction over `contract_*` axes, batched over `batch_*` axes, like
/// an einsum. Each result element sums the products of the `lhs` and `rhs` elements at its batch
/// and free coordinates, over every coordinate of the contracting axes.
///
/// - `batch_lhs` pairs with `batch_rhs`, and `contract_lhs` with `contract_rhs`; the axes of each
///   pair MUST have matching extents. Every axis MUST be in range, a negative axis counting from
///   the end, and an operand's axis MUST appear at most once in its batch and contracting axes.
/// - The result axes are the batch axes, in `batch_lhs` order, then the remaining non-contracting
///   axes of `lhs`, then those of `rhs`, each in axis order.
/// - The operand dtypes MAY differ: integers with integers, and floats or narrow floats with each
///   other. For example, FP8 training multiplies e5m2 gradients by e4m3 weights, and W4A8
///   multiplies int4 weights by int8 activations. Then `accum` and `out` MUST be given.
/// - It accumulates the products in `accum` and casts the sums to `out`. The accumulator policy
///   gives their defaults. A wider `accum` holds the products of narrower inputs exactly, for
///   example bf16 x bf16 with f32 accumulation.
#[derive(crate::Op, Debug, Clone, PartialEq, Eq, Hash)]
#[op(
    "tensor.dot_general",
    operands(lhs: Tensor, rhs: Tensor),
    results(output: Tensor)
)]
pub struct DotGeneral {
    #[attr(default)]
    pub batch_lhs: SmallVec<[i64; 4]>,
    #[attr(default)]
    pub batch_rhs: SmallVec<[i64; 4]>,
    pub contract_lhs: SmallVec<[i64; 4]>,
    pub contract_rhs: SmallVec<[i64; 4]>,
    /// The accumulator dtype, when it differs from [`DType::default_accumulator`] of the operands.
    #[attr(default)]
    pub accum: Option<DType>,
    /// The result dtype, when it differs from the operands'.
    #[attr(default)]
    pub out: Option<DType>,
}

impl DotGeneral {
    /// The contraction of the `lhs` axes with the `rhs` axes, in pairs, without batch axes.
    pub fn contract(lhs: impl AsRef<[i64]>, rhs: impl AsRef<[i64]>) -> Self {
        Self {
            batch_lhs: SmallVec::new(),
            batch_rhs: SmallVec::new(),
            contract_lhs: SmallVec::from_slice(lhs.as_ref()),
            contract_rhs: SmallVec::from_slice(rhs.as_ref()),
            accum: None,
            out: None,
        }
    }

    /// This contraction batched over the `lhs` axes paired with the `rhs` axes.
    pub fn batch(self, lhs: impl AsRef<[i64]>, rhs: impl AsRef<[i64]>) -> Self {
        Self {
            batch_lhs: SmallVec::from_slice(lhs.as_ref()),
            batch_rhs: SmallVec::from_slice(rhs.as_ref()),
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
        })
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

impl OpDef for DotGeneral {
    fn result_types(
        &self,
        ins: DotGeneralOperands<Types<'_>>,
        _cx: &TypeCx<'_, NoRegions>,
    ) -> Result<DotGeneralResults<OwnedTypes>> {
        let (lhs, rhs) = (ins.lhs, ins.rhs);
        // Integers multiply integers, and floats, narrow ones included, multiply floats.
        let family: &[Class] = match Class::of(lhs.dtype) {
            Class::Int => &[Class::Int],
            Class::Float | Class::Narrow => &[Class::Float, Class::Narrow],
            _ => &[Class::Int, Class::Float, Class::Narrow],
        };
        ensure_class("lhs", lhs, family)?;
        ensure_class("rhs", rhs, family)?;
        ensure!(
            lhs.dtype == rhs.dtype || (self.accum.is_some() && self.out.is_some()),
            "operands of different dtypes, {lhs} and {rhs}, need accum and out"
        );
        if let Some(accum) = self.accum {
            ensure_dtype_class("accum", accum, &family[..1])?;
        }
        if let Some(out) = self.out {
            ensure_dtype_class("out", out, family)?;
        }
        ensure!(
            self.batch_lhs.len() == self.batch_rhs.len(),
            "batch_lhs {:?} and batch_rhs {:?} differ in length",
            self.batch_lhs,
            self.batch_rhs
        );
        ensure!(
            self.contract_lhs.len() == self.contract_rhs.len(),
            "contract_lhs {:?} and contract_rhs {:?} differ in length",
            self.contract_lhs,
            self.contract_rhs
        );
        let axes = self.axes(lhs.rank(), rhs.rank())?;
        for (group, lhs_axes, rhs_axes) in [
            ("batch", &axes.batch_lhs, &axes.batch_rhs),
            ("contract", &axes.contract_lhs, &axes.contract_rhs),
        ] {
            for (&l, &r) in lhs_axes.iter().zip(rhs_axes) {
                ensure!(
                    lhs.dims[l] == rhs.dims[r],
                    "{group}_lhs axis {l} has extent {}, but {group}_rhs axis {r} has extent {}",
                    lhs.dims[l],
                    rhs.dims[r]
                );
            }
        }
        let lhs_free = free_axes(lhs.rank(), &axes.batch_lhs, &axes.contract_lhs);
        let rhs_free = free_axes(rhs.rank(), &axes.batch_rhs, &axes.contract_rhs);
        let dims = axes
            .batch_lhs
            .iter()
            .chain(&lhs_free)
            .map(|&axis| lhs.dims[axis].clone())
            .chain(rhs_free.iter().map(|&axis| rhs.dims[axis].clone()))
            .collect();
        Ok(DotGeneralResults {
            output: TensorType {
                dtype: self.out.unwrap_or(lhs.dtype),
                dims,
            },
        })
    }
}

/// The axes of a `tensor.dot_general`, each in `0..rank` of its operand.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DotAxes {
    pub batch_lhs: SmallVec<[usize; 4]>,
    pub batch_rhs: SmallVec<[usize; 4]>,
    pub contract_lhs: SmallVec<[usize; 4]>,
    pub contract_rhs: SmallVec<[usize; 4]>,
}

/// A reduction: it folds its elements with a binary op, starting from its initial value, which a
/// reduction of no elements gives.
#[derive(Attr, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ReduceKind {
    /// Folds with `add`. The initial value is 0.
    Sum,
    /// Folds with `maximum`. The initial value is `-inf` for floats and the smallest value of the
    /// dtype for integers.
    Max,
    /// Folds with `minimum`. The initial value is `+inf` for floats and the largest value of the
    /// dtype for integers.
    Min,
    /// Folds with `and`. The initial value has every bit set, true for `i1`, so on `i1` it is
    /// "all".
    And,
    /// Folds with `or`. The initial value is 0, so on `i1` it is "any".
    Or,
}

impl ReduceKind {
    /// The binary op that combines two partial results. The reduction folds the elements with it.
    pub fn combiner(self) -> BinaryKind {
        match self {
            ReduceKind::Sum => BinaryKind::Add,
            ReduceKind::Max => BinaryKind::Maximum,
            ReduceKind::Min => BinaryKind::Minimum,
            ReduceKind::And => BinaryKind::And,
            ReduceKind::Or => BinaryKind::Or,
        }
    }
}

/// Checks the dtypes of a `kind` reduction of `input` elements, accumulated in `accum` and
/// returned as `out`: the input is of a class the kind computes on, and `accum` and `out` are of
/// the input's class.
pub(super) fn ensure_reduction(
    kind: ReduceKind,
    input: &TensorType,
    accum: Option<DType>,
    out: Option<DType>,
) -> Result<()> {
    ensure_class("the input", input, kind.combiner().classes())?;
    let class = [Class::of(input.dtype)];
    for (what, dtype) in [("accum", accum), ("out", out)] {
        if let Some(dtype) = dtype {
            ensure_dtype_class(what, dtype, &class)?;
        }
    }
    Ok(())
}

/// `tensor.reduce_sum`, `tensor.reduce_max`, ...: reduces the input over `axes` with `kind`.
///
/// - `axes` MUST be unique and in range. A negative axis counts from the end, and the order of
///   `axes` does not matter.
/// - If `keepdims` is true, the reduced axes remain with extent 1; otherwise they are removed.
/// - It accumulates in `accum` and casts the result to `out`. The accumulator policy gives their
///   defaults.
/// - When a reduced axis has extent 0, each result element is the identity of `kind`.
#[derive(crate::Op, Debug, Clone, PartialEq, Eq, Hash)]
#[op(
    "tensor.reduce_{kind}",
    operands(input: Tensor),
    results(output: Tensor)
)]
pub struct Reduce {
    pub kind: ReduceKind,
    pub axes: SmallVec<[i64; 4]>,
    #[attr(default)]
    pub keepdims: bool,
    /// The accumulator dtype, when it differs from [`DType::default_accumulator`] of the input.
    #[attr(default)]
    pub accum: Option<DType>,
    /// The result dtype, when it differs from the input's.
    #[attr(default)]
    pub out: Option<DType>,
}

impl OpDef for Reduce {
    fn result_types(
        &self,
        ins: ReduceOperands<Types<'_>>,
        _cx: &TypeCx<'_, NoRegions>,
    ) -> Result<ReduceResults<OwnedTypes>> {
        let input = ins.input;
        ensure_reduction(self.kind, input, self.accum, self.out)?;
        let axes = self.reduced_axes(input.rank())?;
        Ok(ReduceResults {
            output: TensorType {
                dtype: self.out.unwrap_or(input.dtype),
                dims: self.reduced_dims(&axes, &input.dims, Dim::Static(1)),
            },
        })
    }
}

impl Reduce {
    /// The reduced axes of an input of rank `rank`, negative ones counted from the end, in
    /// increasing order. Fails when an axis is out of range or named twice.
    pub fn reduced_axes(&self, rank: usize) -> Result<SmallVec<[usize; 4]>> {
        let mut axes = SmallVec::new();
        for &axis in &self.axes {
            let index = axis_index(axis, rank)?;
            ensure!(!axes.contains(&index), "axis {axis} is named twice");
            axes.push(index);
        }
        axes.sort_unstable();
        Ok(axes)
    }

    /// `dims` with each of the reduced `axes` set to `one`, or dropped without `keepdims`.
    pub(crate) fn reduced_dims<D: Clone>(
        &self,
        axes: &[usize],
        dims: &[D],
        one: D,
    ) -> SmallVec<[D; 4]> {
        let mut out = SmallVec::new();
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
) -> Result<[SmallVec<[usize; 4]>; 2]> {
    let mut named = SmallVec::<[bool; 8]>::from_elem(false, rank);
    let mut out = [SmallVec::new(), SmallVec::new()];
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

/// `tensor.argmax`: the index of the largest element along `axis`, a negative axis counting from
/// the end.
///
/// - Ties break toward the smallest index. NaN is larger than every number, as in Torch.
/// - If `keepdims` is true, `axis` remains with extent 1; otherwise it is removed. The indices
///   have the dtype `out`, `si32` or `si64`.
/// - If `axis` has extent 0, backends MUST reject the op as a spec violation.
#[derive(crate::Op, Debug, Clone, PartialEq, Eq, Hash)]
#[op(
    "tensor.argmax",
    operands(input: Tensor),
    results(output: Tensor)
)]
pub struct ArgMax {
    pub axis: i64,
    #[attr(default)]
    pub keepdims: bool,
    /// The index dtype: `si32` or `si64`.
    pub out: DType,
}

impl OpDef for ArgMax {
    fn result_types(
        &self,
        ins: ArgMaxOperands<Types<'_>>,
        _cx: &TypeCx<'_, NoRegions>,
    ) -> Result<ArgMaxResults<OwnedTypes>> {
        let input = ins.input;
        ensure_class("the input", input, ARITHMETIC)?;
        ensure!(
            matches!(self.out, DType::Si32 | DType::Si64),
            "indices must be si32 or si64, got {}",
            self.out
        );
        let axis = axis_index(self.axis, input.rank())?;
        ensure!(
            input.dims[axis] != Dim::Static(0),
            "axis {} of {input} is empty",
            self.axis
        );
        let mut dims = input.dims.clone();
        if self.keepdims {
            dims[axis] = Dim::Static(1);
        } else {
            dims.remove(axis);
        }
        Ok(ArgMaxResults {
            output: TensorType {
                dtype: self.out,
                dims,
            },
        })
    }
}

/// `tensor.segment_sum`, `tensor.segment_max`, ...: reduces the rows of `data` into
/// `num_segments` rows, row `i` into row `segment_ids[i]`.
///
/// - `segment_ids` holds one `si32` or `si64` id per row of `data`, along its leading axis. The
///   result has the shape of `data` with `num_segments` rows: `[num_segments, ...]`.
/// - Each id MUST lie in `[0, num_segments)`. A segment that no row reaches holds the initial
///   value of `kind`, as `ReduceKind` states.
/// - It accumulates in the default accumulator of the dtype of `data` and returns that dtype.
/// - `indices_are_sorted` and `unique` state facts about the ids that MUST hold when set, so
///   kernels MAY rely on them.
/// - An extension op: a backend MAY reject it as unimplemented.
#[derive(crate::Op, Debug, Clone, PartialEq, Eq, Hash)]
#[op(
    "tensor.segment_{kind}",
    operands(data: Tensor, segment_ids: Tensor),
    results(output: Tensor)
)]
pub struct Segment {
    pub kind: ReduceKind,
    pub num_segments: usize,
    #[attr(default)]
    pub indices_are_sorted: bool,
    #[attr(default)]
    pub unique: bool,
}

impl OpDef for Segment {
    fn result_types(
        &self,
        ins: SegmentOperands<Types<'_>>,
        _cx: &TypeCx<'_, NoRegions>,
    ) -> Result<SegmentResults<OwnedTypes>> {
        let (data, ids) = (ins.data, ins.segment_ids);
        ensure_reduction(self.kind, data, None, None)?;
        ensure!(
            data.rank() >= 1,
            "data must have rank at least 1, got {data}"
        );
        ensure!(
            matches!(ids.dtype, DType::Si32 | DType::Si64)
                && ids.rank() == 1
                && ids.dims[0] == data.dims[0],
            "segment ids must be si32 or si64 with one id per row of {data}, got {ids}"
        );
        let mut dims = data.dims.clone();
        dims[0] = Dim::Static(self.num_segments);
        Ok(SegmentResults {
            output: data.with_dims(dims),
        })
    }
}

/// `tensor.top_k`: the top-k values along `axis` and their indices.
///
/// - With `largest`, the `k` largest elements along `axis`, largest first; otherwise the `k`
///   smallest, smallest first. A negative axis counts from the end.
/// - Of equal elements, the one with the smaller index comes first, and NaN is larger than every
///   number.
/// - The values have the input's shape with extent `k` on `axis`, and their indices along `axis`
///   have the same shape and the dtype that the attribute `indices` names, `si32` or `si64`.
/// - An extension op: a backend MAY reject it as unimplemented.
#[derive(crate::Op, Debug, Clone, PartialEq, Eq, Hash)]
#[op(
    "tensor.top_k",
    operands(input: Tensor),
    results(values: Tensor, indices: Tensor)
)]
pub struct TopK {
    pub k: usize,
    pub axis: i64,
    pub largest: bool,
    /// The index dtype: `si32` or `si64`.
    pub indices: DType,
}

impl OpDef for TopK {
    fn result_types(
        &self,
        ins: TopKOperands<Types<'_>>,
        _cx: &TypeCx<'_, NoRegions>,
    ) -> Result<TopKResults<OwnedTypes>> {
        let input = ins.input;
        ensure_class("the input", input, ARITHMETIC)?;
        ensure!(
            matches!(self.indices, DType::Si32 | DType::Si64),
            "indices must be si32 or si64, got {}",
            self.indices
        );
        let axis = axis_index(self.axis, input.rank())?;
        if let Dim::Static(extent) = input.dims[axis] {
            ensure!(
                self.k <= extent,
                "k = {} exceeds axis {} of {input}",
                self.k,
                self.axis
            );
        }
        let mut dims = input.dims.clone();
        dims[axis] = Dim::Static(self.k);
        Ok(TopKResults {
            values: input.with_dims(dims.clone()),
            indices: TensorType {
                dtype: self.indices,
                dims,
            },
        })
    }
}
