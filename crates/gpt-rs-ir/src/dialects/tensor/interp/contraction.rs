//! Semantics of the contraction ops.
//!
//! A sum, product or extremum accumulates in its accumulator dtype: each operand element enters it
//! as `tensor.cast` converts it, each step rounds to it, and the result is cast to the result
//! dtype.

use std::cmp::Ordering;
use std::marker::PhantomData;

use anyhow::{anyhow, bail, ensure, Result};

use super::elementwise::Arith;
use super::kernels::{
    accumulate, best_k, from_ints, gather_elements, offsets, strides, values, Accumulation, Values,
};
use crate::dialects::arith;
use crate::dialects::arith::math::Combiner;
use crate::dialects::tensor::ir::*;
use crate::dialects::tensor::utils::contraction::Side;
use crate::dialects::tensor::utils::shape::{axis_index, pick};
use crate::dialects::tensor::Accumulates;
use crate::host::{HostTensor, TensorData};
use crate::ir::{DType, Eval, EvalCx, Host, NoRegions, OwnedHost};

impl Eval for DotGeneral {
    fn eval(
        &self,
        ins: DotGeneralOperands<Host>,
        _cx: &mut EvalCx<NoRegions>,
    ) -> Result<DotGeneralResults<OwnedHost>> {
        let (lhs, rhs) = (ins.lhs, ins.rhs);
        let (accum, out) = (self.accum_dtype(lhs.dtype()), self.out_dtype(lhs.dtype()));
        let (lhs_dims, rhs_dims) = (&lhs.dims, &rhs.dims);
        let (lhs_strides, rhs_strides) = (strides(lhs_dims), strides(rhs_dims));
        let axes = self.axes(lhs_dims.len(), rhs_dims.len())?;

        // The result axes, with the steps each takes through lhs and through rhs.
        let lhs_free = axes.free(Side::Lhs);
        let rhs_free = axes.free(Side::Rhs);
        let out_dims = [
            pick(lhs_dims, &axes.batch_lhs),
            pick(lhs_dims, &lhs_free),
            pick(rhs_dims, &rhs_free),
        ]
        .concat();
        let lhs_steps = [
            pick(&lhs_strides, &axes.batch_lhs),
            pick(&lhs_strides, &lhs_free),
            vec![0; rhs_free.len()],
        ]
        .concat();
        let rhs_steps = [
            pick(&rhs_strides, &axes.batch_rhs),
            vec![0; lhs_free.len()],
            pick(&rhs_strides, &rhs_free),
        ]
        .concat();
        let rows: Vec<_> = offsets(&out_dims, &lhs_steps)
            .into_iter()
            .zip(offsets(&out_dims, &rhs_steps))
            .collect();
        let contract_dims = pick(lhs_dims, &axes.contract_lhs);
        let terms: Vec<_> = offsets(&contract_dims, &pick(&lhs_strides, &axes.contract_lhs))
            .into_iter()
            .zip(offsets(
                &contract_dims,
                &pick(&rhs_strides, &axes.contract_rhs),
            ))
            .collect();
        let data = accumulate(&[&lhs.data, &rhs.data], accum, out, &Dot { rows, terms })?;
        Ok(DotGeneralResults {
            output: HostTensor::new(out_dims, data)?,
        })
    }
}

/// Each result element sums its products from 0, in row-major order of the contracting axes
/// taken in `contract_lhs` order, rounding each product and each sum. `rows` and `terms` hold
/// offsets into `lhs` and `rhs`.
struct Dot {
    rows: Vec<(usize, usize)>,
    terms: Vec<(usize, usize)>,
}

impl Accumulation for Dot {
    fn run<T: Arith>(
        &self,
        inputs: &[&[T]],
        accum: DType,
        round: impl Fn(T) -> T,
    ) -> Result<Vec<T>> {
        let &[lhs, rhs] = inputs else {
            bail!("a dot reads two tensors");
        };
        let add = T::binary::<arith::Add>(accum)?;
        let mul = T::binary::<arith::Mul>(accum)?;
        let zero = T::fold_init::<Add>(accum)?;
        let defined = "add and mul are defined on every pair";
        Ok(self
            .rows
            .iter()
            .map(|&(lhs_row, rhs_row)| {
                self.terms.iter().fold(zero, |sum, &(lhs_term, rhs_term)| {
                    let product = round(
                        mul(lhs[lhs_row + lhs_term], rhs[rhs_row + rhs_term]).expect(defined),
                    );
                    round(add(sum, product).expect(defined))
                })
            })
            .collect())
    }
}

/// The reduction of `input` over the axes of `attrs`, folding with `C`.
pub(crate) fn reduce_axes<C: Combiner>(
    attrs: &ReduceAttrs,
    input: &HostTensor,
) -> Result<HostTensor> {
    let dims = &input.dims;
    let in_strides = strides(dims);
    let axes = attrs.reduced_axes(dims.len())?;
    let kept = attrs.kept_axes(dims.len())?;
    let rows = offsets(&pick(dims, &kept), &pick(&in_strides, &kept));
    // Each result element folds its inputs in row-major order of the reduced axes.
    let terms = offsets(&pick(dims, &axes), &pick(&in_strides, &axes));
    let groups: Vec<Vec<usize>> = rows
        .iter()
        .map(|row| terms.iter().map(|term| row + term).collect())
        .collect();
    let (accum, out) = (
        attrs.accum_dtype(input.dtype()),
        attrs.out_dtype(input.dtype()),
    );
    let data = fold_groups::<C>(input, &groups, accum, out)?;
    HostTensor::new(attrs.reduced_dims(&axes, dims, 1), data)
}

/// The fold with `C` of the elements of `input` at each group of indices, accumulated in
/// `accum`, by default the input's default accumulator, and cast to `out`.
pub(super) fn fold_groups<C: Combiner>(
    input: &HostTensor,
    groups: &[Vec<usize>],
    accum: DType,
    out: DType,
) -> Result<TensorData> {
    let fold = Fold::<C> {
        groups,
        combiner: PhantomData,
    };
    accumulate(&[&input.data], accum, out, &fold)
}

/// Each group of indices folds its elements with `C`, from its initial value, in order.
struct Fold<'a, C> {
    groups: &'a [Vec<usize>],
    combiner: PhantomData<C>,
}

impl<C: Combiner> Accumulation for Fold<'_, C> {
    fn run<T: Arith>(
        &self,
        inputs: &[&[T]],
        accum: DType,
        round: impl Fn(T) -> T,
    ) -> Result<Vec<T>> {
        let &[x] = inputs else {
            bail!("a reduction reads one tensor");
        };
        let init = T::fold_init::<C>(accum)?;
        let combine = T::binary::<C::Scalar>(accum)?;
        let defined = "a fold is defined on every pair";
        Ok(self
            .groups
            .iter()
            .map(|group| {
                group.iter().fold(init, |acc, &index| {
                    round(combine(acc, x[index]).expect(defined))
                })
            })
            .collect())
    }
}

/// How element `first` of `data` orders against element `second`, with NaN above every number.
fn order<'a>(data: &'a TensorData) -> Box<dyn Fn(usize, usize) -> Ordering + 'a> {
    match values(data) {
        Values::Int(elements) => {
            Box::new(move |first, second| i128::order(elements[first], elements[second]))
        }
        Values::F32(elements) => {
            Box::new(move |first, second| f32::order(elements[first], elements[second]))
        }
        Values::F64(elements) => {
            Box::new(move |first, second| f64::order(elements[first], elements[second]))
        }
    }
}

/// The extents before `axis`, at `axis` and after it.
fn split_axis(dims: &[usize], axis: usize) -> (usize, usize, usize) {
    let outer = dims[..axis].iter().product();
    let inner = dims[axis + 1..].iter().product();
    (outer, dims[axis], inner)
}

/// The indices of every line along `axis`: element `line_index` of the line
/// `(outer_index, inner_index)` is at `(outer_index * extent + line_index) * inner + inner_index`.
fn lines(dims: &[usize], axis: usize) -> impl Iterator<Item = Vec<usize>> {
    let (outer, extent, inner) = split_axis(dims, axis);
    (0..outer).flat_map(move |outer_index| {
        (0..inner).map(move |inner_index| {
            (0..extent)
                .map(|line_index| (outer_index * extent + line_index) * inner + inner_index)
                .collect()
        })
    })
}

impl Eval for ArgMax {
    fn eval(
        &self,
        ins: ArgMaxOperands<Host>,
        cx: &mut EvalCx<NoRegions>,
    ) -> Result<ArgMaxResults<OwnedHost>> {
        let compare = order(&ins.input.data);
        let axis = axis_index(self.axis, ins.input.dims.len())?;
        let mut indices = Vec::new();
        for line in lines(&ins.input.dims, axis) {
            let best = best_k(line.len(), 1, true, |first, second| {
                compare(line[first], line[second])
            });
            indices.push(best[0]);
        }
        Ok(ArgMaxResults {
            output: index_tensor(cx.result_dims(0)?, &indices, self.out)?,
        })
    }
}

/// The rows of `data` folded with `C` into the segments of `attrs` that `ids` name.
pub(crate) fn reduce_segments<C: Combiner>(
    attrs: &SegmentAttrs,
    data: &HostTensor,
    ids: &HostTensor,
) -> Result<HostTensor> {
    let segment_ids = ids.as_indices()?;
    if attrs.indices_are_sorted {
        ensure!(
            segment_ids.windows(2).all(|pair| pair[0] <= pair[1]),
            "the segment ids are not sorted"
        );
    }
    if attrs.unique {
        let mut seen = std::collections::HashSet::new();
        ensure!(
            segment_ids.iter().all(|id| seen.insert(id)),
            "the segment ids repeat"
        );
    }
    let row: usize = data.dims[1..].iter().product();
    // Element `e` of segment `s` folds element `e` of every row that `s` holds, in order.
    let mut groups = vec![Vec::new(); attrs.num_segments * row];
    for (row_index, &id) in segment_ids.iter().enumerate() {
        let segment = usize::try_from(id)
            .ok()
            .filter(|&segment| segment < attrs.num_segments)
            .ok_or_else(|| {
                anyhow!(
                    "segment id {id} is out of range for {} segments",
                    attrs.num_segments
                )
            })?;
        for element in 0..row {
            groups[segment * row + element].push(row_index * row + element);
        }
    }
    let accum = data.dtype().default_accumulator();
    let out = fold_groups::<C>(data, &groups, accum, data.dtype())?;
    let dims = [&[attrs.num_segments], &data.dims[1..]].concat();
    HostTensor::new(dims, out)
}

/// A tensor of `indices` in the index dtype `dtype`, si32 or si64.
fn index_tensor(dims: Vec<usize>, indices: &[usize], dtype: DType) -> Result<HostTensor> {
    let data = from_ints(dtype, indices.iter().map(|&index| index as i128))?;
    HostTensor::new(dims, data)
}

/// The semantics of every [`Reduce`] op.
pub(crate) fn reduce<T: Reduce>(
    op: &T,
    ins: ReduceOperands<Host>,
    _cx: &mut EvalCx<NoRegions>,
) -> Result<ReduceResults<OwnedHost>> {
    Ok(ReduceResults {
        output: reduce_axes::<T::Combiner>(op, ins.input)?,
    })
}

/// The semantics of every [`Segment`] op.
pub(crate) fn segment<T: Segment>(
    op: &T,
    ins: SegmentOperands<Host>,
    _cx: &mut EvalCx<NoRegions>,
) -> Result<SegmentResults<OwnedHost>> {
    Ok(SegmentResults {
        output: reduce_segments::<T::Combiner>(op, ins.data, ins.segment_ids)?,
    })
}

/// The semantics of every [`SelectK`] op.
pub(crate) fn select_k<T: SelectK>(
    op: &T,
    ins: SelectKOperands<Host>,
    cx: &mut EvalCx<NoRegions>,
) -> Result<SelectKResults<OwnedHost>> {
    let input = ins.input;
    let compare = order(&input.data);
    // The positions of the kept elements, line by line, then reordered to the result's row-major
    // order, where the `k` kept elements of a line run along `axis`.
    let axis = axis_index(op.axis, input.dims.len())?;
    let (outer, _, inner) = split_axis(&input.dims, axis);
    let mut picked = vec![0usize; outer * op.k * inner];
    let mut indices = vec![0usize; outer * op.k * inner];
    for (line_index, line) in lines(&input.dims, axis).enumerate() {
        let ranking = best_k(line.len(), op.k, T::LARGEST, |first, second| {
            compare(line[first], line[second])
        });
        let (outer_index, inner_index) = (line_index / inner, line_index % inner);
        for (rank, &position) in ranking.iter().enumerate() {
            let at = (outer_index * op.k + rank) * inner + inner_index;
            picked[at] = line[position];
            indices[at] = position;
        }
    }
    let dims = cx.result_dims(0)?;
    Ok(SelectKResults {
        values: HostTensor::new(
            dims.clone(),
            gather_elements(&input.data, picked.len(), |at| picked[at])?,
        )?,
        indices: index_tensor(dims, &indices, op.indices)?,
    })
}
