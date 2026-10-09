//! Semantics of the ops in `ir/contraction.rs`.
//!
//! A sum, product or extremum accumulates in its accumulator dtype: each operand element enters it
//! as `tensor.cast` converts it, each step rounds to it, and the result is cast to the result
//! dtype.

use std::borrow::Cow;
use std::cmp::Ordering;

use anyhow::{anyhow, bail, ensure, Result};

use super::elementwise::{int_format, Arith};
use super::kernels::{
    cast, from_values, gather_elements, rounder, strides, unravel, values, Values,
};
use crate::dialects::tensor::ir::shape::{axis_index, free_axes};
use crate::dialects::tensor::ir::{
    ArgMax, ArgMaxOperands, ArgMaxResults, BinaryKind, DotGeneral, DotGeneralOperands,
    DotGeneralResults, Reduce, ReduceKind, ReduceOperands, ReduceResults, Segment, SegmentOperands,
    SegmentResults, TopK, TopKOperands, TopKResults,
};
use crate::host::{HostTensor, TensorData};
use crate::op::{Eval, EvalCx, Host, NoRegions, OwnedHost};
use crate::types::DType;

impl Eval for DotGeneral {
    fn eval(
        &self,
        ins: DotGeneralOperands<Host<'_>>,
        _cx: &mut EvalCx<'_, NoRegions>,
    ) -> Result<DotGeneralResults<OwnedHost>> {
        let (lhs, rhs) = (ins.lhs, ins.rhs);
        let accum = self.accum.unwrap_or(lhs.dtype().default_accumulator());
        let out = self.out.unwrap_or(lhs.dtype());
        let (lhs_dims, rhs_dims) = (&lhs.dims, &rhs.dims);
        let (lhs_strides, rhs_strides) = (strides(lhs_dims), strides(rhs_dims));
        let axes = self.axes(lhs_dims.len(), rhs_dims.len())?;

        // The result axes, with the steps each takes through lhs and through rhs.
        let lhs_free = free_axes(lhs_dims.len(), &axes.batch_lhs, &axes.contract_lhs);
        let rhs_free = free_axes(rhs_dims.len(), &axes.batch_rhs, &axes.contract_rhs);
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
        let (l, r) = (cast(&lhs.data, accum)?, cast(&rhs.data, accum)?);
        let sums = match (values(&l), values(&r)) {
            (Values::Int(l), Values::Int(r)) => {
                let int = int_format(accum)?;
                Values::Int(dot(&l, &r, &rows, &terms, accum, |x| int.wrap(x)))
            }
            (Values::F32(l), Values::F32(r)) => Values::F32(Cow::Owned(dot(
                &l,
                &r,
                &rows,
                &terms,
                accum,
                rounder(accum),
            ))),
            (Values::F64(l), Values::F64(r)) => {
                Values::F64(Cow::Owned(dot(&l, &r, &rows, &terms, accum, |x| x)))
            }
            _ => bail!("the interpreter cannot accumulate in {accum}"),
        };
        Ok(DotGeneralResults {
            output: HostTensor::new(out_dims, cast(&from_values(accum, sums)?, out)?)?,
        })
    }
}

/// Each result element sums its products from 0, in row-major order of the contracting axes
/// taken in `contract_lhs` order, rounding each product and each sum with `round` to `accum`.
/// `rows` and `terms` hold offsets into `lhs` and `rhs`.
fn dot<T: Arith>(
    lhs: &[T],
    rhs: &[T],
    rows: &[(usize, usize)],
    terms: &[(usize, usize)],
    accum: DType,
    round: impl Fn(T) -> T,
) -> Vec<T> {
    let add = T::binary(BinaryKind::Add).expect("every computation type adds");
    let mul = T::binary(BinaryKind::Mul).expect("every computation type multiplies");
    let zero = T::reduce_init(ReduceKind::Sum, accum).expect("an arithmetic accumulator");
    rows.iter()
        .map(|&(lhs_row, rhs_row)| {
            terms.iter().fold(zero, |sum, &(l, r)| {
                round(add(sum, round(mul(lhs[lhs_row + l], rhs[rhs_row + r]))))
            })
        })
        .collect()
}

impl Eval for Reduce {
    fn eval(
        &self,
        ins: ReduceOperands<Host<'_>>,
        _cx: &mut EvalCx<'_, NoRegions>,
    ) -> Result<ReduceResults<OwnedHost>> {
        let dims = &ins.input.dims;
        let in_strides = strides(dims);
        let axes = self.reduced_axes(dims.len())?;
        let kept = free_axes(dims.len(), &axes, &[]);
        let rows = offsets(&pick(dims, &kept), &pick(&in_strides, &kept));
        // Each result element folds its inputs in row-major order of the reduced axes.
        let terms = offsets(&pick(dims, &axes), &pick(&in_strides, &axes));
        let groups: Vec<Vec<usize>> = rows
            .iter()
            .map(|row| terms.iter().map(|term| row + term).collect())
            .collect();
        let out = self.out.unwrap_or(ins.input.dtype());
        let data = reduce(ins.input, &groups, self.kind, self.accum, out)?;
        Ok(ReduceResults {
            output: HostTensor::new(self.reduced_dims(&axes, dims, 1).into_vec(), data)?,
        })
    }
}

/// The `kind` reduction of the elements of `input` at each group of indices, accumulated in
/// `accum`, by default the input's default accumulator, and cast to `out`.
pub(super) fn reduce(
    input: &HostTensor,
    groups: &[Vec<usize>],
    kind: ReduceKind,
    accum: Option<DType>,
    out: DType,
) -> Result<TensorData> {
    let accum = accum.unwrap_or(input.dtype().default_accumulator());
    let data = cast(&input.data, accum)?;
    let folded = match values(&data) {
        Values::Int(x) => {
            let int = int_format(accum)?;
            Values::Int(fold_groups(&x, groups, kind, accum, |v| int.wrap(v))?)
        }
        Values::F32(x) => Values::F32(Cow::Owned(fold_groups(
            &x,
            groups,
            kind,
            accum,
            rounder(accum),
        )?)),
        Values::F64(x) => Values::F64(Cow::Owned(fold_groups(&x, groups, kind, accum, |v| v)?)),
    };
    cast(&from_values(accum, folded)?, out)
}

/// The `kind` reduction of the elements of `x`, of `dtype`, at each group of indices, folded into
/// the reduction's initial value in order, each step rounded with `round`.
fn fold_groups<T: Arith>(
    x: &[T],
    groups: &[Vec<usize>],
    kind: ReduceKind,
    dtype: DType,
    round: impl Fn(T) -> T,
) -> Result<Vec<T>> {
    let missing = || anyhow!("the interpreter has no {dtype} reduction {kind:?}");
    let init = T::reduce_init(kind, dtype).ok_or_else(missing)?;
    let combine = T::binary(kind.combiner()).ok_or_else(missing)?;
    Ok(groups
        .iter()
        .map(|group| group.iter().fold(init, |acc, &i| round(combine(acc, x[i]))))
        .collect())
}

/// `values[axis]` for each of `axes`.
fn pick(values: &[usize], axes: &[usize]) -> Vec<usize> {
    axes.iter().map(|&axis| values[axis]).collect()
}

/// The element offset of every point of a loop nest, in row-major order: loop `i` runs
/// `extents[i]` times and steps `steps[i]` elements.
fn offsets(extents: &[usize], steps: &[usize]) -> Vec<usize> {
    let count: usize = extents.iter().product();
    (0..count)
        .map(|i| {
            unravel(i, extents)
                .iter()
                .zip(steps)
                .map(|(coord, step)| coord * step)
                .sum()
        })
        .collect()
}

/// How element `a` of `data` orders against element `b`, with NaN above every number.
fn order<'a>(data: &'a TensorData) -> Box<dyn Fn(usize, usize) -> Ordering + 'a> {
    match values(data) {
        Values::Int(v) => Box::new(move |a, b| i128::order(v[a], v[b])),
        Values::F32(v) => Box::new(move |a, b| f32::order(v[a], v[b])),
        Values::F64(v) => Box::new(move |a, b| f64::order(v[a], v[b])),
    }
}

/// The extents before `axis`, at `axis` and after it.
fn split_axis(dims: &[usize], axis: usize) -> (usize, usize, usize) {
    let outer = dims[..axis].iter().product();
    let inner = dims[axis + 1..].iter().product();
    (outer, dims[axis], inner)
}

/// The indices of every line along `axis`: element `j` of line `(o, i)` is at
/// `(o * n + j) * inner + i`.
fn lines(dims: &[usize], axis: usize) -> impl Iterator<Item = Vec<usize>> {
    let (outer, n, inner) = split_axis(dims, axis);
    (0..outer).flat_map(move |o| {
        (0..inner).map(move |i| (0..n).map(|j| (o * n + j) * inner + i).collect())
    })
}

impl Eval for ArgMax {
    fn eval(
        &self,
        ins: ArgMaxOperands<Host<'_>>,
        cx: &mut EvalCx<'_, NoRegions>,
    ) -> Result<ArgMaxResults<OwnedHost>> {
        let compare = order(&ins.input.data);
        let axis = axis_index(self.axis, ins.input.dims.len())?;
        let indices = lines(&ins.input.dims, axis)
            .map(|line| {
                (0..line.len())
                    .reduce(|best, j| match compare(line[j], line[best]) {
                        Ordering::Greater => j,
                        _ => best,
                    })
                    .expect("the axis is not empty")
            })
            .collect::<Vec<usize>>();
        Ok(ArgMaxResults {
            output: index_tensor(cx.result_dims(0)?, &indices, self.out)?,
        })
    }
}

impl Eval for TopK {
    fn eval(
        &self,
        ins: TopKOperands<Host<'_>>,
        cx: &mut EvalCx<'_, NoRegions>,
    ) -> Result<TopKResults<OwnedHost>> {
        let input = ins.input;
        let compare = order(&input.data);
        // The positions of the kept elements, line by line, then reordered to the result's
        // row-major order, where the `k` kept elements of a line run along `axis`.
        let axis = axis_index(self.axis, input.dims.len())?;
        let (outer, _, inner) = split_axis(&input.dims, axis);
        let mut picked = vec![0usize; outer * self.k * inner];
        let mut indices = vec![0usize; outer * self.k * inner];
        for (line_index, line) in lines(&input.dims, axis).enumerate() {
            let mut order: Vec<usize> = (0..line.len()).collect();
            order.sort_by(|&a, &b| {
                let ordering = compare(line[a], line[b]);
                if self.largest {
                    ordering.reverse()
                } else {
                    ordering
                }
            });
            let (o, i) = (line_index / inner, line_index % inner);
            for (rank, &j) in order[..self.k].iter().enumerate() {
                let at = (o * self.k + rank) * inner + i;
                picked[at] = line[j];
                indices[at] = j;
            }
        }
        let dims = cx.result_dims(0)?;
        Ok(TopKResults {
            values: HostTensor::new(
                dims.clone(),
                gather_elements(&input.data, picked.len(), |at| picked[at])?,
            )?,
            indices: index_tensor(dims, &indices, self.indices)?,
        })
    }
}

impl Eval for Segment {
    fn eval(
        &self,
        ins: SegmentOperands<Host<'_>>,
        cx: &mut EvalCx<'_, NoRegions>,
    ) -> Result<SegmentResults<OwnedHost>> {
        let (data, ids) = (ins.data, ins.segment_ids.as_indices()?);
        if self.indices_are_sorted {
            ensure!(
                ids.windows(2).all(|pair| pair[0] <= pair[1]),
                "the segment ids are not sorted"
            );
        }
        if self.unique {
            let mut seen = std::collections::HashSet::new();
            ensure!(
                ids.iter().all(|id| seen.insert(id)),
                "the segment ids repeat"
            );
        }
        let row: usize = data.dims[1..].iter().product();
        // Element `e` of segment `s` folds element `e` of every row that `s` holds, in order.
        let mut groups = vec![Vec::new(); self.num_segments * row];
        for (r, &id) in ids.iter().enumerate() {
            let segment = usize::try_from(id)
                .ok()
                .filter(|&segment| segment < self.num_segments)
                .ok_or_else(|| {
                    anyhow!(
                        "segment id {id} is out of range for {} segments",
                        self.num_segments
                    )
                })?;
            for e in 0..row {
                groups[segment * row + e].push(r * row + e);
            }
        }
        let out = reduce(data, &groups, self.kind, None, data.dtype())?;
        Ok(SegmentResults {
            output: HostTensor::new(cx.result_dims(0)?, out)?,
        })
    }
}

/// A tensor of `indices` in the index dtype `dtype`, si32 or si64.
fn index_tensor(dims: Vec<usize>, indices: &[usize], dtype: DType) -> Result<HostTensor> {
    let data = match dtype {
        DType::Si32 => TensorData::Si32(indices.iter().map(|&i| i as i32).collect()),
        DType::Si64 => TensorData::Si64(indices.iter().map(|&i| i as i64).collect()),
        other => bail!("{other} holds no indices"),
    };
    HostTensor::new(dims, data)
}
