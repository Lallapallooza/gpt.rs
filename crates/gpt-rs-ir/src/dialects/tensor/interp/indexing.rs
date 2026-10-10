//! Semantics of the ops in `ir/indexing.rs`.

use std::borrow::Cow;

use anyhow::{anyhow, bail, Result};

use super::elementwise::{int_format, Arith};
use super::kernels::{
    cast, from_ints, from_values, gather_elements, rounder, strides, unravel, values, Chunks,
    Values,
};
use crate::dialects::tensor::ir::shape::{axis_index, check_size};
use crate::dialects::tensor::ir::{
    BinaryKind, DynamicSlice, DynamicSliceOperands, DynamicSliceResults, DynamicUpdateSlice,
    DynamicUpdateSliceOperands, DynamicUpdateSliceResults, Gather, GatherOperands, GatherResults,
    Iota, IotaOperands, IotaResults, Scatter, ScatterKind, ScatterOperands, ScatterResults, Take,
    TakeOperands, TakeResults,
};
use crate::host::HostTensor;
use crate::op::{Eval, EvalCx, Host, NoRegions, OwnedHost};
use crate::types::DType;

impl Eval for Iota {
    fn eval(
        &self,
        _ins: IotaOperands<Host<'_>>,
        cx: &mut EvalCx<'_, NoRegions>,
    ) -> Result<IotaResults<OwnedHost>> {
        let dims = cx.result_dims(0)?;
        let axis = axis_index(self.axis, dims.len())?;
        let count: usize = dims.iter().product();
        // The counts, converted to `dtype` as `tensor.cast` converts them.
        let counts = from_ints(
            DType::Si64,
            (0..count).map(|i| unravel(i, &dims)[axis] as i128),
        )?;
        Ok(IotaResults {
            output: HostTensor::new(dims, cast(&counts, self.dtype)?)?,
        })
    }
}

impl Eval for Take {
    fn eval(
        &self,
        ins: TakeOperands<Host<'_>>,
        _cx: &mut EvalCx<'_, NoRegions>,
    ) -> Result<TakeResults<OwnedHost>> {
        let indices = ins.indices.as_indices()?;
        let rows = ins.table.dims[0];
        let row_len: usize = ins.table.dims[1..].iter().product();
        let picked = indices
            .iter()
            .map(|&index| {
                usize::try_from(index)
                    .ok()
                    .filter(|&row| row < rows)
                    .ok_or_else(|| anyhow!("index {index} is out of range for {rows} rows"))
            })
            .collect::<Result<Vec<usize>>>()?;
        let table = Chunks::of(&ins.table.data);
        let mut out = Chunks::zeroed(ins.table.dtype(), picked.len() * row_len);
        for (i, &row) in picked.iter().enumerate() {
            out.set_run(i * row_len, table.run(row * row_len, row_len));
        }
        let out = out.into_data()?;
        let dims: Vec<usize> = ins
            .indices
            .dims
            .iter()
            .chain(&ins.table.dims[1..])
            .copied()
            .collect();
        Ok(TakeResults {
            output: HostTensor::new(dims, out)?,
        })
    }
}

impl Eval for DynamicSlice {
    fn eval(
        &self,
        ins: DynamicSliceOperands<Host<'_>>,
        _cx: &mut EvalCx<'_, NoRegions>,
    ) -> Result<DynamicSliceResults<OwnedHost>> {
        let input = ins.input;
        let starts = clamped_starts(&input.dims, &self.sizes, &ins.starts.as_indices()?)?;
        let in_strides = strides(&input.dims);
        let count: usize = self.sizes.iter().product();
        let source = |out_index: usize| -> usize {
            unravel(out_index, &self.sizes)
                .iter()
                .enumerate()
                .map(|(axis, &coord)| (starts[axis] + coord) * in_strides[axis])
                .sum()
        };
        Ok(DynamicSliceResults {
            output: HostTensor::new(&self.sizes, gather_elements(&input.data, count, source)?)?,
        })
    }
}

impl Eval for DynamicUpdateSlice {
    fn eval(
        &self,
        ins: DynamicUpdateSliceOperands<Host<'_>>,
        _cx: &mut EvalCx<'_, NoRegions>,
    ) -> Result<DynamicUpdateSliceResults<OwnedHost>> {
        let base = ins.base;
        let starts = clamped_starts(&base.dims, &ins.update.dims, &ins.starts.as_indices()?)?;
        let base_strides = strides(&base.dims);
        let dest = |update_index: usize| -> usize {
            unravel(update_index, &ins.update.dims)
                .iter()
                .enumerate()
                .map(|(axis, &coord)| (starts[axis] + coord) * base_strides[axis])
                .sum()
        };
        let update = Chunks::of(&ins.update.data);
        let mut out = Chunks::of(&base.data);
        for i in 0..update.len() {
            out.set(dest(i), update.get(i));
        }
        Ok(DynamicUpdateSliceResults {
            output: HostTensor::new(base.dims.clone(), out.into_data()?)?,
        })
    }
}

/// Each start clamped into `[0, extent - size]`, so the window always lies inside the tensor.
fn clamped_starts(dims: &[usize], sizes: &[usize], starts: &[i64]) -> Result<Vec<usize>> {
    dims.iter()
        .zip(sizes)
        .zip(starts)
        .enumerate()
        .map(|(axis, ((&extent, &size), &start))| {
            check_size(axis, size, extent)?;
            Ok((start.max(0) as usize).min(extent - size))
        })
        .collect()
}

/// The element of `input` (of shape `dims`) that element `i` of an index tensor of shape
/// `index_dims` addresses: its coordinates with the one on `axis` replaced by `index`.
fn indexed_element(
    i: usize,
    index: i64,
    axis: usize,
    index_dims: &[usize],
    dims: &[usize],
) -> Result<usize> {
    let mut coord = unravel(i, index_dims);
    coord[axis] = usize::try_from(index)
        .ok()
        .filter(|&row| row < dims[axis])
        .ok_or_else(|| anyhow!("index {index} is out of range for axis {axis} of {dims:?}"))?;
    Ok(coord
        .iter()
        .zip(strides(dims))
        .map(|(coord, stride)| coord * stride)
        .sum())
}

impl Eval for Gather {
    fn eval(
        &self,
        ins: GatherOperands<Host<'_>>,
        _cx: &mut EvalCx<'_, NoRegions>,
    ) -> Result<GatherResults<OwnedHost>> {
        let (input, indices) = (ins.input, ins.indices.as_indices()?);
        let index_dims = &ins.indices.dims;
        let axis = axis_index(self.axis, input.dims.len())?;
        let sources = indices
            .iter()
            .enumerate()
            .map(|(i, &index)| indexed_element(i, index, axis, index_dims, &input.dims))
            .collect::<Result<Vec<_>>>()?;
        Ok(GatherResults {
            output: HostTensor::new(
                index_dims.clone(),
                gather_elements(&input.data, sources.len(), |i| sources[i])?,
            )?,
        })
    }
}

impl Eval for Scatter {
    fn eval(
        &self,
        ins: ScatterOperands<Host<'_>>,
        _cx: &mut EvalCx<'_, NoRegions>,
    ) -> Result<ScatterResults<OwnedHost>> {
        let input = ins.input;
        let index_dims = &ins.indices.dims;
        let axis = axis_index(self.axis, input.dims.len())?;
        let targets = ins
            .indices
            .as_indices()?
            .iter()
            .enumerate()
            .map(|(i, &index)| indexed_element(i, index, axis, index_dims, &input.dims))
            .collect::<Result<Vec<_>>>()?;
        let combine = match self.kind {
            ScatterKind::Add => BinaryKind::Add,
            ScatterKind::Max => BinaryKind::Maximum,
            ScatterKind::Min => BinaryKind::Minimum,
            ScatterKind::Replace => {
                let updates = Chunks::of(&ins.updates.data);
                let mut out = Chunks::of(&input.data);
                for (i, &target) in targets.iter().enumerate() {
                    out.set(target, updates.get(i));
                }
                return Ok(ScatterResults {
                    output: HostTensor::new(input.dims.clone(), out.into_data()?)?,
                });
            }
        };
        // The elements and the updates accumulate in the default accumulator of their dtype.
        let dtype = input.dtype();
        let accum = dtype.default_accumulator();
        let (x, u) = (cast(&input.data, accum)?, cast(&ins.updates.data, accum)?);
        let folded = match (values(&x), values(&u)) {
            (Values::Int(x), Values::Int(u)) => {
                let int = int_format(accum)?;
                Values::Int(scatter_fold(&x, &u, &targets, combine, |v| int.wrap(v))?)
            }
            (Values::F32(x), Values::F32(u)) => Values::F32(Cow::Owned(scatter_fold(
                &x,
                &u,
                &targets,
                combine,
                rounder(accum),
            )?)),
            (Values::F64(x), Values::F64(u)) => {
                Values::F64(Cow::Owned(scatter_fold(&x, &u, &targets, combine, |v| v)?))
            }
            _ => bail!("the interpreter cannot accumulate {dtype} in {accum}"),
        };
        let data = cast(&from_values(accum, folded)?, dtype)?;
        Ok(ScatterResults {
            output: HostTensor::new(input.dims.clone(), data)?,
        })
    }
}

/// `x` with update `i` combined into element `targets[i]`, in order, each step rounded with
/// `round`.
fn scatter_fold<T: Arith>(
    x: &[T],
    updates: &[T],
    targets: &[usize],
    kind: BinaryKind,
    round: impl Fn(T) -> T,
) -> Result<Vec<T>> {
    let combine = T::binary(kind).ok_or_else(|| anyhow!("the interpreter has no {kind:?}"))?;
    let mut out = x.to_vec();
    for (&update, &target) in updates.iter().zip(targets) {
        out[target] = round(combine(out[target], update));
    }
    Ok(out)
}
