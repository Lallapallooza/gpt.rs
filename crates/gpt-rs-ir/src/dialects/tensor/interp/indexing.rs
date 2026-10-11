//! Semantics of the indexing ops.

use std::marker::PhantomData;

use anyhow::{anyhow, bail, Result};

use super::elementwise::Arith;
use super::kernels::{
    accumulate, cast, from_ints, gather_elements, offsets, strided, strides, unravel, Accumulation,
    Chunks,
};
use crate::dialects::tensor::ir::*;
use crate::dialects::tensor::utils::shape::{axis_index, check_size};
use crate::host::HostTensor;
use crate::op::{Eval, EvalCx, Host, NoRegions, OwnedHost};
use crate::scalar::Combiner;
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
        let start = starts.iter().zip(&in_strides).map(|(s, n)| s * n).sum();
        let data = strided(&input.data, start, &self.sizes, &in_strides)?;
        Ok(DynamicSliceResults {
            output: HostTensor::new(self.sizes, data)?,
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
        let start: usize = starts.iter().zip(&base_strides).map(|(s, n)| s * n).sum();
        let update = Chunks::of(&ins.update.data);
        let mut out = Chunks::of(&base.data);
        for (i, offset) in offsets(&ins.update.dims, &base_strides)
            .into_iter()
            .enumerate()
        {
            out.set(start + offset, update.get(i));
        }
        Ok(DynamicUpdateSliceResults {
            output: HostTensor::new(base.dims, out.into_data()?)?,
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
                index_dims,
                gather_elements(&input.data, sources.len(), |i| sources[i])?,
            )?,
        })
    }
}

/// The element of `input` that each update goes to, along `axis` as `indices` say.
fn scatter_targets(axis: i64, input: &HostTensor, indices: &HostTensor) -> Result<Vec<usize>> {
    let axis = axis_index(axis, input.dims.len())?;
    indices
        .as_indices()?
        .iter()
        .enumerate()
        .map(|(i, &index)| indexed_element(i, index, axis, &indices.dims, &input.dims))
        .collect()
}

/// `input` with each update folded into its element with `C`, in the default accumulator of the
/// dtype.
pub(crate) fn scatter_fold<C: Combiner>(
    attrs: &ScatterAttrs,
    input: &HostTensor,
    indices: &HostTensor,
    updates: &HostTensor,
) -> Result<HostTensor> {
    let targets = scatter_targets(attrs.axis, input, indices)?;
    let dtype = input.dtype();
    let fold = ScatterFold::<C> {
        targets,
        combiner: PhantomData,
    };
    let data = accumulate(
        &[&input.data, &updates.data],
        dtype.default_accumulator(),
        dtype,
        &fold,
    )?;
    HostTensor::new(input.dims, data)
}

impl Eval for ScatterReplace {
    fn eval(
        &self,
        ins: ScatterReplaceOperands<Host<'_>>,
        _cx: &mut EvalCx<'_, NoRegions>,
    ) -> Result<ScatterReplaceResults<OwnedHost>> {
        let input = ins.input;
        let targets = scatter_targets(self.axis, input, ins.indices)?;
        let updates = Chunks::of(&ins.updates.data);
        let mut out = Chunks::of(&input.data);
        for (i, &target) in targets.iter().enumerate() {
            out.set(target, updates.get(i));
        }
        Ok(ScatterReplaceResults {
            output: HostTensor::new(input.dims, out.into_data()?)?,
        })
    }
}

/// The input with update `i` folded into element `targets[i]` with `C`, in order, each step
/// rounded.
struct ScatterFold<C> {
    targets: Vec<usize>,
    combiner: PhantomData<C>,
}

impl<C: Combiner> Accumulation for ScatterFold<C> {
    fn run<T: Arith>(
        &self,
        inputs: &[&[T]],
        accum: DType,
        round: impl Fn(T) -> T,
    ) -> Result<Vec<T>> {
        let &[x, updates] = inputs else {
            bail!("a scatter reads its input and its updates");
        };
        let combine = T::binary::<C::Scalar>(accum)?;
        let mut out = x.to_vec();
        for (&update, &target) in updates.iter().zip(&self.targets) {
            let combined = combine(out[target], update).expect("a fold is defined on every pair");
            out[target] = round(combined);
        }
        Ok(out)
    }
}

/// The semantics of every [`Scatter`] op.
pub(crate) fn scatter<T: Scatter>(
    op: &T,
    ins: ScatterOperands<Host<'_>>,
    _cx: &mut EvalCx<'_, NoRegions>,
) -> Result<ScatterResults<OwnedHost>> {
    Ok(ScatterResults {
        output: scatter_fold::<T::Combiner>(op, ins.input, ins.indices, ins.updates)?,
    })
}
