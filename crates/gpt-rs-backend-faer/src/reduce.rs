//! faer's kernel for f32 `reduce`. Each result folds its terms in row-major order of the reduced
//! axes, the order the interpreter folds them in, so sums round the same way.

use anyhow::Result;
use gpt_rs_ir::dialects::tensor::ReduceAttrs;
use gpt_rs_ir::host::HostTensor;
use gpt_rs_ir::ir::{DType, TensorType};

/// The reduction of an f32 `input` with an f32 result that folds with `combine` from `init`,
/// unless `op` takes or gives other dtypes.
pub(crate) fn reduce(
    op: &ReduceAttrs,
    input: &HostTensor,
    out: &TensorType,
    init: f32,
    combine: impl Fn(f32, f32) -> f32,
) -> Option<Result<HostTensor>> {
    let f32_only = input.dtype() == DType::F32
        && out.dtype == DType::F32
        && op.accum.is_none_or(|accum| accum == DType::F32);
    f32_only.then(|| reduce_f32(op, input, out, init, combine))
}

fn reduce_f32(
    op: &ReduceAttrs,
    input: &HostTensor,
    out: &TensorType,
    init: f32,
    combine: impl Fn(f32, f32) -> f32,
) -> Result<HostTensor> {
    let x = input.as_f32()?;
    let dims = &input.dims;
    let mut strides = vec![1; dims.len()];
    for axis in (0..dims.len().saturating_sub(1)).rev() {
        strides[axis] = strides[axis + 1] * dims[axis + 1];
    }
    let axes = op.reduced_axes(dims.len())?;
    let (reduced, kept): (Vec<usize>, Vec<usize>) =
        (0..dims.len()).partition(|axis| axes.contains(axis));
    let rows = offsets(kept.iter().map(|&axis| (dims[axis], strides[axis])));
    // The reduced block as runs: the innermost reduced axes that are contiguous form one run.
    let mut block: Vec<(usize, usize)> = reduced
        .iter()
        .map(|&axis| (dims[axis], strides[axis]))
        .collect();
    let mut run = 1;
    while let Some(&(dim, stride)) = block.last() {
        if stride != run {
            break;
        }
        run *= dim;
        block.pop();
    }
    let runs = offsets(block.into_iter());
    let values = rows
        .iter()
        .map(|&row| {
            runs.iter().fold(init, |acc, &start| {
                x[row + start..][..run]
                    .iter()
                    .fold(acc, |acc, &v| combine(acc, v))
            })
        })
        .collect();
    HostTensor::f32(crate::dims_of(out)?, values)
}

/// The element offset of each index of the axes `(extent, stride)`, in row-major order.
fn offsets(axes: impl Iterator<Item = (usize, usize)>) -> Vec<usize> {
    axes.fold(vec![0], |offsets, (dim, stride)| {
        offsets
            .iter()
            .flat_map(|&offset| (0..dim).map(move |i| offset + i * stride))
            .collect()
    })
}
