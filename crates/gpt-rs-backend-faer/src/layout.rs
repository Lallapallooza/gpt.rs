//! faer's kernels for the ops that move elements without computing on them. Each one copies, with
//! [`copy_strided`], the runs of its input that are contiguous in memory, whatever the dtype.

use anyhow::{anyhow, ensure, Result};
use gpt_rs_ir::dialects::tensor::{
    axis_index, BroadcastTo, Concat, DynamicSlice, Slice, Transpose,
};
use gpt_rs_ir::host::HostTensor;
use gpt_rs_ir::types::TensorType;

/// Row-major strides of `dims`, in elements.
fn strides(dims: &[usize]) -> Vec<usize> {
    let mut strides = vec![1; dims.len()];
    for axis in (0..dims.len().saturating_sub(1)).rev() {
        strides[axis] = strides[axis + 1] * dims[axis + 1];
    }
    strides
}

/// One side of a strided copy: element `c` of the copy is at `offset + c * strides`, in elements.
struct Side<'a> {
    offset: usize,
    strides: &'a [usize],
}

/// Copies the elements of an index space of extents `dims` from `src` to `dst`, both slices of
/// elements of `elem` bytes. Adjacent axes that both sides lay out contiguously merge, so the
/// innermost loop copies whole runs.
fn copy_strided(
    dst: &mut [u8],
    to: Side<'_>,
    src: &[u8],
    from: Side<'_>,
    dims: &[usize],
    elem: usize,
) {
    if dims.contains(&0) {
        return;
    }
    // The axes of extent above 1, merged where both sides are contiguous across them.
    let mut axes: Vec<(usize, usize, usize)> = Vec::new();
    for (axis, &dim) in dims.iter().enumerate().filter(|&(_, &dim)| dim > 1) {
        let (ds, ss) = (to.strides[axis], from.strides[axis]);
        match axes.last_mut() {
            Some(last) if last.1 == ds * dim && last.2 == ss * dim => {
                *last = (last.0 * dim, ds, ss)
            }
            _ => axes.push((dim, ds, ss)),
        }
    }
    let (inner, ds, ss) = axes.pop().unwrap_or((1, 1, 1));
    let run = |dst: &mut [u8], d: usize, s: usize| {
        let (d, s) = (d * elem, s * elem);
        if ds == 1 && ss == 1 {
            dst[d..][..inner * elem].copy_from_slice(&src[s..][..inner * elem]);
        } else if ds == 1 && ss == 0 {
            let value = &src[s..][..elem];
            for chunk in dst[d..][..inner * elem].chunks_exact_mut(elem) {
                chunk.copy_from_slice(value);
            }
        } else {
            for i in 0..inner {
                dst[d + i * ds * elem..][..elem].copy_from_slice(&src[s + i * ss * elem..][..elem]);
            }
        }
    };
    // An odometer over the outer axes, which moves both offsets as it counts.
    let mut index = vec![0usize; axes.len()];
    let (mut d, mut s) = (to.offset, from.offset);
    loop {
        run(dst, d, s);
        let mut axis = axes.len();
        loop {
            let Some(next) = axis.checked_sub(1) else {
                return;
            };
            axis = next;
            let (dim, ds, ss) = axes[axis];
            index[axis] += 1;
            d += ds;
            s += ss;
            if index[axis] < dim {
                break;
            }
            index[axis] = 0;
            d -= ds * dim;
            s -= ss * dim;
        }
    }
}

/// A tensor of type `ty` whose elements a strided copy from `input` fills.
fn gather(
    input: &HostTensor,
    ty: &TensorType,
    offset: usize,
    src_strides: &[usize],
) -> Result<HostTensor> {
    let mut out = HostTensor::zeros(ty)?;
    let elem = elem_bytes(input)?;
    let dims = out.dims;
    let dst_strides = strides(&dims);
    copy_strided(
        out.bytes_mut(),
        Side {
            offset: 0,
            strides: &dst_strides,
        },
        input.to_le_bytes(),
        Side {
            offset,
            strides: src_strides,
        },
        &dims,
        elem,
    );
    Ok(out)
}

fn elem_bytes(tensor: &HostTensor) -> Result<usize> {
    let dtype = tensor.dtype();
    dtype
        .size_in_bytes()
        .ok_or_else(|| anyhow!("{dtype} elements are not whole bytes"))
}

pub(crate) fn broadcast_to(
    _op: &BroadcastTo,
    input: &HostTensor,
    out: &TensorType,
) -> Result<HostTensor> {
    let out_dims = crate::dims_of(out)?;
    let rank_diff = out_dims.len() - input.dims.len();
    let in_strides = strides(&input.dims);
    let src_strides: Vec<usize> = (0..out_dims.len())
        .map(|axis| match axis.checked_sub(rank_diff) {
            Some(a) if input.dims[a] != 1 => in_strides[a],
            _ => 0,
        })
        .collect();
    gather(input, out, 0, &src_strides)
}

pub(crate) fn transpose(
    op: &Transpose,
    input: &HostTensor,
    out: &TensorType,
) -> Result<HostTensor> {
    let in_strides = strides(&input.dims);
    let src_strides: Vec<usize> = op.perm.iter().map(|&axis| in_strides[axis]).collect();
    gather(input, out, 0, &src_strides)
}

pub(crate) fn slice(op: &Slice, input: &HostTensor, out: &TensorType) -> Result<HostTensor> {
    let in_strides = strides(&input.dims);
    let offset = op
        .starts
        .iter()
        .zip(&in_strides)
        .map(|(start, stride)| start * stride)
        .sum();
    let steps: Vec<usize> = in_strides
        .iter()
        .enumerate()
        .map(|(axis, stride)| stride * op.stride(axis))
        .collect();
    gather(input, out, offset, &steps)
}

pub(crate) fn dynamic_slice(
    op: &DynamicSlice,
    input: &HostTensor,
    starts: &HostTensor,
    out: &TensorType,
) -> Result<HostTensor> {
    let starts = clamped_starts(&input.dims, &op.sizes, starts)?;
    window(input, &starts, out)
}

/// The window of `input` at `starts` whose extents are those of `out`.
fn window(input: &HostTensor, starts: &[usize], out: &TensorType) -> Result<HostTensor> {
    let in_strides = strides(&input.dims);
    let offset = starts
        .iter()
        .zip(&in_strides)
        .map(|(start, stride)| start * stride)
        .sum();
    gather(input, out, offset, &in_strides)
}

/// `base` with the window at `starts` replaced by `update`.
pub(crate) fn dynamic_update_slice(
    base: &HostTensor,
    update: &HostTensor,
    starts: &HostTensor,
) -> Result<HostTensor> {
    let starts = clamped_starts(&base.dims, &update.dims, starts)?;
    let elem = elem_bytes(base)?;
    let base_strides = strides(&base.dims);
    let offset = starts
        .iter()
        .zip(&base_strides)
        .map(|(start, stride)| start * stride)
        .sum();
    let mut out = base.clone();
    copy_strided(
        out.bytes_mut(),
        Side {
            offset,
            strides: &base_strides,
        },
        update.to_le_bytes(),
        Side {
            offset: 0,
            strides: &strides(&update.dims),
        },
        &update.dims,
        elem,
    );
    Ok(out)
}

pub(crate) fn concat(op: &Concat, inputs: &[HostTensor], out: &TensorType) -> Result<HostTensor> {
    let mut result = HostTensor::zeros(out)?;
    let out_strides = strides(&result.dims);
    let axis = axis_index(op.axis, result.dims.len())?;
    let mut start = 0;
    for input in inputs {
        let elem = elem_bytes(input)?;
        copy_strided(
            result.bytes_mut(),
            Side {
                offset: start * out_strides[axis],
                strides: &out_strides,
            },
            input.to_le_bytes(),
            Side {
                offset: 0,
                strides: &strides(&input.dims),
            },
            &input.dims,
            elem,
        );
        start += input.dims[axis];
    }
    Ok(result)
}

/// Each start clamped into `[0, extent - size]`, as `dynamic_slice` and `dynamic_update_slice`
/// define, so the window always lies inside the tensor.
fn clamped_starts(dims: &[usize], sizes: &[usize], starts: &HostTensor) -> Result<Vec<usize>> {
    let starts = starts.as_si32()?;
    ensure!(
        starts.len() == dims.len() && sizes.len() == dims.len(),
        "{} starts and {} sizes for a rank-{} tensor",
        starts.len(),
        sizes.len(),
        dims.len()
    );
    dims.iter()
        .zip(sizes)
        .zip(starts)
        .map(|((&extent, &size), &start)| {
            ensure!(
                size <= extent,
                "a window of {size} does not fit an extent of {extent}"
            );
            Ok((start.max(0) as usize).min(extent - size))
        })
        .collect()
}
