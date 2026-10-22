//! Semantics of the window ops.

use anyhow::Result;

use super::contraction::fold_groups;
use super::kernels::{cast, strides, unravel, Chunks};
use crate::dialects::arith::math::Combiner;
use crate::dialects::tensor::ir::*;
use crate::dialects::tensor::Accumulates;
use crate::host::{HostTensor, TensorData};
use crate::ir::{Eval, EvalCx, Host, NoRegions, OwnedHost};

impl Eval for ExtractPatches {
    fn eval(
        &self,
        ins: ExtractPatchesOperands<Host>,
        _cx: &mut EvalCx<NoRegions>,
    ) -> Result<ExtractPatchesResults<OwnedHost>> {
        let x = Chunks::of(&ins.input.data);
        let (windowed, out_dims) = self.geometry(&ins.input.tensor_type())?;
        let in_strides = strides(&ins.input.dims);
        let channels = ins.input.dims[ins.input.dims.len() - 1];
        // A row is one window position `[n, o...]`. Its patch holds the window elements in
        // row-major order, `channels` values each. Elements in the padding keep `pad_value`.
        let rows = &out_dims[..out_dims.len() - 1];
        let offsets: Vec<Vec<usize>> = (0..self.window.iter().product())
            .map(|position| unravel(position, &self.window))
            .collect();
        let pad_data = cast(
            &TensorData::F64(vec![self.pad_value.0].into()),
            ins.input.dtype(),
        )?;
        let pad = Chunks::of(&pad_data);
        let mut out = Chunks::zeroed(ins.input.dtype(), out_dims.iter().product());
        for position in 0..out.len() {
            out.set(position, pad.get(0));
        }
        for row in 0..rows.iter().product::<usize>() {
            let at = unravel(row, rows);
            for (window_index, offset) in offsets.iter().enumerate() {
                let source = windowed.iter().enumerate().try_fold(
                    at[0] * in_strides[0],
                    |acc, (position, axis)| {
                        Some(
                            acc + axis.source(at[1 + position], offset[position])?
                                * in_strides[1 + position],
                        )
                    },
                );
                if let Some(src) = source {
                    let dst = (row * offsets.len() + window_index) * channels;
                    for channel in 0..channels {
                        out.set(dst + channel, x.get(src + channel));
                    }
                }
            }
        }
        Ok(ExtractPatchesResults {
            output: HostTensor::new(out_dims, out.into_data()?)?,
        })
    }
}

/// Every window of `input` that `attrs` places, folded with `C`.
pub(crate) fn reduce_windows<C: Combiner>(
    attrs: &ReduceWindowAttrs,
    input: &HostTensor,
) -> Result<HostTensor> {
    let (windowed, out_dims) = attrs.geometry(&input.tensor_type())?;
    let in_strides = strides(&input.dims);
    let offsets: Vec<Vec<usize>> = (0..attrs.window_dims.iter().product())
        .map(|position| unravel(position, &attrs.window_dims))
        .collect();
    // Each output folds its window elements in row-major order, skipping the ones in the padding
    // or between dilated input elements.
    let groups: Vec<Vec<usize>> = (0..out_dims.iter().product())
        .map(|output_index| {
            let at = unravel(output_index, &out_dims);
            let source = |offset: &Vec<usize>| {
                windowed
                    .iter()
                    .enumerate()
                    .try_fold(0, |src, (position, axis)| {
                        Some(
                            src + axis.source(at[position], offset[position])?
                                * in_strides[position],
                        )
                    })
            };
            offsets.iter().filter_map(source).collect()
        })
        .collect();
    let (accum, out) = (
        attrs.accum_dtype(input.dtype()),
        attrs.out_dtype(input.dtype()),
    );
    let data = fold_groups::<C>(input, &groups, accum, out)?;
    HostTensor::new(out_dims, data)
}

/// The semantics of every [`ReduceWindow`] op.
pub(crate) fn reduce_window<T: ReduceWindow>(
    op: &T,
    ins: ReduceWindowOperands<Host>,
    _cx: &mut EvalCx<NoRegions>,
) -> Result<ReduceWindowResults<OwnedHost>> {
    Ok(ReduceWindowResults {
        output: reduce_windows::<T::Combiner>(op, ins.input)?,
    })
}
