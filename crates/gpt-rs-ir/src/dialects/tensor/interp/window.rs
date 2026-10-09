//! Semantics of the ops in `ir/window.rs`.

use super::contraction::reduce;
use super::kernels::{cast, strides, unravel, Chunks};
use crate::dialects::tensor::ir::{
    ExtractPatches, ExtractPatchesOperands, ExtractPatchesResults, ReduceWindow,
    ReduceWindowOperands, ReduceWindowResults,
};
use crate::host::{HostTensor, TensorData};
use crate::op::{Eval, EvalCx, Host, NoRegions, OwnedHost};
use anyhow::Result;

impl Eval for ExtractPatches {
    fn eval(
        &self,
        ins: ExtractPatchesOperands<Host<'_>>,
        _cx: &mut EvalCx<'_, NoRegions>,
    ) -> Result<ExtractPatchesResults<OwnedHost>> {
        let x = Chunks::of(&ins.input.data);
        let (windowed, out_dims) = self.geometry(&ins.input.tensor_type())?;
        let in_strides = strides(&ins.input.dims);
        let channels = ins.input.dims[ins.input.dims.len() - 1];
        // A row is one window position `[n, o...]`. Its patch holds the window elements in
        // row-major order, `channels` values each. Elements in the padding keep `pad_value`.
        let rows = &out_dims[..out_dims.len() - 1];
        let offsets: Vec<Vec<usize>> = (0..self.window.iter().product())
            .map(|k| unravel(k, &self.window))
            .collect();
        let pad = cast(
            &TensorData::F64(vec![self.pad_value.0].into()),
            ins.input.dtype(),
        )?;
        let pad = Chunks::of(&pad);
        let mut out = Chunks::zeroed(ins.input.dtype(), out_dims.iter().product());
        for i in 0..out.len() {
            out.set(i, pad.get(0));
        }
        for row in 0..rows.iter().product::<usize>() {
            let at = unravel(row, rows);
            for (k, offset) in offsets.iter().enumerate() {
                let source = windowed.iter().enumerate().try_fold(
                    at[0] * in_strides[0],
                    |acc, (i, axis)| {
                        Some(acc + axis.source(at[1 + i], offset[i])? * in_strides[1 + i])
                    },
                );
                if let Some(src) = source {
                    let dst = (row * offsets.len() + k) * channels;
                    for c in 0..channels {
                        out.set(dst + c, x.get(src + c));
                    }
                }
            }
        }
        Ok(ExtractPatchesResults {
            output: HostTensor::new(out_dims, out.into_data()?)?,
        })
    }
}

impl Eval for ReduceWindow {
    fn eval(
        &self,
        ins: ReduceWindowOperands<Host<'_>>,
        _cx: &mut EvalCx<'_, NoRegions>,
    ) -> Result<ReduceWindowResults<OwnedHost>> {
        let input = ins.input;
        let (windowed, out_dims) = self.geometry(&input.tensor_type())?;
        let in_strides = strides(&input.dims);
        let offsets: Vec<Vec<usize>> = (0..self.window_dims.iter().product())
            .map(|k| unravel(k, &self.window_dims))
            .collect();
        // Each output folds its window elements in row-major order, skipping the ones in the
        // padding or between dilated input elements.
        let groups: Vec<Vec<usize>> = (0..out_dims.iter().product())
            .map(|o| {
                let at = unravel(o, &out_dims);
                let source = |offset: &Vec<usize>| {
                    windowed.iter().enumerate().try_fold(0, |src, (i, axis)| {
                        Some(src + axis.source(at[i], offset[i])? * in_strides[i])
                    })
                };
                offsets.iter().filter_map(source).collect()
            })
            .collect();
        let out = self.out.unwrap_or(input.dtype());
        let data = reduce(input, &groups, self.kind, self.accum, out)?;
        Ok(ReduceWindowResults {
            output: HostTensor::new(out_dims, data)?,
        })
    }
}
