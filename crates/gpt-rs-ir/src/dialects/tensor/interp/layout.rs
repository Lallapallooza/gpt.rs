//! Semantics of the ops in `ir/layout.rs`.

use std::sync::Arc;

use anyhow::Result;

use super::kernels::{gather_elements, strides, unravel, Chunks};
use crate::dialects::tensor::ir::shape::{axis_index, check_window};
use crate::dialects::tensor::ir::{
    Bitcast, BitcastOperands, BitcastResults, BroadcastTo, BroadcastToOperands, BroadcastToResults,
    Concat, ConcatOperands, ConcatResults, Reshape, ReshapeOperands, ReshapeResults, Slice,
    SliceOperands, SliceResults, Transpose, TransposeOperands, TransposeResults,
};
use crate::host::{HostTensor, TensorData};
use crate::op::{Eval, EvalCx, Host, NoRegions, OwnedHost};

impl Eval for Reshape {
    fn eval(
        &self,
        ins: ReshapeOperands<Host<'_>>,
        cx: &mut EvalCx<'_, NoRegions>,
    ) -> Result<ReshapeResults<OwnedHost>> {
        Ok(ReshapeResults {
            output: HostTensor::new(cx.result_dims(0)?, ins.input.data.clone())?,
        })
    }
}

impl Eval for Transpose {
    fn eval(
        &self,
        ins: TransposeOperands<Host<'_>>,
        _cx: &mut EvalCx<'_, NoRegions>,
    ) -> Result<TransposeResults<OwnedHost>> {
        let in_dims = &ins.input.dims;
        let out_dims: Vec<usize> = self.perm.iter().map(|&axis| in_dims[axis]).collect();
        let in_strides = strides(in_dims);
        let count: usize = out_dims.iter().product();
        let source = |out_index: usize| -> usize {
            unravel(out_index, &out_dims)
                .iter()
                .enumerate()
                .map(|(out_axis, &coord)| coord * in_strides[self.perm[out_axis]])
                .sum()
        };
        let data = gather_elements(&ins.input.data, count, source)?;
        Ok(TransposeResults {
            output: HostTensor::new(out_dims, data)?,
        })
    }
}

impl Eval for BroadcastTo {
    fn eval(
        &self,
        ins: BroadcastToOperands<Host<'_>>,
        cx: &mut EvalCx<'_, NoRegions>,
    ) -> Result<BroadcastToResults<OwnedHost>> {
        let input = ins.input;
        let in_dims = &input.dims;
        let rank_diff = self.shape.len() - in_dims.len();
        let out_dims = cx.result_dims(0)?;
        let in_strides = strides(in_dims);
        let count: usize = out_dims.iter().product();
        let source = |out_index: usize| -> usize {
            let coord = unravel(out_index, &out_dims);
            (0..in_dims.len())
                .filter(|&axis| in_dims[axis] != 1)
                .map(|axis| coord[rank_diff + axis] * in_strides[axis])
                .sum()
        };
        let data = gather_elements(&input.data, count, source)?;
        Ok(BroadcastToResults {
            output: HostTensor::new(out_dims, data)?,
        })
    }
}

impl Eval for Slice {
    fn eval(
        &self,
        ins: SliceOperands<Host<'_>>,
        _cx: &mut EvalCx<'_, NoRegions>,
    ) -> Result<SliceResults<OwnedHost>> {
        let input = ins.input;
        for (axis, &extent) in input.dims.iter().enumerate() {
            check_window(axis, self.starts[axis], self.span(axis), extent)?;
        }
        let in_strides = strides(&input.dims);
        let count: usize = self.sizes.iter().product();
        let source = |out_index: usize| -> usize {
            unravel(out_index, &self.sizes)
                .iter()
                .enumerate()
                .map(|(axis, &coord)| {
                    (self.starts[axis] + coord * self.stride(axis)) * in_strides[axis]
                })
                .sum()
        };
        Ok(SliceResults {
            output: HostTensor::new(&self.sizes, gather_elements(&input.data, count, source)?)?,
        })
    }
}

impl Eval for Concat {
    fn eval(
        &self,
        ins: ConcatOperands<Host<'_>>,
        _cx: &mut EvalCx<'_, NoRegions>,
    ) -> Result<ConcatResults<OwnedHost>> {
        let first = ins.inputs[0];
        let axis = axis_index(self.axis, first.dims.len())?;
        let mut out_dims = first.dims.clone();
        out_dims[axis] = ins.inputs.iter().map(|input| input.dims[axis]).sum();
        let outer: usize = out_dims[..axis].iter().product();
        // Each of the `outer` rows of the result is the next run of every input in turn.
        let parts: Vec<(Chunks<'_>, usize)> = ins
            .inputs
            .iter()
            .map(|&input| {
                let run = input.dims[axis..].iter().product::<usize>();
                (Chunks::of(&input.data), run)
            })
            .collect();
        let mut out = Chunks::zeroed(first.dtype(), out_dims.iter().product());
        let mut next = 0;
        for row in 0..outer {
            for (part, run) in &parts {
                out.set_run(next, part.run(row * run, *run));
                next += run;
            }
        }
        Ok(ConcatResults {
            output: HostTensor::new(out_dims, out.into_data()?)?,
        })
    }
}

impl Eval for Bitcast {
    fn eval(
        &self,
        ins: BitcastOperands<Host<'_>>,
        cx: &mut EvalCx<'_, NoRegions>,
    ) -> Result<BitcastResults<OwnedHost>> {
        // The elements' bytes are their bit stream: packed from the least significant bit of each
        // byte, little-endian across the bytes of an element.
        let dims = cx.result_dims(0)?;
        let count = dims.iter().product();
        let bytes = ins.input.to_le_bytes().to_vec();
        let data = TensorData::read(self.to, count, Arc::new(bytes))?;
        Ok(BitcastResults {
            output: HostTensor::new(dims, data)?,
        })
    }
}
