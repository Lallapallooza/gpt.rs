//! Semantics of the control-flow ops.

use anyhow::{anyhow, Result};

use super::kernels::{gather_elements, Chunks};
use crate::dialects::tensor::ir::{
    Cond, CondOperands, CondRegions, CondResults, Scan, ScanOperands, ScanRegions, ScanResults,
    While, WhileOperands, WhileRegions, WhileResults, Yield, YieldOperands, YieldResults,
};
use crate::host::HostTensor;
use crate::interp::HostValue;
use crate::ir::{Eval, EvalCx, Host, NoRegions, OwnedHost, Region, TensorType};

impl Eval for Yield {
    fn eval(
        &self,
        _ins: YieldOperands<Host>,
        _cx: &mut EvalCx<NoRegions>,
    ) -> Result<YieldResults<OwnedHost>> {
        Ok(YieldResults::default())
    }
}

impl Eval for Cond {
    fn eval(
        &self,
        ins: CondOperands<Host>,
        cx: &mut EvalCx<CondRegions<Region>>,
    ) -> Result<CondResults<OwnedHost>> {
        let branch = match ins.pred.as_i1()?[0] {
            0 => cx.regions.else_branch,
            _ => cx.regions.then_branch,
        };
        Ok(CondResults {
            outputs: HostValue::into_tensors(cx.run(branch, HostValue::of_tensors(&ins.args))?)?,
        })
    }
}

impl Eval for While {
    fn eval(
        &self,
        ins: WhileOperands<Host>,
        cx: &mut EvalCx<WhileRegions<Region>>,
    ) -> Result<WhileResults<OwnedHost>> {
        let (cond, body) = (cx.regions.cond, cx.regions.body);
        let mut carried = HostValue::of_tensors(&ins.init);
        loop {
            let mut yielded = cx.run(cond, carried)?.into_iter();
            let pred = yielded
                .next()
                .ok_or_else(|| anyhow!("cond yields no predicate"))?
                .into_tensor()?;
            carried = yielded.collect();
            if pred.as_i1()?[0] == 0 {
                return Ok(WhileResults {
                    outputs: HostValue::into_tensors(carried)?,
                });
            }
            carried = cx.run(body, carried)?;
        }
    }
}

impl Eval for Scan {
    fn eval(
        &self,
        ins: ScanOperands<Host>,
        cx: &mut EvalCx<ScanRegions<Region>>,
    ) -> Result<ScanResults<OwnedHost>> {
        let (initial, scanned) = ins.inputs.split_at(self.carry_count);
        let steps = scanned[0].dims[0];
        let mut carried = HostValue::of_tensors(initial);
        let mut outputs: Vec<Vec<HostTensor>> = vec![Vec::with_capacity(steps); self.outputs.len()];
        for step in 0..steps {
            let slices = scanned
                .iter()
                .map(|input| {
                    let row: usize = input.dims[1..].iter().product();
                    let data = gather_elements(&input.data, row, |element| step * row + element)?;
                    Ok(HostValue::Tensor(HostTensor::new(&input.dims[1..], data)?))
                })
                .collect::<Result<Vec<_>>>()?;
            let mut yielded =
                cx.run(cx.regions.body, carried.into_iter().chain(slices).collect())?;
            let per_step = yielded.split_off(self.carry_count);
            carried = yielded;
            for (stack, output) in outputs.iter_mut().zip(per_step) {
                stack.push(output.into_tensor()?);
            }
        }
        let stacked = outputs
            .into_iter()
            .zip(&self.outputs)
            .map(|(stack, ty)| stack_steps(stack, ty, steps))
            .collect::<Result<Vec<_>>>()?;
        Ok(ScanResults {
            outputs: HostValue::into_tensors(carried)?
                .into_iter()
                .chain(stacked)
                .collect(),
        })
    }
}

/// `steps` tensors of type `ty`, stacked along a new leading axis.
fn stack_steps(stack: Vec<HostTensor>, ty: &TensorType, steps: usize) -> Result<HostTensor> {
    let dims = ty
        .static_dims()
        .ok_or_else(|| anyhow!("a scan output of {ty} has a dynamic extent"))?;
    let count: usize = dims.iter().product();
    let mut out = Chunks::zeroed(ty.dtype, steps * count);
    for (step, tensor) in stack.iter().enumerate() {
        let elements = Chunks::of(&tensor.data);
        for element in 0..count {
            out.set(step * count + element, elements.get(element));
        }
    }
    HostTensor::new(
        std::iter::once(steps).chain(dims).collect::<Vec<_>>(),
        out.into_data()?,
    )
}
