//! Semantics of the `host` ops: a call of a dispatch, and the buffers and launches of the buffer
//! form.

use anyhow::{anyhow, ensure, Result};

use super::ir::*;
use crate::dialects::loops::{self, Access};
use crate::interp::{HostBuffer, HostValue};
use crate::ir::{BufferType, Eval, EvalCx, Host, NoRegions, OwnedHost, Space, Type};

impl Eval for Dispatch {
    fn eval(
        &self,
        ins: DispatchOperands<Host>,
        cx: &mut EvalCx<NoRegions>,
    ) -> Result<DispatchResults<OwnedHost>> {
        let name = &self.callee.0;
        let module = cx.module();
        let callee = module
            .symbol(name)
            .and_then(|inst| module.op(inst).get::<loops::Func>())
            .cloned();
        let Some(func) = callee else {
            let args = HostValue::of_tensors(&ins.args);
            let outs = HostValue::into_tensors(cx.call(name, args)?)?;
            return Ok(DispatchResults { outs });
        };
        // A launch: the arguments in `in` buffers, a buffer for each result, and a copy of the
        // argument for each result that updates one.
        let mut args = ins.args.iter();
        let mut results = cx.result_types().iter();
        let mut params = Vec::with_capacity(func.args.len());
        let mut outs = Vec::new();
        for access in &func.access {
            let buffer = match access.takes_argument() {
                true => {
                    let tensor = args
                        .next()
                        .ok_or_else(|| anyhow!("@{name} takes more arguments"))?;
                    if access.gives_result() {
                        results.next();
                    }
                    HostBuffer::of_tensor(tensor, Space::Global)?
                }
                false => {
                    let Some(Type::Tensor(tensor)) = results.next() else {
                        return Err(anyhow!("@{name} gives more results"));
                    };
                    HostBuffer::new(BufferType::of(tensor, Space::Global))?
                }
            };
            if access.gives_result() {
                outs.push(buffer.clone());
            }
            params.push(buffer);
        }
        cx.call(name, launch_args(&func.access, params)?)?;
        let mut tensors = Vec::with_capacity(outs.len());
        for (index, buffer) in outs.iter().enumerate() {
            let tensor = buffer
                .to_tensor()
                .map_err(|err| anyhow!("@{name} result {index}: {err}"))?;
            tensors.push(tensor);
        }
        Ok(DispatchResults { outs: tensors })
    }
}

impl Eval for Arena {
    fn eval(
        &self,
        _ins: ArenaOperands<Host>,
        _cx: &mut EvalCx<NoRegions>,
    ) -> Result<ArenaResults<OwnedHost>> {
        Ok(ArenaResults {
            output: HostBuffer::arena(self.buffer_type()?)?,
        })
    }
}

impl Eval for Slice {
    fn eval(
        &self,
        ins: SliceOperands<Host>,
        _cx: &mut EvalCx<NoRegions>,
    ) -> Result<SliceResults<OwnedHost>> {
        Ok(SliceResults {
            output: ins.arena.slice(self.offset, self.buffer_type())?,
        })
    }
}

impl Eval for Alloc {
    fn eval(
        &self,
        _ins: AllocOperands<Host>,
        _cx: &mut EvalCx<NoRegions>,
    ) -> Result<AllocResults<OwnedHost>> {
        Ok(AllocResults {
            output: HostBuffer::new(self.buffer_type())?,
        })
    }
}

impl Eval for ToBuffer {
    fn eval(
        &self,
        ins: ToBufferOperands<Host>,
        _cx: &mut EvalCx<NoRegions>,
    ) -> Result<ToBufferResults<OwnedHost>> {
        Ok(ToBufferResults {
            output: HostBuffer::read_only(ins.tensor)?,
        })
    }
}

impl Eval for Take {
    fn eval(
        &self,
        ins: TakeOperands<Host>,
        _cx: &mut EvalCx<NoRegions>,
    ) -> Result<TakeResults<OwnedHost>> {
        Ok(TakeResults {
            output: HostBuffer::of_tensor(ins.tensor, Space::Global)?,
        })
    }
}

/// A launch reads the elements that its `in` and `inout` buffers hold, then its `out` and `inout`
/// buffers take their bytes, and no write of the launch may overlap an `in` buffer of another
/// value. A buffer that the launch writes goes to no other parameter, unless an `out<over = k>`
/// parameter shares it with the `in` parameter `k`, which the launch then reads only where the
/// `out` parameter has not written.
impl Eval for Launch {
    fn eval(
        &self,
        ins: LaunchOperands<Host>,
        cx: &mut EvalCx<NoRegions>,
    ) -> Result<LaunchResults<OwnedHost>> {
        let name = &self.callee.0;
        let module = cx.module();
        let func = module
            .symbol(name)
            .and_then(|inst| module.op(inst).get::<loops::Func>())
            .cloned()
            .ok_or_else(|| anyhow!("@{name} is not a loop.func"))?;
        let params = ins.params(&func)?;
        for (index, access) in func.access.iter().enumerate() {
            if *access == Access::In {
                continue;
            }
            for (other, buffer) in params.iter().enumerate() {
                let shared = *access == (Access::Out { over: Some(other) });
                ensure!(
                    other == index || *buffer != params[index] || shared,
                    "parameters {index} and {other} of @{name} get one buffer, which it writes"
                );
            }
        }
        for (buffer, access) in params.iter().zip(&func.access) {
            if !matches!(access, Access::Out { .. }) {
                buffer.check_held()?;
            }
        }
        for (buffer, access) in params.iter().zip(&func.access) {
            if *access != Access::In {
                buffer.hold()?;
            }
        }
        for (buffer, access) in params.iter().zip(&func.access) {
            if *access == Access::In {
                buffer.check_held()?;
            }
        }
        let args = params.into_iter().cloned().collect();
        cx.call(name, launch_args(&func.access, args)?)?;
        Ok(LaunchResults::default())
    }
}

/// The arguments of a launch of a `loop.func` whose parameters `access` describes, on `params`,
/// the buffer of each parameter in order: each `out<over = k>` parameter and the `in` parameter
/// `k` get handles that know which elements the `out` parameter wrote.
fn launch_args(access: &[Access], mut params: Vec<HostBuffer>) -> Result<Vec<HostValue>> {
    for (index, &param_access) in access.iter().enumerate() {
        let Access::Out { over: Some(shared) } = param_access else {
            continue;
        };
        let (input, output) = HostBuffer::written_over(&params[shared], &params[index])?;
        params[shared] = input;
        params[index] = output;
    }
    Ok(params.into_iter().map(HostValue::Buffer).collect())
}

impl Eval for Copy {
    fn eval(
        &self,
        ins: CopyOperands<Host>,
        _cx: &mut EvalCx<NoRegions>,
    ) -> Result<CopyResults<OwnedHost>> {
        ins.from.check_held()?;
        ins.to.hold()?;
        ins.from.check_held()?;
        ins.to.copy_from(ins.from)?;
        Ok(CopyResults::default())
    }
}

impl Eval for ToTensor {
    fn eval(
        &self,
        ins: ToTensorOperands<Host>,
        _cx: &mut EvalCx<NoRegions>,
    ) -> Result<ToTensorResults<OwnedHost>> {
        ins.buffer.check_held()?;
        Ok(ToTensorResults {
            output: ins.buffer.to_tensor()?,
        })
    }
}
