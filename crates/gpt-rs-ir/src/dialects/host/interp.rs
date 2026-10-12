//! Semantics of `host.dispatch`: a call of its dispatch.

use anyhow::{anyhow, Result};

use super::ir::*;
use crate::dialects::r#loop::{self, Access};
use crate::host::{HostBuffer, HostValue};
use crate::op::{Eval, EvalCx, Host, NoRegions, OwnedHost};
use crate::types::{BufferType, Space, Type};

impl Eval for Dispatch {
    fn eval(
        &self,
        ins: DispatchOperands<Host<'_>>,
        cx: &mut EvalCx<'_, NoRegions>,
    ) -> Result<DispatchResults<OwnedHost>> {
        let name = &self.callee.0;
        let module = cx.module();
        let callee = module
            .symbol(name)
            .and_then(|inst| module.op(inst).get::<r#loop::Func>())
            .cloned();
        let Some(func) = callee else {
            let args = HostValue::of_tensors(&ins.args);
            let outs = HostValue::into_tensors(cx.call(name, args)?)?;
            return Ok(DispatchResults { outs });
        };
        // A launch: the arguments in `in` buffers, and an `out` buffer for each result.
        let mut args = ins.args.iter();
        let mut results = cx.result_types().iter();
        let mut params = Vec::with_capacity(func.args.len());
        let mut outs = Vec::new();
        for access in &func.access {
            let buffer = match access {
                Access::In => {
                    let tensor = args
                        .next()
                        .ok_or_else(|| anyhow!("@{name} takes more arguments"))?;
                    HostBuffer::of_tensor(tensor, Space::Global)?
                }
                Access::Out => {
                    let Some(Type::Tensor(tensor)) = results.next() else {
                        return Err(anyhow!("@{name} gives more results"));
                    };
                    let buffer = HostBuffer::new(BufferType::of(tensor, Space::Global))?;
                    outs.push(buffer.clone());
                    buffer
                }
            };
            params.push(HostValue::Buffer(buffer));
        }
        cx.call(name, params)?;
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
