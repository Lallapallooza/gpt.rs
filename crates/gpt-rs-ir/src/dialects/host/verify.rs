//! The type rules of the `host` ops: the types each op gives, and what it rejects.

use anyhow::{anyhow, bail, ensure, Result};

use crate::dialects::host::ir::*;
use crate::dialects::{r#loop, tile};
use crate::module::Module;
use crate::op::{NoRegions, OpDef, OwnedTypes, TypeCx, Types};
use crate::types::{Dim, SymBindings, TensorType, Type};

/// The argument and result types of the dispatch named `name`: a `tile.dispatch`, or a
/// `loop.func` as a call with tensors sees it.
fn callee_types(module: &Module, name: &str) -> Result<(Vec<TensorType>, Vec<TensorType>)> {
    let inst = module
        .symbol(name)
        .ok_or_else(|| anyhow!("no symbol @{name}"))?;
    let op = module.op(inst);
    if let Some(func) = op.get::<r#loop::Func>() {
        return func.call_types();
    }
    let dispatch = op
        .get::<tile::Dispatch>()
        .ok_or_else(|| anyhow!("@{name} is not a tile.dispatch or a loop.func"))?;
    let tensors = |types: &[Type], what: &str| -> Result<Vec<TensorType>> {
        types
            .iter()
            .map(|ty| {
                ty.as_tensor()
                    .cloned()
                    .ok_or_else(|| anyhow!("@{name} {what} {ty}, not a tensor"))
            })
            .collect()
    };
    Ok((
        tensors(&dispatch.args, "takes")?,
        tensors(&dispatch.results, "gives")?,
    ))
}

impl OpDef for Dispatch {
    fn result_types(
        &self,
        ins: DispatchOperands<Types<'_>>,
        cx: &TypeCx<'_, NoRegions>,
    ) -> Result<DispatchResults<OwnedTypes>> {
        let name = &self.callee.0;
        let (params, results) = callee_types(cx.module, name)?;
        ensure!(
            params.len() == ins.args.len(),
            "@{name} takes {} arguments, got {}",
            params.len(),
            ins.args.len()
        );
        let mut bound = SymBindings::default();
        for (i, (param, &arg)) in params.iter().zip(&ins.args).enumerate() {
            bound
                .bind(param, arg)
                .map_err(|err| anyhow!("argument {i}: {err}"))?;
        }
        let outs = results
            .iter()
            .map(|tensor| {
                let unbound = tensor.dims.iter().find(|dim| match dim {
                    Dim::Dynamic(sym) => bound.get(sym).is_none(),
                    Dim::Static(_) => false,
                });
                if let Some(dim) = unbound {
                    bail!("no argument binds {dim}");
                }
                Ok(bound.resolve(tensor))
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(DispatchResults { outs })
    }
}
