//! The type rules of the `host` ops: the types each op gives, and what it rejects.

use anyhow::{anyhow, bail, ensure, Result};

use crate::dialects::host::ir::*;
use crate::dialects::tile;
use crate::op::{NoRegions, OpDef, OwnedTypes, TypeCx, Types};
use crate::types::{Dim, SymBindings};

impl OpDef for Dispatch {
    fn result_types(
        &self,
        ins: DispatchOperands<Types<'_>>,
        cx: &TypeCx<'_, NoRegions>,
    ) -> Result<DispatchResults<OwnedTypes>> {
        let name = &self.callee.0;
        let inst = cx
            .module
            .symbol(name)
            .ok_or_else(|| anyhow!("no symbol @{name}"))?;
        let callee = cx
            .module
            .op(inst)
            .get::<tile::Dispatch>()
            .ok_or_else(|| anyhow!("@{name} is not a tile.dispatch"))?;
        ensure!(
            callee.args.len() == ins.args.len(),
            "@{name} takes {} arguments, got {}",
            callee.args.len(),
            ins.args.len()
        );
        let mut bound = SymBindings::default();
        for (i, (param, &arg)) in callee.args.iter().zip(&ins.args).enumerate() {
            let param = param
                .as_tensor()
                .ok_or_else(|| anyhow!("argument {i} of @{name} is not a tensor"))?;
            bound
                .bind(param, arg)
                .map_err(|err| anyhow!("argument {i}: {err}"))?;
        }
        let outs = callee
            .results
            .iter()
            .map(|ty| {
                let tensor = ty
                    .as_tensor()
                    .ok_or_else(|| anyhow!("@{name} gives {ty}, not a tensor"))?;
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
