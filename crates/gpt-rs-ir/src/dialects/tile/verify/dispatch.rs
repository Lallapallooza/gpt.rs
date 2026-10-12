//! The type rules of `tile.group` and `tile.dispatch`: the types each op gives, and what it
//! rejects.

use std::collections::BTreeSet;

use anyhow::{anyhow, ensure, Result};

use crate::dialects::tile::ir::*;
use crate::module::Region;
use crate::op::{OpDef, OwnedTypes, RegionOpDef, RegionSig, TypeCx, Types};
use crate::types::{Dim, Type};

impl RegionOpDef for Group {
    fn region_signatures(&self, _ins: GroupOperands<Types<'_>>) -> Result<GroupRegions<RegionSig>> {
        Ok(GroupRegions {
            body: RegionSig::args_only([]),
        })
    }
}

impl OpDef for Group {
    fn result_types(
        &self,
        _ins: GroupOperands<Types<'_>>,
        cx: &TypeCx<'_, GroupRegions<Region>>,
    ) -> Result<GroupResults<OwnedTypes>> {
        let terminator = cx
            .module
            .last_inst(cx.regions.body)
            .ok_or_else(|| anyhow!("the body is empty"))?;
        let outs = cx
            .module
            .operand_types(terminator)
            .into_iter()
            .map(|ty| {
                ty.as_tensor()
                    .cloned()
                    .ok_or_else(|| anyhow!("a group yields tensors, got {ty}"))
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(GroupResults { outs })
    }
}

impl RegionOpDef for Dispatch {
    fn region_signatures(
        &self,
        _ins: DispatchOperands<Types<'_>>,
    ) -> Result<DispatchRegions<RegionSig>> {
        Ok(DispatchRegions {
            body: RegionSig::new(self.args.clone(), self.results.clone()),
        })
    }
}

impl OpDef for Dispatch {
    fn result_types(
        &self,
        _ins: DispatchOperands<Types<'_>>,
        cx: &TypeCx<'_, DispatchRegions<Region>>,
    ) -> Result<DispatchResults<OwnedTypes>> {
        for types in [&self.args, &self.results] {
            for ty in types {
                ensure!(
                    ty.as_tensor().is_some(),
                    "a dispatch takes and gives tensors, got {ty}"
                );
            }
        }
        let mut bound = BTreeSet::new();
        for tensor in self.args.iter().filter_map(Type::as_tensor) {
            for dim in &tensor.dims {
                if let Dim::Dynamic(sym) = dim {
                    bound.insert(sym.as_str().to_string());
                }
            }
        }
        for (name, fact) in &self.facts {
            ensure!(
                bound.contains(name),
                "facts name {{{name}}}, which no argument binds"
            );
            ensure!(
                fact.min <= fact.max,
                "facts give {{{name}}} the empty range [{}, {}]",
                fact.min,
                fact.max
            );
            ensure!(fact.divisor > 0, "facts give {{{name}}} the divisor 0");
        }
        // A dispatch runs on its arguments alone, so they bind every symbol its blocks use.
        let module = cx.module;
        for inst in module.walk(cx.regions.body) {
            let Some(block) = module.op(inst).get::<Block>() else {
                continue;
            };
            for sym in block.syms(module, module.regions(inst)[0]) {
                ensure!(
                    bound.contains(sym.as_str()),
                    "a block uses {{{}}}, which no argument binds",
                    sym.as_str()
                );
            }
        }
        self.schedule.legal(module, cx.regions.body)?;
        Ok(DispatchResults::default())
    }
}
