//! The type rules of `tile.group` and `tile.dispatch`: the types each op gives, and what it
//! rejects.

use std::collections::BTreeSet;

use anyhow::{anyhow, bail, ensure, Result};

use crate::dialects::r#loop::Level;
use crate::dialects::tile::ir::*;
use crate::ir::{
    Dim, Module, OpDef, OwnedTypes, Region, RegionOpDef, RegionSig, Type, TypeCx, Types,
};

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

impl Schedule {
    /// Fails with the rule the schedule breaks for the dispatch whose body is `body`.
    pub fn legal(&self, module: &Module, body: Region) -> Result<()> {
        if self.nest.is_empty() {
            return Ok(());
        }
        let blocks: Vec<&Block> = module
            .insts(body)
            .filter_map(|inst| module.op(inst).get::<Block>())
            .collect();
        let &[block] = blocks.as_slice() else {
            bail!(
                "a nest schedules the one block of a dispatch, and this one holds {}",
                blocks.len()
            );
        };
        let iters = &block.iters;
        let mut tiles: Vec<Option<usize>> = vec![None; iters.len()];
        let mut innermost = None;
        for (index, entry) in self.nest.iter().enumerate() {
            let position = entry.iter.position();
            let Some(iter) = iters.get(position) else {
                bail!(
                    "loop {index} walks {}, which the block does not have",
                    entry.iter
                );
            };
            ensure!(entry.extent > 0, "loop {index} walks empty tiles");
            if let Some(outer) = tiles[position] {
                ensure!(
                    entry.extent < outer,
                    "loop {index} walks tiles of {} points of {}, inside tiles of {outer}",
                    entry.extent,
                    entry.iter
                );
            }
            tiles[position] = Some(entry.extent);
            ensure!(
                iter.kind != IterKind::Prefix,
                "loop {index} walks {}, a prefix iterator, which a nest leaves whole",
                entry.iter
            );
            let Parallelism::Par(par) = &entry.parallelism else {
                continue;
            };
            ensure!(
                !iter.kind.is_reduced(),
                "loop {index} runs tiles of {}, a reduction iterator, in parallel: that splits \
                 the reduction",
                entry.iter
            );
            ensure!(par.units != Some(0), "loop {index} runs on no units");
            if let Some(outer) = innermost {
                ensure!(
                    par.level >= outer,
                    "loop {index} runs at level {}, inside a loop at level {}",
                    level_name(par.level),
                    level_name(outer)
                );
            }
            innermost = Some(par.level);
        }
        for fold in &block.folds {
            ensure!(
                fold.writes.iter().all(|write| write.into.is_none()),
                "a nest of a block that updates an operand would reorder its writes"
            );
            ensure!(
                !matches!(
                    fold.combine,
                    CombineKind::OnlineSoftmax(_) | CombineKind::Logsumexp
                ),
                "a nest of `online_softmax` or `logsumexp` needs their update of a tile"
            );
            if fold.combine == CombineKind::None {
                continue;
            }
            let steps = self.steps(block, true);
            for (position, extent) in self.state_tiles(block, &steps).1 {
                ensure!(
                    extent.as_static().is_some(),
                    "the state of a fold keeps the tile of d{position}, whose extent {extent} is \
                     not static"
                );
            }
        }
        Ok(())
    }
}

fn level_name(level: Level) -> &'static str {
    crate::ir::Mnemonic::mnemonic(level)
}
