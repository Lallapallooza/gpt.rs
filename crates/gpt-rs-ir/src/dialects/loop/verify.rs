//! The type rules of the `loop` ops: the types each op gives, and what it rejects.

use std::collections::BTreeSet;

use anyhow::{anyhow, ensure, Result};

use crate::dialects::r#loop::ir::*;
use crate::index::{Expr, ReadKind};
use crate::module::{Region, ValueDef};
use crate::op::{NoRegions, OpDef, OwnedTypes, RegionOpDef, RegionSig, TypeCx, Types};
use crate::types::{BufferType, DType, Dim, Space, Type};

impl RegionOpDef for Func {
    fn region_signatures(&self, _ins: FuncOperands<Types<'_>>) -> Result<FuncRegions<RegionSig>> {
        Ok(FuncRegions {
            body: RegionSig::new(self.args.clone(), []),
        })
    }
}

impl OpDef for Func {
    fn result_types(
        &self,
        _ins: FuncOperands<Types<'_>>,
        cx: &TypeCx<'_, FuncRegions<Region>>,
    ) -> Result<FuncResults<OwnedTypes>> {
        ensure!(
            self.access.len() == self.args.len(),
            "the access list has length {}, but the parameter list has length {}",
            self.access.len(),
            self.args.len()
        );
        let mut bound = BTreeSet::new();
        for (index, ty) in self.args.iter().enumerate() {
            let buffer = ty
                .as_buffer()
                .filter(|buffer| buffer.space == Space::Global)
                .ok_or_else(|| anyhow!("parameter {index} is {ty}, not a global buffer"))?;
            if self.access[index] == Access::In {
                bound.extend(syms(buffer));
            }
        }
        for (index, ty) in self.args.iter().enumerate() {
            let Some(buffer) = ty.as_buffer() else {
                continue;
            };
            if let Some(sym) = syms(buffer).find(|sym| !bound.contains(sym)) {
                return Err(anyhow!(
                    "parameter {index} has the extent {{{sym}}}, which no `in` parameter binds"
                ));
            }
        }
        let module = cx.module;
        for inst in module.walk(cx.regions.body) {
            if module.op(inst).get::<Store>().is_none() {
                continue;
            }
            let buffer = module.operands(inst)[1];
            let ValueDef::Arg(region, index) = module.def(buffer) else {
                continue;
            };
            ensure!(
                region != cx.regions.body || self.access[index] != Access::In,
                "it stores into parameter {index}, which is `in`"
            );
        }
        Ok(FuncResults::default())
    }
}

/// The names of the dimension symbols of `buffer`'s extents.
fn syms(buffer: &BufferType) -> impl Iterator<Item = String> + '_ {
    buffer.dims.iter().filter_map(|dim| match dim {
        Dim::Dynamic(sym) => Some(sym.as_str().to_string()),
        Dim::Static(_) => None,
    })
}

impl OpDef for Return {
    fn result_types(
        &self,
        _ins: ReturnOperands<Types<'_>>,
        _cx: &TypeCx<'_, NoRegions>,
    ) -> Result<ReturnResults<OwnedTypes>> {
        Ok(ReturnResults::default())
    }
}

impl RegionOpDef for For {
    fn region_signatures(&self, _ins: ForOperands<Types<'_>>) -> Result<ForRegions<RegionSig>> {
        Ok(ForRegions {
            body: RegionSig::new([Type::Index], []),
        })
    }
}

impl OpDef for For {
    fn result_types(
        &self,
        _ins: ForOperands<Types<'_>>,
        _cx: &TypeCx<'_, ForRegions<Region>>,
    ) -> Result<ForResults<OwnedTypes>> {
        Ok(ForResults::default())
    }
}

impl RegionOpDef for Forall {
    fn region_signatures(
        &self,
        _ins: ForallOperands<Types<'_>>,
    ) -> Result<ForallRegions<RegionSig>> {
        Ok(ForallRegions {
            body: RegionSig::new([Type::Index], []),
        })
    }
}

impl OpDef for Forall {
    fn result_types(
        &self,
        _ins: ForallOperands<Types<'_>>,
        _cx: &TypeCx<'_, ForallRegions<Region>>,
    ) -> Result<ForallResults<OwnedTypes>> {
        Ok(ForallResults::default())
    }
}

impl RegionOpDef for If {
    fn region_signatures(&self, _ins: IfOperands<Types<'_>>) -> Result<IfRegions<RegionSig>> {
        Ok(IfRegions {
            then: RegionSig::new([], []),
        })
    }
}

impl OpDef for If {
    fn result_types(
        &self,
        ins: IfOperands<Types<'_>>,
        _cx: &TypeCx<'_, IfRegions<Region>>,
    ) -> Result<IfResults<OwnedTypes>> {
        ensure!(
            ins.condition == DType::I1,
            "the condition is {}, not i1",
            ins.condition
        );
        Ok(IfResults::default())
    }
}

impl OpDef for Yield {
    fn result_types(
        &self,
        _ins: YieldOperands<Types<'_>>,
        _cx: &TypeCx<'_, NoRegions>,
    ) -> Result<YieldResults<OwnedTypes>> {
        Ok(YieldResults::default())
    }
}

/// Checks that `expr` names operands below `count` only.
fn ensure_operands<R: ReadKind>(expr: &Expr<R>, count: usize) -> Result<()> {
    if let Some(&position) = expr.iters().iter().find(|&&position| position >= count) {
        return Err(anyhow!(
            "the expression names d{position}, but the op's operand count is {count}"
        ));
    }
    Ok(())
}

impl OpDef for Index {
    fn result_types(
        &self,
        ins: IndexOperands<Types<'_>>,
        _cx: &TypeCx<'_, NoRegions>,
    ) -> Result<IndexResults<OwnedTypes>> {
        ensure_operands(&self.expr, ins.ins.len())?;
        Ok(IndexResults { output: () })
    }
}

impl OpDef for Predicate {
    fn result_types(
        &self,
        ins: PredicateOperands<Types<'_>>,
        _cx: &TypeCx<'_, NoRegions>,
    ) -> Result<PredicateResults<OwnedTypes>> {
        for constraint in &self.constraints {
            for expr in constraint.exprs() {
                ensure!(!expr.has_reads(), "a constraint reads an operand element");
                ensure_operands(expr, ins.ins.len())?;
            }
        }
        Ok(PredicateResults { output: DType::I1 })
    }
}

impl OpDef for Alloc {
    fn result_types(
        &self,
        _ins: AllocOperands<Types<'_>>,
        _cx: &TypeCx<'_, NoRegions>,
    ) -> Result<AllocResults<OwnedTypes>> {
        Ok(AllocResults {
            output: BufferType {
                dtype: self.dtype,
                dims: self
                    .shape
                    .iter()
                    .map(|&extent| Dim::Static(extent))
                    .collect(),
                space: Space::Private,
            },
        })
    }
}

/// Checks that `indices` gives one index per axis of `buffer`.
fn ensure_rank(buffer: &BufferType, indices: usize) -> Result<()> {
    ensure!(
        indices == buffer.rank(),
        "the index count is {indices}, but {buffer} has rank {}",
        buffer.rank()
    );
    Ok(())
}

impl OpDef for Load {
    fn result_types(
        &self,
        ins: LoadOperands<Types<'_>>,
        _cx: &TypeCx<'_, NoRegions>,
    ) -> Result<LoadResults<OwnedTypes>> {
        ensure_rank(ins.buffer, ins.indices.len())?;
        if let Some(pad) = self.pad {
            ensure!(
                ins.buffer.dtype.represents(pad.0),
                "it pads with {}, which {} cannot hold",
                pad.0,
                ins.buffer.dtype
            );
        }
        Ok(LoadResults {
            output: ins.buffer.dtype,
        })
    }
}

impl OpDef for Store {
    fn result_types(
        &self,
        ins: StoreOperands<Types<'_>>,
        _cx: &TypeCx<'_, NoRegions>,
    ) -> Result<StoreResults<OwnedTypes>> {
        ensure_rank(ins.buffer, ins.indices.len())?;
        ensure!(
            ins.value == ins.buffer.dtype,
            "the value is {}, but {} holds {}",
            ins.value,
            ins.buffer,
            ins.buffer.dtype
        );
        Ok(StoreResults::default())
    }
}
