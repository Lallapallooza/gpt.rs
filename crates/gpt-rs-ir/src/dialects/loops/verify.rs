//! The type rules of the `loop` ops: the types each op gives, and what it rejects.

use std::collections::BTreeSet;

use anyhow::{anyhow, bail, ensure, Result};

use crate::dialects::loops::ir::*;
use crate::ir::{
    Attr, BufferType, DType, Dim, Expr, Module, NoRegions, OpDef, OwnedTypes, ReadKind, Region,
    RegionOpDef, RegionSig, Space, Type, TypeCx, Types, ValueDef, MAX_GROUP_RANK,
};
use crate::targets::Target;

impl RegionOpDef for Func {
    fn region_signatures(&self, _ins: FuncOperands<Types>) -> Result<FuncRegions<RegionSig>> {
        Ok(FuncRegions {
            body: RegionSig::new(self.args.clone(), []),
        })
    }
}

impl OpDef for Func {
    fn result_types(
        &self,
        _ins: FuncOperands<Types>,
        cx: &TypeCx<FuncRegions<Region>>,
    ) -> Result<FuncResults<OwnedTypes>> {
        ensure!(
            self.access.len() == self.args.len(),
            "the access list has length {}, but the parameter list has length {}",
            self.access.len(),
            self.args.len()
        );
        let mut bound = BTreeSet::new();
        let mut buffers = Vec::with_capacity(self.args.len());
        let mut sharer = vec![None; self.args.len()];
        for (index, ty) in self.args.iter().enumerate() {
            let buffer = ty
                .as_buffer()
                .filter(|buffer| buffer.space == Space::Global)
                .ok_or_else(|| anyhow!("parameter {index} is {ty}, not a global buffer"))?;
            if matches!(self.access[index], Access::In | Access::Inout) {
                bound.extend(syms(buffer));
            }
            buffers.push(buffer);
        }
        for (index, buffer) in buffers.iter().enumerate() {
            if let Some(sym) = syms(buffer).find(|sym| !bound.contains(sym)) {
                return Err(anyhow!(
                    "parameter {index} has the extent {{{sym}}}, which no `in` or `inout` \
                     parameter binds"
                ));
            }
            let Access::Out { over: Some(over) } = self.access[index] else {
                continue;
            };
            ensure!(
                self.access.get(over) == Some(&Access::In),
                "parameter {index} may share the buffer of parameter {over}, which is not `in`"
            );
            if let Some(first) = sharer[over].replace(index) {
                bail!(
                    "parameters {first} and {index} may both share the buffer of parameter {over}"
                );
            }
            let shared = buffers[over];
            ensure!(
                shared.dims == buffer.dims
                    && shared.dtype.size_in_bytes().is_some()
                    && shared.dtype.size_in_bytes() == buffer.dtype.size_in_bytes(),
                "parameter {index}, {buffer}, may share the buffer of parameter {over}, \
                 {shared}, whose elements differ in size or position"
            );
        }
        let module = cx.module;
        for inst in module.walk(cx.regions.body) {
            let Some(store) = module.op_as::<Store>(inst) else {
                continue;
            };
            // A store into a pick writes one of the buffers it picks from.
            let mut written = vec![store.buffer()];
            while let Some(buffer) = written.pop() {
                if let Some(pick) = buffer.defining_op::<Pick>() {
                    written.extend_from_slice(pick.buffers());
                    continue;
                }
                let ValueDef::Arg(region, index) = module.def(buffer.id()) else {
                    continue;
                };
                ensure!(
                    region != cx.regions.body || self.access[index] != Access::In,
                    "it stores into parameter {index}, which is `in`"
                );
            }
        }
        if let Some(reference) = &self.target {
            check_levels(module, cx.regions.body, &reference.target()?)?;
        }
        Ok(FuncResults::default())
    }
}

/// Fails unless `target`, the target the function names, has each level that `body` names: the
/// execution level of each `loop.forall`, and for each `loop.prefetch` a memory that the target
/// prefetches into.
fn check_levels(module: &Module, body: Region, target: &Target) -> Result<()> {
    for inst in module.walk(body) {
        let op = module.op(inst);
        if let Some(forall) = op.get::<Forall>() {
            ensure!(
                target.units(&forall.level).is_some(),
                "a loop.forall runs on the units of {}, which {} does not have",
                forall.level,
                target.name
            );
        }
        if let Some(prefetch) = op.get::<Prefetch>() {
            let prefetched = target
                .holding_memory(&prefetch.level)
                .is_some_and(|memory| memory.prefetch.is_some());
            ensure!(
                prefetched,
                "a loop.prefetch asks {} for lines, which is no memory that {} prefetches into",
                prefetch.level,
                target.name
            );
        }
    }
    Ok(())
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
        _ins: ReturnOperands<Types>,
        _cx: &TypeCx<NoRegions>,
    ) -> Result<ReturnResults<OwnedTypes>> {
        Ok(ReturnResults::default())
    }
}

impl RegionOpDef for For {
    fn region_signatures(&self, _ins: ForOperands<Types>) -> Result<ForRegions<RegionSig>> {
        Ok(ForRegions {
            body: RegionSig::new([Type::Index], []),
        })
    }
}

impl OpDef for For {
    fn result_types(
        &self,
        _ins: ForOperands<Types>,
        _cx: &TypeCx<ForRegions<Region>>,
    ) -> Result<ForResults<OwnedTypes>> {
        Ok(ForResults::default())
    }
}

impl RegionOpDef for Atom {
    fn region_signatures(&self, _ins: AtomOperands<Types>) -> Result<AtomRegions<RegionSig>> {
        Ok(AtomRegions {
            body: RegionSig::new(vec![Type::Index; self.tile.len()], []),
        })
    }
}

impl OpDef for Atom {
    fn result_types(
        &self,
        ins: AtomOperands<Types>,
        _cx: &TypeCx<AtomRegions<Region>>,
    ) -> Result<AtomResults<OwnedTypes>> {
        let rank = self.tile.len();
        ensure!(
            (1..=MAX_GROUP_RANK).contains(&rank),
            "a register tile has rank 1 to {MAX_GROUP_RANK}, not {rank}"
        );
        ensure!(
            ins.extents.len() == rank,
            "the extent count is {}, but the tile has rank {rank}",
            ins.extents.len()
        );
        let lanes_extent = self.tile[rank - 1];
        ensure!(
            self.lanes > 0 && lanes_extent.is_multiple_of(self.lanes),
            "vectors of {} lanes do not fill the {lanes_extent} points of the tile's last axis",
            self.lanes
        );
        Ok(AtomResults::default())
    }
}

impl RegionOpDef for Forall {
    fn region_signatures(&self, _ins: ForallOperands<Types>) -> Result<ForallRegions<RegionSig>> {
        Ok(ForallRegions {
            body: RegionSig::new([Type::Index], []),
        })
    }
}

impl OpDef for Forall {
    fn result_types(
        &self,
        _ins: ForallOperands<Types>,
        _cx: &TypeCx<ForallRegions<Region>>,
    ) -> Result<ForallResults<OwnedTypes>> {
        Ok(ForallResults::default())
    }
}

impl RegionOpDef for If {
    fn region_signatures(&self, _ins: IfOperands<Types>) -> Result<IfRegions<RegionSig>> {
        Ok(IfRegions {
            then: RegionSig::new([], []),
        })
    }
}

impl OpDef for If {
    fn result_types(
        &self,
        ins: IfOperands<Types>,
        _cx: &TypeCx<IfRegions<Region>>,
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
        _ins: YieldOperands<Types>,
        _cx: &TypeCx<NoRegions>,
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
        ins: IndexOperands<Types>,
        _cx: &TypeCx<NoRegions>,
    ) -> Result<IndexResults<OwnedTypes>> {
        ensure_operands(&self.expr, ins.ins.len())?;
        Ok(IndexResults { output: () })
    }
}

impl OpDef for Predicate {
    fn result_types(
        &self,
        ins: PredicateOperands<Types>,
        _cx: &TypeCx<NoRegions>,
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
        _ins: AllocOperands<Types>,
        _cx: &TypeCx<NoRegions>,
    ) -> Result<AllocResults<OwnedTypes>> {
        let mut space = String::new();
        self.space.print(&mut space);
        ensure!(
            matches!(self.space, Space::Private | Space::Reg),
            "it allocates {space} memory, which only the caller of a launch owns"
        );
        Ok(AllocResults {
            output: BufferType {
                dtype: self.dtype,
                dims: self
                    .shape
                    .iter()
                    .map(|&extent| Dim::Static(extent))
                    .collect(),
                space: self.space,
            },
        })
    }
}

impl OpDef for Pick {
    fn result_types(
        &self,
        ins: PickOperands<Types>,
        _cx: &TypeCx<NoRegions>,
    ) -> Result<PickResults<OwnedTypes>> {
        let Some((first, rest)) = ins.buffers.split_first() else {
            return Err(anyhow!("it picks from no buffer"));
        };
        for (position, buffer) in rest.iter().enumerate() {
            ensure!(
                buffer == first,
                "it picks from buffers of two types, {first} and {buffer} at position {}",
                position + 1
            );
        }
        Ok(PickResults {
            output: (*first).clone(),
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
        ins: LoadOperands<Types>,
        _cx: &TypeCx<NoRegions>,
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
        ins: StoreOperands<Types>,
        _cx: &TypeCx<NoRegions>,
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

impl OpDef for Prefetch {
    fn result_types(
        &self,
        ins: PrefetchOperands<Types>,
        _cx: &TypeCx<NoRegions>,
    ) -> Result<PrefetchResults<OwnedTypes>> {
        ensure_rank(ins.buffer, ins.indices.len())?;
        Ok(PrefetchResults::default())
    }
}
