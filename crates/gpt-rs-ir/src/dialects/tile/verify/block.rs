//! The type rules of `tile.block`, `tile.index` and `tile.yield`: the types each op gives, and what
//! it rejects.

use std::collections::{BTreeSet, HashMap};

use anyhow::{anyhow, bail, ensure, Result};

use super::schedule::check_schedule;
use crate::dialects::tensor::utils::class::{ensure_dtype_class, ARITHMETIC};
use crate::dialects::tile::ir::*;
use crate::dialects::tile::{Iter, IterKind, Read, Write};
use crate::ir::{
    ranked, DType, Dim, Expr, IndexEnv, Module, NoRegions, OpDef, OperandRef, OwnedTypes, ReadKind,
    Region, RegionOpDef, RegionSig, Sym, TensorType, Type, TypeCx, Types, Value, MAX_ITERS,
};

impl RegionOpDef for Block {
    fn region_signatures(&self, ins: BlockOperands<Types>) -> Result<BlockRegions<RegionSig>> {
        let args = self
            .reads
            .iter()
            .enumerate()
            .map(|(index, read)| {
                let first = read.source.operands().start;
                let ty = ins.ins.get(first).ok_or_else(|| {
                    anyhow!(
                        "read {index} names operand ${first}, but the block has {} operands",
                        ins.ins.len()
                    )
                })?;
                Ok(Type::Scalar(ty.dtype))
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(BlockRegions {
            body: RegionSig::args_only(args),
        })
    }
}

impl OpDef for Block {
    fn result_types(
        &self,
        ins: BlockOperands<Types>,
        cx: &TypeCx<BlockRegions<Region>>,
    ) -> Result<BlockResults<OwnedTypes>> {
        let operands: &[&TensorType] = &ins.ins;
        let scope = Scope {
            iters: &self.iters,
            operands,
        };
        let prefixes = self
            .iters
            .iter()
            .filter(|iter| iter.kind == IterKind::Prefix)
            .count();
        let reductions = self
            .iters
            .iter()
            .filter(|iter| iter.kind.is_reduced())
            .count();
        ensure!(prefixes <= 1, "a block has at most one prefix iterator");
        ensure!(
            prefixes == 0 || reductions == 0,
            "a block with a prefix iterator has no reduction iterators"
        );

        check_reads(self, &scope)?;
        check_schedule(self)?;
        for (index, constraint) in self.constraints.iter().enumerate() {
            for expr in constraint.exprs() {
                scope.check(expr, &format!("constraint {index}"))?;
            }
        }

        let module = cx.module;
        let body = cx.regions.body;
        for inst in module.insts(body) {
            if let Some(index) = module.op(inst).get::<Index>() {
                scope.check(&index.expr, "tile.index")?;
            }
        }
        let yields = Block::yield_types(module, body)?;
        let depends = yield_iters(self, module, body);
        let ranges = self.fold_ranges();
        let taken = ranges.last().map_or(0, |(yielded, _)| yielded.end);
        ensure!(
            taken == yields.len(),
            "the folds take {taken} yielded values, but the body yields {}",
            yields.len()
        );
        let folding = self
            .folds
            .iter()
            .any(|fold| fold.combine != CombineKind::None);
        ensure!(
            reductions + prefixes == 0 || folding,
            "a block with reduction or prefix iterators needs a fold other than none"
        );
        let outs = fold_types(self, &scope, &yields, &depends)?;
        Ok(BlockResults { outs })
    }
}

impl OpDef for Index {
    fn result_types(
        &self,
        _ins: IndexOperands<Types>,
        _cx: &TypeCx<NoRegions>,
    ) -> Result<IndexResults<OwnedTypes>> {
        ensure_dtype_class("dtype", self.dtype, ARITHMETIC)?;
        Ok(IndexResults { output: self.dtype })
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

/// Checks the reads of `block`: the operands each names and their types, its coordinates and
/// its pad.
fn check_reads(block: &Block, scope: &Scope) -> Result<()> {
    let operands = scope.operands;
    for (index, read) in block.reads.iter().enumerate() {
        let range = read.source.operands();
        ensure!(
            range.end <= operands.len(),
            "read {index} names operand ${}, but the block has {} operands",
            range.end - 1,
            operands.len()
        );
        let ty = operands[range.start];
        for &other in &operands[range.clone()] {
            ensure!(
                other == ty,
                "the operands of read {index} differ in type: {ty} and {other}"
            );
        }
        ensure!(
            read.coords.len() == ty.rank(),
            "read {index} gives {} coordinates for {ty}",
            read.coords.len()
        );
        for expr in read.exprs() {
            scope.check(expr, &format!("read {index}"))?;
        }
        if read.pad.is_none() && block.constraints.is_empty() {
            check_corners(block, index, read, ty)?;
        }
        if let Some(pad) = read.pad {
            ensure!(
                ty.dtype.represents(pad.0),
                "read {index} pads with {}, which is not a {} value",
                pad.0,
                ty.dtype
            );
        }
    }
    Ok(())
}

/// The types of the results of the folds of `block`, which take the values of dtypes `yields`
/// that its body yields, each depending on the iterators `depends` holds for it.
fn fold_types(
    block: &Block,
    scope: &Scope,
    yields: &[DType],
    depends: &[BTreeSet<usize>],
) -> Result<Vec<TensorType>> {
    let mut outs = Vec::new();
    let ranges = block.fold_ranges();
    for (index, (fold, (taken, _))) in block.folds.iter().zip(ranges).enumerate() {
        let kind = &fold.combine;
        ensure!(
            fold.writes.len() == kind.rule().results(),
            "fold {index} has {} writes, but {} gives {} results",
            fold.writes.len(),
            kind.name(),
            kind.rule().results()
        );
        check_value_iters(block, kind, taken.start, depends)?;
        let dtypes = kind.rule().result_dtypes(kind.name(), &yields[taken])?;
        let first_result = outs.len();
        for (result, (&dtype, write)) in dtypes.iter().zip(&fold.writes).enumerate() {
            let what = format!("write {}", outs.len());
            ensure!(
                block.constraints.is_empty() || write.into.is_some() || *kind != CombineKind::None,
                "result {} is a plain new tensor, which a block with constraints cannot write",
                outs.len()
            );
            outs.push(write_type(block, scope, write, &what, kind, result, dtype)?);
        }
        check_init(fold, scope.operands, &outs[first_result..])?;
    }
    Ok(outs)
}

/// Checks that each of the value's iterators of `kind`, the combine kind of a fold whose yields
/// start at `first_yield`, is a parallel iterator of `block`, and that no value the fold takes and
/// keeps apart from them depends on one.
fn check_value_iters(
    block: &Block,
    kind: &CombineKind,
    first_yield: usize,
    depends: &[BTreeSet<usize>],
) -> Result<()> {
    for value_iter in kind.rule().value_iters().positions() {
        ensure!(
            block.iters.get(value_iter).map(|iter| iter.kind) == Some(IterKind::Par),
            "the value's iterator d{value_iter} is not a parallel iterator of the block"
        );
        for yielded in 0..kind.rule().yields() {
            if !kind.rule().independent_of_value_iters(yielded) {
                continue;
            }
            let position = first_yield + yielded;
            ensure!(
                !depends[position].contains(&value_iter),
                "yielded value {position}, which {} keeps apart from the value's iterators, \
                 depends on d{value_iter}",
                kind.name()
            );
        }
    }
    Ok(())
}

/// What the type rule of a block knows: its iterators and its operands.
struct Scope<'a> {
    iters: &'a [Iter],
    operands: &'a [&'a TensorType],
}

impl Scope<'_> {
    /// Checks that `expr` uses only the block's iterators and reads only integer operands, at
    /// their rank. `what` names it in the error.
    fn check<R: ReadKind>(&self, expr: &Expr<R>, what: &str) -> Result<()> {
        for position in expr.iters() {
            ensure!(
                position < self.iters.len(),
                "{what} uses d{position}, which is not an iterator of the block"
            );
        }
        for (operand, count) in expr.reads() {
            let ty = self.operands.get(operand).ok_or_else(|| {
                anyhow!(
                    "{what} reads ${operand}, but the block has {} operands",
                    self.operands.len()
                )
            })?;
            ensure!(
                ty.dtype.int_format().is_some(),
                "{what} reads ${operand} as an index, but it is {ty}"
            );
            ensure!(
                count == ty.rank(),
                "{what} reads ${operand} at {count} coordinates, but it is {ty}"
            );
        }
        ensure!(
            expr.is_affine(),
            "{what} multiplies or divides by an expression of iterators: {expr}"
        );
        Ok(())
    }
}

/// Fails when read `index` of `block`, which has no pad, leaves `ty` at a corner of the block's
/// points, where each iterator stands at its first or its last point: a point the block computes,
/// unless constraints skip it, which the caller rules out. A coordinate that reads an operand or
/// names a symbol, and a block without points, are left to the interpreter.
fn check_corners(block: &Block, index: usize, read: &Read, ty: &TensorType) -> Result<()> {
    let count = block.iters.len();
    let mut lasts = [0_i64; MAX_ITERS];
    for (position, iter) in block.iters.iter().enumerate() {
        let Some(extent) = iter.extent.as_static() else {
            return Ok(());
        };
        if extent == 0 {
            return Ok(());
        }
        lasts[position] = extent as i64 - 1;
    }
    for (axis, coord) in read.coords.iter().enumerate() {
        let Some(extent) = ty.dims[axis].as_static() else {
            continue;
        };
        let used: Vec<usize> = coord.iters().into_iter().collect();
        for corner in 0..1u32 << used.len() {
            let mut point = [0_i64; MAX_ITERS];
            for (bit, &position) in used.iter().enumerate() {
                if corner >> bit & 1 == 1 {
                    point[position] = lasts[position];
                }
            }
            let at_corner = &point[..count];
            let Ok(value) = coord.eval(&Corner(at_corner)) else {
                break;
            };
            ensure!(
                (0..extent as i64).contains(&value),
                "read {index} has no pad, but coordinate {axis} is {value} at the point \
                 {at_corner:?}, outside {ty}"
            );
        }
    }
    Ok(())
}

/// A point of a block's iterators, for an expression that reads nothing and names no symbol.
struct Corner<'p>(&'p [i64]);

impl IndexEnv for Corner<'_> {
    fn iter(&self, position: usize) -> Option<i64> {
        self.0.get(position).copied()
    }

    fn sym(&self, _sym: &Sym) -> Option<i64> {
        None
    }

    fn read(&self, _operand: usize, _coords: &[i64]) -> Result<i64> {
        bail!("a corner reads no operand")
    }
}

/// The iterators that each value the body of `block` yields depends on: through the reads whose
/// values it uses, at the iterators their coordinates and picks name, and through the
/// `tile.index` values it uses.
fn yield_iters(block: &Block, module: &Module, body: Region) -> Vec<BTreeSet<usize>> {
    let mut depends: HashMap<Value, BTreeSet<usize>> = HashMap::new();
    for (arg, read) in module.args(body).iter().zip(&block.reads) {
        let mut iters = BTreeSet::new();
        for expr in read.exprs() {
            iters.extend(expr.iters());
        }
        depends.insert(*arg, iters);
    }
    for inst in module.insts(body) {
        let mut iters = BTreeSet::new();
        for operand in module.operands(inst) {
            if let Some(known) = depends.get(operand) {
                iters.extend(known.iter().copied());
            }
        }
        if let Some(index) = module.op(inst).get::<Index>() {
            iters.extend(index.expr.iters());
        }
        if module.op(inst).get::<Yield>().is_some() {
            let mut yielded = Vec::with_capacity(module.operands(inst).len());
            for operand in module.operands(inst) {
                yielded.push(depends.get(operand).cloned().unwrap_or_default());
            }
            return yielded;
        }
        for &result in module.results(inst) {
            depends.insert(result, iters.clone());
        }
    }
    Vec::new()
}

/// The type of the result that `write` places, result `result` of `kind`, of dtype `dtype`.
/// `what` names the write in errors.
fn write_type(
    block: &Block,
    scope: &Scope,
    write: &Write,
    what: &str,
    kind: &CombineKind,
    result: usize,
    dtype: DType,
) -> Result<TensorType> {
    if let Some(OperandRef(into)) = write.into {
        let target = scope.operands.get(into).ok_or_else(|| {
            anyhow!(
                "{what} updates ${into}, but the block has {} operands",
                scope.operands.len()
            )
        })?;
        ensure!(
            write.coords.len() == target.rank(),
            "{what} gives {} coordinates for {target}",
            write.coords.len()
        );
        for expr in &write.coords {
            scope.check(expr, what)?;
        }
        ensure!(
            kind.rule().folds_into_operand(),
            "{what} folds into an operand with {}, but only none, sum, max, min, and and or \
             fold into one",
            kind.name()
        );
        ensure!(
            dtype == target.dtype,
            "{what} writes {dtype} values into {target}"
        );
        return Ok((*target).clone());
    }

    let mut named = BTreeSet::new();
    let mut dims = Vec::new();
    for coord in &write.coords {
        if coord.as_const() == Some(0) {
            dims.push(Dim::Static(1));
            continue;
        }
        let Some(position) = coord.as_iter() else {
            bail!("{what} writes a new tensor at {coord}; its coordinates are iterators or 0")
        };
        let iter = scope.iters.get(position).ok_or_else(|| {
            anyhow!("{what} uses d{position}, which is not an iterator of the block")
        })?;
        ensure!(named.insert(position), "{what} names d{position} twice");
        dims.push(iter.extent.clone());
    }
    let value_iters = kind.rule().value_iters();
    let mut expected = BTreeSet::new();
    for (position, iter) in block.iters.iter().enumerate() {
        let folded = *kind != CombineKind::None && iter.kind.is_reduced();
        let dropped = kind.rule().drops_value_iters(result) && value_iters.contains(position);
        if !folded && !dropped {
            expected.insert(position);
        }
    }
    let list = |positions: &BTreeSet<usize>| {
        let mut names = Vec::new();
        for position in positions {
            names.push(format!("d{position}"));
        }
        format!("[{}]", names.join(", "))
    };
    ensure!(
        named == expected,
        "{what} must name each of the iterators {} once, got {}",
        list(&expected),
        list(&named)
    );
    dims.extend(kind.rule().extra_dims(result));
    Ok(TensorType {
        dtype,
        dims: ranked(dims)?,
    })
}

/// Checks that each init operand of `fold` has the type of the state component it starts. That
/// type is the fold's result of the same position, because the results of a fold that starts
/// from a state are its state.
fn check_init(fold: &Fold, operands: &[&TensorType], outs: &[TensorType]) -> Result<()> {
    if fold.init.is_empty() {
        return Ok(());
    }
    let kind = &fold.combine;
    ensure!(
        fold.init.len() == kind.rule().init_count(),
        "init lists {} operands, but {} starts from {} states",
        fold.init.len(),
        kind.name(),
        kind.rule().init_count()
    );
    for (component, &OperandRef(operand)) in fold.init.iter().enumerate() {
        let ty = operands.get(operand).ok_or_else(|| {
            anyhow!(
                "init names ${operand}, but the block has {} operands",
                operands.len()
            )
        })?;
        let target = &outs[component];
        ensure!(
            *ty == target,
            "init ${operand} is {ty}, but the state it starts is {target}"
        );
    }
    Ok(())
}
