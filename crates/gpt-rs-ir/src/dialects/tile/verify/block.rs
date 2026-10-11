//! The type rules of `tile.block`, `tile.index` and `tile.yield`: the types each op gives, and what
//! it rejects.

use std::collections::BTreeSet;

use anyhow::{anyhow, bail, ensure, Result};

use crate::dialects::tensor::utils::class::{ensure_dtype_class, ARITHMETIC};
use crate::dialects::tile::ir::*;
use crate::index::{Expr, Iter, IterKind, ReadKind, Write};
use crate::module::Region;
use crate::op::{NoRegions, OpDef, OwnedTypes, RegionOpDef, RegionSig, TypeCx, Types};
use crate::types::{ranked, DType, Dim, TensorType, Type};

impl RegionOpDef for Block {
    fn region_signatures(&self, ins: BlockOperands<Types<'_>>) -> Result<BlockRegions<RegionSig>> {
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
        ins: BlockOperands<Types<'_>>,
        cx: &TypeCx<'_, BlockRegions<Region>>,
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

        for (index, read) in self.reads.iter().enumerate() {
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
            if let Some(pad) = read.pad {
                ensure!(
                    ty.dtype.represents(pad.0),
                    "read {index} pads with {}, which is not a {} value",
                    pad.0,
                    ty.dtype
                );
            }
        }
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
        let taken: usize = self
            .folds
            .iter()
            .map(|fold| fold.combine.rule().yields())
            .sum();
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
        let mut outs = Vec::new();
        let mut first_yield = 0;
        for (index, fold) in self.folds.iter().enumerate() {
            let kind = &fold.combine;
            ensure!(
                fold.writes.len() == kind.rule().results(),
                "fold {index} has {} writes, but {} gives {} results",
                fold.writes.len(),
                kind.name(),
                kind.rule().results()
            );
            for lane in kind.rule().lanes().positions() {
                ensure!(
                    self.iters.get(lane).map(|iter| iter.kind) == Some(IterKind::Par),
                    "lane d{lane} is not a parallel iterator of the block"
                );
            }
            let dtypes = kind
                .rule()
                .result_dtypes(&yields[first_yield..first_yield + kind.rule().yields()])?;
            let first_result = outs.len();
            for (result, (&dtype, write)) in dtypes.iter().zip(&fold.writes).enumerate() {
                let what = format!("write {}", outs.len());
                ensure!(
                    self.constraints.is_empty()
                        || write.into.is_some()
                        || *kind != CombineKind::None,
                    "result {} is a plain new tensor, which a block with constraints cannot write",
                    outs.len()
                );
                outs.push(write_type(self, &scope, write, &what, kind, result, dtype)?);
            }
            check_init(fold, operands, &outs[first_result..])?;
            first_yield += kind.rule().yields();
        }
        Ok(BlockResults { outs })
    }
}

impl OpDef for Index {
    fn result_types(
        &self,
        _ins: IndexOperands<Types<'_>>,
        _cx: &TypeCx<'_, NoRegions>,
    ) -> Result<IndexResults<OwnedTypes>> {
        ensure_dtype_class("dtype", self.dtype, ARITHMETIC)?;
        Ok(IndexResults { output: self.dtype })
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

/// The type of the result that `write` places, result `result` of `kind`, of dtype `dtype`.
/// `what` names the write in errors.
fn write_type(
    block: &Block,
    scope: &Scope<'_>,
    write: &Write,
    what: &str,
    kind: &CombineKind,
    result: usize,
    dtype: DType,
) -> Result<TensorType> {
    if let Some(into) = write.into {
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
    let lanes = kind.rule().lanes();
    let mut expected = BTreeSet::new();
    for (position, iter) in block.iters.iter().enumerate() {
        let folded = *kind != CombineKind::None && iter.kind.is_reduced();
        let dropped = kind.rule().drops_lanes(result) && lanes.contains(position);
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
    for (component, &operand) in fold.init.iter().enumerate() {
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
