//! Reading and rewriting the attributes of `tile.block`, and the text form of its folds.

use std::collections::BTreeSet;

use anyhow::{anyhow, Result};
use smallvec::SmallVec;

use crate::attr::Attr;
use crate::dialects::tile::ir::*;
use crate::index::{AccessExpr, Source, Write};
use crate::module::{Module, Region, Value};
use crate::text::Cursor;
use crate::types::{DType, Dim, Sym};

impl Block {
    /// The writes of the results, in result order.
    pub fn writes(&self) -> impl Iterator<Item = &Write> {
        self.folds.iter().flat_map(|fold| &fold.writes)
    }

    /// The number of results.
    pub fn result_count(&self) -> usize {
        self.writes().count()
    }

    /// The scalar types the body yields, from its terminator.
    pub fn yield_types(module: &Module, body: Region) -> Result<Vec<DType>> {
        let terminator = module
            .last_inst(body)
            .ok_or_else(|| anyhow!("the body is empty"))?;
        module
            .operand_types(terminator)
            .into_iter()
            .map(|ty| {
                ty.as_scalar()
                    .ok_or_else(|| anyhow!("the body yields scalars, got {ty}"))
            })
            .collect()
    }

    /// Every dimension symbol the block uses: in its extents, its expressions and its body's
    /// `tile.index` values.
    pub fn syms(&self, module: &Module, body: Region) -> BTreeSet<Sym> {
        let mut syms = BTreeSet::new();
        for iter in &self.iters {
            if let Dim::Dynamic(sym) = &iter.extent {
                syms.insert(sym.clone());
            }
        }
        let mut exprs: Vec<&AccessExpr> = Vec::new();
        for read in &self.reads {
            exprs.extend(read.exprs());
        }
        for write in self.writes() {
            exprs.extend(&write.coords);
        }
        for constraint in &self.constraints {
            exprs.extend(constraint.exprs());
        }
        for expr in exprs {
            expr.collect_syms(&mut syms);
        }
        for inst in module.insts(body) {
            if let Some(index) = module.op(inst).get::<Index>() {
                index.expr.collect_syms(&mut syms);
            }
        }
        syms
    }

    /// The extent of the iterator at `position`.
    pub fn extent(&self, position: usize) -> Option<Dim> {
        self.iters.get(position).map(|iter| iter.extent.clone())
    }

    /// The block with every index expression rewritten by `rewrite`: reads, writes and
    /// constraints.
    pub fn map_exprs(&self, rewrite: &dyn Fn(&AccessExpr) -> AccessExpr) -> Block {
        let mut block = self.clone();
        for (read, original) in block.reads.iter_mut().zip(&self.reads) {
            *read = original.map_exprs(rewrite);
        }
        for fold in &mut block.folds {
            for write in &mut fold.writes {
                for coord in &mut write.coords {
                    *coord = rewrite(coord);
                }
            }
        }
        for (constraint, original) in block.constraints.iter_mut().zip(&self.constraints) {
            *constraint = original.map_exprs(rewrite);
        }
        block
    }

    /// The block with operand `i` renamed to `rename(i)` everywhere it is named.
    pub fn rename_operands(&self, rename: &dyn Fn(usize) -> usize) -> Block {
        let mut block = self.map_exprs(&|expr| expr.rename_operands(rename));
        for read in &mut block.reads {
            *read = read.rename_operands(rename);
        }
        for fold in &mut block.folds {
            for write in &mut fold.writes {
                write.into = write.into.map(rename);
            }
            for operand in &mut fold.init {
                *operand = rename(*operand);
            }
        }
        block
    }

    /// The operands the block names: in its reads, index expressions, writes and init.
    pub fn named_operands(&self) -> BTreeSet<usize> {
        let mut named = BTreeSet::new();
        for read in &self.reads {
            named.extend(read.source.operands());
            for expr in read.exprs() {
                expr.collect_operands(&mut named);
            }
        }
        for fold in &self.folds {
            for write in &fold.writes {
                named.extend(write.into);
                for expr in &write.coords {
                    expr.collect_operands(&mut named);
                }
            }
            named.extend(fold.init.iter().copied());
        }
        for constraint in &self.constraints {
            for expr in constraint.exprs() {
                expr.collect_operands(&mut named);
            }
        }
        named
    }

    /// The operands that a read takes from a run of operands: these keep their positions.
    pub fn listed_operands(&self) -> BTreeSet<usize> {
        let mut listed = BTreeSet::new();
        for read in &self.reads {
            if matches!(read.source, Source::List { .. }) {
                listed.extend(read.source.operands());
            }
        }
        listed
    }

    /// The block with operands `ins` compacted: operands that hold the same value merged into
    /// the first, and operands it does not name dropped. Operands of a run that a read picks from
    /// keep their places.
    pub fn compact(&self, ins: Vec<Value>) -> (Block, Vec<Value>) {
        let listed = self.listed_operands();
        let mut target = Vec::with_capacity(ins.len());
        for (index, value) in ins.iter().enumerate() {
            let first =
                (0..index).find(|&earlier| ins[earlier] == *value && !listed.contains(&earlier));
            target.push(match first {
                Some(earlier) if !listed.contains(&index) => earlier,
                _ => index,
            });
        }
        let merged = self.rename_operands(&|operand| target[operand]);
        let named = merged.named_operands();
        let kept: Vec<usize> = (0..ins.len())
            .filter(|operand| named.contains(operand) || listed.contains(operand))
            .collect();
        let position = |operand: usize| {
            kept.iter()
                .position(|&kept_operand| kept_operand == operand)
                .expect("a named operand stays")
        };
        let mut values = Vec::with_capacity(kept.len());
        for &operand in &kept {
            values.push(ins[operand]);
        }
        (merged.rename_operands(&position), values)
    }

    /// Whether the block writes one value per point of its domain: one plain fold of one new
    /// result, no constraints, and no read from a run of operands. A reader can compute such a
    /// block's elements itself.
    pub fn is_map(&self) -> bool {
        let [fold] = &self.folds[..] else {
            return false;
        };
        fold.combine == CombineKind::None
            && fold.writes.len() == 1
            && fold.writes[0].into.is_none()
            && self.constraints.is_empty()
            && self
                .reads
                .iter()
                .all(|read| matches!(read.source, Source::Operand(_)))
    }
}

impl Fold {
    /// The fold of `combine` with the writes of its results and no start state.
    pub fn new(combine: CombineKind, writes: Vec<Write>) -> Self {
        Self {
            combine,
            writes: writes.into(),
            init: SmallVec::new(),
        }
    }

    /// One yielded value, written at each point.
    pub fn plain(write: Write) -> Self {
        Self::new(CombineKind::None, vec![write])
    }
}

impl Attr for Fold {
    fn print(&self, out: &mut String) {
        self.combine.print(out);
        out.push_str(" -> ");
        for (position, write) in self.writes.iter().enumerate() {
            if position > 0 {
                out.push_str(", ");
            }
            write.print(out);
        }
        if !self.init.is_empty() {
            out.push_str(" init ");
            for (position, operand) in self.init.iter().enumerate() {
                if position > 0 {
                    out.push_str(", ");
                }
                out.push_str(&format!("${operand}"));
            }
        }
    }

    /// The writes and the init operands continue past a comma that a write or an operand
    /// follows; any other comma separates folds.
    fn parse(c: &mut Cursor<'_>) -> Result<Self> {
        let combine = CombineKind::parse(c)?;
        c.expect("->")?;
        let mut writes = SmallVec::new();
        writes.push(Write::parse(c)?);
        while c.peek_pair(',', '[') || c.peek_pair(',', '$') {
            c.expect(",")?;
            writes.push(Write::parse(c)?);
        }
        let mut init = SmallVec::new();
        if c.eat("init") {
            loop {
                c.expect("$")?;
                let operand = c.int()?;
                init.push(usize::try_from(operand).map_err(|_| c.error("bad operand number"))?);
                if !c.peek_pair(',', '$') {
                    break;
                }
                c.expect(",")?;
            }
        }
        Ok(Self {
            combine,
            writes,
            init,
        })
    }
}
