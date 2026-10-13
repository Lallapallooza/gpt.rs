//! Reading and rewriting the attributes of `tile.block`.

use std::collections::BTreeSet;
use std::ops::Range;

use anyhow::{anyhow, ensure, Result};
use smallvec::SmallVec;
use tinyvec::ArrayVec;

use crate::dialects::tile::ir::*;
use crate::ir::{AccessExpr, DType, Dim, IterRef, Module, OperandRef, Region, Sym, Value};

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
                write.into = write.into.map(|operand| OperandRef(rename(operand.0)));
            }
            for operand in &mut fold.init {
                operand.0 = rename(operand.0);
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
                named.extend(write.into.map(|operand| operand.0));
                for expr in &write.coords {
                    expr.collect_operands(&mut named);
                }
            }
            named.extend(fold.init.iter().map(|operand| operand.0));
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

impl Source {
    /// The operands the read may take elements from.
    pub fn operands(&self) -> Range<usize> {
        match self {
            Source::Operand(operand) => *operand..*operand + 1,
            Source::List { first, count, .. } => *first..*first + *count,
        }
    }
}

/// The text form of a [`Source`]: one operand, `$2`, or a list of operands and the pick,
/// `$2..$4[$1[d0]]`.
#[derive(crate::Attr, Debug, Clone, PartialEq, Eq, Hash)]
#[attr(format = "$first (`..` $list^)?")]
pub(in crate::dialects::tile) struct SourceText {
    first: OperandRef,
    list: Option<OperandList>,
}

/// The last operand of a list and the pick, after `..`.
#[derive(crate::Attr, Debug, Clone, PartialEq, Eq, Hash)]
#[attr(format = "$last `[` $pick `]`")]
struct OperandList {
    last: OperandRef,
    pick: AccessExpr,
}

impl From<&Source> for SourceText {
    fn from(source: &Source) -> Self {
        match source {
            Source::Operand(operand) => SourceText {
                first: OperandRef(*operand),
                list: None,
            },
            Source::List { first, count, pick } => SourceText {
                first: OperandRef(*first),
                list: Some(OperandList {
                    last: OperandRef(first + count - 1),
                    pick: (**pick).clone(),
                }),
            },
        }
    }
}

impl TryFrom<SourceText> for Source {
    type Error = anyhow::Error;

    fn try_from(text: SourceText) -> Result<Self> {
        let first = text.first.0;
        let Some(list) = text.list else {
            return Ok(Source::Operand(first));
        };
        let last = list.last.0;
        ensure!(last >= first, "operand list ${first}..${last} is empty");
        Ok(Source::List {
            first,
            count: last - first + 1,
            pick: Box::new(list.pick),
        })
    }
}

/// The coordinates `coords`, in the form of an access.
fn coords_of(coords: impl IntoIterator<Item = impl Into<AccessExpr>>) -> Coords {
    coords.into_iter().map(Into::into).collect()
}

impl Read {
    /// A read of operand `operand` at `coords`, without padding.
    pub fn new(operand: usize, coords: impl IntoIterator<Item = impl Into<AccessExpr>>) -> Self {
        Self {
            source: Source::Operand(operand),
            coords: coords_of(coords),
            pad: None,
        }
    }

    /// The read with every expression rewritten by `rewrite`.
    pub fn map_exprs(&self, rewrite: &dyn Fn(&AccessExpr) -> AccessExpr) -> Read {
        Read {
            source: match &self.source {
                Source::Operand(operand) => Source::Operand(*operand),
                Source::List { first, count, pick } => Source::List {
                    first: *first,
                    count: *count,
                    pick: Box::new(rewrite(pick)),
                },
            },
            coords: self.coords.iter().map(rewrite).collect(),
            pad: self.pad,
        }
    }

    /// The read with operand `i` renamed to `rename(i)`, in its source and in its expressions.
    pub fn rename_operands(&self, rename: &dyn Fn(usize) -> usize) -> Read {
        let mut read = self.map_exprs(&|expr| expr.rename_operands(rename));
        read.source = match read.source {
            Source::Operand(operand) => Source::Operand(rename(operand)),
            Source::List { first, count, pick } => Source::List {
                first: rename(first),
                count,
                pick,
            },
        };
        read
    }

    /// The expressions of the read: its coordinates and its pick.
    pub fn exprs(&self) -> Vec<&AccessExpr> {
        let mut out: Vec<&AccessExpr> = self.coords.iter().collect();
        if let Source::List { pick, .. } = &self.source {
            out.push(pick);
        }
        out
    }
}

impl Write {
    /// A write of a new result at `coords`.
    pub fn new(coords: impl IntoIterator<Item = impl Into<AccessExpr>>) -> Self {
        Self {
            into: None,
            coords: coords_of(coords),
        }
    }

    /// A write of an updated copy of operand `operand`, at `coords`.
    pub fn update(operand: usize, coords: impl IntoIterator<Item = impl Into<AccessExpr>>) -> Self {
        Self {
            into: Some(OperandRef(operand)),
            coords: coords_of(coords),
        }
    }
}

impl IterList {
    /// The iterators at `positions`.
    #[track_caller]
    pub fn new(positions: impl IntoIterator<Item = usize>) -> Self {
        let mut list = ArrayVec::new();
        for position in positions {
            list.push(IterRef::new(position));
        }
        Self(list)
    }

    /// The positions, in order.
    pub fn positions(&self) -> impl Iterator<Item = usize> + '_ {
        self.0.iter().map(|iter| iter.position())
    }

    pub fn contains(&self, position: usize) -> bool {
        self.positions().any(|listed| listed == position)
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl IterKind {
    /// Whether the combine folds the iterator.
    pub fn is_reduced(self) -> bool {
        matches!(self, IterKind::Red | IterKind::RedOrdered)
    }
}

impl Iter {
    pub fn new(kind: IterKind, extent: Dim) -> Self {
        Self { kind, extent }
    }

    pub fn par(extent: Dim) -> Self {
        Self::new(IterKind::Par, extent)
    }

    pub fn red(extent: Dim) -> Self {
        Self::new(IterKind::Red, extent)
    }
}
