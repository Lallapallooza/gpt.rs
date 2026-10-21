//! The canonical patterns of the `loop` dialect: index math composed into one expression over the
//! values that no `loop.index` computes, and reduced where the loops around bound those values,
//! and loops of one trip written as their bodies.

use std::collections::HashMap;

use tinyvec::ArrayVec;

use crate::dialects::loops::utils::loop_bounds;
use crate::dialects::loops::{Alloc, Atom, For, ForOp, Index, IndexOp, IndexRanges};
use crate::ir::{BinaryOp, Dim, Expr, IndexExpr, Module, Root, Value, ValueDef, MAX_ITERS};
use crate::require;
use crate::rewrite::{pattern, NoMatch, Pattern, Rewriter};

/// A `loop.index` is one expression over the values that no `loop.index` computes, such as loop
/// variables and the coordinates of atoms, in canonical form where the loops around bound those
/// values: the variable of a loop is its start plus an offset below the most points the loop
/// runs, and a coordinate of a `loop.atom` is below the atom's tile. The op becomes that
/// expression when it computes fewer `floordiv` and `mod`, or as many and fewer operations. It
/// becomes the value its expression names when the expression is one value. The indices it took
/// stay for their other uses.
#[pattern(canonical)]
struct ComposeIndices;

impl Pattern for ComposeIndices {
    type Root = Index;

    fn rewrite(&self, index: IndexOp, rewriter: &mut Rewriter) -> Result<(), NoMatch> {
        let mut leaves = Leaves::new(index.module());
        let mut operands = Vec::with_capacity(index.ins().len());
        for operand in index.ins() {
            operands.push(leaves.compose(operand.id())?);
        }
        let composed = index
            .expr
            .substitute(&|position| operands.get(position).cloned());
        let reduced = leaves.reduce(&composed)?;
        let cost = |expr: &IndexExpr| (expr.divisions(), expr.operations());
        require!(
            reduced.as_iter().is_some() || cost(&reduced) < cost(&index.expr),
            "the composed index computes no fewer divisions and operations"
        );
        if let Some(position) = reduced.as_iter() {
            rewriter.replace_op(&index, leaves.values[position]);
            return Ok(());
        }
        let mut used: ArrayVec<[usize; MAX_ITERS]> = ArrayVec::new();
        used.extend(reduced.iters());
        let renumbered = reduced.substitute(&|position| {
            used.iter()
                .position(|&used_position| used_position == position)
                .map(Expr::iter)
        });
        let mut values = Vec::with_capacity(used.len());
        for &position in &used {
            values.push(leaves.values[position]);
        }
        let value = rewriter.create(Index { expr: renumbered }, values);
        rewriter.replace_op(&index, value);
        Ok(())
    }
}

/// A `loop.for` whose end is its start plus 1, where the loops around bound them, is its body,
/// with the index at the start, when the body allocates no buffer: a buffer lives as long as the
/// region that allocates it, so the loop bounds its life.
#[pattern(canonical)]
struct InlineSingleTrip;

impl Pattern for InlineSingleTrip {
    type Root = For;

    fn rewrite(&self, looped: ForOp, rewriter: &mut Rewriter) -> Result<(), NoMatch> {
        let mut leaves = Leaves::new(looped.module());
        let start = leaves.compose(looped.start().id())?;
        let end = leaves.compose(looped.end().id())?;
        let trips = leaves.reduce(&(end - start))?;
        require!(
            trips.as_const() == Some(1),
            "the loop may run other than one trip"
        );
        let body = looped.body();
        let allocates = body
            .insts()
            .any(|inst| body.module().op(inst).get::<Alloc>().is_some());
        require!(
            !allocates,
            "the body allocates a buffer, whose life the loop bounds"
        );
        rewriter.inline_region(body.id(), &looped, looped.start().id());
        rewriter.erase(&looped);
        Ok(())
    }
}

/// Where the loops around put a leaf: at `start`, an expression of the leaves, plus an offset
/// below `points`.
#[derive(Clone)]
struct Offset {
    start: IndexExpr,
    points: i64,
}

/// The values that no `loop.index` computes, by their position in the expressions composed over
/// them, and the expression of each index composed so far.
struct Leaves<'m> {
    module: &'m Module,
    values: Vec<Value>,
    composed: HashMap<Value, IndexExpr>,
    ranges: IndexRanges<'m>,
}

impl<'m> Leaves<'m> {
    fn new(module: &'m Module) -> Self {
        Self {
            module,
            values: Vec::new(),
            composed: HashMap::new(),
            ranges: IndexRanges::new(module),
        }
    }

    /// The index `value` as an expression of leaves: the expression of the `loop.index` that
    /// computes it, of its operands composed, or a leaf of its own.
    fn compose(&mut self, value: Value) -> Result<IndexExpr, NoMatch> {
        if let Some(known) = self.composed.get(&value) {
            return Ok(known.clone());
        }
        let expr = match self.module.value_ref(value).defining_op::<Index>() {
            Some(index) => {
                let mut operands = Vec::with_capacity(index.ins().len());
                for operand in index.ins() {
                    operands.push(self.compose(operand.id())?);
                }
                index
                    .expr
                    .substitute(&|position| operands.get(position).cloned())
            }
            None => {
                let position = self.values.len();
                require!(
                    position < MAX_ITERS,
                    "the composed index takes more values than an expression names"
                );
                self.values.push(value);
                Expr::iter(position)
            }
        };
        self.composed.insert(value, expr.clone());
        Ok(expr)
    }

    /// Where the loops around put the leaf at `position`: a coordinate of a `loop.atom` at 0, below
    /// the atom's tile, and the variable of a `loop.for` or a `loop.forall` at the loop's start,
    /// below the most points the loop runs.
    fn offset(&mut self, position: usize) -> Result<Option<Offset>, NoMatch> {
        let module = self.module;
        let ValueDef::Arg(region, axis) = module.def(self.values[position]) else {
            return Ok(None);
        };
        let Some(parent) = module.parent_op(region) else {
            return Ok(None);
        };
        if let Some(atom) = module.op(parent).get::<Atom>() {
            return Ok(Some(Offset {
                start: Expr::constant(0),
                points: atom.tile[axis] as i64,
            }));
        }
        let Some((start_value, end_value)) = loop_bounds(module, parent) else {
            return Ok(None);
        };
        let start = self.compose(start_value)?;
        let end = self.compose(end_value)?;
        let Some(points) = self.most_points(&end, &start) else {
            return Ok(None);
        };
        Ok((points > 0).then_some(Offset { start, points }))
    }

    /// The most points a loop from `start` up to `end` runs: a bound of `end - start`, with each
    /// `min` and `max` of `end` taken apart, as `min(a, b) - s` is `min(a - s, b - s)`.
    fn most_points(&mut self, end: &IndexExpr, start: &IndexExpr) -> Option<i64> {
        if let Root::Binary(op @ (BinaryOp::Min | BinaryOp::Max), lhs, rhs) = end.root() {
            let left_most = self.most_points(&lhs, start);
            let right_most = self.most_points(&rhs, start);
            return match (op, left_most, right_most) {
                (BinaryOp::Min, Some(left), Some(right)) => Some(left.min(right)),
                (BinaryOp::Min, one, other) => one.or(other),
                (_, one, other) => Some(one?.max(other?)),
            };
        }
        let difference = (end.clone() - start.clone()).canonical(&|_| None);
        let mut ranges = Vec::with_capacity(self.values.len());
        for &leaf in &self.values {
            ranges.push(self.ranges.of(leaf));
        }
        let (_, highest) = difference.bounds(&|position| ranges[position])?;
        Some(highest)
    }

    /// `expr`, an expression of the leaves, in canonical form where the loops around bound them.
    /// While it takes the canonical form, each leaf at a start other than 0 is that start plus an
    /// offset of its own, so that `floordiv` and `mod` reduce by the multiples in the start and the
    /// range of the offset. After that, each offset is its leaf less the start again.
    fn reduce(&mut self, expr: &IndexExpr) -> Result<IndexExpr, NoMatch> {
        // Composing a start finds the leaves of the loops around, whose offsets come after.
        let mut offsets: ArrayVec<[Option<Offset>; MAX_ITERS]> = ArrayVec::new();
        while offsets.len() < self.values.len() {
            let offset = self.offset(offsets.len())?;
            offsets.push(offset);
        }
        let leaves = self.values.len();
        // The slot of each leaf's own offset, none for a leaf that starts at 0 or past the leaves.
        let mut moved = [None; MAX_ITERS];
        let mut slots = leaves;
        for (leaf_moved, leaf_offset) in moved.iter_mut().zip(&offsets) {
            let starts_at_zero = leaf_offset
                .as_ref()
                .is_none_or(|offset| offset.start.as_const() == Some(0));
            if !starts_at_zero {
                *leaf_moved = Some(slots);
                slots += 1;
            }
        }
        require!(
            slots <= MAX_ITERS,
            "the spread index takes more values than an expression names"
        );
        let spread = Spread {
            offsets: &offsets,
            moved: &moved,
        };
        let mut expansions: ArrayVec<[Option<IndexExpr>; MAX_ITERS]> = ArrayVec::new();
        expansions.resize(leaves, None);
        for position in expr.iters() {
            spread.expand(position, &mut expansions);
        }
        let expanded = expr.substitute(&|position| expansions[position].clone());
        let reduced = expanded.canonical(&|position| spread.extent(position));
        let restored = reduced.substitute(&|slot| {
            let leaf = moved.iter().position(|&moved_to| moved_to == Some(slot))?;
            let offset = offsets[leaf].as_ref()?;
            Some(Expr::iter(leaf) - offset.start.clone())
        });
        Ok(restored.canonical(&|position| match position < leaves {
            true => spread.extent(position),
            false => None,
        }))
    }
}

/// The leaves spread over their starts: the offset of each leaf, and the position of its own
/// offset for a leaf at a start other than 0.
struct Spread<'o> {
    offsets: &'o [Option<Offset>],
    moved: &'o [Option<usize>],
}

impl Spread<'_> {
    /// The leaf at `position` as its start, itself spread, plus its offset, or the leaf itself
    /// when it starts at 0 or has no offset; each kept in `expansions`.
    fn expand(&self, position: usize, expansions: &mut [Option<IndexExpr>]) -> IndexExpr {
        if let Some(known) = &expansions[position] {
            return known.clone();
        }
        let expansion = match (self.moved[position], &self.offsets[position]) {
            (Some(slot), Some(offset)) => {
                // A loop's start comes before the loop, so its leaves are those of loops around.
                for inner in offset.start.iters() {
                    self.expand(inner, expansions);
                }
                let start = offset.start.substitute(&|inner| expansions[inner].clone());
                start + Expr::iter(slot)
            }
            _ => Expr::iter(position),
        };
        expansions[position] = Some(expansion.clone());
        expansion
    }

    /// The extent of the leaf or the offset at `position`, while the leaves are spread: the
    /// points of a leaf that starts at 0, and those of each offset of its own.
    fn extent(&self, position: usize) -> Option<Dim> {
        let leaf = match position < self.offsets.len() {
            true => Some(position).filter(|&leaf| self.moved[leaf].is_none()),
            false => self
                .moved
                .iter()
                .position(|&moved_to| moved_to == Some(position)),
        }?;
        let offset = self.offsets[leaf].as_ref()?;
        Some(Dim::Static(offset.points as usize))
    }
}
