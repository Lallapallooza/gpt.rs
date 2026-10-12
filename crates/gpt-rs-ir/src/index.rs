//! The index language of the `tile` dialect: one expression language for a block's reads, writes
//! and constraints and for `tile.index` values.
//!
//! An expression combines iterators (`d0`, the block's iterator at position 0), dimension symbols
//! (`{T}`), integer constants and integer elements of the block's operands (`$2[d0]`) with `+`,
//! `-`, `*`, `floordiv`, `mod`, `min` and `max`. Values are `i64`: `+`, `-` and `*` wrap as si64
//! arithmetic does, and `floordiv` and `mod` round toward minus infinity.
//!
//! An expression is one list of nodes in postfix order: an operation follows the nodes of its
//! operands, so each subexpression is a run of nodes that ends at its root. [`IndexExpr`], the
//! expression of `tile.index`, reads no operand; [`AccessExpr`], the expression of a block's
//! reads, writes and constraints, may.

use std::collections::BTreeSet;
use std::fmt;
use std::hash::Hash;
use std::ops::Range;

use anyhow::{anyhow, bail, ensure, Result};
use smallvec::SmallVec;
use tinyvec::ArrayVec;

use crate::attr::{Attr, F64};
use crate::text::{Cursor, TokenKind};
use crate::types::{Dim, Sym, MAX_RANK};

/// The largest number of iterators of a `tile.block`.
pub const MAX_ITERS: usize = 16;

/// What the reads of an expression name: nothing for an [`IndexExpr`], whose reads are
/// [`NoRead`], and the operand for an [`AccessExpr`], whose reads are `usize`.
pub trait ReadKind: Copy + fmt::Debug + PartialEq + Eq + Hash + 'static {
    /// The name of the expressions whose reads are of this kind.
    const EXPR: &'static str;

    /// The read of operand `operand`, or `None` when the expression reads no operand.
    fn of_operand(operand: usize) -> Option<Self>;
    /// The operand that the read reads.
    fn operand(self) -> usize;
}

/// The reads of an expression that reads no operand: there are none.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NoRead {}

impl ReadKind for NoRead {
    const EXPR: &'static str = "IndexExpr";

    fn of_operand(_operand: usize) -> Option<Self> {
        None
    }

    fn operand(self) -> usize {
        match self {}
    }
}

impl ReadKind for usize {
    const EXPR: &'static str = "AccessExpr";

    fn of_operand(operand: usize) -> Option<Self> {
        Some(operand)
    }

    fn operand(self) -> usize {
        self
    }
}

/// An operation on two index values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    FloorDiv,
    Mod,
    Min,
    Max,
}

impl BinaryOp {
    /// The operation on `lhs` and `rhs`. `floordiv` and `mod` fail on a divisor that is not
    /// positive.
    pub fn apply(self, lhs: i64, rhs: i64) -> Result<i64> {
        Ok(match self {
            BinaryOp::Add => lhs.wrapping_add(rhs),
            BinaryOp::Sub => lhs.wrapping_sub(rhs),
            BinaryOp::Mul => lhs.wrapping_mul(rhs),
            BinaryOp::FloorDiv => {
                ensure!(rhs > 0, "floordiv by {rhs}");
                lhs.div_euclid(rhs)
            }
            BinaryOp::Mod => {
                ensure!(rhs > 0, "mod by {rhs}");
                lhs.rem_euclid(rhs)
            }
            BinaryOp::Min => lhs.min(rhs),
            BinaryOp::Max => lhs.max(rhs),
        })
    }

    /// The operations written between their operands; `min` and `max` are written as calls.
    const INFIX: [BinaryOp; 5] = [
        BinaryOp::Add,
        BinaryOp::Sub,
        BinaryOp::Mul,
        BinaryOp::FloorDiv,
        BinaryOp::Mod,
    ];

    /// Binding strength in the text form, which the printer and the parser share: sums bind
    /// loosest, and `min` and `max` are calls.
    fn precedence(self) -> u8 {
        match self {
            BinaryOp::Add | BinaryOp::Sub => 0,
            BinaryOp::Mul | BinaryOp::FloorDiv | BinaryOp::Mod => 1,
            BinaryOp::Min | BinaryOp::Max => 2,
        }
    }

    /// The token of the operation: the operator between its operands, or the name of the call.
    fn symbol(self) -> &'static str {
        match self {
            BinaryOp::Add => "+",
            BinaryOp::Sub => "-",
            BinaryOp::Mul => "*",
            BinaryOp::FloorDiv => "floordiv",
            BinaryOp::Mod => "mod",
            BinaryOp::Min => "min",
            BinaryOp::Max => "max",
        }
    }
}

/// One node of an expression, after the nodes of its operands.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum Node<R> {
    Const(i64),
    /// The iterator at this position of the block.
    Iter(u8),
    /// The run-time extent of a dimension symbol.
    Sym(Sym),
    /// The element of an integer operand at the coordinates of the subexpressions before it, as
    /// many as it counts.
    Read(R, u8),
    Binary(BinaryOp),
    Neg,
}

impl<R: ReadKind> Node<R> {
    /// How many operands the node takes.
    fn arity(&self) -> usize {
        match self {
            Node::Const(_) | Node::Iter(_) | Node::Sym(_) => 0,
            Node::Read(_, count) => usize::from(*count),
            Node::Binary(_) => 2,
            Node::Neg => 1,
        }
    }
}

/// An index expression, whose reads name `R`: see [`IndexExpr`] and [`AccessExpr`].
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct Expr<R> {
    /// The nodes in postfix order, the root last.
    nodes: SmallVec<[Node<R>; 3]>,
}

/// An expression that reads no operand: the expression of `tile.index`.
pub type IndexExpr = Expr<NoRead>;

/// An expression that may read integer elements of the block's operands: a coordinate of a
/// block's read or write, or a side of a constraint.
pub type AccessExpr = Expr<usize>;

/// The root of an expression, with its operands.
pub enum Root<R> {
    Const(i64),
    Iter(usize),
    Sym(Sym),
    Read(R, Vec<Expr<R>>),
    Binary(BinaryOp, Expr<R>, Expr<R>),
    Neg(Expr<R>),
}

/// What evaluating an index expression reads: iterator values, symbol extents and operand
/// elements.
pub trait IndexEnv {
    /// The value of the iterator at `position`.
    fn iter(&self, position: usize) -> Option<i64>;
    fn sym(&self, sym: &Sym) -> Option<i64>;
    /// Element `coords` of operand `operand`, an integer. Fails when the coordinates are out of
    /// range.
    fn read(&self, operand: usize, coords: &[i64]) -> Result<i64>;
}

/// The start of the subexpression of `nodes` whose root is at `end`.
fn subexpression_start<R: ReadKind>(nodes: &[Node<R>], end: usize) -> usize {
    let mut needed = 1;
    let mut start = end + 1;
    while needed > 0 {
        start -= 1;
        needed = needed - 1 + nodes[start].arity();
    }
    start
}

/// The top value of an evaluation stack, which a well-formed expression always has.
fn pop<T>(stack: &mut Vec<T>) -> T {
    stack
        .pop()
        .expect("an expression gives each operation its operands")
}

impl<R: ReadKind> Expr<R> {
    /// The expression of `node` on `operands`.
    fn on(node: Node<R>, operands: Vec<Self>) -> Self {
        let mut nodes = SmallVec::new();
        for operand in operands {
            nodes.extend(operand.nodes);
        }
        nodes.push(node);
        Self { nodes }
    }

    pub fn constant(value: i64) -> Self {
        Self::on(Node::Const(value), Vec::new())
    }

    /// The iterator at `position` of the block.
    #[track_caller]
    pub fn iter(position: usize) -> Self {
        assert!(
            position < MAX_ITERS,
            "a block has at most {MAX_ITERS} iterators, not d{position}"
        );
        Self::on(Node::Iter(position as u8), Vec::new())
    }

    pub fn sym(sym: Sym) -> Self {
        Self::on(Node::Sym(sym), Vec::new())
    }

    /// The expression for an extent: a constant or a symbol.
    pub fn dim(dim: &Dim) -> Self {
        match dim {
            Dim::Static(size) => Self::constant(*size as i64),
            Dim::Dynamic(sym) => Self::sym(sym.clone()),
        }
    }

    pub fn binary(op: BinaryOp, lhs: Self, rhs: Self) -> Self {
        Self::on(Node::Binary(op), vec![lhs, rhs])
    }

    pub fn floordiv(dividend: Self, divisor: Self) -> Self {
        Self::binary(BinaryOp::FloorDiv, dividend, divisor)
    }

    pub fn modulo(dividend: Self, divisor: Self) -> Self {
        Self::binary(BinaryOp::Mod, dividend, divisor)
    }

    pub fn min(lhs: Self, rhs: Self) -> Self {
        Self::binary(BinaryOp::Min, lhs, rhs)
    }

    pub fn max(lhs: Self, rhs: Self) -> Self {
        Self::binary(BinaryOp::Max, lhs, rhs)
    }

    /// The pure expression `pure`, in this form.
    pub fn from_pure(pure: &IndexExpr) -> Self {
        let mut nodes = SmallVec::new();
        for node in &pure.nodes {
            nodes.push(match node {
                Node::Const(value) => Node::Const(*value),
                Node::Iter(position) => Node::Iter(*position),
                Node::Sym(sym) => Node::Sym(sym.clone()),
                Node::Read(never, _) => match *never {},
                Node::Binary(op) => Node::Binary(*op),
                Node::Neg => Node::Neg,
            });
        }
        Self { nodes }
    }

    /// The element count of a tensor of `dims`.
    pub fn volume(dims: &[Dim]) -> Self {
        let mut volume = Self::constant(1);
        for dim in dims {
            volume = (volume * Self::dim(dim)).simplify();
        }
        volume
    }

    /// The row-major position of the point `coords` in a tensor of `dims`.
    pub fn linearize(coords: &[Self], dims: &[Dim]) -> Self {
        let mut position = Self::constant(0);
        for (coord, dim) in coords.iter().zip(dims) {
            position = ((position * Self::dim(dim)) + coord.clone()).simplify();
        }
        position
    }

    /// The coordinates in a tensor of `dims` of the row-major position this expression gives.
    pub fn delinearize(&self, dims: &[Dim]) -> Vec<Self> {
        let mut coords = Vec::with_capacity(dims.len());
        for axis in 0..dims.len() {
            let quotient = Self::floordiv(self.clone(), Self::volume(&dims[axis + 1..])).simplify();
            coords.push(if axis == 0 {
                quotient
            } else {
                Self::modulo(quotient, Self::dim(&dims[axis])).simplify()
            });
        }
        coords
    }

    /// The root of the expression and its operands.
    pub fn root(&self) -> Root<R> {
        let last = self.nodes.len() - 1;
        let mut operands = Vec::new();
        for run in self.operand_runs(last) {
            operands.push(Self {
                nodes: self.nodes[run].iter().cloned().collect(),
            });
        }
        match &self.nodes[last] {
            Node::Const(value) => Root::Const(*value),
            Node::Iter(position) => Root::Iter(usize::from(*position)),
            Node::Sym(sym) => Root::Sym(sym.clone()),
            Node::Read(read, _) => Root::Read(*read, operands),
            Node::Binary(op) => {
                let [lhs, rhs]: [Self; 2] = operands.try_into().expect("two operands");
                Root::Binary(*op, lhs, rhs)
            }
            Node::Neg => Root::Neg(operands.pop().expect("one operand")),
        }
    }

    /// The runs of nodes of the operands of the node at `end`, in order.
    fn operand_runs(&self, end: usize) -> Vec<Range<usize>> {
        let count = self.nodes[end].arity();
        let mut runs = Vec::with_capacity(count);
        let mut operand_end = end;
        for _ in 0..count {
            let start = subexpression_start(&self.nodes, operand_end - 1);
            runs.push(start..operand_end);
            operand_end = start;
        }
        runs.reverse();
        runs
    }

    /// The iterator position, when the expression is a bare iterator.
    pub fn as_iter(&self) -> Option<usize> {
        match self.nodes.as_slice() {
            [Node::Iter(position)] => Some(usize::from(*position)),
            _ => None,
        }
    }

    pub fn as_const(&self) -> Option<i64> {
        match self.nodes.as_slice() {
            [Node::Const(value)] => Some(*value),
            _ => None,
        }
    }

    /// The positions of the iterators the expression uses.
    pub fn iters(&self) -> BTreeSet<usize> {
        let mut positions = BTreeSet::new();
        for node in &self.nodes {
            if let Node::Iter(position) = node {
                positions.insert(usize::from(*position));
            }
        }
        positions
    }

    /// Whether the expression uses the iterator at `position`.
    pub fn uses_iter(&self, position: usize) -> bool {
        self.nodes
            .iter()
            .any(|node| matches!(node, Node::Iter(used) if usize::from(*used) == position))
    }

    /// Adds the symbols the expression uses to `out`.
    pub fn collect_syms(&self, out: &mut BTreeSet<Sym>) {
        for node in &self.nodes {
            if let Node::Sym(sym) = node {
                out.insert(sym.clone());
            }
        }
    }

    /// The operand and the coordinate count of each read of the expression.
    pub fn reads(&self) -> Vec<(usize, usize)> {
        let mut reads = Vec::new();
        for node in &self.nodes {
            if let Node::Read(read, count) = node {
                reads.push((read.operand(), usize::from(*count)));
            }
        }
        reads
    }

    /// Adds the operands the expression reads to `out`.
    pub fn collect_operands(&self, out: &mut BTreeSet<usize>) {
        for (operand, _) in self.reads() {
            out.insert(operand);
        }
    }

    /// Whether the expression reads any operand.
    pub fn has_reads(&self) -> bool {
        self.nodes.iter().any(|node| matches!(node, Node::Read(..)))
    }

    /// The value at the point `env` describes.
    pub fn eval(&self, env: &dyn IndexEnv) -> Result<i64> {
        let mut stack: Vec<i64> = Vec::with_capacity(self.nodes.len());
        for node in &self.nodes {
            let value = match node {
                Node::Const(value) => *value,
                Node::Iter(position) => env
                    .iter(usize::from(*position))
                    .ok_or_else(|| anyhow!("no iterator d{position}"))?,
                Node::Sym(sym) => env
                    .sym(sym)
                    .ok_or_else(|| anyhow!("the symbol {{{}}} is not bound", sym.as_str()))?,
                Node::Read(read, count) => {
                    let start = stack.len() - usize::from(*count);
                    let value = env.read(read.operand(), &stack[start..])?;
                    stack.truncate(start);
                    value
                }
                Node::Binary(op) => {
                    let rhs = pop(&mut stack);
                    let lhs = pop(&mut stack);
                    op.apply(lhs, rhs)?
                }
                Node::Neg => pop(&mut stack).wrapping_neg(),
            };
            stack.push(value);
        }
        Ok(pop(&mut stack))
    }

    /// The expression with every subexpression rewritten by `rewrite`, operands first.
    pub fn map(&self, rewrite: &mut dyn FnMut(Self) -> Self) -> Self {
        let mut stack: Vec<Self> = Vec::new();
        for node in &self.nodes {
            let operands = stack.split_off(stack.len() - node.arity());
            stack.push(rewrite(Self::on(node.clone(), operands)));
        }
        pop(&mut stack)
    }

    /// The expression with each iterator that `subst` maps replaced by its expression.
    pub fn substitute(&self, subst: &dyn Fn(usize) -> Option<IndexExpr>) -> Self {
        self.map(&mut |expr| match expr.as_iter().and_then(subst) {
            Some(replacement) => Self::from_pure(&replacement),
            None => expr,
        })
    }

    /// Whether the expression is affine in the iterators: no product of two subexpressions that
    /// both use iterators, and no divisor that uses them.
    pub fn is_affine(&self) -> bool {
        // Per subexpression: whether it uses iterators, and whether it is affine.
        let mut stack: Vec<(bool, bool)> = Vec::new();
        for node in &self.nodes {
            let operands = stack.split_off(stack.len() - node.arity());
            let uses = matches!(node, Node::Iter(_)) || operands.iter().any(|&(uses, _)| uses);
            let operands_affine = operands.iter().all(|&(_, affine)| affine);
            let affine = operands_affine
                && match node {
                    Node::Binary(BinaryOp::Mul) => !(operands[0].0 && operands[1].0),
                    Node::Binary(BinaryOp::FloorDiv | BinaryOp::Mod) => !operands[1].0,
                    _ => true,
                };
            stack.push((uses, affine));
        }
        pop(&mut stack).1
    }

    /// Folds constants and drops the identities of `+`, `-` and `*`. Keeps the value.
    pub fn simplify(&self) -> Self {
        self.map(&mut Self::fold_root)
    }

    /// The expression with its root folded, its operands folded already.
    fn fold_root(self) -> Self {
        let pure = |expr: &Self| !expr.has_reads();
        match self.root() {
            Root::Binary(op, lhs, rhs) => match (op, lhs.as_const(), rhs.as_const()) {
                (BinaryOp::FloorDiv | BinaryOp::Mod, Some(_), Some(divisor)) if divisor <= 0 => {
                    self
                }
                (_, Some(left), Some(right)) => {
                    Self::constant(op.apply(left, right).expect("a positive divisor"))
                }
                (BinaryOp::Add, Some(0), None) | (BinaryOp::Mul, Some(1), None) => rhs,
                (BinaryOp::Add | BinaryOp::Sub, None, Some(0))
                | (BinaryOp::Mul | BinaryOp::FloorDiv, None, Some(1)) => lhs,
                (BinaryOp::Sub, ..) if lhs == rhs && pure(&lhs) => Self::constant(0),
                (BinaryOp::Mul, Some(0), None) | (BinaryOp::Mul, None, Some(0))
                    if pure(&lhs) && pure(&rhs) =>
                {
                    Self::constant(0)
                }
                (BinaryOp::Mod, None, Some(1)) if pure(&lhs) => Self::constant(0),
                (BinaryOp::Min | BinaryOp::Max, ..) if lhs == rhs => lhs,
                _ => self,
            },
            Root::Neg(operand) => match operand.as_const() {
                Some(value) => Self::constant(value.wrapping_neg()),
                None => self,
            },
            _ => self,
        }
    }

    /// An interval that holds every value, given the interval of each iterator, or `None` when
    /// the expression uses a symbol or reads an operand, or an interval is unknown.
    pub fn bounds(&self, iter_range: &dyn Fn(usize) -> Option<(i64, i64)>) -> Option<(i64, i64)> {
        let mut stack: Vec<Option<(i64, i64)>> = Vec::new();
        for node in &self.nodes {
            let operands = stack.split_off(stack.len() - node.arity());
            let bounds = match node {
                Node::Const(value) => Some((*value, *value)),
                Node::Iter(position) => iter_range(usize::from(*position)),
                Node::Sym(_) | Node::Read(..) => None,
                Node::Binary(op) => match (operands[0], operands[1]) {
                    (Some(lhs), Some(rhs)) => binary_bounds(*op, lhs, rhs),
                    _ => None,
                },
                Node::Neg => operands[0]
                    .and_then(|(low, high)| Some((high.checked_neg()?, low.checked_neg()?))),
            };
            stack.push(bounds);
        }
        pop(&mut stack)
    }

    /// The expression in canonical form, with the same value at every point. `extent` gives the
    /// extent of the iterator at a position. The canonical form:
    /// - writes affine parts as sums of terms with merged coefficients;
    /// - reduces `floordiv` and `mod` by the multiples of their divisor, and by remainders that
    ///   `extent` shows to be in range;
    /// - decides `min` and `max` where the ranges decide them.
    pub fn canonical(&self, extent: &dyn Fn(usize) -> Option<Dim>) -> Self {
        linear(self, extent).expr()
    }
}

/// The interval of `lhs op rhs` for operands in the intervals `lhs` and `rhs`.
fn binary_bounds(op: BinaryOp, lhs: (i64, i64), rhs: (i64, i64)) -> Option<(i64, i64)> {
    let ((lhs_low, lhs_high), (rhs_low, rhs_high)) = (lhs, rhs);
    match op {
        BinaryOp::Add => Some((
            lhs_low.checked_add(rhs_low)?,
            lhs_high.checked_add(rhs_high)?,
        )),
        BinaryOp::Sub => Some((
            lhs_low.checked_sub(rhs_high)?,
            lhs_high.checked_sub(rhs_low)?,
        )),
        BinaryOp::Mul => {
            let corners = [
                lhs_low.checked_mul(rhs_low)?,
                lhs_low.checked_mul(rhs_high)?,
                lhs_high.checked_mul(rhs_low)?,
                lhs_high.checked_mul(rhs_high)?,
            ];
            Some((*corners.iter().min()?, *corners.iter().max()?))
        }
        BinaryOp::FloorDiv => {
            if rhs_low <= 0 {
                return None;
            }
            let corners = [
                lhs_low.div_euclid(rhs_low),
                lhs_low.div_euclid(rhs_high),
                lhs_high.div_euclid(rhs_low),
                lhs_high.div_euclid(rhs_high),
            ];
            Some((*corners.iter().min()?, *corners.iter().max()?))
        }
        BinaryOp::Mod => {
            if rhs_low <= 0 {
                return None;
            }
            if rhs_low == rhs_high && lhs_low >= 0 && lhs_high < rhs_low {
                return Some((lhs_low, lhs_high));
            }
            Some((0, rhs_high - 1))
        }
        BinaryOp::Min => Some((lhs_low.min(rhs_low), lhs_high.min(rhs_high))),
        BinaryOp::Max => Some((lhs_low.max(rhs_low), lhs_high.max(rhs_high))),
    }
}

impl AccessExpr {
    /// The element of operand `operand` at `coords`, read as an integer.
    #[track_caller]
    pub fn read(operand: usize, coords: impl IntoIterator<Item = impl Into<AccessExpr>>) -> Self {
        let coords: Vec<AccessExpr> = coords.into_iter().map(Into::into).collect();
        assert!(
            coords.len() <= MAX_RANK,
            "a read has one coordinate per axis, at most {MAX_RANK}"
        );
        Self::on(Node::Read(operand, coords.len() as u8), coords)
    }

    /// The expression with operand `i` renamed to `rename(i)`.
    pub fn rename_operands(&self, rename: &dyn Fn(usize) -> usize) -> Self {
        let mut nodes = SmallVec::new();
        for node in &self.nodes {
            nodes.push(match node {
                Node::Read(operand, count) => Node::Read(rename(*operand), *count),
                other => other.clone(),
            });
        }
        Self { nodes }
    }
}

impl From<IndexExpr> for AccessExpr {
    fn from(pure: IndexExpr) -> Self {
        Self::from_pure(&pure)
    }
}

impl<R: ReadKind> std::ops::Add for Expr<R> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self::binary(BinaryOp::Add, self, rhs)
    }
}

impl<R: ReadKind> std::ops::Sub for Expr<R> {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self::binary(BinaryOp::Sub, self, rhs)
    }
}

impl<R: ReadKind> std::ops::Mul for Expr<R> {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        Self::binary(BinaryOp::Mul, self, rhs)
    }
}

impl<R: ReadKind> std::ops::Neg for Expr<R> {
    type Output = Self;
    fn neg(self) -> Self {
        Self::on(Node::Neg, vec![self])
    }
}

/// A sum of terms with integer coefficients, plus a constant: the normal form of affine parts of
/// expressions.
#[derive(Clone)]
struct Linear<R> {
    terms: Vec<(Expr<R>, i64)>,
    constant: i64,
}

impl<R: ReadKind> Linear<R> {
    fn constant(value: i64) -> Self {
        Self {
            terms: Vec::new(),
            constant: value,
        }
    }

    fn atom(expr: Expr<R>) -> Self {
        Self {
            terms: vec![(expr, 1)],
            constant: 0,
        }
    }

    fn as_constant(&self) -> Option<i64> {
        self.terms.is_empty().then_some(self.constant)
    }

    fn add(mut self, other: Linear<R>, sign: i64) -> Self {
        for (atom, coefficient) in other.terms {
            match self
                .terms
                .iter_mut()
                .find(|(existing_atom, _)| *existing_atom == atom)
            {
                Some((_, existing)) => {
                    *existing = existing.wrapping_add(coefficient.wrapping_mul(sign))
                }
                None => self.terms.push((atom, coefficient.wrapping_mul(sign))),
            }
        }
        self.terms.retain(|&(_, coefficient)| coefficient != 0);
        self.constant = self
            .constant
            .wrapping_add(other.constant.wrapping_mul(sign));
        self
    }

    fn scale(mut self, factor: i64) -> Self {
        if factor == 0 {
            return Self::constant(0);
        }
        for (_, coefficient) in &mut self.terms {
            *coefficient = coefficient.wrapping_mul(factor);
        }
        self.constant = self.constant.wrapping_mul(factor);
        self
    }

    /// The expression: the terms with positive coefficients in order, then the negative ones
    /// subtracted, then the constant.
    fn expr(&self) -> Expr<R> {
        let scaled = |atom: &Expr<R>, coefficient: i64| {
            if coefficient == 1 {
                atom.clone()
            } else {
                atom.clone() * Expr::constant(coefficient)
            }
        };
        let mut out: Option<Expr<R>> = None;
        for (atom, coefficient) in &self.terms {
            if *coefficient > 0 {
                let term = scaled(atom, *coefficient);
                out = Some(match out {
                    None => term,
                    Some(sum) => sum + term,
                });
            }
        }
        for (atom, coefficient) in &self.terms {
            if *coefficient < 0 {
                let term = scaled(atom, coefficient.wrapping_neg());
                out = Some(match out {
                    None => -term,
                    Some(sum) => sum - term,
                });
            }
        }
        match (out, self.constant) {
            (None, constant) => Expr::constant(constant),
            (Some(sum), 0) => sum,
            (Some(sum), constant) if constant > 0 => sum + Expr::constant(constant),
            (Some(sum), constant) => sum - Expr::constant(constant.wrapping_neg()),
        }
    }
}

/// The interval of `expr` when the iterators range over their extents.
fn range_of<R: ReadKind>(
    expr: &Expr<R>,
    extent: &dyn Fn(usize) -> Option<Dim>,
) -> Option<(i64, i64)> {
    expr.bounds(&|position| match extent(position)? {
        Dim::Static(size) if size > 0 => Some((0, size as i64 - 1)),
        _ => None,
    })
}

/// Whether `0 <= remainder < divisor` at every point: by intervals, or because `remainder` is an
/// iterator whose extent is `divisor`.
fn below<R: ReadKind>(
    remainder: &Linear<R>,
    divisor: &Expr<R>,
    extent: &dyn Fn(usize) -> Option<Dim>,
) -> bool {
    if let (Some((low, high)), Some(constant_divisor)) =
        (range_of(&remainder.expr(), extent), divisor.as_const())
    {
        return low >= 0 && high < constant_divisor;
    }
    let [(atom, 1)] = remainder.terms.as_slice() else {
        return false;
    };
    let Some(position) = atom.as_iter().filter(|_| remainder.constant == 0) else {
        return false;
    };
    extent(position).is_some_and(|dim| Expr::dim(&dim) == *divisor)
}

/// `dividend` split by `divisor`: the quotient of the terms that are multiples of `divisor`, and
/// the rest.
fn split<R: ReadKind>(dividend: Linear<R>, divisor: &Expr<R>) -> (Linear<R>, Linear<R>) {
    let mut quotient = Linear::constant(0);
    let mut rest = Linear::constant(0);
    let constant_divisor = divisor.as_const();
    for (atom, coefficient) in dividend.terms {
        if let Some(constant) = constant_divisor {
            if coefficient % constant == 0 {
                quotient.terms.push((atom, coefficient / constant));
            } else {
                rest.terms.push((atom, coefficient));
            }
            continue;
        }
        if atom == *divisor {
            quotient.constant += coefficient;
            continue;
        }
        match atom.root() {
            Root::Binary(BinaryOp::Mul, lhs, rhs) if rhs == *divisor => {
                quotient.terms.push((lhs, coefficient))
            }
            Root::Binary(BinaryOp::Mul, lhs, rhs) if lhs == *divisor => {
                quotient.terms.push((rhs, coefficient))
            }
            _ => rest.terms.push((atom, coefficient)),
        }
    }
    match constant_divisor {
        Some(constant) => {
            quotient.constant += dividend.constant.div_euclid(constant);
            rest.constant = dividend.constant.rem_euclid(constant);
        }
        None => rest.constant = dividend.constant,
    }
    (quotient, rest)
}

/// The canonical linear form of `expr`.
fn linear<R: ReadKind>(expr: &Expr<R>, extent: &dyn Fn(usize) -> Option<Dim>) -> Linear<R> {
    match expr.root() {
        Root::Const(value) => Linear::constant(value),
        Root::Iter(_) | Root::Sym(_) => Linear::atom(expr.clone()),
        Root::Read(read, coords) => {
            let mut canonical = Vec::with_capacity(coords.len());
            for coord in &coords {
                canonical.push(coord.canonical(extent));
            }
            Linear::atom(Expr::on(Node::Read(read, coords.len() as u8), canonical))
        }
        Root::Neg(operand) => linear(&operand, extent).scale(-1),
        Root::Binary(BinaryOp::Add, lhs, rhs) => linear(&lhs, extent).add(linear(&rhs, extent), 1),
        Root::Binary(BinaryOp::Sub, lhs, rhs) => linear(&lhs, extent).add(linear(&rhs, extent), -1),
        Root::Binary(BinaryOp::Mul, lhs, rhs) => {
            let (left, right) = (linear(&lhs, extent), linear(&rhs, extent));
            match (left.as_constant(), right.as_constant()) {
                (_, Some(factor)) => left.scale(factor),
                (Some(factor), _) => right.scale(factor),
                _ => Linear::atom(left.expr() * right.expr()),
            }
        }
        Root::Binary(BinaryOp::FloorDiv, lhs, rhs) => linear_floordiv(&lhs, &rhs, extent),
        Root::Binary(BinaryOp::Mod, lhs, rhs) => {
            let divisor = rhs.canonical(extent);
            if divisor.as_const().is_some_and(|value| value <= 0) {
                return Linear::atom(Expr::modulo(lhs.canonical(extent), divisor));
            }
            let (_, rest) = split(linear(&lhs, extent), &divisor);
            if let (Some(remainder), Some(constant_divisor)) =
                (rest.as_constant(), divisor.as_const())
            {
                return Linear::constant(remainder.rem_euclid(constant_divisor));
            }
            if below(&rest, &divisor, extent) {
                return rest;
            }
            Linear::atom(Expr::modulo(rest.expr(), divisor))
        }
        Root::Binary(op @ (BinaryOp::Min | BinaryOp::Max), lhs, rhs) => {
            linear_extreme(op == BinaryOp::Min, &lhs, &rhs, extent)
        }
    }
}

/// The canonical linear form of `dividend floordiv divisor`.
fn linear_floordiv<R: ReadKind>(
    dividend: &Expr<R>,
    divisor: &Expr<R>,
    extent: &dyn Fn(usize) -> Option<Dim>,
) -> Linear<R> {
    let divisor = divisor.canonical(extent);
    // `(x floordiv c1) floordiv c2` is `x floordiv (c1 * c2)`.
    if let (Root::Binary(BinaryOp::FloorDiv, inner_dividend, inner_divisor), Some(outer)) =
        (dividend.root(), divisor.as_const())
    {
        if let Some(inner) = inner_divisor
            .as_const()
            .filter(|&value| value > 0 && outer > 0)
        {
            let merged = Expr::floordiv(inner_dividend, Expr::constant(inner * outer));
            return linear(&merged, extent);
        }
    }
    if divisor.as_const().is_some_and(|value| value <= 0) {
        return Linear::atom(Expr::floordiv(dividend.canonical(extent), divisor));
    }
    let (quotient, rest) = split(linear(dividend, extent), &divisor);
    if let (Some(remainder), Some(constant_divisor)) = (rest.as_constant(), divisor.as_const()) {
        return quotient.add(Linear::constant(remainder.div_euclid(constant_divisor)), 1);
    }
    if below(&rest, &divisor, extent) {
        return quotient;
    }
    quotient.add(Linear::atom(Expr::floordiv(rest.expr(), divisor)), 1)
}

/// The canonical linear form of `min(lhs, rhs)` when `is_min`, else of `max(lhs, rhs)`.
fn linear_extreme<R: ReadKind>(
    is_min: bool,
    lhs: &Expr<R>,
    rhs: &Expr<R>,
    extent: &dyn Fn(usize) -> Option<Dim>,
) -> Linear<R> {
    let (left, right) = (lhs.canonical(extent), rhs.canonical(extent));
    if left == right {
        return linear(&left, extent);
    }
    if let (Some((left_low, left_high)), Some((right_low, right_high))) =
        (range_of(&left, extent), range_of(&right, extent))
    {
        let (first, second) = if is_min {
            (left_high <= right_low, right_high <= left_low)
        } else {
            (left_low >= right_high, right_low >= left_high)
        };
        if first {
            return linear(&left, extent);
        }
        if second {
            return linear(&right, extent);
        }
    }
    // `min(max(x, c), c)` and `max(min(x, c), c)` are `c`.
    let inner = match (left.root(), is_min) {
        (Root::Binary(BinaryOp::Max, _, bound), true)
        | (Root::Binary(BinaryOp::Min, _, bound), false) => Some(bound),
        _ => None,
    };
    if inner.is_some_and(|bound| bound == right && right.as_const().is_some()) {
        return linear(&right, extent);
    }
    let op = if is_min { BinaryOp::Min } else { BinaryOp::Max };
    Linear::atom(Expr::binary(op, left, right))
}

impl<R: ReadKind> fmt::Display for Expr<R> {
    /// The right operand of `-` and of a product binds tighter, so `a - (b + c)` keeps its
    /// parentheses and `a - b + c` reads left to right.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Per subexpression: its text and its binding strength.
        let mut stack: Vec<(String, u8)> = Vec::new();
        let side = |(text, precedence): &(String, u8), min: u8| {
            if *precedence < min {
                format!("({text})")
            } else {
                text.clone()
            }
        };
        for node in &self.nodes {
            let operands = stack.split_off(stack.len() - node.arity());
            let text = match node {
                Node::Const(value) => value.to_string(),
                Node::Iter(position) => IterRef(*position).to_string(),
                Node::Sym(sym) => format!("{{{}}}", sym.as_str()),
                Node::Read(read, _) => {
                    let mut coords = Vec::with_capacity(operands.len());
                    for (coord, _) in &operands {
                        coords.push(coord.as_str());
                    }
                    format!("{}[{}]", OperandRef(read.operand()), coords.join(", "))
                }
                Node::Binary(op @ (BinaryOp::Min | BinaryOp::Max)) => {
                    format!("{}({}, {})", op.symbol(), operands[0].0, operands[1].0)
                }
                Node::Binary(op) => {
                    let precedence = op.precedence();
                    let lhs = side(&operands[0], precedence);
                    let rhs = side(&operands[1], precedence + 1);
                    stack.push((format!("{lhs} {} {rhs}", op.symbol()), precedence));
                    continue;
                }
                Node::Neg => format!("-{}", side(&operands[0], 2)),
            };
            stack.push((text, 2));
        }
        f.write_str(&pop(&mut stack).0)
    }
}

impl<R: ReadKind> fmt::Debug for Expr<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

/// An iterator of a block by its position, written `d` and the position: `d2`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct IterRef(u8);

impl IterRef {
    #[track_caller]
    pub fn new(position: usize) -> Self {
        assert!(
            position < MAX_ITERS,
            "a block has at most {MAX_ITERS} iterators, not d{position}"
        );
        Self(position as u8)
    }

    pub fn position(self) -> usize {
        usize::from(self.0)
    }
}

/// The position of the iterator that the name `name` writes.
fn iter_position(name: &str) -> Option<usize> {
    let digits = name.strip_prefix('d')?;
    if digits.is_empty() || !digits.bytes().all(|byte| byte.is_ascii_digit()) {
        return None;
    }
    digits.parse().ok()
}

impl Attr for IterRef {
    const ITEM: &'static str = "IterRef";

    fn print(&self, out: &mut String) {
        out.push_str(&self.to_string());
    }
    fn parse(c: &mut Cursor<'_>) -> Result<Self> {
        let start = c.position();
        let name = c.ident()?;
        let Some(position) = iter_position(name) else {
            bail!(c.error_at(
                start,
                format!("`{name}` is no iterator: an iterator is `d` and its position, such as d0")
            ));
        };
        ensure!(
            position < MAX_ITERS,
            c.error_at(
                start,
                format!("a block has at most {MAX_ITERS} iterators, not {name}")
            )
        );
        Ok(Self(position as u8))
    }
    fn peek(c: &Cursor<'_>) -> bool {
        let next = c.peek();
        next.kind == TokenKind::Ident && iter_position(next.text).is_some()
    }
}

impl fmt::Display for IterRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "d{}", self.0)
    }
}

/// An operand of an op by its position, written `$` and the position: `$2`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct OperandRef(pub usize);

impl Attr for OperandRef {
    const ITEM: &'static str = "OperandRef";

    fn print(&self, out: &mut String) {
        out.push_str(&self.to_string());
    }
    fn parse(c: &mut Cursor<'_>) -> Result<Self> {
        let start = c.position();
        let text = c.token(TokenKind::Operand, "an operand")?;
        let operand = text[1..]
            .parse()
            .map_err(|_| c.error_at(start, format!("no operand is {text}")))?;
        Ok(Self(operand))
    }
    fn peek(c: &Cursor<'_>) -> bool {
        c.peek_kind(TokenKind::Operand)
    }
}

impl fmt::Display for OperandRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "${}", self.0)
    }
}

/// An expression whose infix operations bind at least as tightly as `min`: a factor, then each
/// infix operation with its right side, left to right.
fn parse_expr<R: ReadKind>(c: &mut Cursor<'_>, min: u8) -> Result<Expr<R>> {
    let mut lhs = parse_factor(c)?;
    loop {
        let infix = BinaryOp::INFIX
            .into_iter()
            .find(|op| op.precedence() >= min && c.peek_is(op.symbol()));
        let Some(op) = infix else {
            return Ok(lhs);
        };
        c.next_token();
        let rhs = parse_expr(c, op.precedence() + 1)?;
        lhs = Expr::binary(op, lhs, rhs);
    }
}

/// A factor: an integer, `-` and a factor, an expression in parentheses, a symbol `{T}`, a read
/// `$k[...]`, a call of `min` or `max`, or an iterator.
fn parse_factor<R: ReadKind>(c: &mut Cursor<'_>) -> Result<Expr<R>> {
    if c.eat("(") {
        let inner = parse_expr(c, 0)?;
        c.expect(")")?;
        return Ok(inner);
    }
    if c.peek_unsigned().kind == TokenKind::Int {
        return Ok(Expr::constant(i64::parse(c)?));
    }
    if c.eat("-") {
        return Ok(-parse_factor(c)?);
    }
    if c.eat("{") {
        let sym = Sym::parse(c)?;
        c.expect("}")?;
        return Ok(Expr::sym(sym));
    }
    if OperandRef::peek(c) {
        let start = c.position();
        let operand = OperandRef::parse(c)?;
        let Some(read) = R::of_operand(operand.0) else {
            bail!(c.error_at(
                start,
                format!("the expression of tile.index reads no operand, not {operand}")
            ));
        };
        let coords = Vec::<Expr<R>>::parse(c)?;
        ensure!(
            coords.len() <= MAX_RANK,
            c.error_at(
                start,
                format!(
                    "{} coordinates, but a tensor has at most {MAX_RANK} axes",
                    coords.len()
                )
            )
        );
        return Ok(Expr::on(Node::Read(read, coords.len() as u8), coords));
    }
    for op in [BinaryOp::Min, BinaryOp::Max] {
        if c.eat(op.symbol()) {
            c.expect("(")?;
            let lhs = parse_expr(c, 0)?;
            c.expect(",")?;
            let rhs = parse_expr(c, 0)?;
            c.expect(")")?;
            return Ok(Expr::binary(op, lhs, rhs));
        }
    }
    if c.peek_kind(TokenKind::Ident) {
        return Ok(Expr::iter(IterRef::parse(c)?.position()));
    }
    Err(c.unexpected("an index expression"))
}

/// The grammar of [`parse_expr`]: the operations take their binding strength from
/// [`BinaryOp::precedence`], which the printer reads too.
impl<R: ReadKind> Attr for Expr<R> {
    const ITEM: &'static str = R::EXPR;

    fn print(&self, out: &mut String) {
        out.push_str(&self.to_string());
    }
    fn parse(c: &mut Cursor<'_>) -> Result<Self> {
        parse_expr(c, 0)
    }
    fn peek(c: &Cursor<'_>) -> bool {
        let next = c.peek();
        matches!(
            next.kind,
            TokenKind::Int | TokenKind::Operand | TokenKind::Ident
        ) || matches!(next.text, "(" | "-" | "{")
    }
}

/// Which operand a read takes its elements from.
#[derive(crate::Attr, Debug, Clone, PartialEq, Eq, Hash)]
#[attr(via = SourceText)]
pub enum Source {
    /// Operand `i`.
    Operand(usize),
    /// One of the operands `first..first + count`, all of one type, picked at each point by the
    /// value of `pick`: per-expert weights that stay separate tensors.
    List {
        first: usize,
        count: usize,
        pick: Box<AccessExpr>,
    },
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
struct SourceText {
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

/// The coordinates of a read or a write, one per axis.
pub type Coords = SmallVec<[AccessExpr; 4]>;

/// The coordinates `coords`, in the form of an access.
fn coords_of(coords: impl IntoIterator<Item = impl Into<AccessExpr>>) -> Coords {
    coords.into_iter().map(Into::into).collect()
}

/// A read of a `tile.block`: an operand element at index expressions, one per axis. With `pad`, a
/// point whose coordinates fall outside the operand reads that value; without it, every point
/// MUST be inside.
#[derive(crate::Attr, Debug, Clone, PartialEq, Eq, Hash)]
#[attr(format = "$source `` $coords (`pad` $pad^)?")]
pub struct Read {
    pub source: Source,
    pub coords: Coords,
    pub pad: Option<F64>,
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

/// Where a `tile.block` writes one of its results: the coordinates of a new tensor, or of an
/// operand that the result updates (`into`).
#[derive(crate::Attr, Debug, Clone, PartialEq, Eq, Hash)]
#[attr(format = "($into^)? `` $coords")]
pub struct Write {
    /// The operand that the result is an updated copy of.
    pub into: Option<OperandRef>,
    pub coords: Coords,
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

/// The relation of a comparison of two index expressions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Relation {
    Less,
    LessEqual,
    Equal,
    NotEqual,
    GreaterEqual,
    Greater,
}

impl Relation {
    const ALL: [Relation; 6] = [
        Relation::Less,
        Relation::LessEqual,
        Relation::Equal,
        Relation::NotEqual,
        Relation::GreaterEqual,
        Relation::Greater,
    ];

    /// The symbol of the relation in index constraints.
    pub fn symbol(self) -> &'static str {
        match self {
            Relation::Less => "<",
            Relation::LessEqual => "<=",
            Relation::Equal => "==",
            Relation::NotEqual => "!=",
            Relation::GreaterEqual => ">=",
            Relation::Greater => ">",
        }
    }

    pub fn holds<T: PartialOrd>(self, lhs: T, rhs: T) -> bool {
        match self {
            Relation::Less => lhs < rhs,
            Relation::LessEqual => lhs <= rhs,
            Relation::Equal => lhs == rhs,
            Relation::NotEqual => lhs != rhs,
            Relation::GreaterEqual => lhs >= rhs,
            Relation::Greater => lhs > rhs,
        }
    }
}

/// A comparison of two index expressions.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Cmp {
    pub lhs: AccessExpr,
    pub op: Relation,
    pub rhs: AccessExpr,
}

impl Cmp {
    pub fn new(lhs: impl Into<AccessExpr>, op: Relation, rhs: impl Into<AccessExpr>) -> Self {
        Self {
            lhs: lhs.into(),
            op,
            rhs: rhs.into(),
        }
    }

    pub fn holds(&self, env: &dyn IndexEnv) -> Result<bool> {
        Ok(self.op.holds(self.lhs.eval(env)?, self.rhs.eval(env)?))
    }
}

/// A constraint of a `tile.block`: it holds when any of its comparisons does.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Constraint(pub SmallVec<[Cmp; 1]>);

impl Constraint {
    /// The constraint `lhs op rhs`.
    pub fn cmp(lhs: impl Into<AccessExpr>, op: Relation, rhs: impl Into<AccessExpr>) -> Self {
        let mut cmps = SmallVec::new();
        cmps.push(Cmp::new(lhs, op, rhs));
        Self(cmps)
    }

    pub fn holds(&self, env: &dyn IndexEnv) -> Result<bool> {
        for cmp in &self.0 {
            if cmp.holds(env)? {
                return Ok(true);
            }
        }
        Ok(false)
    }

    pub fn exprs(&self) -> impl Iterator<Item = &AccessExpr> {
        self.0.iter().flat_map(|cmp| [&cmp.lhs, &cmp.rhs])
    }

    pub fn map_exprs(&self, rewrite: &dyn Fn(&AccessExpr) -> AccessExpr) -> Constraint {
        let mut cmps = SmallVec::new();
        for cmp in &self.0 {
            cmps.push(Cmp::new(rewrite(&cmp.lhs), cmp.op, rewrite(&cmp.rhs)));
        }
        Constraint(cmps)
    }
}

impl fmt::Display for Constraint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (position, cmp) in self.0.iter().enumerate() {
            if position > 0 {
                f.write_str(" or ")?;
            }
            write!(f, "{} {} {}", cmp.lhs, cmp.op.symbol(), cmp.rhs)?;
        }
        Ok(())
    }
}

impl Attr for Constraint {
    const ITEM: &'static str = "Constraint";

    fn print(&self, out: &mut String) {
        out.push_str(&self.to_string());
    }
    /// Comparisons joined by `or`, each two expressions of [`parse_expr`] around a relation.
    fn parse(c: &mut Cursor<'_>) -> Result<Self> {
        let mut cmps = SmallVec::new();
        loop {
            let lhs = parse_expr(c, 0)?;
            let op = Relation::ALL
                .into_iter()
                .find(|&relation| c.eat(relation.symbol()))
                .ok_or_else(|| c.unexpected("a comparison"))?;
            let rhs = parse_expr(c, 0)?;
            cmps.push(Cmp { lhs, op, rhs });
            if !c.eat("or") {
                return Ok(Constraint(cmps));
            }
        }
    }
    fn peek(c: &Cursor<'_>) -> bool {
        AccessExpr::peek(c)
    }
}

/// Iterators by position, written by name: `[d0, d2]`.
#[derive(crate::Attr, Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
#[attr(format = "$0")]
pub struct IterList(ArrayVec<[IterRef; MAX_ITERS]>);

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

/// How a `tile.block` treats an iterator.
#[derive(crate::Attr, Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum IterKind {
    /// The results are indexed by it.
    #[default]
    Par,
    /// The combine folds it.
    Red,
    /// The combine folds it in order: no schedule may split or reorder it.
    RedOrdered,
    /// The combine folds it, and every prefix of the fold is a result.
    Prefix,
}

impl IterKind {
    /// Whether the combine folds the iterator.
    pub fn is_reduced(self) -> bool {
        matches!(self, IterKind::Red | IterKind::RedOrdered)
    }
}

/// An iterator of a `tile.block`: a kind and an extent, `red 8`. Its position in the block's list
/// names it: `d0` is the first.
#[derive(crate::Attr, Debug, Clone, Default, PartialEq, Eq, Hash)]
#[attr(format = "$kind $extent")]
pub struct Iter {
    pub kind: IterKind,
    pub extent: Dim,
}

/// The iterators of a `tile.block`, at most [`MAX_ITERS`].
pub type Iters = ArrayVec<[Iter; MAX_ITERS]>;

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

/// What a dispatch knows of a dimension symbol: a range of extents, and a divisor of every one,
/// `[1, 64] % 8`.
#[derive(crate::Attr, Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[attr(format = "`[` $min `,` $max `]` (`%` $divisor^)?")]
pub struct SymFact {
    pub min: u64,
    pub max: u64,
    #[attr(default = 1)]
    pub divisor: u64,
}
