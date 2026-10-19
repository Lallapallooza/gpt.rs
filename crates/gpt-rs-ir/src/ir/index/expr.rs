//! Expressions: their nodes, their constructors, how they evaluate and print, and the references
//! to iterators and operands they hold.

use std::collections::BTreeSet;
use std::fmt;
use std::hash::Hash;
use std::ops::Range;

use anyhow::{anyhow, bail, ensure, Result};
use smallvec::SmallVec;

use crate::ir::{Attr, Dim, Sym, MAX_RANK};
use crate::text::{Cursor, TokenKind};

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
    pub(super) const INFIX: [BinaryOp; 5] = [
        BinaryOp::Add,
        BinaryOp::Sub,
        BinaryOp::Mul,
        BinaryOp::FloorDiv,
        BinaryOp::Mod,
    ];

    /// Binding strength in the text form, which the printer and the parser share: sums bind
    /// loosest, and `min` and `max` are calls.
    pub(super) fn precedence(self) -> u8 {
        match self {
            BinaryOp::Add | BinaryOp::Sub => 0,
            BinaryOp::Mul | BinaryOp::FloorDiv | BinaryOp::Mod => 1,
            BinaryOp::Min | BinaryOp::Max => 2,
        }
    }

    /// The token of the operation: the operator between its operands, or the name of the call.
    pub(super) fn symbol(self) -> &'static str {
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
pub(super) enum Node<R> {
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
    pub(super) fn arity(&self) -> usize {
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
    pub(super) nodes: SmallVec<[Node<R>; 3]>,
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

/// Every iterator at 0 and no symbol bound: an expression evaluates to its value at the origin, and
/// fails where it uses a symbol or reads an operand.
pub struct AtOrigin;

impl IndexEnv for AtOrigin {
    fn iter(&self, _position: usize) -> Option<i64> {
        Some(0)
    }

    fn sym(&self, _sym: &Sym) -> Option<i64> {
        None
    }

    fn read(&self, operand: usize, _coords: &[i64]) -> Result<i64> {
        bail!("the expression reads operand {operand}")
    }
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
pub(super) fn pop<T>(stack: &mut Vec<T>) -> T {
    stack
        .pop()
        .expect("an expression gives each operation its operands")
}

impl<R: ReadKind> Expr<R> {
    /// The expression of `node` on `operands`.
    pub(super) fn on(node: Node<R>, operands: Vec<Self>) -> Self {
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

    /// How the expression moves with the iterator at `position`. It is 0 when the expression does
    /// not use the iterator, and the constant `c` when the expression is `c` times the iterator
    /// plus terms without it. It is `None` when the iterator appears any other way, as inside
    /// `floordiv` or a read.
    pub fn coefficient(&self, position: usize) -> Option<i64> {
        if !self.uses_iter(position) {
            return Some(0);
        }
        match self.root() {
            Root::Iter(_) => Some(1),
            Root::Neg(operand) => operand.coefficient(position).map(|value| -value),
            Root::Binary(BinaryOp::Add, lhs, rhs) => {
                Some(lhs.coefficient(position)? + rhs.coefficient(position)?)
            }
            Root::Binary(BinaryOp::Sub, lhs, rhs) => {
                Some(lhs.coefficient(position)? - rhs.coefficient(position)?)
            }
            Root::Binary(BinaryOp::Mul, lhs, rhs) => match (lhs.as_const(), rhs.as_const()) {
                (Some(factor), _) => Some(factor * rhs.coefficient(position)?),
                (_, Some(factor)) => Some(factor * lhs.coefficient(position)?),
                (None, None) => None,
            },
            _ => None,
        }
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
    pub fn substitute(&self, subst: &dyn Fn(usize) -> Option<Self>) -> Self {
        self.map(&mut |expr| match expr.as_iter().and_then(subst) {
            Some(replacement) => replacement,
            None => expr,
        })
    }
}

impl AccessExpr {
    /// The element of operand `operand` at `coords`, read as an integer.
    #[track_caller]
    pub fn read(operand: usize, coords: impl IntoIterator<Item = impl Into<AccessExpr>>) -> Self {
        let coord_exprs: Vec<AccessExpr> = coords.into_iter().map(Into::into).collect();
        assert!(
            coord_exprs.len() <= MAX_RANK,
            "a read has one coordinate per axis, at most {MAX_RANK}"
        );
        Self::on(Node::Read(operand, coord_exprs.len() as u8), coord_exprs)
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

impl<R: ReadKind> fmt::Display for Expr<R> {
    /// The right operand of `-` and of a product binds tighter, so `a - (b + c)` keeps its
    /// parentheses and `a - b + c` reads left to right.
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
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
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
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
    fn parse(cursor: &mut Cursor) -> Result<Self> {
        let start = cursor.position();
        let name = cursor.ident()?;
        let Some(position) = iter_position(name) else {
            bail!(cursor.error_at(
                start,
                format!("`{name}` is no iterator: an iterator is `d` and its position, such as d0")
            ));
        };
        ensure!(
            position < MAX_ITERS,
            cursor.error_at(
                start,
                format!("a block has at most {MAX_ITERS} iterators, not {name}")
            )
        );
        Ok(Self(position as u8))
    }
    fn peek(cursor: &Cursor) -> bool {
        let next = cursor.peek();
        next.kind == TokenKind::Ident && iter_position(next.text).is_some()
    }
}

impl fmt::Display for IterRef {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
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
    fn parse(cursor: &mut Cursor) -> Result<Self> {
        let start = cursor.position();
        let text = cursor.token(TokenKind::Operand, "an operand")?;
        let operand = text[1..]
            .parse()
            .map_err(|_| cursor.error_at(start, format!("no operand is {text}")))?;
        Ok(Self(operand))
    }
    fn peek(cursor: &Cursor) -> bool {
        cursor.peek_kind(TokenKind::Operand)
    }
}

impl fmt::Display for OperandRef {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "${}", self.0)
    }
}
