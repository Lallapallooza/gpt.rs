//! The text form of expressions and constraints.

use anyhow::{bail, ensure, Result};
use smallvec::SmallVec;

use super::expr::Node;
use super::{AccessExpr, BinaryOp, Cmp, Constraint, Expr, IterRef, OperandRef, ReadKind, Relation};
use crate::ir::{Attr, Sym, MAX_RANK};
use crate::text::{Cursor, TokenKind};

/// An expression whose infix operations bind at least as tightly as `min`: a factor, then each
/// infix operation with its right side, left to right.
fn parse_expr<R: ReadKind>(cursor: &mut Cursor, min: u8) -> Result<Expr<R>> {
    let mut lhs = parse_factor(cursor)?;
    loop {
        let infix = BinaryOp::INFIX
            .into_iter()
            .find(|op| op.precedence() >= min && cursor.peek_is(op.symbol()));
        let Some(op) = infix else {
            return Ok(lhs);
        };
        cursor.next_token();
        let rhs = parse_expr(cursor, op.precedence() + 1)?;
        lhs = Expr::binary(op, lhs, rhs);
    }
}

/// A factor: an integer, `-` and a factor, an expression in parentheses, a symbol `{T}`, a read
/// `$k[...]`, a call of `min` or `max`, or an iterator.
fn parse_factor<R: ReadKind>(cursor: &mut Cursor) -> Result<Expr<R>> {
    if cursor.eat("(") {
        let inner = parse_expr(cursor, 0)?;
        cursor.expect(")")?;
        return Ok(inner);
    }
    if cursor.peek_unsigned().kind == TokenKind::Int {
        return Ok(Expr::constant(i64::parse(cursor)?));
    }
    if cursor.eat("-") {
        return Ok(-parse_factor(cursor)?);
    }
    if cursor.eat("{") {
        let sym = Sym::parse(cursor)?;
        cursor.expect("}")?;
        return Ok(Expr::sym(sym));
    }
    if OperandRef::peek(cursor) {
        let start = cursor.position();
        let operand = OperandRef::parse(cursor)?;
        let Some(read) = R::of_operand(operand.0) else {
            bail!(cursor.error_at(
                start,
                format!("the expression of tile.index reads no operand, not {operand}")
            ));
        };
        let coords = Vec::<Expr<R>>::parse(cursor)?;
        ensure!(
            coords.len() <= MAX_RANK,
            cursor.error_at(
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
        if cursor.eat(op.symbol()) {
            cursor.expect("(")?;
            let lhs = parse_expr(cursor, 0)?;
            cursor.expect(",")?;
            let rhs = parse_expr(cursor, 0)?;
            cursor.expect(")")?;
            return Ok(Expr::binary(op, lhs, rhs));
        }
    }
    if cursor.peek_kind(TokenKind::Ident) {
        return Ok(Expr::iter(IterRef::parse(cursor)?.position()));
    }
    Err(cursor.unexpected("an index expression"))
}

/// The grammar of [`parse_expr`]: the operations take their binding strength from
/// [`BinaryOp::precedence`], which the printer reads too.
impl<R: ReadKind> Attr for Expr<R> {
    const ITEM: &'static str = R::EXPR;

    fn print(&self, out: &mut String) {
        out.push_str(&self.to_string());
    }
    fn parse(cursor: &mut Cursor) -> Result<Self> {
        parse_expr(cursor, 0)
    }
    fn peek(cursor: &Cursor) -> bool {
        let next = cursor.peek();
        matches!(
            next.kind,
            TokenKind::Int | TokenKind::Operand | TokenKind::Ident
        ) || matches!(next.text, "(" | "-" | "{")
    }
}

impl Attr for Constraint {
    const ITEM: &'static str = "Constraint";

    fn print(&self, out: &mut String) {
        out.push_str(&self.to_string());
    }
    /// Comparisons joined by `or`, each two expressions of [`parse_expr`] around a relation.
    fn parse(cursor: &mut Cursor) -> Result<Self> {
        let mut cmps = SmallVec::new();
        loop {
            let lhs = parse_expr(cursor, 0)?;
            let op = Relation::ALL
                .into_iter()
                .find(|&relation| cursor.eat(relation.symbol()))
                .ok_or_else(|| cursor.unexpected("a comparison"))?;
            let rhs = parse_expr(cursor, 0)?;
            cmps.push(Cmp { lhs, op, rhs });
            if !cursor.eat("or") {
                return Ok(Constraint(cmps));
            }
        }
    }
    fn peek(cursor: &Cursor) -> bool {
        AccessExpr::peek(cursor)
    }
}
