//! The index language: one expression language for the reads, writes and constraints of a
//! `tile.block`, for `tile.index` values, and for `loop.index` and `loop.predicate`.
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

mod constraint;
mod expr;
mod simplify;
mod text;

pub use self::constraint::*;
pub use self::expr::*;
