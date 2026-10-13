//! Analyses of a module.
//!
//! What an op costs to run: the arithmetic it does and the bytes it reads and writes, from its
//! operand and result types. A cost is of the op as a backend runs it, not of a naive reading of
//! its semantics: a `tensor.gather` reads the rows it gathers, not its whole table.
//!
//! Most ops cost what [`default_cost`] says. An op that costs something else implements
//! [`OpCost`] with `#[cost]`, which registers it.

mod cost;

pub use self::cost::*;
