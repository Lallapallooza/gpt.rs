//! The `func` dialect's ops and their attributes.

use crate::ir::Type;
use crate::traits::{IsolatedFromAbove, Terminator};

/// Stable identity of a model parameter: derived from its checkpoint name, or from the text of the
/// load-time function that computes it.
#[derive(crate::Attr, Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[attr(format = "$0")]
pub struct ParamId(pub u128);

/// `func.func`: a named function. Its body's block arguments are the function arguments, and its
/// `func.return` yields the results.
///
/// A function whose results are bound to parameters is a load-time function. It reads only
/// parameters, and the runtime runs it once when it loads them. The runtime binds each result to
/// its parameter wherever an argument names that parameter.
#[crate::op(
    "func.func",
    regions(body(Block(Return))),
    traits(IsolatedFromAbove),
    symbol
)]
pub struct Func {
    pub name: String,
    pub args: Vec<Type>,
    pub results: Vec<Type>,
    /// For each argument, the parameter it binds, or `None` for a run-time input. Empty when no
    /// argument binds one.
    #[attr(default, arg(param))]
    pub params: Vec<Option<ParamId>>,
    /// For each result of a load-time function, the parameter it computes. Empty otherwise.
    #[attr(default, result(param))]
    pub result_params: Vec<ParamId>,
}

/// `func.return`: ends a function body. Its operands are the function results.
#[crate::op("func.return", operands(values(Variadic(Any))), traits(Terminator))]
pub struct Return;

#[derive(crate::Dialect, Debug, Clone, PartialEq, Eq, Hash)]
#[dialect(name = "func", op = Func)]
pub enum FuncOps {
    Func(Func),
    Return(Return),
}
