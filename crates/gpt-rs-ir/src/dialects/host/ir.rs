//! `host.dispatch`.

use crate::ir::SymbolRef;

/// `host.dispatch`: calls the dispatch named `callee` on `args` and gives its results. The callee
/// is a `tile.dispatch`, or the `loop.func` it lowers to.
///
/// - The arguments have the callee's argument types, with each of its dimension symbols bound to
///   one extent across all of them. The results have its result types under that binding.
/// - A `loop.func` takes each argument in an `in` parameter, in order, and gives each `out`
///   parameter as a result, in order.
/// - It is pure: a call computes its results from its arguments only.
#[crate::op(
    "host.dispatch",
    operands(args(Variadic(Tensor))),
    results(outs(Variadic(Tensor)))
)]
pub struct Dispatch {
    pub callee: SymbolRef,
}

#[derive(crate::Dialect, Debug, Clone, PartialEq, Eq, Hash)]
#[dialect(name = "host", op = Host)]
pub enum HostOp {
    Dispatch(Dispatch),
}
