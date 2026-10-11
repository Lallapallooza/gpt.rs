//! `host.dispatch`.

use crate::attr::SymbolRef;

/// `host.dispatch`: calls the `tile.dispatch` named `callee` on `args` and gives its results.
///
/// - The arguments have the callee's argument types, with each of its dimension symbols bound to
///   one extent across all of them. The results have its result types under that binding.
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
