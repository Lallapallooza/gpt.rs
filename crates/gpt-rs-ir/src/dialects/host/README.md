# The host dialect

The `host` dialect is the program that runs dispatches. A function calls dispatches on tensors with
`host.dispatch`: `tile.dispatch` symbols, or the `loop.func` symbols they lower to. Each call is
pure: its results depend on its arguments alone.

## Conventions

- Each op is defined in `ir.rs`. Its type rule is in `verify.rs`, and its semantics, its `Eval`
  impl, are in `interp.rs`.
- A call binds the callee's dimension symbols to the extents of its arguments, one extent per
  symbol across all of them.
