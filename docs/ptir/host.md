<!-- Generated from the code by `cargo run -p gpt-rs-docs`. Do not edit: change the code and run it again. -->

# The host dialect

The `host` dialect is the program that runs dispatches. A function calls dispatches on tensors with
`host.dispatch`: `tile.dispatch` symbols, or the `loop.func` symbols they lower to. Each call is
pure: its results depend on its arguments alone.

## Conventions

- Each op is defined in `ir.rs`. Its type rule is in `verify.rs`, and its semantics, its `Eval`
  impl, are in `interp.rs`.
- A call binds the callee's dimension symbols to the extents of its arguments, one extent per
  symbol across all of them.

## Ops

- `host.dispatch`

### `host.dispatch`

Calls the dispatch named `callee` on `args` and gives its results. The callee
is a `tile.dispatch`, or the `loop.func` it lowers to.

- The arguments have the callee's argument types, with each of its dimension symbols bound to
  one extent across all of them. The results have its result types under that binding.
- A `loop.func` takes each argument in an `in` parameter, in order, and gives each `out`
  parameter as a result, in order.
- It is pure: a call computes its results from its arguments only.

Defined by `Dispatch` in [crates/gpt-rs-ir/src/dialects/host/ir.rs](../../crates/gpt-rs-ir/src/dialects/host/ir.rs).

| | |
|---|---|
| Operands | `args`: tensors |
| Results | `outs`: tensors |
| Attributes | `callee`: `SymbolRef` |
| Effects | none |

Example, from `filetests/tile/roundtrip/block.ptir`:

```text
tile.dispatch @rms(%input: tensor<{T}x8xf32>) -> tensor<{T}xf32> attributes {grid = [d0], facts = {T = [16, 4096] % 16}} {
  %sum_of_squares = tile.block(%input) {iters = [par {T}, red 8], reads = [$0[d0, d1]], folds = [sum -> [d0]]} ({
  ^body(%element: f32):
    %square = arith.mul(%element, %element) : (f32, f32) -> f32
    tile.yield(%square) : (f32) -> ()
  }) : (tensor<{T}x8xf32>) -> tensor<{T}xf32>
  func.return(%sum_of_squares) : (tensor<{T}xf32>) -> ()
}
func.func @call(%input: tensor<{T}x8xf32>) -> (tensor<{T}xf32>, tensor<{T}x8xf32>) {
  %sum_of_squares = host.dispatch(%input) {callee = @rms} : (tensor<{T}x8xf32>) -> tensor<{T}xf32>
  %exponentials = tile.group() {grid = [d0]} ({
    %block_exponentials = tile.block(%input) {iters = [par {T}, par 8], reads = [$0[d0, d1]], folds = [none -> [d0, d1]]} ({
    ^body(%element: f32):
      %exponential = arith.exp(%element) : (f32) -> f32
      tile.yield(%exponential) : (f32) -> ()
    }) : (tensor<{T}x8xf32>) -> tensor<{T}x8xf32>
    tile.yield(%block_exponentials) : (tensor<{T}x8xf32>) -> ()
  }) : () -> tensor<{T}x8xf32>
  func.return(%sum_of_squares, %exponentials) : (tensor<{T}xf32>, tensor<{T}x8xf32>) -> ()
}
```

Rejected, as the verify filetests pin it:

- `host.dispatch: @double is not a tile.dispatch or a loop.func`
- `host.dispatch: no symbol @missing`
- `host.dispatch: argument 0: tensor<4xsi32> does not match tensor<{T}xf32>`

## Traits

A trait states a fact about the ops that have it. The folds and patterns it implies apply to every one of them; [passes.md](passes.md) says which pass runs each.
