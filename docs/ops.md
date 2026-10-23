# Ops layer: recording, graphs, and execution

This document describes the `gpt_rs::ops` layer: how tensor ops record into lazy graphs, how graphs
compile, and how backends run them. The ops are the `tensor` dialect of `gpt-rs-ir`
(`crates/gpt-rs-ir/src/dialects/tensor/`), and their semantics are its `Eval` impls (`interp/`).
[ptir/tensor.md](ptir/tensor.md) is the reference of the dialect, and
[backends/README.md](backends/README.md) describes the functions that backends compile and run.

## Where things live

- `crates/gpt-rs/src/tensor/ops.rs`: the ops of `DeviceTensor`, each of which records one `tensor`
  op.
- `crates/gpt-rs/src/ops/functional/`: portable kernels (validate, then record ops).
- `crates/gpt-rs/src/ops/graph/`: the graph arena and the program cache.
- `crates/gpt-rs/src/ops/trace/`: dump and trace hooks around compilation and execution.
- `crates/gpt-rs/src/tensor/device_tensor.rs`: lazy handles and materialization triggers.

## Validate, then record (functionals)

Every functional is a `#[functional]` with the same structure:

1. Validate shape and dtype with the validation macros (`ensure_rank!`, `ensure_dtype!`, ...).
2. Compute with the ops of `DeviceTensor` and other functionals. Each op records one `tensor` op
   into the current default arena (else the graph of its first operand that has one, else a new
   graph) and returns a lazy `DeviceTensor`. Its type comes from the op's type rule
   (`OpDef::result_types`), so an op that its operands do not fit panics where it is recorded.
   An op checks that its operands live on the graph's backend instance.

See [frontend.md](frontend.md) for an example.

## Lazy execution model

`DeviceTensor<B>` is a thin wrapper around a backend and a *lazy handle*.

The handle can represent:
- a tensor already on the backend (an argument or a parameter)
- a node in a `GraphArena<B>` (a recorded value)

Materialization happens when caller code needs the underlying data (for example via `.to_host()`),
or when a downstream op forces execution. The arena groups the requested values into a plan, builds a
PTIR module, runs the backend's pipeline ([ptir/passes.md](ptir/passes.md)), runs any
load-time functions whose results the backend's parameter store lacks, and compiles and runs the
result on the backend. The arena caches plans and compiled functions by a key of the graph's
structure and specialization.

### Traced entry points

Recording does work for each op, so a forward of hundreds of ops pays it on every call before its
program runs. The runtime's fixed entry points, the forward of an image model and a decoder's decode
step, run through `ops::graph::Traced`. A function that borrows what it runs uses its `Traces`
instead.

The first call with inputs of some signature records the function and keeps the program it ran: the
signature is the inputs' dtypes, their extents, and which of them are the same tensor. Later calls
with the same signature run that program on the new inputs directly, without recording. A decode
step takes the token, its position, the generator state where sampling is random, and the caches as
inputs, and gives the next ones as results. So every step replays one program. Code below the entry
points, layers and functionals, stays ordinary lazy tensor code.

A trace replays only when its graph can depend on nothing but the signature of its inputs. A trace
falls back to recording every call when the function, during its first recording:

- reads a value back with `.to_host()`, or materializes a tensor;
- reads a tensor that is neither an input nor a parameter, such as one made from host data;
- does not run exactly one program, returns a tensor that its program does not compute, or marks
  for export a value that it does not return.

Calls also record while `GPTRS_EAGER` is set, or while `nn::capture` records module outputs.
`GPTRS_TRACE_CHECK=1` records every call of a traced function, and fails when its graph is not the
one its trace replays. It catches a forward that bakes a value of each call, such as a Rust number,
into its graph.

### Eager debugging mode

Setting `GPTRS_EAGER=1` forces tensors created from recorded nodes to flush immediately (useful for
debugging functionals in "eager-like" mode).

## Tracing and dumps

The ops layer exposes hooks around compilation and execution via `crates/gpt-rs/src/ops/trace/`:

- `ExecutionTraceSink`: callbacks before and after each executed program, with the module it was
  compiled from, `on_pass(&PassRun, &Module)` after each pass with the pass's name, options,
  whether it changed the module and its statistics, and `on_artifact(name, contents)` with a file
  a backend made, such as its generated C. The arena runs a backend's pipeline this way, through
  `trace::run_pipeline`.
- `FileTraceSink`: writes each module as `program.ptir` with `meta.json` and `stats.json`, the
  runs to `executions.jsonl`, the module after each pass and each artifact under `passes/`, and,
  with `GPTRS_PASS_STATS=1`, the pass events to `passes.jsonl`.

Dumps print each op's location and its number, `// #47`, its place in print order. A location
holds the layer and the line of Rust that recorded the op and, after `via`, its trail: each pass
that made the op from another, with that op's name and number in the module before the pass,
which is the dump of the pass before. With the x86 backend the dumps go on through `tile`, `loop`
and `isa`, and each line of its C ends with the number of the op it comes from, so a line of C
leads through every dump back to the line of Rust:

```text
isa.mul(...) loc("convolution"("crates/gpt-rs/src/nn/layers/batch_norm.rs":34:9) via
  ["dot-scale-fold"("tensor.mul":15), "tensor-to-tile"("tensor.dot_general":5),
   "seal-dispatches"("arith.mul":8), "tile-to-loop"("arith.mul":10),
   "loop-to-isa"("arith.mul":112)]) // #117
```

`gpt-rs-cli ir-opt --locations` prints the same.

`gpt-rs-cli` wires this up via:
- `--dump-dir <DIR>` and `--dump-mode all|compile`

See [testing.md](testing.md) for CLI usage examples.

## How to debug a bad kernel

Suggested loop:

1. Reproduce with `gpt-rs-cli --dump-dir ...` and keep the dumped `program.ptir`.
2. If debugging shape rules, focus on the functional validation (the `ensure_*!` macros in
   `ops/functional/validate.rs`) and the op's type rule (`OpDef::result_types`).
3. If debugging a pass, cut the dumped module down to a filetest and run the pass on it
   (`docs/testing.md`).
4. If debugging execution, inspect the module and then the backend implementation.
5. If the bug disappears under fusion, try `GPTRS_EAGER=1` to force earlier materialization.
