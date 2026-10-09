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

### Eager debugging mode

Setting `GPTRS_EAGER=1` forces tensors created from recorded nodes to flush immediately (useful for
debugging functionals in "eager-like" mode).

## Tracing and dumps

The ops layer exposes hooks around compilation and execution via `crates/gpt-rs/src/ops/trace/`:

- `ExecutionTraceSink`: callbacks before and after each executed program, with the module it was
  compiled from, and `on_pass(&PassRun, &Module)` after each pass with the pass's name, options,
  whether it changed the module and its statistics.
- `FileTraceSink`: writes each module as `program.ptir` with `meta.json` and `stats.json`, the
  runs to `executions.jsonl`, the module after each pass under `passes/`, and, with
  `GPTRS_PASS_STATS=1`, the pass events to `passes.jsonl`.

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
