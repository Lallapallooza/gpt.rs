# Ops layer: PTIR capture, graphs, and execution

This document describes the `gpt_rs::ops` layer: how portable functionals capture PTIR graphs, how graphs
are cached/optimized, and how backends execute PTIR programs.

If you are looking for the *spec*, see [backend.md](backend.md).

## Where things live

- `crates/gpt-rs/src/ops/functional/`: portable kernels (validate + capture).
- `crates/gpt-rs/src/ops/ptir/`: PTIR DSL (`PtirSession`, `Tensor<'ctx, ...>`, op builders).
- `crates/gpt-rs/src/ops/graph/`: graph arena + plan cache + optimizer pipeline.
- `crates/gpt-rs/src/ops/trace/`: dumping/profiling hooks around PTIR execution.
- `crates/gpt-rs/src/tensor/device_tensor.rs`: lazy handles and materialization triggers.

## Validate + capture pattern (functionals)

Every functional is a `#[functional]` with the same structure:

1. Validate shape/dtype/backend invariants with the validation macros (`ensure_rank!`, `ensure_dtype!`, ...).
2. Capture PTIR with `capture!` (emits graph nodes and returns lazy `DeviceTensor`s).

See [frontend.md](frontend.md) for an example.

## Lazy execution model

`DeviceTensor<B>` is a thin wrapper around a backend and a *lazy handle*.

The handle can represent:
- an argument tensor already materialized on the backend
- a parameter reference (stable id + `ParamSource<B>`, loaded on demand)
- a node in a `GraphArena<B>` (a captured PTIR value id)

Materialization happens when caller code needs the underlying data (for example via `.to_host()`), or when a
downstream op forces execution. The graph arena groups requested values, compiles a PTIR `Program`, runs the
optimizer pipeline, and executes via the backend.

### Eager debugging mode

Setting `GPTRS_EAGER=1` forces tensors created from captured nodes to flush immediately (useful for debugging
functionals in "eager-like" mode).

## Tracing and dumps

The ops layer exposes hooks around PTIR execution via `crates/gpt-rs/src/ops/trace/`:

- `ExecutionTraceSink`: callbacks before/after each executed PTIR program, plus optional pass events.
- `FileTraceSink`: writes `.ptir` and `.json` artifacts to a directory.

`gpt-rs-cli` wires this up via:
- `--dump-dir <DIR>` and `--dump-mode all|compile`

See [testing.md](testing.md) for CLI usage examples.

## How to debug a bad kernel

Suggested loop:

1. Reproduce with `gpt-rs-cli --dump-dir ...` and keep the dumped `.ptir` + `.json`.
2. If debugging shape rules, focus on the functional validation (the `ensure_*!` macros in `ops/functional/validate.rs`).
3. If debugging lowering/execution, inspect the PTIR program and then the backend implementation.
4. If the bug disappears under fusion, try `GPTRS_EAGER=1` to force earlier materialization.
