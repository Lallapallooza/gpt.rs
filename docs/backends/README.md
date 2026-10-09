# Backends

A backend compiles functions of the IR and runs them on its own tensors. The trait is
`PortableBackend` in [crates/gpt-rs/src/backend/mod.rs](../../crates/gpt-rs/src/backend/mod.rs);
its doc comments are the reference for each method. The functions hold ops of the `tensor`
dialect ([../ptir/tensor.md](../ptir/tensor.md)), after the passes of the backend's pipeline
([../ptir/passes.md](../ptir/passes.md)). [../ops.md](../ops.md) describes how the ops are
recorded and run.

## Data flow

```text
model / layers
  -> ops::functional::* record `tensor` ops into a GraphArena
  -> the arena builds a PTIR module and runs the backend's pipeline
  -> PortableBackend::compile(module, "captured"), once per distinct graph
  -> PortableBackend::run(executable, inputs)
```

## What a backend receives

The graph arena asks the backend to compile two kinds of function:

- `@captured`, the run-time function. Its arguments are the plan's inputs: run-time arguments and
  parameters, which the function's `params` attribute binds by their stable `u128` ids.
- Load-time functions, which `outline-static` creates. Each computes values from parameters alone,
  such as a packed weight, and its `result_params` attribute names the parameter each result
  becomes. The arena runs each one once per backend and keeps its results in the backend's
  parameter store (`PortableBackend::params`), where later runs read them.

A backend compiles a function once and runs it many times. The arena caches compiled functions by
the structure of the graph, so a decode step reuses the function of the step before it.

## Op semantics

Each op's semantics is its `Eval` impl in `crates/gpt-rs-ir/src/dialects/tensor/interp/`. This is
the one definition. The reference backend runs it, the filetests check every pass against it, and
a backend's kernel for an op must compute what it computes. A backend chooses its kernels when it
compiles: it recognizes op sequences, such as a convolution or a projection over a stored bf16
weight, and runs one kernel in their place.

## Tensors

Host tensors are `gpt_rs_ir::host::HostTensor`: little-endian elements in owned memory or in a view
of memory another allocation owns, such as a memory-mapped checkpoint. `upload` takes either kind.
A backend that runs on host memory keeps the view, so checkpoint weights are never copied into
process memory. A kernel that reads bf16 or FP8 weights reads them in place.

## Backends in this repo

- [ref_cpu.md](ref_cpu.md): the reference backend, which runs the IR's interpreter (`gpt-rs-backend-ref-cpu`).
- [faer.md](faer.md): a CPU backend with its own kernels for the hot ops (`gpt-rs-backend-faer`).
- [c.md](c.md): single-file C codegen with an on-disk compilation cache (`gpt-rs-backend-c`).

## Selecting a backend

- Rust CLI: `gpt-rs-cli --backend <name>` (or `GPTRS_BACKEND=<name>`).
- Python extension: `gpt_rs.load_model(..., backend="<name>")` (see `crates/gpt-rs-py/README.md`).

Both load through `gpt_rs_backends::load_model` (`crates/gpt-rs-backends`), the one place that maps
a name (`ref-cpu`, `faer`, `c`) to a backend type. The returned `ModelHandle` runs the model without
naming that type.

Notes:
- `c` backend support is feature-gated. Build `gpt-rs-cli` with `--features conversion-c` (same for `gpt-rs-py`).
