# Backend implementations

This directory describes how each backend in this repository executes PTIR programs.

If you are looking for the *portable* contract (op semantics, shapes/dtypes, etc), start with:
- [../backend.md](../backend.md) (PTIR backend contract, ptir.v0.4)
- [../backend_optimizer.md](../backend_optimizer.md) (optimizer/pattern system)
- [../ops.md](../ops.md) (capture/graph/execution)

## Common data flow

High level:

```text
model / layers
  -> ops::functional::* captures PTIR
  -> GraphArena builds a Program + EntrySignature
  -> optimizer runs (default + backend hooks)
  -> PortableBackend::run_program(program, inputs)
```

Parameter streaming and derived-weight caching are wired through stable `u128` ids:
- The runtime feeds parameters as *inputs* to the compiled plan by stable id (see `PortableBackend::param_resolver()`).
- Backends may memoize derived representations (packed weights, constants, etc) keyed by stable id.

## Backends in this repo

- [ref_cpu.md](ref_cpu.md): the reference portable CPU interpreter (`gpt-rs-backend-ref-cpu`).
- [faer.md](faer.md): a fused CPU backend that accelerates key ops and patterns via faer (`gpt-rs-backend-faer`).
- [c.md](c.md): PTIR -> single-file C codegen + on-disk compilation cache (`gpt-rs-backend-c`).

## Selecting a backend

- Rust CLI: `gpt-rs-cli --backend <name>` (or `GPTRS_BACKEND=<name>`).
- Python extension: `gpt_rs.load_model(..., backend="<name>")` (see `crates/gpt-rs-py/README.md`).

Notes:
- `c` backend support is feature-gated. Build `gpt-rs-cli` with `--features conversion-c` (same for `gpt-rs-py`).
- Backends self-register via static initializers in their crate (see each backend crate `src/lib.rs`).
