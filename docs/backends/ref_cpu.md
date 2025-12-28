# Reference CPU backend (`gpt-rs-backend-ref-cpu`)

The reference CPU backend is the simplest PTIR executor in this repo:
- It directly **interprets PTIR instructions** in program order.
- It is the "spec bring-up" backend: correctness first, performance last.
- Other backends (like `faer`) reuse its semantics and selectively fuse/override hot ops.

Source of truth:
- [../../crates/gpt-rs-backend-ref-cpu/src/lib.rs](../../crates/gpt-rs-backend-ref-cpu/src/lib.rs)
- [../../crates/gpt-rs-backend-ref-cpu/src/cpu.rs](../../crates/gpt-rs-backend-ref-cpu/src/cpu.rs)

## Data flow

```text
PTIR Program
  -> GenericCpuBackend::run_program
       - map entry params -> tensors
       - for each instruction:
           - materialize literal operands
           - execute op
           - store result by ValueId
       - collect entry results
```

This backend implements `PortableBackend` and supports:
- `materialize(TensorInit)`: host literal / zero init into a `CpuTensor`.
- `execute_instruction`: execute one instruction (used by the graph arena executor).
- `run_program`: execute the whole program (used by some drivers).

## Kernel interception (extension point)

`GenericCpuBackend<I>` accepts a `CpuKernelInterceptor`:
- `NoopInterceptor` (default) executes everything via the reference implementation.
- Other backends can install an interceptor to override specific ops while keeping
  the fallback semantics stable (see [faer.md](faer.md)).

## Parameter resolver

The backend exposes an `InMemoryParamResolver<CpuTensor>` via `PortableBackend::param_resolver()`.
This is where stable `u128` keys can map to already-materialized parameters or derived values.

## Backend registry names

This crate registers the backend under:
- `cpu`
- `cpu-portable`

See `register_cpu_backend()` in
[../../crates/gpt-rs-backend-ref-cpu/src/lib.rs](../../crates/gpt-rs-backend-ref-cpu/src/lib.rs).
