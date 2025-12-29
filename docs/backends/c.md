# C backend (`gpt-rs-backend-c`)

The C backend converts a PTIR `Program` into a **single C translation unit**, compiles it into a
shared library, and executes the generated entrypoint.

This is useful as a "portable codegen" backend:
- PTIR stays the portable source of truth (see [../backend.md](../backend.md)).
- Backends can still run optimizer passes and fuse patterns before lowering to C.
- Generated artifacts are cached on disk so repeated runs are fast.

Source of truth: [../../crates/gpt-rs-backend-c/src/lib.rs](../../crates/gpt-rs-backend-c/src/lib.rs).

## Data flow

```text
PTIR Program
  -> legality + buffer planning (static shapes/dtypes)
  -> optimizer (backend pipeline)
  -> PTIR -> C conversion (codegen)
  -> write program_<fp>.c
  -> compile libgpt_rs_c_<fp>.(so|dylib|dll)
  -> dlopen + call entrypoint(inputs, outputs)
```

Key entrypoints:
- `CBackend::run_program`: convert + compile + execute.
- `CConversionTarget::convert`: PTIR -> `ConvertedIr { module: String, entrypoints }`.
- `CBackend::get_or_compile`: on-disk compile cache + `libloading` of the compiled module.

## Single-file codegen

The generated module is intentionally self-contained:
- It includes a small runtime (`PtirTensor` ABI, shape checks, helpers).
- It inlines a kernel library emitted from `crates/gpt-rs-backend-c/src/kernels/*` via
  [../../crates/gpt-rs-backend-c/src/kernels/mod.rs](../../crates/gpt-rs-backend-c/src/kernels/mod.rs).

Code generation lives in:
- [../../crates/gpt-rs-backend-c/src/codegen/mod.rs](../../crates/gpt-rs-backend-c/src/codegen/mod.rs)

## On-disk cache

The cache directory defaults to a temp folder and can be overridden:
- `GPTRS_C_CACHE_DIR=/path/to/dir`

Artifacts (names are stable per converted program fingerprint):
- `program_<fingerprint>.c`
- `libgpt_rs_c_<fingerprint>.(so|dylib|dll)`

See `c_cache_dir()` and `CBackend::get_or_compile()` in
[../../crates/gpt-rs-backend-c/src/lib.rs](../../crates/gpt-rs-backend-c/src/lib.rs).

## Profiling (C backend)

When `GPTRS_PROFILE_BACKEND=1`, the generated module exports per-op counters and the Rust runtime
ingests them into the usual profiler tables.

- Enable: `GPTRS_PROFILE_BACKEND=1`
- Implemented in: [../../crates/gpt-rs-backend-c/src/codegen/profile.rs](../../crates/gpt-rs-backend-c/src/codegen/profile.rs)

## Limitations (current)

The C backend is strict today:
- Requires **static shapes** and **known dtypes** (buffer planning rejects dynamic dims).
- Accepts a limited dtype set for execution (see `c_legality_spec()`).
- `execute_instruction` is not implemented; only `run_program` is supported.

## Build / run

The C backend is feature-gated in runners:

- CLI (build with C backend support):
  ```bash
  cargo run --release -p gpt-rs-cli --features conversion-c -- --help
  cargo run --release -p gpt-rs-cli --features conversion-c -- generate --backend c ...
  ```

- Compiler selection:
  - `CC=clang` (defaults to `cc`)
