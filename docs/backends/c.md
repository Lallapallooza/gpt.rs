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

Before emission, the backend pipeline ([../../crates/gpt-rs-backend-c/src/optimizer/](../../crates/gpt-rs-backend-c/src/optimizer/))
applies these rewrites, among others:
- Linear bf16 weights: the legalize stage rewrites `dot_general(x: f32 [m, k], cast(w: bf16 [n, k]))`
  into a backend-private custom call. Its kernel reads the bf16 weights and widens them in registers,
  so no f32 copy of the weights exists. See
  [optimizer/linear.rs](../../crates/gpt-rs-backend-c/src/optimizer/linear.rs). bf16 x bf16 dots
  in linear layout stay plain `dot_general`. Codegen has no kernel for other dots with bf16 operands
  and rejects them.
- Elementwise fusion: each fused kernel starts from the last op of a chain and absorbs its
  single-use producers. The kernel reads `broadcast_to` and `slice` inputs in place and does not
  materialise them. When the only user of a kernel is an f32 -> bf16 `cast`, the kernel writes bf16
  directly.

Emission then outlines each op into a `static` function, with the operands renamed to parameters,
and deduplicates identical bodies. This keeps compile times low for deep models that repeat the same
layers ([../../crates/gpt-rs-backend-c/src/codegen/outline.rs](../../crates/gpt-rs-backend-c/src/codegen/outline.rs)).

## Kernels, threads, and compiler flags

- Kernels live in [../../crates/gpt-rs-backend-c/src/kernels/](../../crates/gpt-rs-backend-c/src/kernels/).
  Compiler flags and host defines live in `compiler_command()` and `host_defines()` in
  [../../crates/gpt-rs-backend-c/src/lib.rs](../../crates/gpt-rs-backend-c/src/lib.rs).
- `dot_kernel` in [codegen/emit/dot.rs](../../crates/gpt-rs-backend-c/src/codegen/emit/dot.rs) sends
  each dot whose operands are both contiguous along K to
  [kernels/linear.inc.c](../../crates/gpt-rs-backend-c/src/kernels/linear.inc.c). The exception is
  an f32 dot with at least `LINEAR_PACK_MIN_ROWS` rows. It goes to the packed GEMM with every other
  f32 dot, and the GEMM packing absorbs the operand strides.
- Threads: the default is one thread per physical core. Each entry call sets the team size to
  `GPTRS_NUM_THREADS`. A team with one thread per SMT sibling stalls at every barrier whenever
  another process is runnable. `OMP_NUM_THREADS` overrides the default.
- Numerics: the build does not use `-ffast-math`. The vectorised `expf`, `logf`, `tanhf` and `erff`
  come from glibc's libmvec. They are accurate to within 4 ulp, but not correctly rounded.

## On-disk cache

The cache directory defaults to a temp folder and can be overridden:
- `GPTRS_C_CACHE_DIR=/path/to/dir`

Artifacts (names are stable per converted program fingerprint):
- `program_<fingerprint>.c`
- `libgpt_rs_c_<fingerprint>.(so|dylib|dll)`

See `c_cache_dir()` and `CBackend::get_or_compile()` in
[../../crates/gpt-rs-backend-c/src/lib.rs](../../crates/gpt-rs-backend-c/src/lib.rs).

## Profiling (C backend)

With `GPTRS_PROFILE_BACKEND=1`, the backend compiles the module with `-DGPTRS_C_PROFILE`. This wraps
every op in `GPTRS_OP_BEGIN(id)` / `GPTRS_OP_END(id)` counters, and the Rust runtime ingests the
counters into the usual profiler tables. Without the flag, the macros compile to nothing.

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
