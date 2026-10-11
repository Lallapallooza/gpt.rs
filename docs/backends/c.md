# C backend (`gpt-rs-backend-c`)

The C backend generates a **single C translation unit** from a function of the IR, compiles it into
a shared library, and runs the generated entrypoint.

- The IR stays the portable source of truth (see [../ptir/tensor.md](../ptir/tensor.md)).
- The backend's pipeline adds a pass that lays out GEMM and grouped-convolution weights for its
  kernels, and the backend chooses its kernels as it generates C.
- Compiled libraries are cached on disk by a fingerprint of their source, the compiler, its flags
  and the host, so repeated runs are fast.

Source of truth: [../../crates/gpt-rs-backend-c/src/lib.rs](../../crates/gpt-rs-backend-c/src/lib.rs).

## Data flow

```text
recorded module (`tensor` dialect)
  -> `CBackend::pipeline`: the default passes, with `c-pack-weights` before the load-time passes
  -> kernel selection, buffer planning (static shapes and dtypes) and C codegen
  -> write program_<fp>.c
  -> compile libgpt_rs_c_<fp>.(so|dylib|dll)
  -> dlopen + call gpt_rs_c_entry(inputs, outputs)
```

Key entrypoints:
- `CBackend::compile`: generate the C source, then load the library compiled from it.
- `generate_c_module`: the C source of a function (`gpt-rs-cli convert` writes it to a file).
- `get_or_compile`: the on-disk compile cache, and `libloading` of the compiled library.
- `CBackend::run`: call the entrypoint on the input tensors and on output buffers, reused from
  earlier runs when one is free (`OutputPool`).

## Single-file codegen

The generated module is intentionally self-contained:
- It includes a small runtime (`PtirTensor` ABI, shape checks, helpers).
- It inlines a kernel library emitted from `crates/gpt-rs-backend-c/src/kernels/*` via
  [../../crates/gpt-rs-backend-c/src/kernels/mod.rs](../../crates/gpt-rs-backend-c/src/kernels/mod.rs).

Code generation lives in:
- [../../crates/gpt-rs-backend-c/src/codegen/mod.rs](../../crates/gpt-rs-backend-c/src/codegen/mod.rs)

`c-pack-weights` ([passes/pack_weights.rs](../../crates/gpt-rs-backend-c/src/passes/pack_weights.rs)) runs
before `outline-static`. It rewrites an f32 `dot_general` that goes to the packed GEMM. A reshape
and a transpose put its weight `w`, `[K, N]` or `[N, K]`, in the GEMM's panel layout
`[N / 64, K, 64]`. The dot then contracts `x` with the panels into `[.., N / 64, 64]`, and a reshape
gives that back as `[.., N]`. The `[KH * KW * C_in, C_out]` weight of a convolution, which
`merge-contracted-axes` writes, gets the same layout.

The pass packs only a static `w`, computed from parameters and constants only, whose packed form
takes no more bytes than those parameters. So it leaves `w` unpacked when N is not a multiple of
64, which would pad the panels with zeros, or when `w` widens a parameter of a narrower dtype. Other
weights stay unpacked, and the GEMM packs them on each call, as it packs activations.

At load time, `outline-static` computes the packed weight and stores it as a parameter. Its id is a
hash of the load-time function, which names the ids of the parameters it reads, so each model gets
its own packed weights.

The same pass transposes the weight of a grouped convolution from `[G, C_out / G, C_in / G, KH, KW]`
to `[KH, KW, C_in / G, G, C_out / G]`, the layout that the grouped kernel reads. `outline-static`
moves this transpose to load time too.

Codegen chooses the kernels ([codegen/select.rs](../../crates/gpt-rs-backend-c/src/codegen/select.rs)):
- Convolution: a `dot_general` that contracts the last axis of `extract_patches` with a
  `[KH * KW * C_in, C_out]` weight, packed or not, runs on the conv kernel, which builds the patches
  tile by tile and never materializes them. With AVX-512, its tasks, blocks of output pixels by
  runs of output channels, split across the team. The portable build splits output rows. The
  kernel also covers a bias add on the channel axis that follows it. That add is a conv bias or a
  folded BatchNorm shift.
- Grouped convolution, depthwise included: the patches, the dot batched over the groups, and the
  transpose and reshape back to `[N, OH, OW, C_out]` run on one grouped kernel. The kernel reads
  the input and the laid-out weight in place and splits output rows across the team. It covers a
  bias add as the conv kernel does.
- GEMM: `reads_panels` in [codegen/emit/dot.rs](../../crates/gpt-rs-backend-c/src/codegen/emit/dot.rs)
  sends a `dot_general` over a panel weight to the packed GEMM.
- Linear layers with bf16 or FP8 weights: `gpt_rs::ops::functional::linear_weight` matches a
  `dot_general` in linear layout (`x [m, k] . w [n, k]`) whose weight operand is `cast(w)` or the
  dequantized FP8 weight of `nn::Linear`, `cast(w) * broadcast_to(scale)`. The weight operand may
  also be cast to the dtype of `x`. It runs on a linear kernel when `linear_kernel` in
  [targets.rs](../../crates/gpt-rs-backend-c/src/targets.rs) has one for the dtypes of `x`, `w` and
  the scale. The kernels read the weights in place and widen them as they multiply, so no f32 or
  bf16 copy of a weight exists. The scaled kernels round each `w * scale` to the dtype of the scale,
  as `transformers` dequantizes it, and then to the dtype of `x`. bf16 x bf16 dots, projections
  included, run on the bf16 GEMM, which widens the panels it packs to f32. A single row of `x`
  against a `[K, N]` weight reads the bf16 weight in place, without packing.
- Elementwise fusion: each fused kernel starts from the last op of a chain and covers its
  single-use producers. The kernel reads `broadcast_to` and `slice` inputs in place and does not
  materialize them. When the only user of a kernel is an f32 -> bf16 `cast`, the kernel writes bf16
  directly.

An op that a kernel covers, such as the cast of a weight that a linear kernel reads in place, is
emitted only when another emitted op reads it.

Emission then outlines each op into a `static` function, with the operands renamed to parameters,
and deduplicates identical bodies. This keeps compile times low for deep models that repeat the same
layers ([../../crates/gpt-rs-backend-c/src/codegen/outline.rs](../../crates/gpt-rs-backend-c/src/codegen/outline.rs)).

## Kernels, threads, and compiler flags

- Kernels live in [../../crates/gpt-rs-backend-c/src/kernels/](../../crates/gpt-rs-backend-c/src/kernels/).
  Compiler flags and host defines live in `compiler_command()` and `host_defines()` in
  [../../crates/gpt-rs-backend-c/src/lib.rs](../../crates/gpt-rs-backend-c/src/lib.rs).
- `dot_kernel` in [codegen/emit/dot.rs](../../crates/gpt-rs-backend-c/src/codegen/emit/dot.rs) sends
  each dot whose operands are both contiguous along K to
  [kernels/linear.inc.c](../../crates/gpt-rs-backend-c/src/kernels/linear.inc.c). The exception
  is an f32 dot with at least `LINEAR_PACK_MIN_ROWS` rows, which goes to the packed GEMM. The
  packed GEMM also takes every other f32 or bf16 x bf16 dot, and its packing absorbs the operand
  strides. A bf16 x bf16 linear kernel packs its own operands for `vdpbf16ps` from half that many
  rows.
- Threads: the default is one thread per physical core. Each entry call sets the team size to
  `GPTRS_NUM_THREADS`, a define that the build computes from the physical core count. A team with
  one thread per SMT sibling stalls at every barrier whenever another process is runnable. When
  `OMP_NUM_THREADS` is set, the build leaves the define out and OpenMP's setting decides.
- Numerics: the build does not use `-ffast-math`. The vectorized `expf`, `logf`, `tanhf` and `erff`
  come from glibc's libmvec. They are accurate to within 4 ulp, but not correctly rounded.

## On-disk cache

The cache directory defaults to a temp folder and can be overridden:
- `GPTRS_C_CACHE_DIR=/path/to/dir`

Artifacts (named by the fingerprint of the source, the compiler, its flags and the host):
- `program_<fingerprint>.c`
- `libgpt_rs_c_<fingerprint>.(so|dylib|dll)`

See `c_cache_dir()` and `get_or_compile()` in
[../../crates/gpt-rs-backend-c/src/lib.rs](../../crates/gpt-rs-backend-c/src/lib.rs).

A loaded module stays loaded until the process exits, and every `CBackend` in the process shares
it. So the generated code keeps no data from one call to the next: the next call can come from
another backend or model, with other values at the same addresses.

## Profiling (C backend)

With `GPTRS_PROFILE_BACKEND=1`, the backend compiles the module with `-DGPTRS_C_PROFILE`. This wraps
every op in `GPTRS_OP_BEGIN(id)` / `GPTRS_OP_END(id)` counters, and the Rust runtime ingests the
counters into the usual profiler tables. Without the flag, the macros compile to nothing.

- Enable: `GPTRS_PROFILE_BACKEND=1`
- Implemented in: [../../crates/gpt-rs-backend-c/src/codegen/profile.rs](../../crates/gpt-rs-backend-c/src/codegen/profile.rs)

## Limitations

- Requires **static shapes** and whole-byte dtypes: f32, bf16, f8e4m3, si32, ui64 and i1 (`c_type`
  in [codegen/utils.rs](../../crates/gpt-rs-backend-c/src/codegen/utils.rs)).
- Draws only uniform f32 values (`tensor.rng_uniform`, the Philox stream of `random.inc.c`), and
  searches (`argmax`, `top_k`) only f32 tensors, which covers sampling tokens from logits.

## Build / run

The C backend is feature-gated in runners:

- CLI (build with C backend support):
  ```bash
  cargo run --release -p gpt-rs-cli --features conversion-c -- --help
  cargo run --release -p gpt-rs-cli --features conversion-c -- generate --backend c ...
  ```

- Compiler selection:
  - `CC=clang` (defaults to `cc`)
