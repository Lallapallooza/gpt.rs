# Faer backend (`gpt-rs-backend-faer`)

The faer backend is a **fused CPU backend**:
- It keeps the reference CPU backend semantics as a fallback.
- It accelerates a subset of hot ops/patterns (matmul, conv patterns, etc).

Source of truth:
- [../../crates/gpt-rs-backend-faer/src/lib.rs](../../crates/gpt-rs-backend-faer/src/lib.rs)
- [../../crates/gpt-rs-backend-faer/src/optimizer.rs](../../crates/gpt-rs-backend-faer/src/optimizer.rs)

## Data flow

```text
PTIR Program
  -> optimizer pipeline (FaerPipeline) rewrites known patterns to CustomCall
  -> FaerPortableBackend::run_program interprets the entry function
       - fast paths for known fusions
       - otherwise fall back to GenericCpuBackend::execute_instruction
           - interceptor tries to execute selected ops (faer matmul, custom calls, ...)
           - otherwise uses the reference CPU kernel for the op
```

## Optimizer pipeline

`PortableBackend::pipeline()` returns `FaerPipeline`, which installs a legalization pass that
rewrites common portable patterns to `Operation::CustomCall` with a stable target string.

Example targets:
- `gpt_rs.faer.conv2d.nhwc.f32.v1`
- `gpt_rs.faer.depthwise_conv2d.nhwc.f32.v1`

See:
- `FaerPipeline` / `FaerCustomCallFusionPass` in
  [../../crates/gpt-rs-backend-faer/src/optimizer.rs](../../crates/gpt-rs-backend-faer/src/optimizer.rs)

## Execution strategy

Two layers participate:

1) `FaerPortableBackend` (outer)
- Runs the program and recognizes a few multi-op sequences directly (fusion-style evaluation).

2) `GenericCpuBackend<FaerCpuInterceptor>` (inner)
- Executes single PTIR ops.
- Uses `FaerCpuInterceptor` to override specific operations:
  - `DotGeneral` (f32 matmul paths) via faer
  - selected elementwise/transpose/reduce-window cases
  - `CustomCall` targets produced by the optimizer pipeline

## Derived parameter cache

The backend provides `PortableBackend::param_resolver()` which currently caches `CpuTensor`s by
stable `u128` key. This is the intended place to memoize derived representations (packed weights,
constants, etc) without changing model code.

See `FaerDerivedParamResolver` in
[../../crates/gpt-rs-backend-faer/src/lib.rs](../../crates/gpt-rs-backend-faer/src/lib.rs).
