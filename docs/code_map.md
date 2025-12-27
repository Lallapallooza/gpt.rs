# Code Map

## Functional Layer Overview
- Location: `crates/gpt-rs/src/ops/functional`.
- Goal: expose backend-portable tensor kernels implemented via graph capture.
- Entry points: `mod.rs` re-exports modules so callers import `gpt_rs::ops::functional::*`.

## Core Data Flow
```
DeviceTensorOps method           support_runtime_overload wrapper
         |                                   |
         v                                   v
  capture_ptir! helper (common.rs) --> GraphArena::capture --> PtirSession graph nodes
         |                                   |
         v                                   v
CaptureIntoDeviceTensor -------> DeviceTensor::from_lazy (spec inferred from arena)
```
- `DeviceTensorOps` routes arithmetic through `capture_ptir!`, so call sites never touch import boilerplate.
- `GraphArena` retains lazy programs while `PtirSession` emits portable IR nodes for backend execution.
- `CaptureIntoDeviceTensor` looks up the recorded `TensorSpec` and then calls `DeviceTensor::from_lazy`, keeping gradient flags intact.

## Module Notes
- `activation.rs`: unary kernels like `softmax_last_dim` and `gelu`.
- `attention.rs`: owns cache structs, the portable forward kernel, concat/slice helpers, and cache-key
  traits used by the registry.
- `common.rs`: central capture helpers, backend validation (e.g., `ensure_same_dtype`, `ensure_rank`,
  `ensure_shape_matches`), and the `DeviceTensorOps` trait.
- `common.rs` also defines `CaptureIntoDeviceTensor`, enabling `(graph, ValueId)` pairs produced by
  `capture_ptir!` to materialise as `DeviceTensor` instances without manually threading shape/dtype.
- `embedding.rs`: builds backend `Gather` programs with optional arena reuse to avoid redundant captures.
- `linalg.rs`: wraps dot-general matmul variants (2D and batched 3D inputs) behind `#[support_runtime_overload]`.
- `normalization.rs`: implements `layer_norm`, returning intermediates so downstream consumers can reuse mean and `1/std`.
- `portable_utils.rs`: bridges frontend shapes/dtypes/literals into backend descriptors; currently retains a few
  `#[allow(dead_code)]` helpers for upcoming kernels.
- `registry.rs` / `runtime.rs`: host the runtime override stack, benchmark cache, and functional registry plumbing.
- `stochastic.rs`: RNG-backed operators like `dropout`, composed from scalar broadcasts and PTIR selects.
- `tensor_ops.rs`: small helpers built atop memoized binary captures (e.g., `add_bias`), with validation
  delegated to `common.rs`.
- Autograd scaffolding is temporarily removed while we stabilise the forward capture pipeline; future gradient
  support will reintroduce a dedicated module once the PTIR backprop story solidifies.

## Audit Checklist (2025-10-24)
- Verified every file under `crates/gpt-rs/src/ops/functional/` avoids host tensor materialization, panics,
  and disallowed constructors.
- Ensured all public forward helpers use `#[support_runtime_overload]` and return `Result<DeviceTensor<_>>`.
- Confirmed `attention.rs` no longer suppresses `clippy::disallowed_types` and relies on runtime errors instead.
- Audited `registry.rs`/`runtime.rs` for consistent mutex usage; `unwrap()` calls are safe due to prevalidated inputs.
- Checked RNG and broadcast helpers (`stochastic.rs`, `common.rs`) for scalar literal creation via PTIR rather than
  host tensors.
- Noted remaining follow-up: implement captured derivatives for GELU and attention once PTIR backprop support is ready.

## Runtime Dispatch Sketch
```
call functional::foo(...) --> support_runtime_overload --> runtime::with_registry
                                     |                                |
                                     v                                v
                        registry::FunctionalRegistry ----> selected implementation
```
- `support_runtime_overload` instruments functions so registries can swap implementations.
- Registries benchmark, force, or default per functional key based on overrides.

## Capture Macro Guide
- Prefer `capture_ptir!({ bindings }, |session| { expression })` to build portable graphs; the macro performs
  all imports upfront, instantiates a `PtirSession`, and returns whichever value the closure produced alongside
  the active `GraphArena`. User code can destructure directly with `let (graph, value) = capture_ptir!(...) ?;`.
  `(graph, value)` pairs implement `CaptureIntoDeviceTensor`, which consults the arena for the recorded `TensorSpec`
  so call sites can simply invoke `.into_device_tensor(requires_grad)` without threading shape or dtype manually.
  Optional debug assertions remain where we still derive closed-form shapes (e.g., `matmul`) so mismatches are
  surfaced in tests.
- Each functional follows a `validate_*`/`capture_*` pattern: validation helpers reuse `common.rs`
  utilities to check dtype/rank/shape/backend and derive metadata (axes, gather specs, cache lengths),
  while capture helpers focus purely on PTIR emission. Public entry points chain validation then
  capture to keep responsibilities separate.
- Multi-output captures (e.g., `layer_norm`) can return tuples from the closure and destructure with
  `let (graph, (a, b, c)) = capture_ptir!(...) ?;`, avoiding separate macro variants.
- Supply `graph = existing_graph;` when reusing a preallocated arena (e.g., embedding lookups that stitch
  into an existing capture); otherwise the macro resolves or allocates a graph from the tensors.

## Follow-Ups
- Expand this map with backend-specific overrides when we document device kernels.
