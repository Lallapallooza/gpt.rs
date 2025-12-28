# Code Map

This file is a quick pointer map for contributors. It intentionally avoids duplicating the deeper
design docs; it just tells you "where to look".

## Key directories

- `crates/gpt-rs/src/ops/functional/`: portable kernels (validate + capture).
- `crates/gpt-rs/src/ops/ptir/`: PTIR DSL (typed builder API).
- `crates/gpt-rs/src/ops/graph/`: graph arena + plan cache + optimizer hooks.
- `crates/gpt-rs/src/backend/`: PTIR spec + backend hook points.
- `crates/gpt-rs/src/nn/layers/`: layers built from functionals.
- `crates/gpt-rs/src/model/`: model assemblies (GPT, ResNet, MobileNetV2).
- `crates/gpt-rs/src/runtime/`: checkpoint loading + model capability adapters.

## Core data flow (portable op)

```
DeviceTensorOps method           support_runtime_overload wrapper
         |                                   |
         v                                   v
  capture_ptir! helper (common.rs) --> GraphArena::capture --> PtirSession graph nodes
         |                                   |
         v                                   v
CaptureIntoDeviceTensor -------> DeviceTensor::from_lazy (spec inferred from arena)
```
In practice:
- layers use `DeviceTensorOps` for ergonomic math
- functionals validate and capture PTIR
- the graph arena caches plans and runs the optimizer passes before backend execution

## Functional layer notes

Most contributors touch:
- `crates/gpt-rs/src/ops/functional/common.rs`: validation helpers + `DeviceTensorOps`
- `crates/gpt-rs/src/ops/functional/registry.rs`: runtime dispatch and overrides
- `crates/gpt-rs/src/ops/functional/runtime.rs`: thread-local registry stack

## Runtime Dispatch Sketch
```
call functional::foo(...) --> support_runtime_overload --> runtime::with_registry
                                     |                                |
                                     v                                v
                        registry::FunctionalRegistry ----> selected implementation
```
- Registries select implementations based on `FunctionalOverrides` and per-op support predicates.

## Capture Macro Guide
- Prefer `capture_ptir!({ bindings }, |session| { expression })` to build portable graphs; the macro performs
  all imports upfront, instantiates a `PtirSession`, and returns whichever value the closure produced alongside
  the active `GraphArena`. User code can destructure directly with `let (graph, value) = capture_ptir!(...) ?;`.
- Each functional follows a `validate_*`/`capture_*` pattern: validation helpers reuse `common.rs`
  utilities to check dtype/rank/shape/backend and derive metadata (axes, gather specs, cache lengths),
  while capture helpers focus purely on PTIR emission. Public entry points chain validation then
  capture to keep responsibilities separate.
- Multi-output captures (e.g., `layer_norm`) can return tuples from the closure and destructure with
  `let (graph, (a, b, c)) = capture_ptir!(...) ?;`, avoiding separate macro variants.
- Supply `graph = existing_graph;` when reusing a preallocated arena (e.g., embedding lookups that stitch
  into an existing capture); otherwise the macro resolves or allocates a graph from the tensors.
