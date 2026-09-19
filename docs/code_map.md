# Code Map

This file is a quick pointer map for contributors. It intentionally avoids duplicating the deeper
design docs; it just tells you "where to look".

## Key directories

- `crates/gpt-rs/src/ops/functional/`: portable kernels (validate + capture).
- `crates/gpt-rs/src/ops/ptir/`: PTIR DSL (typed builder API).
- `crates/gpt-rs/src/ops/graph/`: graph arena + plan cache + optimizer hooks.
- `crates/gpt-rs/src/backend/`: PTIR spec + backend hook points.
- `crates/gpt-rs/src/nn/layers/`: layers built from functionals. Each layer is declared with
  `#[nn::module]` (`crates/gpt-rs-macros/src/nn_module.rs`) and built by `nn::LayerLoader`
  (`nn/loader.rs`). The `Module` and `Layer` traits live in `crates/gpt-rs/src/module.rs`.
- `crates/gpt-rs/src/model/`: model configs and assemblies. `registry.rs` maps each checkpoint
  `kind` to its builder.
- `crates/gpt-rs/src/inference/`: the shared causal decoder (`decoder.rs`), KV caches, generation
  and sampling.
- `crates/gpt-rs/src/runtime/`: checkpoint loading + model capability adapters.

## Core data flow (portable op)

```
functional::foo(&x) / x.add(&y)       (#[functional], DeviceTensorOps)
         |
         v
  validation macros (ensure_rank!, ensure_dtype!, ...)
         |
         v
  capture!(|x| ...) --> GraphArena::capture --> PtirSession graph nodes
         |
         v
  lazy DeviceTensor(s) (DeviceTensor::from_lazy, spec inferred from the arena)
```
In practice:
- layers call functionals and use `DeviceTensorOps` for elementwise math
- functionals validate and capture PTIR
- the graph arena caches plans and runs the optimizer passes before backend execution

## Functional layer notes

- `#[functional]` and `capture!` are documented on their definitions in
  `crates/gpt-rs-macros/src/lib.rs`. The validation macros are documented in
  `crates/gpt-rs/src/ops/functional/validate.rs`.
- Each `#[functional]` also generates a pattern view, for example `GeluPattern` for `gelu` with the
  target `gpt_rs.gelu`. Backends match these views with `OpRewritePattern<View>` and rewrite them to
  faster kernels. See [backend_optimizer.md](backend_optimizer.md). `gpt-rs-cli patterns` lists the
  views.
