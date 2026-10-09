# Code Map

This file is a quick pointer map for contributors. It intentionally avoids duplicating the deeper
design docs; it just tells you "where to look".

## Key directories

- `crates/gpt-rs/src/tensor/`: `DeviceTensor` and its ops (`ops.rs`), each recording one PTIR op.
- `crates/gpt-rs/src/ops/functional/`: portable kernels (validate, then record ops).
- `crates/gpt-rs/src/ops/graph/`: graph arena + program cache.
- `crates/gpt-rs/src/backend/`: the `PortableBackend` trait and the parameter store.
- `crates/gpt-rs/src/nn/layers/`: layers built from functionals. Each layer is declared with
  `#[nn::module]` (`crates/gpt-rs-macros/src/nn_module.rs`) and built from checkpoint tensors
  through `nn::Params` (`nn/params.rs`). The `Module` and `Layer` traits live in
  `crates/gpt-rs/src/module.rs`.
- `crates/gpt-rs/src/model/`: model configs and assemblies. `registry.rs` maps each checkpoint
  `kind` to its builder.
- `crates/gpt-rs/src/inference/`: the shared causal decoder (`decoder.rs`), KV caches, generation
  with sampling on the device (`generate.rs`), and the image models (`vision.rs`).
- `crates/gpt-rs/src/runtime/`: checkpoint loading + model capability adapters.
- `crates/gpt-rs-backends/`: the backends of a build, chosen by name (`load_model`).

## Core data flow (portable op)

```
functional::foo(&x)                    (#[functional])
         |
         v
  validation macros (ensure_rank!, ensure_dtype!, ...)
         |
         v
  x.exp(), &x + &y, ...  --> GraphArena: one node per op, typed by the op's type rule
         |
         v
  lazy DeviceTensor (DeviceTensor::from_lazy)
```
In practice:
- layers call functionals and use the ops of `DeviceTensor` for elementwise math
- functionals validate, then record PTIR ops
- the graph arena caches plans and runs the optimizer passes before backend execution

## Functional layer notes

- `#[functional]` is documented on its definition in `crates/gpt-rs-macros/src/lib.rs`. The
  validation macros are documented in `crates/gpt-rs/src/ops/functional/validate.rs`.
- Each `#[functional]` runs inside its own profiler scope, named after the function.
