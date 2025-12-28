# Runtime: model loading and capability dispatch

This document describes the `gpt_rs::runtime` layer: how checkpoints are loaded into a dynamic model
handle, how runtime functional overrides are applied, and how the CLI calls into models without
hardcoding model kinds.

Source of truth: `crates/gpt-rs/src/runtime/mod.rs` and `crates/gpt-rs-cli/src/main.rs`.

## Core entrypoint: `load_model`

The canonical loader is:

- `gpt_rs::runtime::load_model(backend, checkpoint_path) -> Box<dyn LoadedModel<B>>`

The loader:

1. Opens a self-describing checkpoint (`GPTRSCHK`) and reads `ModelConfig` + tensor index.
2. Builds a `FunctionalRegistry<B>` from `ModelConfig.runtime.functional_overrides`.
3. Creates a checkpoint-backed `ParamSource<B>` for random-access parameter loads.
4. Builds a `get(name)` closure that returns `DeviceTensor::lazy_param(...)` for each parameter.
5. Selects a model factory by `ModelConfig.kind` and constructs the model using `get(name)`.
6. Wraps the model in `ModelHandle<B>`, which ensures the functional registry is installed for every call.

## Dynamic model interface (`LoadedModel`)

Models are exposed through a small dynamic trait:

- `LoadedModel<B>`: `kind()`, `forward(ModelInput) -> ModelOutput`
- Optional capabilities exposed via trait methods returning `Option<...>`:
  - `as_causal_lm() -> Option<&dyn CausalLanguageModel<B>>`

This is what makes the CLI generic: it asks the model for a capability (e.g. "causal LM") instead of
switching on an enum of model kinds.

See:
- `crates/gpt-rs/src/runtime/mod.rs` (`LoadedModel`, `ModelInput`, `ModelOutput`, `ModelHandle`)
- `crates/gpt-rs/src/inference/mod.rs` (`CausalLanguageModel`)

## Functional overrides: registry installation

Most functionals are dispatched through a `FunctionalRegistry<B>` (portable baseline by default).

`runtime::load_model` builds a registry using `ModelConfig.runtime.functional_overrides` and wraps the
inner model with `ModelHandle<B>`. `ModelHandle` calls `ops::functional::with_registry(...)` around:

- `LoadedModel::forward`
- `CausalLanguageModel` methods when the model is used as a causal LM

This means:
- model and layer code stays backend-agnostic (it calls portable functionals)
- runtime decides which implementations are active for a given model instance

## Namespacing and parameter streaming

Parameter identity is split into two layers (see `crates/gpt-rs/src/params.rs`):

- `BaseParamId(u128)`: deterministic hash of the parameter name (stable across runs).
- `ModelNamespaceId(u128)`: runtime-assigned namespace per loaded model instance.
- `ParamKey(u128)`: stable key derived from `(namespace, base_id)` used for caching/resolvers.

`load_model` picks a fresh namespace (`next_namespace()`), then for each parameter name in the checkpoint index:

- computes a `ParamKey` for the model instance
- creates `DeviceTensor::lazy_param(backend, shape, dtype, stable_id=ParamKey, base_id, source, ...)`

The `ParamSource<B>` is checkpoint-backed and loads tensors by `BaseParamId` on demand. This keeps memory
usage proportional to the set of parameters actually touched (important for sparse models like MoE).

Backends may provide a `ParamResolver` (`PortableBackend::param_resolver`) so derived parameter formats (packed
weights, layouts) can be memoized by stable id without changing model code.

## How the CLI uses runtime

`gpt-rs-cli` is capability-based:

- `generate`: requires `model.as_causal_lm()` and uses `CausalLanguageModel` (greedy/sampling + optional KV cache).
- `forward`: calls `model.forward(...)` for either token inputs or vision inputs.

See: `crates/gpt-rs-cli/src/main.rs` (`generate` / `forward` subcommands).

## Adding a new model kind (runtime wiring)

After implementing your model and `LoadedModel<B>` impl, register it with the runtime factory list:

- `crates/gpt-rs/src/runtime/mod.rs`: `model_factories()` / `model_factory(kind)`

For a full checklist (model + layer + functional + backend), see [docs/howto.md](howto.md).
