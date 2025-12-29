# How-to: add models, layers, functionals, and backends

This is the contributor-oriented "recipe book" for gpt.rs. It is intentionally short and points
to concrete code locations.

If anything here stops matching the code, fix it or delete it.

## Add a model (checkpoint-loadable)

Goal: make `runtime::load_model(backend, checkpoint_path)` construct your model from a self-describing
checkpoint (`GPTRSCHK`) and expose it through the dynamic capability API used by `gpt-rs-cli`.

Checklist:

1. Add `crates/gpt-rs/src/model/<your_model>.rs`.
   - Keep the config struct in the same file (see `model/gpt.rs`, `model/resnet.rs`).
2. Implement the model struct as `struct YourModel<B: PortableBackend> { ... }`.
   - Store `Arc<B>` and submodules/layers.
   - Use NHWC/NCHW conventions explicitly (see "Layouts" below).
3. Implement `Module<B>` for parameter enumeration:
   - `visit_params` / `visit_params_mut` must produce stable ASCII parameter names.
   - Use `ParamVisitor::scoped("path", |v| ...)` to build hierarchical names.
4. Add a builder that loads tensors by name:
   - Pattern: `fn build_from_params(backend, get: &mut dyn FnMut(&str) -> Result<DeviceTensor<B>>)`.
   - `get("path.to.weight")?` returns a lazily-loaded param tensor when loaded from a checkpoint.
5. Implement `runtime::LoadedModel<B>` for your model:
   - `kind()` must match `ModelConfig.kind` stored in the checkpoint.
   - `forward(ModelInput)` returns `ModelOutput`.
   - If applicable, expose capabilities:
     - Causal LM generation: return `Some(self)` from `as_causal_lm()`.
6. Register the model factory:
   - Today this is a list in `crates/gpt-rs/src/runtime/mod.rs` (`model_factories()`).
7. Add a checkpoint exporter and baseline:
   - For Torch models, extend `scripts/export_model_weights.py` and `scripts/gptrs_eval/models/`.
   - Ensure `scripts/eval.py --model <kind> --workload validate` can run end-to-end.

## Write a layer

Goal: a reusable module that owns parameters and calls portable functionals.

Rules of thumb (see `crates/gpt-rs/src/nn/layers/linear.rs`):

- Store parameters as `DeviceTensor<B>` (usually created via `.as_param()?` / `.as_buffer()?`).
- Keep `Arc<B>` in the layer for:
  - calling non-method functionals that still take `&_backend` (e.g. `functional::relu(backend, ...)`)
  - profiling scopes
- Use `DeviceTensorOps` for math (`x.matmul(&w)?`, `x.add(&y)?`, ...).
- Avoid host materialization in forward paths (no `.to_host()?` inside layers).
- Implement `Module<B>` so parameters can be bound to stable ids (`params::bind_namespace`).

## Write a functional (portable kernel)

Goal: a backend-agnostic op that validates inputs and captures PTIR.

Pattern (see `crates/gpt-rs/src/ops/functional/*`):

1. Validate with shared helpers in `ops/functional/common.rs`:
   - dtype/rank/shape/backend checks (`ensure_same_dtype`, `ensure_rank`, ...)
2. Capture with `capture_ptir!` and return a `DeviceTensor<B>` via `CaptureIntoDeviceTensor`.
3. Annotate the public entry point:
   - `#[support_runtime_overload]` so the runtime can swap/benchmark implementations.
   - `#[ptir_pattern(...)]` if you want backends to match this lowering as a pattern.

Testing expectations:

- Add capture/shape/error-message tests under `crates/gpt-rs/tests/` when changing validation rules.
- Add numerical parity under `crates/gpt-rs-backend-tests/src/torch_parity/` when changing math.
  (See [docs/testing.md](testing.md).)

## Override a functional (custom kernel without touching model code)

There are two routes:

1. Backend-side rewrite (recommended for "fused kernel" work):
   - Add optimizer passes in your backend crate via `PortableBackend::pipeline()`.
   - Match portable lowerings using `#[ptir_pattern]`-generated views (example: C backend conv2d pass in
     `crates/gpt-rs-backend-c/src/optimizer/conv2d.rs` uses `ops::functional::conv::Conv2dPattern`).
   - Replace the matched subgraph with a `CustomCall` or a different PTIR sequence.

2. Runtime functional registry (useful for algorithmic variants):
   - `#[support_runtime_overload]` functionals are dispatched through `FunctionalRegistry<B>`.
   - `ModelConfig.runtime.functional_overrides` can force or benchmark per-op policies.
   - Override syntax lives in `crates/gpt-rs/src/ops/functional/registry.rs`:
     - `"force=<impl_name>"` or `"benchmark(cache=<N>)"`.

## Implement a new backend

Start from an existing backend crate:

- Reference interpreter: `crates/gpt-rs-backend-ref-cpu`
- Optimized CPU: `crates/gpt-rs-backend-faer`

Steps:

1. Create a new crate `crates/gpt-rs-backend-<name>`.
2. Implement `PortableBackend` in your backend type.
   - The contract is defined in [crates/gpt-rs/src/backend/spec.rs](../crates/gpt-rs/src/backend/spec.rs)
     (and summarized in [docs/backend.md](backend.md)).
3. Optional but recommended:
   - `PortableBackend::pipeline()` to inject legalization/fusion passes.
   - `PortableBackend::param_resolver()` to cache derived param representations keyed by stable ids.
4. Wire it into `gpt-rs-cli`:
   - Extend the `--backend` match in `crates/gpt-rs-cli/src/main.rs`.

## Parameter identity + streaming (why it is flexible)

Key types live in `crates/gpt-rs/src/params.rs`:

- `BaseParamId(u128)`: deterministic hash of the parameter name (`base_param_id("a.b.weight")`).
- `ModelNamespaceId(u128)`: runtime-assigned namespace so multiple models can coexist without collisions.
- `ParamKey(u128)`: stable key derived from `(namespace, base_id)`; used for caches and resolvers.
- `ParamSource<B>`: random-access source that can load a backend handle by `BaseParamId`.

What makes it "streaming":

- `runtime::load_model` builds `DeviceTensor::lazy_param(...)` handles for checkpoint weights.
- The underlying `ParamSource` (e.g. checkpoint reader) only loads a tensor when the backend needs it.
- Backends can cache derived formats (packed weights, layouts) in `PortableBackend::param_resolver()`,
  keyed by the stable param id, without leaking backend-specific code into models.

## Layouts (NCHW vs NHWC)

gpt.rs treats layout as an explicit convention in shapes:

- Many vision models accept input as NCHW (Torch convention) and immediately transpose to NHWC
  because the portable conv/pool kernels are written for NHWC (see `model/resnet.rs`).

Why this is low-friction:

- Layout conversion is just `transpose(...)` in the same lazy graph, so backends can:
  - fuse/absorb transposes into kernels via optimizer passes, or
  - execute them as explicit ops when needed.
