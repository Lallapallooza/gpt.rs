# Frontend Execution Model

The gpt.rs "frontend" is the portion of the stack that *defines computation* without committing to a
particular kernel implementation:

- **Models** (`gpt_rs::model::*`) compose layers into end-to-end networks.
- **Layers** (`gpt_rs::nn::layers::*`) own parameters and control flow and expose ergonomic `forward` helpers.
- **Functionals** (`gpt_rs::ops::functional::*`) implement portable math that captures PTIR graphs.
- **Backends** (`gpt_rs::backend::spec::PortableBackend`) execute PTIR programs.

The key idea is: layers do orchestration, functionals do portable math, and backends do execution.

## Models

Models live in `crates/gpt-rs/src/model/` and are responsible for wiring submodules and deciding what the
public outputs look like (logits, traces, etc).

For runtime usage, `gpt_rs::runtime::load_model` loads a checkpoint into a `dyn LoadedModel<B>`, which
exposes optional "capabilities" like `CausalLanguageModel` (generation) or vision tracing.

## Layers

Layers live in `crates/gpt-rs/src/nn/layers/` and typically contain:

- `Arc<B>` for some backend `B: PortableBackend`
- parameter tensors (`DeviceTensor<B>`) and buffers
- lightweight validation + orchestration logic

Layers also implement the small `Module` trait (`crates/gpt-rs/src/module.rs`) so parameters can be
enumerated/updated by stable name (this is what checkpoint tooling and future training utilities build on).

Automatic differentiation is not implemented yet; some layers expose `*_with_state` helpers that return
the minimal forward state that a future derivative pass would need.

## Functionals (portable kernels)

Functionals live in `crates/gpt-rs/src/ops/functional/`. They follow a consistent pattern:

- validate inputs (dtype/shape/backend invariants)
- capture PTIR graphs via `capture_ptir!` / `PtirSession`
- return `DeviceTensor<B>` results (often lazily executed by the backend)

Most public functionals are annotated with `#[support_runtime_overload]`, which wires them into the runtime
registry/override system.

### `DeviceTensorOps`

`DeviceTensorOps` is an extension trait implemented for `DeviceTensor<B>`. It provides method syntax like
`a.matmul(&b)?` while still routing through the functional implementations (so layers stay backend-agnostic).

## Runtime overrides (swapping implementations)

A `FunctionalRegistry<B>` selects an implementation per functional family (portable reference by default).

There are two supported override routes:

1. **From checkpoint config**: `ModelConfig.runtime.functional_overrides` can request specific implementations
   by name; `runtime::load_model` installs a registry configured by those overrides.
2. **Programmatic**: callers can build a registry and install it for a scope:

```rust
let registry = std::sync::Arc::new(gpt_rs::ops::functional::FunctionalRegistry::<B>::default());
let _guard = gpt_rs::ops::functional::runtime::push_registry(registry);
// run model forward here
```

If you are adding a new overrideable functional family, keep the contract strict: validate shape/dtype in the
portable path, and ensure custom implementations only use backend primitives (so correctness tooling still applies).

## Backends

Backends implement the PTIR contract (`gpt_rs::backend::spec::PortableBackend`) and live in crates like:

- `gpt-rs-backend-faer` (optimized CPU backend)
- `gpt-rs-backend-ref-cpu` (reference interpreter)

Backends can be wrapped with hooks for dumping/profiling/debugging (see [testing.md](testing.md)).

## Typical call flow

1. Layer calls a functional (e.g. attention, layer norm, conv2d).
2. Functional captures PTIR (or hits a cached plan) and asks the backend to execute.
3. Backend returns lazy handles; materialization happens only when needed.
