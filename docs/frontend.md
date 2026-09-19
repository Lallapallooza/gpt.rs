# Frontend Execution Model

The gpt.rs "frontend" is the portion of the stack that *defines computation* without committing to a
particular kernel implementation:

- **Models** (`gpt_rs::model::*`) compose layers into end-to-end networks.
- **Layers** (`gpt_rs::nn::layers::*`) own parameters and control flow in their `forward` (`nn::module`).
- **Functionals** (`gpt_rs::ops::functional::*`) implement portable math that captures PTIR graphs.
- **Backends** (`gpt_rs::backend::spec::PortableBackend`) execute PTIR programs.

The key idea is: layers do orchestration, functionals do portable math, and backends do execution.

## Models

Models live in `crates/gpt-rs/src/model/` and are responsible for wiring submodules and deciding what the
public outputs look like (logits, traces, etc).

For runtime usage, `gpt_rs::runtime::load_model` loads a checkpoint into a `dyn LoadedModel<B>`, which
exposes optional "capabilities" like `CausalLanguageModel` (generation) or vision tracing.

## Layers

Layers live in `crates/gpt-rs/src/nn/layers/` and are declared with `#[nn::module]` (see
[howto.md](howto.md#write-a-layer)). A layer is a struct and an impl. The struct holds sublayers,
parameter tensors (`DeviceTensor<B>`) and plain configuration. The impl holds the layer's `forward`,
which does lightweight validation and orchestration. The macro makes the layer generic over the
backend `B: PortableBackend` and implements two traits:

- `Module` (`crates/gpt-rs/src/module.rs`) enumerates and updates parameters by stable name.
  Checkpoint tooling builds on it.
- `Layer`, whose `call` runs `forward` inside the profiler's layer scope, named after the type. A
  parent calls a sublayer field as a method of the same name.

Layers hold no backend handle, and functionals take none. The tensors carry the backend.

## Functionals (portable kernels)

Functionals live in `crates/gpt-rs/src/ops/functional/`, each declared with `#[functional]`:

```rust
#[functional]
pub fn gelu(x: &Tensor) -> Result<Tensor> {
    ensure_rank_at_least!(x, 1);
    capture!(|x| {
        let half = 0.5f32 * x;
        let erf = ptir::erf(x / ptir::sqrt(2.0f32));
        half * (1.0f32 + erf)
    })
}
```

- The validation macros (`ensure_rank!`, `ensure_dtype!`, `ensure_same_shape!`, ...) check the inputs.
  Their errors name the functional and the argument, for example `gelu: x must have rank >= 1, got []`.
- `capture!` records the PTIR of the body into the lazy graph of the operands and returns lazy
  `DeviceTensor<B>`s.
- The attribute adds the backend generic `B`. Tensors carry the backend, so callers write
  `functional::gelu(&x)`. The attribute also generates the pattern view `GeluPattern`, which backends
  rewrite to faster kernels.

The macros are documented on their definitions (`crates/gpt-rs-macros/src/lib.rs`,
`crates/gpt-rs/src/ops/functional/validate.rs`).

### `DeviceTensorOps`

`DeviceTensorOps` is an extension trait implemented for `DeviceTensor<B>`. It provides method syntax like
`a.matmul(&b)?` over the functionals, so layers stay backend-agnostic.

### Faster implementations

A functional has one implementation: its portable PTIR. Backends make it fast by rewriting the captured
ops. For example, a backend can lower `Conv2dPattern` to its own convolution kernel. See
[backend_optimizer.md](backend_optimizer.md).

## Backends

Backends implement the PTIR contract (`gpt_rs::backend::spec::PortableBackend`) and live in crates like:

- `gpt-rs-backend-faer` (optimized CPU backend)
- `gpt-rs-backend-ref-cpu` (reference interpreter)

Backends can be wrapped with hooks for dumping/profiling/debugging (see [testing.md](testing.md)).

## Typical call flow

1. Layer calls a functional (e.g. attention, layer norm, conv2d).
2. Functional captures PTIR (or hits a cached plan) and asks the backend to execute.
3. Backend returns lazy handles; materialization happens only when needed.
