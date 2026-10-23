# Frontend Execution Model

The gpt.rs "frontend" is the portion of the stack that *defines computation* without committing to a
particular kernel implementation:

- **Models** (`gpt_rs::model::*`) compose layers into end-to-end networks.
- **Layers** (`gpt_rs::nn::layers::*`) own parameters and control flow in their `forward` (`nn::module`).
- **Functionals** (`gpt_rs::ops::functional::*`) implement portable math from the ops of `DeviceTensor`.
- **Backends** (`gpt_rs::backend::PortableBackend`) compile and run functions of the IR.

The key idea is: layers do orchestration, functionals do portable math, and backends do execution.

## Models

Models live in `crates/gpt-rs/src/model/` and are responsible for wiring submodules and deciding what the
public outputs look like (logits, or pooled features for an image model).

For runtime usage, `gpt_rs::runtime::load_model` loads a checkpoint into a `ModelHandle`, whose
capabilities are `CausalLanguageModel` (generation) and `VisionModel` (a forward pass); see
[runtime.md](runtime.md).

## Layers

Layers live in `crates/gpt-rs/src/nn/layers/` and are declared with `#[nn::module]` (see
[howto.md](howto.md#write-a-layer)). A layer is a struct and an impl. The struct holds sublayers,
parameter tensors (`DeviceTensor<B>`) and plain configuration. The impl holds the layer's `forward`,
which does lightweight validation and orchestration. The macro makes the layer generic over the
backend `B: PortableBackend` and implements two traits:

- `Module` (`crates/gpt-rs/src/module.rs`) enumerates and updates parameters by stable name.
- `Layer`, whose `call` runs `forward` inside the profiler's layer scope, named after the type. A
  parent calls a sublayer field as a method of the same name.

Layers hold no backend handle, and functionals take none. The tensors carry the backend.

## Functionals (portable kernels)

Functionals live in `crates/gpt-rs/src/ops/functional/`, each declared with `#[functional]`:

```rust
#[functional]
pub fn gelu(x: &Tensor) -> Tensor {
    ensure_rank_at_least!(x, 1);
    0.5 * x * (1.0 + (x / 2.0f32.sqrt()).erf())
}
```

- The validation macros (`ensure_rank!`, `ensure_dtype!`, `ensure_same_shape!`, ...) check the inputs.
  They panic with a message that names the functional and the argument, for example
  `gelu: x must have rank >= 1, got []`, and the panic names the caller's line.
- The body computes with the ops of `DeviceTensor` (`crates/gpt-rs/src/tensor/ops.rs`) and with
  other functionals. Each op records one PTIR op into the lazy graph and returns a lazy
  `DeviceTensor<B>`. Binary ops and the operators `+ - * /` broadcast as Torch does, through
  explicit `broadcast_to` ops, and take an `f32` for a tensor: `x.maximum(0.0)`, `x * 0.5`.
  `slice` takes a Python-style index that `s!` writes: `x.slice(s![.., -3.., ..;2])` is
  `x[:, -3:, ::2]`, with `...`, `None` and single positions as in Python.
- The attribute adds the backend generic `B`. Tensors carry the backend, so callers write
  `functional::gelu(&x)`. The attribute also opens the profiler scope of the functional.

The macros are documented on their definitions (`crates/gpt-rs-macros/src/lib.rs`,
`crates/gpt-rs/src/ops/functional/validate.rs`).

### Errors

Tensor math returns tensors, not `Result`s. An op or functional whose inputs do not fit is a bug in
the caller, so it panics there. Loading, uploading, compiling and running return `Result`, as they
can fail at run time: `to_host()`, `DeviceTensor::from_host`, checkpoint loading.

### Faster implementations

A functional has one implementation: its portable PTIR. Backends make it fast when they compile
the recorded ops. faer and C recognize op sequences: they run the ops that `conv2d` records on
their own convolution kernels. x86 lowers every op through `tile`, `loop` and `isa` to C. See
[backends/README.md](backends/README.md).

## Backends

Backends implement `gpt_rs::backend::PortableBackend` ([backends/README.md](backends/README.md)) and live in crates like:

- `gpt-rs-backend-faer` (optimized CPU backend)
- `gpt-rs-backend-c` (generated C)
- `gpt-rs-backend-x86` (C with AVX-512 intrinsics, through the multi-level IR)
- `gpt-rs-backend-ref-cpu` (reference interpreter)

Dumps and profiles need no backend wrapper: dumps come from a trace sink that the graph arena
calls (`gpt_rs::ops::trace`), and the profiler records ambient scopes around layers, functionals,
passes and backend ops.

## Typical call flow

1. Layer calls a functional (e.g. attention, layer norm, conv2d).
2. The functional records PTIR ops into the lazy graph. Reading a result compiles the graph (or
   hits a cached plan) and runs it on the backend.
3. The backend returns its tensor handles; a `DeviceTensor` stays lazy until its value is needed.
