# How-to: add models, layers, functionals, and backends

This is the contributor-oriented "recipe book" for gpt.rs. It is intentionally short and points
to concrete code locations.

If anything here stops matching the code, fix it or delete it.

## Add a model (checkpoint-loadable)

Goal: make `runtime::load_model(backend, model_dir)` construct your model from its Hugging Face
model directory and expose it through the dynamic capability API used by `gpt-rs-cli`. The model's
`KIND` is the Hugging Face `model_type`, and its parameter names are the Hugging Face names, so the
published checkpoint loads unchanged.

A decoder-only text model only describes its structure:

1. Add `crates/gpt-rs/src/model/<your_model>.rs` with the config struct and
   `pub type YourModel<B> = CausalDecoder<B, YourConfig>` (see `model/ministral.rs`). The config
   struct holds the keys of the Hugging Face text config that the model reads, under their Hugging
   Face names.
2. Implement `inference::decoder::DecoderConfig` for the config: `KIND`, `BASE_MODEL_PREFIX`,
   `context_length`, `tie_word_embeddings`, and `layout`, which validates the config and returns the
   `DecoderLayout` of the base model (see `MinistralConfig::layout`).
   - Implement `nn::DecoderNames` for the config too. Its defaults are the Llama family's module
     names; a family with other names overrides them (see `GptConfig`).
   - A building block that no layout can express yet goes into the shared layers: a new token mixer
     into `MixerConfig` and `Mixer` in `nn/layers/decoder_block.rs`, an attention variant into
     `nn::AttentionConfig`, an MLP variant into `nn::MlpConfig`.
3. Register `decoder::build_from_model_config::<B, YourConfig>` in `model/registry.rs`.
4. Add the eval case as in step 6 below.

`CausalDecoder` implements the rest.

Other models follow the full checklist:

1. Add `crates/gpt-rs/src/model/<your_model>.rs`.
   - Keep the config struct in the same file (see `model/resnet.rs`).
2. Declare the model and its blocks as `#[nn::module]` structs whose fields are sublayers (see
   "Write a layer" below). Mirror the module tree of the source framework, so that the field names
   are the parameter names. Build from the shared blocks, such as `nn::ConvNormActivation`, before
   you write new ones.
   - Use NHWC/NCHW conventions explicitly (see "Layouts" below).
3. Give each block a constructor `new(p: &mut nn::Params<B>, ...)` in its `#[nn::module]` impl:
   - Each field initializer of a `Self { .. }` literal reads its tensors under the field's name.
     For the field `up_proj` of a block at `model.layers.0.mlp`, `Linear::new(p)?` reads
     `model.layers.0.mlp.up_proj.weight`.
   - Shapes come from the checkpoint. Pass only what the tensors do not hold, such as a stride or
     an epsilon.
   - `p.optional(build)` builds a sublayer that the checkpoint may leave out. `p.each(items, build)`
     builds the entries of a `ModuleList`.
   - Name parameters after the modules of the source framework
     ([formats.md](formats.md#parameter-names-and-layouts)).
4. Implement `inference::vision::VisionConfig` for an image model: `KIND`, which must match the
   checkpoint's `model_type`, the model type `Model`, and `build`. `inference::vision::ImageModel`
   then loads it and runs it as one program per forward, as `runtime::LoadedModel` with the
   `as_vision()` capability.
5. Register the model factory:
   - Add `inference::vision::build_from_model_config::<B, YourConfig>` to `model_factories()` in
     `crates/gpt-rs/src/model/registry.rs`.
6. Add an eval case:
   - Add a case class under `scripts/gptrs_eval/models/`.
   - List it in `scripts/gptrs_eval/registry.py`, with its Hugging Face repo as the `hf_model`
     default.
   - Ensure `scripts/eval.py --model <case> --workload validate` can run end-to-end.

## Write a layer

Goal: a reusable module that owns parameters and calls portable functionals. A layer is a struct
and an impl that holds its `forward`. Mark both with `#[nn::module]` (see
`crates/gpt-rs/src/nn/layers/conv.rs` and the macro's documentation):

```rust
#[nn::module]
pub struct ConvNormActivation {
    pub convolution: Conv2d,
    pub normalization: BatchNorm2d,
    pub activation: Option<Activation>,
}

#[nn::module]
impl ConvNormActivation {
    pub fn new(
        p: &mut Params<B>,
        stride: usize,
        dilation: usize,
        padding: ConvPadding,
        eps: f32,
        activation: Option<ActivationFunction>,
    ) -> Result<Self> {
        Ok(Self {
            convolution: Conv2d::new(p, stride, dilation, padding)?,
            normalization: BatchNorm2d::new(p, eps)?,
            activation: activation.map(ActivationFunction::layer),
        })
    }

    fn forward(&self, x: &Tensor) -> Tensor {
        let x = self.convolution(x);
        let x = self.normalization(&x);
        self.activation(&x).unwrap_or(x)
    }
}
```

- The macro adds the backend parameter `B`. Write sublayer fields without it; it goes before any
  other generic argument (`DecoderBlock<N>` is `DecoderBlock<B, N>`). `Tensor` is
  `DeviceTensor<B>`. Mark plain data (dimensions, epsilons, configs) with `#[module(config)]`.
- Field names are the parameter names, because the macro generates `Module<B>`. Mirror the module
  tree of the source framework. Use `#[module(rename = ...)]` only for names that do not follow Rust's
  snake_case field naming (`"A_log"`) or that differ between decoder families (`nn::DecoderNames`).
- Call a sublayer field as a method: `self.up_proj(x)`. When its `forward` takes several arguments,
  pass a tuple: `self.self_attn((x, cache, positions))`. The call runs inside the profiler scope of
  the sublayer. Run `Vec` elements and enum variants with `layer.call(args)` (`module::Layer`).
- Call functionals with tensors only (`functional::gelu(&x)`). The tensors carry the backend.
- Build layers only through their constructors and `nn::Params`, including in tests
  (`gpt_rs_backend_tests::layers::load_layer`).
- `forward` returns its output, not a `Result`: tensor math panics on inputs that do not fit. A
  forward that uploads inputs (token ids, positions) returns a `Result` for the upload.
- Use the ops of `DeviceTensor` for math (`&x + &y`, `x.exp()`, ...) and `nn::Linear` for projections:
  `Linear::new` for PyTorch `[out_features, in_features]` weights, and
  `Linear::with_layout(p, WeightLayout::InOut)` where the source module is a `transformers` `Conv1D`
  (`[in_features, out_features]` weights).
- Avoid host materialization in forward paths (no `.to_host()?` inside layers).

## Write a functional (portable kernel)

Goal: a backend-agnostic op that validates inputs and records PTIR.

Pattern (see `crates/gpt-rs/src/ops/functional/*`):

1. Declare it with `#[functional]`. `Tensor` is `DeviceTensor<B>`, and the functional takes no
   backend argument.
2. Validate with the validation macros (`ensure_rank!`, `ensure_dtype!`, `ensure_same_shape!`, ...).
   For other checks, use `assert!` with a message that starts with `{FUNCTIONAL}: `.
3. Compute with the ops of `DeviceTensor` and other functionals, and return the result. Binary ops
   broadcast as Torch does and take an `f32` for a tensor (`x.maximum(0.0)`). `x.export()` keeps
   a value that has no reader, like an updated cache, as a result of the graph.

Testing expectations:

- Add tests of the exact panic messages under `crates/gpt-rs/tests/` when changing validation rules.
- Add numerical parity under `crates/gpt-rs-backend-tests/src/torch_parity/` when changing math.
  (See [docs/testing.md](testing.md).)

## Speed up a functional (custom kernel without touching model code)

- The backend chooses its kernels when it compiles a function: it recognizes the op sequence the
  functional records and runs its kernel in place of those ops. faer does this in
  `crates/gpt-rs-backend-faer/src/select.rs` and C in
  `crates/gpt-rs-backend-c/src/codegen/select.rs`. The ops a kernel covers still run when another
  op reads their results.
- When the kernel needs a parameter in another layout, add a pass before `outline-static` to the
  backend's pipeline (`PortableBackend::pipeline`). Write the layout with plain ops, as the C
  backend's `c-pack-weights` does, so `outline-static` moves the work to load time and the
  interpreter checks the pass ([ptir/passes.md](ptir/passes.md)).
- The kernel computes what the `Eval` impls of the ops it covers compute. Torch parity checks its
  numbers.

## Implement a new backend

Start from an existing backend crate:

- Reference interpreter: `crates/gpt-rs-backend-ref-cpu`
- Optimized CPU: `crates/gpt-rs-backend-faer`
- Compiled through `tile`, `loop` and `isa` to C: `crates/gpt-rs-backend-x86`

Steps:

1. Create a new crate `crates/gpt-rs-backend-<name>`.
2. Implement `PortableBackend` in your backend type.
   - The trait is in [crates/gpt-rs/src/backend/mod.rs](../crates/gpt-rs/src/backend/mod.rs), and
     [backends/README.md](backends/README.md) describes what a backend compiles and runs.
   - A backend without a kernel for an op can run it on the interpreter
     (`gpt_rs_ir::interp::eval_inst`), as faer does.
3. Optional: `PortableBackend::pipeline()` to add passes, such as a layout pass before
   `outline-static`.
4. Make it loadable by name: add it, behind a feature, to `load_model` in
   `crates/gpt-rs-backends/src/lib.rs`, which the CLI and the Python bindings both call.
5. Add `tests/backend_suite.rs` with `define_backend_tests!` for Torch parity, and the backend to
   `backends()` in `crates/gpt-rs-docs/src/main.rs`, so the reference lists its pipeline.

## Parameter identity

The loader binds each checkpoint tensor the model reads under the `ParamKey` of its name in the
model's namespace ([runtime.md](runtime.md#parameter-identity)).
A backend's passes can derive formats (packed weights, layouts) from parameters. `outline-static`
moves that work into load-time functions, which run once, and the backend's parameter store keeps
their results, without backend-specific code in models.

## Layouts (NCHW vs NHWC)

gpt.rs treats layout as an explicit convention in shapes:

- Many vision models accept input as NCHW (Torch convention) and immediately transpose to NHWC
  because the portable conv/pool kernels are written for NHWC (see `model/resnet.rs`).

Why this is low-friction:

- Layout conversion is just `transpose(...)` in the same lazy graph: `canonicalize` merges and
  removes transposes, a backend's kernels cover others, and the rest run as ops.
