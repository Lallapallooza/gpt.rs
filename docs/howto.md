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
   `context_length`, `tie_word_embeddings`, the base model type `Model`, and `base_model`, which
   builds it.
   - A Llama-style decoder uses `DecoderModel` and builds it from the `DecoderLayout` that the
     config returns after validation (see `MinistralConfig::layout`). To add a new kind of token
     mixer, add it to `MixerConfig` and `Mixer` in `nn/layers/decoder_block.rs`, with their match
     arms.
   - A model with another Hugging Face module tree declares that tree as `#[nn::module]` structs
     (see "Write a layer" below) and implements `inference::decoder::BaseModel` for its base model
     (see `model/gpt.rs`).
3. Register `decoder::build_from_model_config::<B, YourConfig>` in `model/registry.rs`.
4. Add the eval case as in step 6 below.

`CausalDecoder` implements the rest.

Other models follow the full checklist:

1. Add `crates/gpt-rs/src/model/<your_model>.rs`.
   - Keep the config struct in the same file (see `model/resnet.rs`).
2. Declare the model and its blocks as `#[nn::module]` structs whose fields are sublayers (see
   "Write a layer" below). Mirror the module tree of the source framework, so that the field names
   are the parameter names.
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
   - The registry factory receives `get: &mut dyn FnMut(&str) -> Result<Option<DeviceTensor<B>>>`.
     `get` returns the parameter tensor of a checkpoint name, or `None` when the checkpoint has no
     such tensor. Wrap it in `Params::new`, and pass `runtime.matmul_input_dtype` to
     `with_linear_input_dtype`.
4. Implement `runtime::LoadedModel<B>` for your model:
   - `kind()` must match the checkpoint's `model_type`.
   - `forward(ModelInput)` returns `ModelOutput`. Run the model with `Layer::call`.
   - If applicable, expose capabilities:
     - Causal LM generation: return `Some(self)` from `as_causal_lm()`.
5. Register the model factory:
   - Add it to `model_factories()` in `crates/gpt-rs/src/model/registry.rs`.
6. Add an eval case:
   - Add a case class under `scripts/gptrs_eval/models/`.
   - List it in `scripts/gptrs_eval/registry.py`, with its Hugging Face repo as the `hf_model`
     default.
   - Ensure `scripts/eval.py --model <case> --workload validate` can run end-to-end.

## Write a layer

Goal: a reusable module that owns parameters and calls portable functionals. A layer is a struct
and an impl that holds its `forward`. Mark both with `#[nn::module]` (see
`crates/gpt-rs/src/nn/layers/gated_feed_forward.rs` and the macro's documentation):

```rust
#[nn::module]
pub struct GatedFeedForward {
    pub gate_proj: Linear,
    pub up_proj: Linear,
    pub down_proj: Linear,
}

#[nn::module]
impl GatedFeedForward {
    pub fn new(p: &mut Params<B>) -> Result<Self> {
        Ok(Self {
            gate_proj: Linear::new(p)?,
            up_proj: Linear::new(p)?,
            down_proj: Linear::new(p)?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = self.gate_proj(x)?;
        let up = self.up_proj(x)?;
        let hidden = functional::swiglu(&gate, &up)?;
        self.down_proj(&hidden)
    }
}
```

- The macro adds the backend parameter `B`. Write sublayer fields without it. `Tensor` is
  `DeviceTensor<B>`. Mark plain data (dimensions, epsilons, configs) with `#[module(config)]`.
- Field names are the parameter names, because the macro generates `Module<B>`. Mirror the module
  tree of the source framework. Use `#[module(rename = "A_log")]` only for names that do not follow Rust's
  snake_case field naming.
- Call a sublayer field as a method: `self.up_proj(x)`. When its `forward` takes several arguments,
  pass a tuple: `self.self_attn((x, cache, positions))`. The call runs inside the profiler scope of
  the sublayer. Run `Vec` elements and enum variants with `layer.call(args)` (`module::Layer`).
- Call functionals with tensors only (`functional::gelu(&x)`). The tensors carry the backend.
- Build layers only through their constructors and `nn::Params`, including in tests
  (`gpt_rs_backend_tests::load_layer`).
- Use `DeviceTensorOps` for math (`x.add(&y)?`, ...) and `nn::Linear` for projections (PyTorch
  `[out_features, in_features]` weights), or `nn::Conv1D` where the source module is a
  `transformers` `Conv1D` (`[in_features, out_features]` weights).
- Avoid host materialization in forward paths (no `.to_host()?` inside layers).

## Write a functional (portable kernel)

Goal: a backend-agnostic op that validates inputs and captures PTIR.

Pattern (see `crates/gpt-rs/src/ops/functional/*`):

1. Declare it with `#[functional]`. `Tensor` is `DeviceTensor<B>`, and the functional takes no
   backend argument.
2. Validate with the validation macros (`ensure_rank!`, `ensure_dtype!`, `ensure_same_shape!`, ...).
   For other checks, use `ensure!` with a message that starts with `{FUNCTIONAL}: `.
3. Capture with `capture!(|x, y| body)`. The body returns the PTIR tensor (or a tuple), and
   `capture!` returns the lazy `DeviceTensor`(s). Name the session (`capture!(session, |x| ...)`)
   when the body needs constants or `session.export(value)`. `export` keeps a value that has no
   reader, like an updated cache.
4. Name each op that a backend rewrite needs with `let`. These ops become fields of the generated
   pattern view.

Testing expectations:

- Add capture/shape/error-message tests under `crates/gpt-rs/tests/` when changing validation rules.
- Add numerical parity under `crates/gpt-rs-backend-tests/src/torch_parity/` when changing math.
  (See [docs/testing.md](testing.md).)

## Speed up a functional (custom kernel without touching model code)

- Add optimizer passes in your backend crate via `PortableBackend::pipeline()`.
- Match portable lowerings with the pattern views that `#[functional]` generates. For example, the C
  backend conv2d pass in `crates/gpt-rs-backend-c/src/optimizer/conv2d.rs` uses
  `ops::functional::conv::Conv2dPattern`.
- Replace the matched subgraph with a `CustomCall` or a different PTIR sequence.

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

- With weight streaming on (`LoadOptions.runtime.weight_streaming`),
  `runtime::load_model_with_options` builds `DeviceTensor::lazy_param(...)` handles for checkpoint
  weights.
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
