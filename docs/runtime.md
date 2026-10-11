# Runtime: model loading and capability dispatch

This document describes the `gpt_rs::runtime` layer: how checkpoints are loaded into a dynamic model
handle and how the CLI calls into models without hardcoding model kinds.

Source of truth: `crates/gpt-rs/src/runtime/`, `crates/gpt-rs-backends/` and
`crates/gpt-rs-cli/src/main.rs`.

## Core entrypoint: `load_model`

The canonical loader is:

- `gpt_rs::runtime::load_model(backend, model_dir) -> ModelHandle`

`backend` is an `Arc` of a concrete backend type, and the handle runs the model on it without
naming that type. `gpt_rs_backends::load_model(name, model_dir, options)` picks the backend by
name instead: `ref-cpu` always, `faer` and `c` with the features of those names. The CLI and the
Python bindings load through it; the CLI always has faer, and both enable C with their
`conversion-c` feature.

The loader:

1. Opens the checkpoint, a Hugging Face model directory ([formats.md](formats.md)), and reads its
   `ModelConfig`: the kind (`model_type`), the config, and the end-of-sequence ids.
2. Builds a `get(name)` closure. It returns the checkpoint tensor of that name as a parameter, and
   `None` for a name the checkpoint does not have.
3. Selects a model factory by `ModelConfig.kind` and constructs the model using `get(name)`. Models
   build their layers through `nn::Params`, which reads each tensor by its parameter name. Shapes
   come from the checkpoint, and the functionals check them when the model runs. For names and
   layouts, see [formats.md](formats.md#parameter-names-and-layouts).
4. Wraps the model in a `ModelHandle` with the checkpoint's `eos_token_ids` and the runtime
   settings it runs with.

`load_model_with_options(backend, path, options)` is the general form. `LoadOptions` in
`crates/gpt-rs/src/runtime/loader.rs` holds the parameter namespace, plus runtime settings
(`ModelRuntimeConfig`: `matmul_input_dtype`) and config keys, both of which replace what the
checkpoint declares. The keys are those of the model's config struct, so `num_hidden_layers`
truncates a Ministral or Qwen3.5 model and `n_layer` truncates a GPT-2 model.

`nn::Params` passes `matmul_input_dtype` to every projection layer (`nn::Linear`) it builds. With
`bf16`, each layer casts its input to bf16 before `functional::linear`, which multiplies with f32
accumulation and f32 outputs. Norms, attention and the recurrent state stay in f32. These numerics
differ from a PyTorch bf16 model, which also rounds matmul outputs and residuals to bf16. A
checkpoint whose config declares bf16 (`dtype`, or `torch_dtype` in configs saved before
transformers 4.56) gets `bf16` by default. Any other checkpoint keeps f32 inputs, and the matmul
widens bf16 weights to f32 exactly. This includes an f16 checkpoint, since f16 activations can
overflow. The CLI's `--matmul-input-dtype` and the Qwen3.5 eval case's flag override it.

## Dynamic model interface (`LoadedModel`)

Models are exposed through a small dynamic trait, which does not name the backend type:

- `LoadedModel`: `kind()` and the capabilities, trait methods that return `Option<...>`:
  - `as_causal_lm() -> Option<&dyn CausalLanguageModel>`: token sequences in, logits out.
  - `as_vision() -> Option<&dyn VisionModel>`: NCHW pixels in, class logits out.

`ModelHandle::causal_lm()` and `ModelHandle::vision()` return a capability or an error that names
the model kind.

`CausalLanguageModel::forward` returns logits `[T, vocab]` for every position. It computes them as
a prefill into fresh caches. `CausalLanguageModel::decode` prefills a prompt and returns a `Decode`,
the state that produces the tokens after it one at a time. `inference::generate::Generator` drives
it: each `step` returns one token, and the generation ends on a stop token, after the requested
number of tokens, or when it fills the context length.

The model picks each token on the device, as `inference::generate::Sampling` says: the most likely
token (`argmax`), or a draw from the softmax of the logits divided by the temperature, over the
`top_k` most likely tokens when that is set. The draw adds Gumbel noise from a Philox generator
state that the decode carries from step to step (`functional::sample`). A step downloads one token
id, not the logits.

End-of-sequence tokens are checkpoint metadata, not a model capability. The checkpoint reader stores
the `eos_token_id` values of `generation_config.json` and `config.json` in
`ModelConfig.eos_token_ids`, and
`ModelHandle::eos_token_ids()` returns them. `gpt-rs-cli generate` and the Python `generate_tokens`
stop after one of these tokens. `--ignore-eos` and `ignore_eos=True` turn this off.

This is what makes the CLI generic: it asks the model for a capability (e.g. "causal LM") instead of
switching on an enum of model kinds.

See:
- `crates/gpt-rs/src/runtime/handle.rs` (`LoadedModel`, `ModelHandle`)
- `crates/gpt-rs/src/inference/mod.rs` (`CausalLanguageModel`, `Decode`) and
  `inference/generate.rs` (`Generator`, `Sampling`)
- `crates/gpt-rs/src/inference/vision.rs` (`VisionModel`)

## Decoder models (`CausalDecoder`)

Every text model is an `inference::decoder::CausalDecoder<B, C>`. Its config type `C` holds the keys
of the Hugging Face text config that the model reads, and implements `DecoderConfig`
(`inference/decoder.rs`).

Like a Hugging Face `*ForCausalLM`, `CausalDecoder` holds the base model, under `BASE_MODEL_PREFIX`,
and `lm_head`. Every model shares the base model `DecoderModel`: the token embedding, optional
learned position embeddings, the layers and the final norm. A model's config gives its layout
(`DecoderLayout`): the norm kind, the token mixer of each layer, the MLP, the rotary embedding and
whether positions are learned. It also gives its module names (`nn::DecoderNames`), which default to
the Llama family's, so GPT-2's `transformer.h.{i}.attn.c_attn` and Ministral's
`model.layers.{i}.self_attn.q_proj` come from the same modules. Each `nn::DecoderBlock` is a pre-norm
residual block with a token mixer (`self_attn` or `linear_attn`) and a gated or plain MLP
(`nn::FeedForward`). `CausalDecoder` implements `LoadedModel`, `CausalLanguageModel` and `Module`
once for all decoder models.

`nn::AttentionConfig` configures every softmax attention variant in one place: separate or fused
query, key and value projections, the weight layout, q/k norms and an output gate. The rotary
embedding belongs to the model: the base model computes the `(cos, sin)` tables of the positions in
the graph once per forward, and `functional::attention_kv_cache` rotates the queries and keys with
them. The positions (`si32 [T]`) are the only per-step upload besides the tokens of a prompt: the KV
cache write offset, the causal mask and GPT-2's position embedding all derive from them.

Every call runs its tokens in chunks of at most `PREFILL_CHUNK` tokens. The decoder records
each chunk into one graph arena: the caller's default arena when one is installed, otherwise a fresh
arena per chunk. So computations that read only parameters join the program of the chunk and do not
run as programs of their own. An example is the `1 + weight` scale of a unit-offset RMSNorm. The
decoder materializes the caches between chunks, so each chunk runs as exactly one compiled program.

The KV caches hold the sequence length rounded up to a power of two, capped at the context length:
the prompt for `forward`, and the prompt plus the generated tokens for a decode. The rounding lets
sequences of different lengths reuse compiled programs. A decode step after the first runs as one
program, which the steps after it reuse.

`nn::capture::module_outputs` runs a closure with module-output capture enabled on the current
thread. It returns the output of every module, named by Hugging Face module path, such as
`model.layers.{i}.mlp`. Layers report through `capture::record`. Without a running capture, `record`
costs only a thread-local check. The Python `LoadedModel.debug_token_activations` wraps a forward in
`module_outputs`.

## Parameter identity

Parameter identity is split into two layers (see `crates/gpt-rs/src/params.rs`):

- `BaseParamId(u128)`: deterministic hash of the parameter name (stable across runs).
- `ModelNamespaceId(u128)`: runtime-assigned namespace per loaded model instance.
- `ParamKey(u128)`: stable key derived from `(namespace, base_id)`, the parameter's id in the
  backend's parameter store.

`load_model` picks a fresh namespace (`next_namespace()`) unless `LoadOptions.namespace` names one.
`get(name)` binds each tensor the model reads under the `ParamKey` of its name, and hands it to the
backend as a view of the memory-mapped checkpoint (`PortableBackend::upload` of a
`HostTensor::view`). The OS reads a weight from disk when a backend first reads it. An entry that
the file does not align for its dtype is copied, and the load prints how many bytes it copied.

Each backend keeps parameters in its parameter store (`PortableBackend::params`). The load-time
functions that the pipeline outlines compute derived formats (packed weights, layouts) once and
store them there under ids of their own, without changes to model code.

## How the CLI uses runtime

`gpt-rs-cli` is capability-based:

- `generate` and `benchmark`: require `model.causal_lm()` and generate with `Generator`.
- `forward`: calls `CausalLanguageModel::forward` for token inputs, or `VisionModel::forward` for
  vision inputs.

See: `crates/gpt-rs-cli/src/main.rs` (`generate` / `forward` subcommands).

## Adding a new model kind (runtime wiring)

After implementing your model's config, register it with the runtime factory list:

- `crates/gpt-rs/src/model/registry.rs`: `model_factories()` / `model_factory(kind)`. Decoder models
  register `inference::decoder::build_from_model_config::<B, TheirConfig>`, and image models
  `inference::vision::build_from_model_config::<B, TheirConfig>`.

For a full checklist (model + layer + functional + backend), see [docs/howto.md](howto.md).
