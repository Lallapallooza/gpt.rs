# Runtime: model loading and capability dispatch

This document describes the `gpt_rs::runtime` layer: how checkpoints are loaded into a dynamic model
handle and how the CLI calls into models without hardcoding model kinds.

Source of truth: `crates/gpt-rs/src/runtime/` and `crates/gpt-rs-cli/src/main.rs`.

## Core entrypoint: `load_model`

The canonical loader is:

- `gpt_rs::runtime::load_model(backend, checkpoint_path) -> Box<dyn LoadedModel<B>>`

The loader:

1. Opens a self-describing checkpoint (`GPTRSCHK`) and reads `ModelConfig` + tensor index.
2. Creates a checkpoint-backed `ParamSource<B>` for random-access parameter loads.
3. Builds a `get(name)` closure that returns `DeviceTensor::lazy_param(...)` for each parameter.
4. Selects a model factory by `ModelConfig.kind` and constructs the model using `get(name)`. Models
   build their layers through `nn::LayerLoader`, which checks the shape of every tensor. For names
   and layouts, see [formats.md](formats.md#parameter-names-and-layouts).
5. Wraps the model in `ModelHandle<B>` with the checkpoint's `eos_token_ids`.

`load_model_with_options(backend, path, options)` is the general form. `LoadOptions` in
`crates/gpt-rs/src/runtime/loader.rs` documents the options, among them a `matmul_input_dtype` that
overrides `runtime.matmul_input_dtype` from the checkpoint.

Every model passes `matmul_input_dtype` to its `nn::Linear` layers at build time. With `bf16`, each
layer casts its input to bf16 before `functional::linear`, which multiplies with f32 accumulation and
f32 outputs. Norms, attention and the recurrent state stay in f32. These numerics differ from a
PyTorch bf16 model, which also rounds matmul outputs and residuals to bf16. The default keeps
activations in f32 and widens bf16 weights exactly.

## Dynamic model interface (`LoadedModel`)

Models are exposed through a small dynamic trait:

- `LoadedModel<B>`: `kind()`, `forward(ModelInput) -> ModelOutput`
- Optional capabilities exposed via trait methods returning `Option<...>`:
  - `as_causal_lm() -> Option<&dyn CausalLanguageModel<B>>`

`CausalLanguageModel` carries per-layer decode state as `LayerCache<B>` values: a fixed-capacity KV
cache for attention layers, and convolution and recurrent state for linear-attention layers.
`functional::attention_kv_cache` reads and writes the KV cache for prefill and decode alike.

`forward_with_decode_cache` returns logits `[1, vocab]` for the last position of each call. `forward`
returns logits `[T, vocab]` for every position. It computes them as a prefill into fresh caches.

End-of-sequence tokens are checkpoint metadata, not a model capability. Exporters write the Hugging
Face `eos_token_id` values as the top-level `ModelConfig.eos_token_ids`, and
`ModelHandle::eos_token_ids()` returns them. `gpt-rs-cli generate` and the Python `generate_tokens`
stop after one of these tokens. `--ignore-eos` and `ignore_eos=True` turn this off.

This is what makes the CLI generic: it asks the model for a capability (e.g. "causal LM") instead of
switching on an enum of model kinds.

See:
- `crates/gpt-rs/src/runtime/handle.rs` (`LoadedModel`, `ModelHandle`) and `runtime/types.rs` (`ModelInput`, `ModelOutput`)
- `crates/gpt-rs/src/inference/mod.rs` (`CausalLanguageModel`)

## Decoder models (`CausalDecoder`)

Every text model is an `inference::decoder::CausalDecoder<B, C>`. Its config type `C` holds the keys
of the Hugging Face text config that the model reads. `C` implements `DecoderConfig`: the checkpoint
`KIND` and `layout()`, which validates the config and describes the decoder's layers.

`nn::AttentionConfig` configures every attention variant in one place. The attention layers share
one rotary embedding.

Like a Hugging Face `*ForCausalLM`, `CausalDecoder` holds the decoder `model` and `lm_head`. The
`model` holds `nn::DecoderBlock` layers. Each block is a pre-norm residual block with a token mixer,
`self_attn` or `linear_attn`, and a plain or gated MLP. `CausalDecoder` implements `LoadedModel`,
`CausalLanguageModel` and `Module` once for all decoder models.

Every call runs its tokens in chunks of at most `PREFILL_CHUNK` tokens. The decoder captures
each chunk into one graph arena: the caller's default arena when one is installed, otherwise a fresh
arena per chunk. So computations that read only parameters join the program of the chunk and do not
run as programs of their own. An example is the `1 + weight` scale of a unit-offset RMSNorm. The
decoder materialises the caches between chunks, so each chunk runs as exactly one compiled program.

`forward` sizes the KV caches to the sequence. Cached calls use the pinned `capacity`, or the end
position rounded up to a power of two. `forward_with_decode_cache_sample_next` hands the
last-position logits to `PortableBackend::sample_decode_token` when the backend supports the request.

`nn::capture::module_outputs` runs a closure with module-output capture enabled on the current
thread. It returns the output of every module, named by Hugging Face module path, such as
`model.layers.{i}.mlp`. Layers report through `capture::record`. Without a running capture, `record`
costs only a thread-local check. The Python `LoadedModel.debug_token_activations` wraps a forward in
`module_outputs`.

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

- `crates/gpt-rs/src/model/registry.rs`: `model_factories()` / `model_factory(kind)`. Decoder models
  register `inference::decoder::build_from_model_config::<B, TheirConfig>`.

For a full checklist (model + layer + functional + backend), see [docs/howto.md](howto.md).
