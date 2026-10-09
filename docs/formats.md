# File formats: checkpoints and tensor files

gpt.rs reads Hugging Face model directories as they are published, with no conversion step, and
uses safetensors for every other tensor file.

Source of truth is the code:
- Checkpoint readers: `crates/gpt-rs/src/checkpoint/` (the `Checkpoint` trait and
  `checkpoint::open`)
- Safetensors reading and writing: `crates/gpt-rs/src/io/safetensors.rs`

## Checkpoints: Hugging Face model directories

`runtime::load_model(backend, model_dir)` loads a directory with:

- `config.json`. Its `model_type` (the `text_config` one for a multimodal checkpoint) is the model
  kind that picks the Rust model in `crates/gpt-rs/src/model/registry.rs`. The loader passes the
  config to that model, which reads only the keys that its config struct declares.
- `model.safetensors`, or the shards that `model.safetensors.index.json` lists. The loader ignores
  other weight files, such as Mistral's `consolidated.safetensors`.
- `generation_config.json` (optional). Its `eos_token_id` values end generation, as do those of
  `config.json`.

`gpt-rs-cli` reads `tokenizer.json` from the model directory by default. The CLI and
`gpt_rs.Tokenizer` read byte-level BPE files such as GPT-2's and Qwen's, but not Ministral's.

`uv run hf download <repo> --include "*.json" --include "model*.safetensors"` or
`huggingface_hub.snapshot_download` fetches such a directory.

### How tensors load

- Tensors keep their Hugging Face names. A decoder model also finds tensors saved without its
  `DecoderConfig::BASE_MODEL_PREFIX` (GPT-2's `transformer`), as `transformers` does.
- A multimodal checkpoint loads as its text model. The vision tower is not loaded. The text model
  is:
  - the `text_config`, with the top-level `tie_word_embeddings` when the text config has none;
  - the text decoder's tensors, renamed from `model.language_model.` or `language_model.model.` to
    `model.`;
  - `lm_head.weight`.
- 2-D tensors keep their stored dtype: the projection and embedding matrices, FP8 (`F8_E4M3`)
  weights included. Other f16 and bf16 tensors load as f32, and tensors of every other dtype load
  as stored.
- An FP8 tensor that is not 2-D is a load error. An FP8 weight's scale, `<name>_scale_inv`, is per
  tensor: a float of rank 0, which keeps its dtype. A scale of another rank or dtype is a load
  error. The scale is a parameter of the weight's `nn::Linear`, whose weight is
  `weight * weight_scale_inv` in the dtype of the scale, as `transformers` dequantizes it. Backends
  with a linear kernel for it read the FP8 weight in place and dequantize it as they multiply. The
  reference backend computes the dequantization as recorded.
- The `activation_scale` tensors are not loaded, and complex tensors, a dtype the runtime does not
  have, are skipped. A model uploads only the tensors it reads, so BatchNorm's
  `num_batches_tracked` never reaches a backend.
- Tensors stored in their load dtype reach the backend as views of the memory-mapped file, and
  backends that run on host memory execute on those views (`PortableBackend::upload` of a
  `HostTensor::view`): the weights stay in the OS page cache. Tensors that load as f32 from another
  stored dtype are copies, and so are entries that the file does not align for their dtype.

`load_model_with_options` also takes runtime settings and config overrides
([runtime.md](runtime.md)).

### Adding a format

A format implements `checkpoint::Checkpoint`: the model config (Hugging Face `model_type`, config,
and end-of-sequence ids), the tensors under their Hugging Face names, and each tensor's
little-endian bytes in its load dtype. Add the reader to `checkpoint::open`. A format whose names or
config keys differ from Hugging Face's (for example GGUF) maps them in its reader, so models see
only Hugging Face names.

## Parameter names and layouts

The Rust modules mirror the Hugging Face modules, so parameter names are the module paths that
`Module::visit_params` yields and match the checkpoint names (see "How tensors load"). `nn::Linear`
is PyTorch's `nn.Linear`: `weight: [out_features, in_features]`, `y = x @ weight^T + bias`. With
`WeightLayout::InOut` it is GPT-2's `Conv1D`: `weight: [in_features, out_features]`,
`y = x @ weight + bias`. A decoder family whose module names differ from the Llama family's names
them in its `nn::DecoderNames`.
The language-model head reuses the token embedding weight when `tie_word_embeddings` is true. When
`config.json` omits the key, the default of the Hugging Face config class applies (true for
`GPT2Config`).

The Ministral and Qwen3.5 decoders share the Llama-style names:

| Tensor | Shape |
|---|---|
| `model.embed_tokens.weight` | `[vocab, hidden]` |
| `model.layers.{i}.input_layernorm.weight` | `[hidden]` |
| `model.layers.{i}.self_attn.{q,k,v,o}_proj.weight` | `[out, in]` |
| `model.layers.{i}.post_attention_layernorm.weight` | `[hidden]` |
| `model.layers.{i}.mlp.{gate,up,down}_proj.weight` | `[out, in]` |
| `model.norm.weight` | `[hidden]` |
| `lm_head.weight` | `[vocab, hidden]` |

- `ministral3`: RMSNorm scales, no biases.
- `qwen3_5_text`: RMSNorm weights are zero-centred, and the graph computes the scale `1 + weight`.
  The model builds one layer for each of the first `num_hidden_layers` entries of `layer_types`.
  Full-attention layers add `self_attn.q_norm` and `self_attn.k_norm`, and their `q_proj` emits
  `[query, gate]` per head. Linear-attention layers have these tensors under `linear_attn.`:
  - the projections `in_proj_qkv`, `in_proj_z`, `in_proj_b`, `in_proj_a` and `out_proj`
  - `conv1d.weight` (`[conv_dim, 1, kernel]`)
  - `dt_bias`
  - `A_log` (the graph computes the decay `-exp(A_log)`)
  - `norm.weight` (a plain RMSNorm scale)

GPT-2 (`gpt2`) has the `GPT2LMHeadModel` modules `transformer.wte`, `transformer.wpe`,
`transformer.h.{i}.{ln_1,attn.c_attn,attn.c_proj,ln_2,mlp.c_fc,mlp.c_proj}`, and `transformer.ln_f`.
Each has a `.weight` and a `.bias`, except `wte` and `wpe`, which have only a `.weight`.
`attn.c_attn` projects to the fused `[query, key, value]`.

The vision models (`resnet`, `mobilenet_v2`) have the names of `ResNetForImageClassification` and
`MobileNetV2ForImageClassification`. Convolutions keep the PyTorch `[out, in / groups, kh, kw]`
layout. BatchNorm loads its `weight`, `bias`, `running_mean`, and `running_var`.

## Tensor files

The CLI reads vision inputs (`gpt-rs-cli forward --input ... --input-key input`) and writes logits
(`--out`) as safetensors, which `safetensors.numpy.load_file` and `safetensors.torch.load_file`
read.

## Parameter identity

How parameters get their stable ids, and how backends key derived formats by them, is in
[runtime.md](runtime.md#parameter-identity).
