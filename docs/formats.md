# File formats: checkpoints and tensor archives

This repo uses two small, indexed binary formats:

- **Checkpoint**: `GPTRSCHK` (model config + tensors, random-access by name/id)
- **Tensor archive**: `GPTRSTEN` (named tensor bundle for inputs/outputs/traces)

Source of truth is the code:
- Checkpoints: `crates/gpt-rs/src/checkpoint/*` and `scripts/gptrs_eval/checkpoint.py`
- Tensor archives: `crates/gpt-rs/src/io/tensor_archive.rs`
- Shared index and payload layout: `crates/gpt-rs/src/io/tensor_index.rs`

## Checkpoint (`GPTRSCHK`, v2)

High-level layout:

```
magic[8] = "GPTRSCHK"
version[u32] = 2
config_len[u32]
config_json[config_len]
index_len[u32]
index_bytes[index_len]
data_bytes[...]
```

### `config_json`

JSON object matching `crates/gpt-rs/src/model/config.rs::ModelConfig`:

- `kind`: string (e.g. `"gpt"`, `"ministral"`, `"resnet34"`)
- `config`: model-specific JSON payload. For the decoder models, this is the Hugging Face text config,
  which multimodal checkpoints nest under `text_config`. The model reads only the keys that its
  config struct declares.
- `runtime` (optional): runtime-only knobs, `ModelRuntimeConfig` in the same file. See
  [runtime.md](runtime.md).
- `eos_token_ids` (optional): token ids that end a generated sequence. The Python exporters write the
  Hugging Face `eos_token_id` values here.

### `index_bytes`

Little-endian binary index enabling random access:

```
tensor_count[u32]
repeat tensor_count times:
  name_len[u32]
  name_bytes[name_len]          // UTF-8; parameter naming rules enforce ASCII in Rust module traversal
  stored_base_id[u128]          // may be 0 (meaning "not stored"; loader computes from name)
  rank[u32]
  dims[rank * u64]
  dtype_tag[u32]
  reserved[u8]
  offset_abs[u64]
  byte_len[u64]
```

The Rust loader computes `BaseParamId` from the name (and validates `stored_base_id` if it is non-zero).
Readers ignore `reserved`. The Python writer stores a `requires_grad` flag there.

### `data_bytes`

Raw tensor payloads:

- Little-endian, row-major (C-order) bytes for the tensor's logical shape.
- `dtype_tag`: `0 = f32`, `1 = f16`, `2 = bf16`, `3 = i32` (`crates/gpt-rs/src/tensor/dtype.rs`).
- Both writers align every payload to 64 bytes. They write `<path>.tmp` and then rename it over
  `path`, so processes that memory-mapped the previous file keep a valid mapping.
- Readers check each entry's byte length against its shape and dtype, and its range against the file
  size.

### Parameter names and layouts

Checkpoints mirror the Hugging Face text models. Tensor names are the module paths that
`Module::visit_params` yields. Projections use the PyTorch `nn.Linear` layout
`weight: [out_features, in_features]`, and `nn::Linear` computes `x @ weight^T + bias`. Exporters
copy tensors unchanged apart from dtype. They strip multimodal prefixes (`model.language_model.`
becomes `model.`). They always write `lm_head.weight`, also for tied embeddings.

The decoder models (`gpt`, `ministral`, `qwen3_5`) share the Llama-style names:

| Tensor | Shape |
|---|---|
| `model.embed_tokens.weight` | `[vocab, hidden]` |
| `model.layers.{i}.input_layernorm.weight` | `[hidden]` |
| `model.layers.{i}.self_attn.{q,k,v,o}_proj.weight` | `[out, in]` |
| `model.layers.{i}.post_attention_layernorm.weight` | `[hidden]` |
| `model.layers.{i}.mlp.{gate,up,down}_proj.weight` | `[out, in]` |
| `model.norm.weight` | `[hidden]` |
| `lm_head.weight` | `[vocab, hidden]` |

- `gpt` (GPT-2): LayerNorms and every projection also have a `.bias`. The learned positions are
  `model.embed_positions.weight` (`[context, hidden]`). The MLP is not gated and has only
  `mlp.up_proj` and `mlp.down_proj`. The config's `activation_function` is the Hugging Face name of
  the MLP activation. The exporter transposes GPT-2's `Conv1D` weights, which GPT-2 stores as
  `[in, out]`, and splits the fused `attn.c_attn` into `q_proj`, `k_proj` and `v_proj`.
- `ministral`: RMSNorm scales, no biases.
- `qwen3_5`: RMSNorm weights are zero-centred, and the graph computes the scale `1 + weight`.
  Full-attention layers add `self_attn.q_norm` and `self_attn.k_norm`, and their `q_proj` emits
  `[query, gate]` per head. Linear-attention layers have these tensors under `linear_attn.`:
  - the projections `in_proj_qkv`, `in_proj_z`, `in_proj_b`, `in_proj_a` and `out_proj`
  - `conv1d.weight` (`[conv_dim, 1, kernel]`)
  - `dt_bias`
  - `A_log` (the graph computes the decay `-exp(A_log)`)
  - `norm.weight` (a plain RMSNorm scale)

  The exporter stores matrices as bf16 (`--weight-dtype`) and every other tensor as f32.

The vision models keep the torchvision names, with BatchNorm folded into the convolutions. Their
`fc` / `classifier` weights use the same `[out, in]` layout.

## Tensor archive (`GPTRSTEN`, v2)

Tensor archives are used for:

- CLI vision inputs (`gpt-rs-cli forward --input ... --input-key input`)
- Optional CLI logits output (`gpt-rs-cli forward --out ...`)

Layout:

```
magic[8] = "GPTRSTEN"
version[u32] = 2
index_len[u32]
index_bytes[index_len]
data_bytes[...]
```

Index entry layout is the checkpoint format *without* config/base_id:

```
tensor_count[u32]
repeat tensor_count times:
  name_len[u32]
  name_bytes[name_len]
  rank[u32]
  dims[rank * u64]
  dtype_tag[u32]
  reserved[u8]
  offset_abs[u64]
  byte_len[u64]
```

Tensor archives share these rules with checkpoints: the dtype tags, the 64-byte payload alignment,
the write-then-rename, and the reader checks.

## Parameter identity and streaming (why it stays flexible)

Key ideas (see `crates/gpt-rs/src/params.rs`):

- Parameters are identified by stable **names** (ASCII, dot-separated path) produced by `Module::visit_params`.
- `BaseParamId(u128)` is a deterministic hash of the parameter name.
- `ModelNamespaceId(u128)` is assigned at runtime so multiple models can coexist without collisions.
- `ParamKey(u128)` is derived from `(namespace, base_id)` and becomes the stable id used for caching.

Streaming path:

- `runtime::load_model` wraps checkpoint tensors as `DeviceTensor::lazy_param(...)`, storing:
  - `base_id` (for loading bytes) and
  - `stable_id` (the `ParamKey`, used for caching and backend resolvers).
- When a param is first needed, the tensor's `ParamSource<B>` loads it by `BaseParamId`.
- Backends may expose a `ParamResolver` (`PortableBackend::param_resolver`) so optimized/derived parameter
  representations (packed weights, layouts) can be memoized by stable id without changing model code.
