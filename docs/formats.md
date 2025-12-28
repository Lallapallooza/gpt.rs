# File formats: checkpoints and tensor archives

This repo uses two small, indexed binary formats:

- **Checkpoint**: `GPTRSCHK` (model config + tensors, random-access by name/id)
- **Tensor archive**: `GPTRSTEN` (named tensor bundle for inputs/outputs/traces)

Source of truth is the code:
- Checkpoints: `crates/gpt-rs/src/checkpoint/*` and `scripts/gptrs_eval/checkpoint.py`
- Tensor archives: `crates/gpt-rs/src/io/tensor_archive.rs`

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

- `kind`: string (e.g. `"gpt"`, `"resnet34"`)
- `config`: model-specific JSON payload
- `runtime` (optional): runtime-only knobs (e.g. functional overrides)

Notes:
- Python exporters currently write only `{ "kind": ..., "config": ... }` (no `runtime`).
- For legacy compatibility, Rust may accept untagged GPT configs and wrap them as `{ kind: "gpt", config: <value> }`.

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
  requires_grad[u8]
  offset_abs[u64]
  byte_len[u64]
```

The Rust loader computes `BaseParamId` from the name (and validates `stored_base_id` if it is non-zero).

### `data_bytes`

Raw tensor payloads:

- Row-major (C-order) bytes for the tensor's logical shape.
- Current Rust reader supports `f32` and `i32`. (`f16` / `bf16` are rejected today.)

## Tensor archive (`GPTRSTEN`, v2)

Tensor archives are used for:

- CLI vision inputs (`gpt-rs-cli forward --input ... --input-key input`)
- CLI trace output (`gpt-rs-cli trace --out ...`)
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
  requires_grad[u8]
  offset_abs[u64]
  byte_len[u64]
```

Like checkpoints, the current Rust reader supports `f32` and `i32` only.

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
