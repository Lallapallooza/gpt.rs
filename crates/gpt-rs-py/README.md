# gpt-rs-py

Minimal Python bindings for `gpt-rs` focused on **checkpoint-backed inference**:
- `load_model(...)` (runtime loader for Hugging Face model directories)
- `LoadedModel` forward + generation helpers
- `Tokenizer`
- dump/profiler hooks

Tensor/layer/functional APIs are intentionally not exposed in Python (for now).

## Build / install (from repo root)

```bash
uv sync
uv pip install maturin

# Build the extension (recommended: enable all backends you want available at runtime):
uv --project "$(pwd)" --directory "$(pwd)/crates/gpt-rs-py" run maturin develop --release --features faer,conversion-c

# Or enable a single backend to reduce build time:
uv --project "$(pwd)" --directory "$(pwd)/crates/gpt-rs-py" run maturin develop --release --features faer
uv --project "$(pwd)" --directory "$(pwd)/crates/gpt-rs-py" run maturin develop --release --features conversion-c
```

Then pick a backend at runtime:

```python
import gpt_rs

gpt_rs.set_backend("faer")  # or: "c", "cpu", "cpu-portable"
```

## Quick smoke test

```bash
python crates/gpt-rs-py/test_install.py
```

## Usage

`load_model` takes a local Hugging Face model directory, for example from
`huggingface_hub.snapshot_download("gpt2", allow_patterns=["*.json", "model*.safetensors"])`.
`load_model(model_dir, config_overrides={...})` replaces keys of the model config.

GPT-2 style generation (tokens in / tokens out):

```python
import gpt_rs
from huggingface_hub import snapshot_download

gpt_rs.set_backend("faer")
model_dir = snapshot_download("gpt2", allow_patterns=["*.json", "model*.safetensors"])
tok = gpt_rs.Tokenizer.from_file(f"{model_dir}/tokenizer.json")
model = gpt_rs.load_model(model_dir)

prompt = tok.encode("Hello")
out = model.generate_tokens(prompt, 64, temperature=0.8, kv_cache=True)
print(tok.decode(out))
```

Vision forward (float32 NCHW in, logits out):

```python
import numpy as np
import gpt_rs
from huggingface_hub import snapshot_download

gpt_rs.set_backend("faer")
model_dir = snapshot_download(
    "microsoft/resnet-34", allow_patterns=["*.json", "model*.safetensors"]
)
model = gpt_rs.load_model(model_dir)
x = np.random.randn(1, 3, 224, 224).astype(np.float32)
logits = model.forward_vision(x)
print(logits.shape)
```
