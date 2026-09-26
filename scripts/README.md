# scripts/

Small, maintained Python utilities for gpt.rs.

## Canonical entrypoints

- `scripts/eval.py`: Torch-baselined validate / bench / run runner. gpt-rs and Torch load the same
  Hugging Face checkpoint (`--hf-model`, a Hub id or a local directory).
- `scripts/rebuild_py.py`: Rebuild/install the `gpt_rs` Python bindings via `uv` + `maturin`.

## Common commands

```bash
# list the eval cases
uv run python scripts/eval.py --list-models

# validate Ministral against Torch
uv run python scripts/eval.py --model ministral_3_3b_instruct_2512 --workload validate

# the same, on a local copy of the checkpoint
uv run python scripts/eval.py --model ministral_3_3b_instruct_2512 --workload validate \
  --hf-model checkpoints/ministral-3-3b
```
