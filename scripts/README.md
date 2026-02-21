# scripts/

Small, maintained Python utilities for gpt.rs.

## Canonical entrypoints

- `scripts/eval.py`: Torch-baselined validate / bench / run runner.
- `scripts/export.py`: Unified metadata-driven exporter CLI (`list`, `inspect`, `export`, `validate`).
- `scripts/rebuild_py.py`: Rebuild/install the `gpt_rs` Python bindings via `uv` + `maturin`.

## Common commands

```bash
# discover available exporters
uv run python scripts/export.py list

# inspect Ministral exporter metadata/default artifacts
uv run python scripts/export.py inspect --exporter ministral_3_3b_instruct_2512 --format json

# export Ministral artifacts (checkpoint + config + HF tokenizer json)
uv run python scripts/export.py export --exporter ministral_3_3b_instruct_2512 \
  --checkpoint-out checkpoints/ministral_3_3b_instruct_2512.bin \
  --config-out configs/ministral_3_3b_instruct_2512_model.json \
  --tokenizer-out configs/ministral_3_3b_instruct_2512_tokenizer.json

# run Torch parity validation for the exported checkpoint
uv run python scripts/eval.py --model ministral_3_3b_instruct_2512 --workload validate \
  --checkpoint checkpoints/ministral_3_3b_instruct_2512.bin \
  --torch-model mistralai/Ministral-3-3B-Instruct-2512
```
