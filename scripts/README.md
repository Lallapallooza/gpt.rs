# scripts/

Small, maintained Python utilities for gpt.rs.

## Canonical entrypoints

- `scripts/eval.py`: Torch-baselined validate / trace / bench / run runner.
- `scripts/export_model_weights.py`: Export torchvision weights into a gpt.rs checkpoint (`*.bin`).
- `scripts/export_gpt2.py`: Export Hugging Face GPT-2 weights into a gpt.rs checkpoint + configs.
- `scripts/rebuild_py.py`: Rebuild/install the `gpt_rs` Python bindings via `uv` + `maturin`.

## Misc helpers

- `scripts/check_speedup.py`: Check `scripts/eval.py --format json` bench output for a minimum speedup.
