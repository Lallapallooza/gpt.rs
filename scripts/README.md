# scripts/

Small, maintained Python utilities for gpt.rs.

## Canonical entrypoints

- `scripts/eval.py`: Torch-baselined validate / bench / run runner.
- `scripts/export.py`: Unified metadata-driven exporter CLI (`list`, `inspect`, `export`, `validate`).
- `scripts/rebuild_py.py`: Rebuild/install the `gpt_rs` Python bindings via `uv` + `maturin`.
