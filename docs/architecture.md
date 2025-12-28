# Architecture Overview

gpt.rs is structured as a small set of composable layers:

- **Models** (`gpt_rs::model`) wire blocks together (GPT, ResNet34, MobileNetV2, ...).
- **Layers** (`gpt_rs::nn::layers`) own parameters and orchestration (`forward`, caching, shape checks).
- **Functionals** (`gpt_rs::ops::functional`) are portable math kernels that capture PTIR graphs.
- **Backends** (`gpt_rs::backend::spec::PortableBackend` + `crates/gpt-rs-backend-*`) execute PTIR programs.

For a quick diagram, see the README’s “Architecture (high level)” section. For more detail on the
frontend layering and functional overrides, see `docs/frontend.md`.

## Core crates

- `crates/gpt-rs`: library (tensors, PTIR capture, layers, models, tokenizer, checkpoints).
- `crates/gpt-rs-cli`: model runner CLI with consistent dump/profile flags across models.
- `crates/gpt-rs-backend-tests`: shared backend suite + Torch parity harness (via `tch` / libtorch).
- `crates/gpt-rs-backend-faer`: optimized CPU backend (recommended).
- `crates/gpt-rs-backend-ref-cpu`: slow reference backend for debugging/spec bring-up.

## Core modules (inside `crates/gpt-rs`)

- `tensor`: host `Tensor` plus `DeviceTensor<B>` (shape/dtype tracking, lazy handles, materialization).
- `backend`: PTIR backend contract and hook points (profiling/debug wrappers).
- `ops`: PTIR capture (graph arena), functional kernels, tracing/dump sinks.
- `nn`: parameterized layers composed from functionals (e.g. `Linear`, `LayerNorm`, attention blocks).
- `model`: concrete model assemblies and weight loading helpers.
- `checkpoint`: self-describing model checkpoint format + loader.
- `io`: portable tensor archives (used for inputs/outputs/traces).
- `inference`: sampling + incremental generation (`Generator`).
- `tokenizer`: tokenizer config + encode/decode.
- `train`: small training utilities (experimental; not the primary focus right now).

## How a forward pass runs

1. A layer calls a functional (e.g. `matmul`, `layer_norm`, `conv2d_nhwc`).
2. The functional validates shape/dtype and captures a PTIR program (or picks a cached one).
3. The selected backend executes the PTIR program and returns a lazy handle.
4. The result materializes when the caller needs host data (or when chaining forces it).

## Testing + correctness

- Portable kernel correctness is checked via backend suites + Torch parity (see `docs/testing.md`).
- Full-model baselines use `scripts/eval.py` (validate / trace / bench) against Torch.
