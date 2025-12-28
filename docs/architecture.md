# Architecture Overview

gpt.rs is a Rust workspace for experimenting with a portable tensor IR (PTIR) and a small set of
models/layers built on top of it.

At a glance:
- **Models** (`gpt_rs::model::*`) wire blocks together (GPT-2, ResNet34, MobileNetV2, ...).
- **Layers** (`gpt_rs::nn::layers::*`) own parameters and orchestration (`forward`, cache plumbing, shape checks).
- **Functionals** (`gpt_rs::ops::functional::*`) are portable kernels that validate inputs and capture PTIR graphs.
- **Backends** (`gpt_rs::backend::spec::PortableBackend`) execute PTIR programs (faer, ref-cpu, ...).
- **Runtime** (`gpt_rs::runtime::load_model`) loads a self-describing checkpoint into a dynamic model handle.

For the layering model and runtime functional overrides, see `docs/frontend.md`.

## Project layout

Core crates:
- `crates/gpt-rs`: library (tensors, PTIR capture, layers, models, tokenizer, checkpoints).
- `crates/gpt-rs-cli`: thin runner with `generate` / `forward` / `trace` and dump/profile flags.
- `crates/gpt-rs-backend-tests`: shared backend suite + Torch parity harness (via `tch` / libtorch).
- `crates/gpt-rs-backend-faer`: optimized CPU backend (recommended default).
- `crates/gpt-rs-backend-ref-cpu`: slow reference backend for debugging/spec bring-up.

Core modules (inside `crates/gpt-rs/src/`):
- `tensor`: host `Tensor` and device `DeviceTensor<B>` (shape/dtype tracking, lazy handles, materialization).
- `module`: a small `Module` trait for enumerating/updating parameters by stable name.
- `params`: parameter identity (`u128` ids) and streaming sources for lazy param loading.
- `checkpoint`: self-describing model checkpoints (`GPTRSCHK`) and tensor archive IO.
- `ops`: PTIR DSL (`ops/ptir`), graph arena (`ops/graph`), functional kernels, trace/dump sinks.
- `backend`: PTIR backend contract + hook points (profiling/debug wrappers).
- `nn`: parameterized layers composed from functionals.
- `model`: concrete model assemblies (and any model-specific helpers).
- `runtime`: model loading, model capability adapters (causal LM / vision tracing), and functional override plumbing.
- `inference`: sampling + incremental generation (`Generator`).
- `tokenizer`: encode/decode and tokenizer config.

## End-to-end execution (what actually happens)

1. A caller loads a checkpoint via `runtime::load_model(backend, path)`.
2. Model code calls layers, and layers call functionals.
3. Each functional validates shape/dtype and captures a PTIR program (or reuses a cached one).
4. The backend executes the PTIR program and returns lazy handles.
5. Host materialization happens only when explicitly requested (e.g. for printing/logits export).

## Correctness and validation

- Kernel-level correctness: backend suite + Torch parity (see `docs/testing.md`).
- End-to-end model baselines: `scripts/eval.py` (validate / trace / bench) against Torch/HF.
