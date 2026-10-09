# Architecture Overview

gpt.rs is a Rust workspace for experimenting with a portable tensor IR (PTIR) and a small set of
models/layers built on top of it.

At a glance:
- **Models** (`gpt_rs::model::*`) wire blocks together (GPT-2, ResNet, MobileNetV2, ...).
- **Layers** (`gpt_rs::nn::layers::*`) own parameters and orchestration (`forward`, cache plumbing, shape checks).
- **Functionals** (`gpt_rs::ops::functional::*`) are portable kernels that validate inputs and record PTIR ops.
- **Backends** (`gpt_rs::backend::PortableBackend`) compile and run functions of the IR (faer, ref-cpu, c).
- **Runtime** (`gpt_rs::runtime::load_model`) loads a Hugging Face model directory into a dynamic
  model handle.

For the layering model, see [frontend.md](frontend.md).

## Project layout

The crates are listed in the [README](../README.md#repository-layout).

Core modules (inside `crates/gpt-rs/src/`):
- `tensor`: the lazy `DeviceTensor<B>` and its ops, which record PTIR into a graph arena (shape/dtype tracking, lazy handles, materialization).
- `module`: a small `Module` trait for enumerating/updating parameters by stable name.
- `params`: parameter identity: stable `u128` ids, model namespaces and `ParamKey`.
- `checkpoint`: checkpoint readers (Hugging Face model directories) behind the `Checkpoint` trait.
- `io`: safetensors reading and writing.
- `ops`: graph arena (`ops/graph`), functional kernels, trace/dump sinks.
- `backend`: the `PortableBackend` contract and the parameter store (`ParamStore`).
- `nn`: parameterized layers composed from functionals.
- `model`: one file per model family. A decoder's file holds its config, its module names and its
  layout. An image model's file holds its `#[nn::module]` model and its config.
- `inference`: the shared models those configs build: `decoder.rs` (`CausalDecoder`, decoding and
  sampling) and `vision.rs` (`ImageModel`).
- `runtime`: model loading into a `ModelHandle`, whose capabilities are a causal LM and a vision
  model.
- `profiling`: ambient profiler scopes for layers, functionals, passes and backend ops.
- `tokenizer`: encode/decode and tokenizer config.

## End-to-end execution (what actually happens)

1. A caller loads a checkpoint via `runtime::load_model(backend, path)`.
2. Model code calls layers, and layers call functionals.
3. Each functional validates shape/dtype, records `tensor` ops into the graph arena, and returns
   lazy `DeviceTensor`s.
4. When a value is needed, the arena builds a PTIR module from the recorded ops, runs the pass
   pipeline ([ptir/passes.md](ptir/passes.md)), compiles the module on the backend, caches the
   program, and runs it on the backend.
5. Host materialization happens only when explicitly requested (e.g. for printing/logits export).

## Correctness and validation

- Kernel-level correctness: backend suite + Torch parity (see [testing.md](testing.md)).
- End-to-end model baselines: `scripts/eval.py` (validate / bench) against Torch/HF.
