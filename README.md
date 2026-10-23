# gpt.rs

Rust-first **research project** for portable tensor programs (PTIR) and model runtimes.

PTIR lets you keep model math portable while backends compete on execution:
- Layers call portable functionals (no open-coded kernels).
- Functionals record PTIR ops into lazy graphs.
- Backends execute PTIR: faer and C run recognized op sequences, such as a convolution, on their own
  kernels, and x86 lowers each program through the `tile`, `loop` and `isa` dialects to C.
- Parameters have stable `u128` ids, and weights load as views of the memory-mapped checkpoint.

Warning: expect churn. APIs, formats, and model coverage change quickly.

## Demo
https://github.com/user-attachments/assets/4a99a79d-b98c-4b11-9f2d-64906b832ca7



## Benchmarks (early)

CPU numbers from `scripts/eval.py --workload bench --backend c` against Torch 2.14 in eager mode,
on an AMD Ryzen 9 9950X3D (16 cores, Linux). Images are batch 1 at 224x224. GPT-2 generates 64
tokens.

| Workload | Threads | gpt.rs | torch | Speedup |
| --- | ---: | ---: | ---: | ---: |
| ResNet34 | 1 | 25.66 images/s | 24.45 images/s | 1.05x |
| ResNet34 | 16 | 141.18 images/s | 127.01 images/s | 1.11x |
| MobileNetV2 | 1 | 141.92 images/s | 78.36 images/s | 1.81x |
| MobileNetV2 | 16 | 323.38 images/s | 108.71 images/s | 2.97x |
| GPT-2 | 1 | 92.08 tokens/s | 50.46 tokens/s | 1.82x |
| GPT-2 | 16 | 108.50 tokens/s | 49.10 tokens/s | 2.21x |

Reproduce (C backend):
```bash
# Build the Python extension with the C backend enabled:
uv sync
uv pip install maturin
uv run python scripts/rebuild_py.py --features conversion-c

# Bench (gpt.rs and Torch load the same Hugging Face checkpoints):
for model in resnet34 mobilenet_v2; do
  GPTRS_C_CACHE_DIR=./.cache/gpt_rs_c_backend uv run python scripts/eval.py --model "$model" \
    --workload bench --backend c --threads 1 16 --warmup 3 --iters 30 --batch 1 --image-size 224
done
GPTRS_C_CACHE_DIR=./.cache/gpt_rs_c_backend uv run python scripts/eval.py \
  --model gpt2 --workload bench --backend c --threads 1 16 --warmup 2 --iters 5 --bench-tokens 64
```

## Highlights

- **Capability-based runtime**: `runtime::load_model` returns a dynamic model handle; the CLI runs `generate` / `forward` without hardcoding model kinds. See [docs/runtime.md](docs/runtime.md).
- **Stable parameter ids**: weights are views of the memory-mapped checkpoint; backends can memoize derived parameter formats by stable id. See [docs/runtime.md](docs/runtime.md#parameter-identity) and [docs/formats.md](docs/formats.md).
- **Kernel selection**: backends pick their kernels by matching the plain ops they compile, so the IR keeps its meaning through every pass. See [docs/backends/README.md](docs/backends/README.md) and [crates/gpt-rs-backend-faer/src/select.rs](crates/gpt-rs-backend-faer/src/select.rs).
- **Correctness tooling**: Torch parity at the kernel level and end-to-end model baselines via Python runners. See [docs/testing.md](docs/testing.md).
- **Debuggability**: PTIR dumps (`--dump-dir`), profiling (`--profile`, with the `profiler` feature), and eager debugging (`GPTRS_EAGER=1`).

## Docs

Start here:
- [docs/README.md](docs/README.md) (doc index + policy)
- [docs/howto.md](docs/howto.md) (add models/layers/functionals/backends)
- [docs/runtime.md](docs/runtime.md) (loader, capability dispatch)
- [docs/testing.md](docs/testing.md) (Torch parity + dumps/profiling + Python baselines)
- [docs/formats.md](docs/formats.md) (Hugging Face checkpoints + safetensors tensor files)

Reference:
- [docs/backends/README.md](docs/backends/README.md) (how each backend works: ref-cpu, faer, c, x86)
- [docs/ptir/tensor.md](docs/ptir/tensor.md) (the `tensor` dialect: the PTIR spec, generated from the code)
- [docs/ptir/arith.md](docs/ptir/arith.md), [docs/ptir/tile.md](docs/ptir/tile.md), [docs/ptir/loop.md](docs/ptir/loop.md), [docs/ptir/isa.md](docs/ptir/isa.md) and [docs/ptir/host.md](docs/ptir/host.md) (the scalar, tile, loop, vector and host dialects, generated from the code)
- [docs/ptir/passes.md](docs/ptir/passes.md) (the passes and the pipeline of each backend)
- [docs/ops.md](docs/ops.md) (PTIR recording/graphs/execution)
- [docs/frontend.md](docs/frontend.md) (frontend layering)

Scripts:
- [scripts/README.md](scripts/README.md) (Python utilities: eval against Torch)

## Roadmap (TODO)

Backends:
- [ ] NVIDIA GPU (CUDA + cuDNN) backend
- [ ] XLA backend
- [ ] IREE backend

Training:
- [ ] Full training loop
- [ ] Autograd
- [ ] Distributed training

Inference:
- [ ] Quantized models (end-to-end)
- [ ] Paged attention
- [ ] Speculative decoding

Tooling / interop:
- [ ] Full Python interop (define/train models in Python)
- [ ] PyTorch importer

Models:
- [x] Qwen3.5
- [ ] Llama
- [ ] DeepSeek
- [ ] GPT-OSS
- [ ] Diffusion models
- [ ] Speech models

Platforms:
- [ ] Windows
- [ ] MacOS (partially works)

Misc
- [ ] WebUI

## Repository layout

- `crates/gpt-rs`: core library (tensors, PTIR recording, layers, models, tokenizer, checkpoints, runtime).
- `crates/gpt-rs-macros`: the `#[nn::module]` and `#[functional]` proc macros.
- `crates/gpt-rs-ir`: the IR: dialects, text form, verifier, interpreter and passes.
- `crates/gpt-rs-ir-macros`: the macros that define ops, dialects, attributes, passes, patterns and folds.
- `crates/gpt-rs-ir-lexer`: the tokens of the IR text form, which its parser and the attribute formats share.
- `crates/gpt-rs-filetests`: the filetest runner.
- `crates/gpt-rs-docs`: writes the IR reference pages, `docs/ptir/`, from the code.
- `filetests/`: IR filetests (`cargo test -p gpt-rs-ir --test filetests`) and the C backend's (`cargo test -p gpt-rs-backend-c --test filetests`).
- `crates/gpt-rs-backend-ref-cpu`, `crates/gpt-rs-backend-faer`, `crates/gpt-rs-backend-c`, `crates/gpt-rs-backend-x86`: the backends (reference interpreter, faer, generated C, C through the multi-level IR).
- `crates/gpt-rs-backends`: maps a backend name to its type; the CLI and Python load models through it.
- `crates/gpt-rs-backend-tests`: backend suite + Torch parity harness (via `tch` / libtorch).
- `crates/gpt-rs-cli`: model runner CLI (`--help` lists its commands) with dump/profile hooks.
- `crates/gpt-rs-py`: Python bindings for checkpoint-backed inference.
- `scripts/`: Python baselines against Torch (`scripts/eval.py`).

## Architecture (high level)

```text
inputs (tokens / images)
        |
        v
  model::* (Gpt, ResNet, MobileNetV2, ...)
        |
        v
   nn::layers::* (Linear, LayerNorm, CausalSelfAttention, Conv2d, ...)
        |
        v
ops::functional::* (matmul, layer_norm, conv2d, ...)
        |
        v
gpt_rs_ir tensor dialect -> the backend's pass pipeline
        |
        v
backend::PortableBackend (compile and run a function of the IR)
        |
        v
backend impl (faer, ref-cpu, ...)
```

## Quick Start

```bash
# build the workspace (requires Rust toolchain)
cargo build

# explore the CLI surface
cargo run -p gpt-rs-cli -- --help

# download Hugging Face checkpoints, which gpt.rs loads as published
uv sync
uv run hf download gpt2 --local-dir checkpoints/gpt2 \
  --include "*.json" --include "model*.safetensors"
uv run hf download mistralai/Ministral-3-3B-Instruct-2512 \
  --local-dir checkpoints/ministral-3-3b --include "*.json" --include "model*.safetensors"
uv run hf download microsoft/resnet-34 --local-dir checkpoints/resnet-34 \
  --include "*.json" --include "model*.safetensors"

# run generation (the tokenizer defaults to the model directory's tokenizer.json)
cargo run --release -p gpt-rs-cli -- generate --prompt "Hello" --max-tokens 64 \
  --checkpoint checkpoints/gpt2

# the CLI cannot read Ministral's tokenizer.json, so validate Ministral against Torch with
# scripts/eval.py on the local copy
uv run python scripts/eval.py --model ministral_3_3b_instruct_2512 --workload validate \
  --hf-model checkpoints/ministral-3-3b

# run an image model (deterministic random input by default)
cargo run --release -p gpt-rs-cli -- forward --checkpoint checkpoints/resnet-34
```

## Torch Baselines (Python)

The canonical entrypoint is `scripts/eval.py` (validate / bench) which runs both:

```bash
uv sync
uv pip install maturin
uv run python scripts/rebuild_py.py --features faer

uv run python scripts/eval.py --model resnet34 --workload validate
uv run python scripts/eval.py --model gpt2 --workload bench --threads 1 4 --bench-tokens 1 64
uv run python scripts/eval.py --model ministral_3_3b_instruct_2512 --workload validate
```

Notes:
- Select a backend with `--backend` (default: `faer`).
- `--profile` prints tables only when the extension is built with the `profiler` feature
  (`uv run python scripts/rebuild_py.py --features faer,profiler`).
- Use `--dump-dir` / `--dump-mode` to capture PTIR programs (Rust CLI or Python runner).

## Testing

```bash
# The tokenizer tests read configs/gpt2_tokenizer.json (CI downloads it the same way):
mkdir -p configs && curl -fsSL https://huggingface.co/gpt2/resolve/main/tokenizer.json -o configs/gpt2_tokenizer.json

cargo test  # workspace unit + integration tests (no Torch)
cargo test --workspace --exclude gpt-rs-py --exclude gpt-rs-backend-c  # the subset CI runs

GPTRS_TORCH_PARITY_BACKENDS=faer cargo test -p gpt-rs-backend-faer --features torch --test backend_suite  # Torch parity + timings (libtorch via tch)
```

Torch parity tests live under `crates/gpt-rs-backend-tests/src/torch_parity/` and are wired into each backend via
`define_backend_tests!` (behind the `torch` feature). See [docs/testing.md](docs/testing.md).

## Status

Forward inference for causal LMs (`gpt2`, `ministral3`, `qwen3_5_text`) and image classification
models (`resnet`, `mobilenet_v2`, timm's `mobilenetv3_large_100`) is implemented, with portable
PTIR kernels and Torch baselines for correctness.

`DeviceTensor` exposes the elementwise ops and operators (with Torch-style broadcasting), and the
functional layer the composite math (matmul, normalization, attention, conv/pool), all recorded as
PTIR for the backends to run.

## License

Apache License 2.0. See `LICENSE`.
