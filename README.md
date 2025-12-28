# gpt.rs

Pure Rust experimentation toolkit for portable tensor programs (PTIR) and small model implementations.

This workspace contains:

- `crates/gpt-rs`: core library (tensors, PTIR capture, layers, models, tokenizer, checkpoints).
- `crates/gpt-rs-cli`: model runner CLI (`generate` / `forward` / `trace`) with PTIR dumping + profiling hooks.
- `scripts/eval.py`: Torch-baselined validate/trace/bench runner for full models.

## Architecture (high level)

```text
inputs (tokens / images)
        |
        v
  model::* (Gpt, ResNet34, MobileNetV2, ...)
        |
        v
   nn::layers::* (Linear, LayerNorm, Attention, Conv2d, ...)
        |
        v
ops::functional::* (matmul, layer_norm, conv2d_nhwc, ...)
        |                  \
        |                   +-- runtime overrides (FunctionalRegistry / FunctionalOverrides)
        v
backend::spec::PortableBackend (dot_general, reduce_window, gather, elementwise, ...)
        |
        v
backend impl (faer, ref-cpu, ...)
```

- Layers never “open code” kernels; they call portable functionals.
- Functionals are pure math + PTIR capture; they can be swapped at runtime via the registry/overrides.

## Quick Start

```bash
# build the workspace (requires Rust toolchain)
cargo build

# run GPT-2 generation (checkpoint + tokenizer)
# export the checkpoint/tokenizer with: `uv run python scripts/export_gpt2.py --help`
cargo run --release -p gpt-rs-cli -- generate --prompt "Hello" --max-tokens 64 \
  --checkpoint checkpoints/gpt2.bin --tokenizer configs/gpt2_tokenizer.json

# export torchvision weights (gpt.rs checkpoint)
uv sync
uv run python scripts/export_model_weights.py --model resnet34 --out checkpoints/resnet34.bin

# run an image model (deterministic random input by default)
cargo run --release -p gpt-rs-cli -- forward --checkpoint checkpoints/resnet34.bin
```

## Torch Baselines (Python)

The canonical entrypoint is `scripts/eval.py` (validate / trace / bench) which runs both:

```bash
uv sync
uv pip install maturin
cd crates/gpt-rs-py && uv run maturin develop --release --features faer && cd ../..

uv run python scripts/eval.py --model resnet34 --workload validate
uv run python scripts/eval.py --model mobilenet_v2 --workload trace --stop-on-first-mismatch
uv run python scripts/eval.py --model gpt2 --workload bench --threads 1 4 --bench-tokens 1 64
```

See `docs/README.md` for a doc index and `docs/testing.md` for dumps/profiling details.

## Testing

```bash
cargo test  # workspace unit + integration tests (no Torch)

cargo test -p gpt-rs-backend-faer --test backend_suite  # backend smoke tests
cargo test -p gpt-rs-backend-faer --features torch --test backend_suite -- --nocapture  # Torch parity + timings (libtorch via tch)
cargo test -p gpt-rs --features torch --test torch_parity  # smaller parity set (ref-cpu backend)
```

Torch parity tests live under `crates/gpt-rs-backend-tests/src/torch_parity/` and are wired into each backend via
`define_backend_tests!` (behind the `torch` feature). See `docs/testing.md`.

## Status

Forward inference for GPT-2 generation and image classification models (ResNet34, MobileNetV2) is implemented, with
portable PTIR kernels and Torch baselines for correctness.

The functional layer exposes portable math (elementwise ops, matmul, normalization, attention, conv/pool) via the
`DeviceTensorOps` extension trait while still delegating to backend PTIR execution.
