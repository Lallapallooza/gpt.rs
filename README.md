# gpt.rs

Rust-first experimentation toolkit for **portable tensor programs** (PTIR) and small model implementations.

Write models once, keep the math portable, and let backends compete on execution:
- Layers call portable functionals (no open-coded kernels).
- Functionals capture PTIR graphs and can be overridden at runtime.
- Backends execute PTIR and can rewrite/fuse portable graphs into custom kernels.
- Parameters have stable `u128` ids and can be streamed lazily from checkpoints.

## Highlights

- **Capability-based runtime**: `runtime::load_model` returns a dynamic model handle; the CLI runs `generate` / `forward` / `trace` without hardcoding model kinds. See [docs/runtime.md](docs/runtime.md).
- **Parameter streaming + stable ids**: checkpoint-backed `ParamSource` loads weights on demand; backends can memoize derived parameter formats by stable id. See [docs/howto.md](docs/howto.md) and [docs/formats.md](docs/formats.md).
- **Backend rewrites**: pattern-driven PTIR rewrites via `#[ptir_pattern]` views and backend optimizer passes. See [docs/backend_optimizer.md](docs/backend_optimizer.md) (and [crates/gpt-rs-backend-c/src/optimizer/conv2d.rs](crates/gpt-rs-backend-c/src/optimizer/conv2d.rs) for a real example).
- **Correctness tooling**: Torch parity at the kernel level and end-to-end model baselines via Python runners. See [docs/testing.md](docs/testing.md).
- **Debuggability**: PTIR dumps (`--dump-dir`), profiling (`--profile` with `-F gpt-rs/profiler`), and eager debugging (`GPTRS_EAGER=1`).

## Docs

Start here:
- [docs/README.md](docs/README.md) (doc index + policy)
- [docs/howto.md](docs/howto.md) (add models/layers/functionals/backends)
- [docs/runtime.md](docs/runtime.md) (loader, capability dispatch, overrides)
- [docs/testing.md](docs/testing.md) (Torch parity + dumps/profiling + Python baselines)
- [docs/formats.md](docs/formats.md) (checkpoint + tensor archive formats)

Reference:
- [docs/backends/README.md](docs/backends/README.md) (how each backend works: ref-cpu, faer, c)
- [docs/backend.md](docs/backend.md) (PTIR backend contract, ptir.v0.4)
- [docs/backend_optimizer.md](docs/backend_optimizer.md) (optimizer pipeline + patterns)
- [docs/ops.md](docs/ops.md) (PTIR capture/graphs/execution)
- [docs/frontend.md](docs/frontend.md) (frontend layering + runtime overrides)

Scripts:
- [scripts/README.md](scripts/README.md) (Python utilities: export + eval)

## Repository layout

- `crates/gpt-rs`: core library (tensors, PTIR capture, layers, models, tokenizer, checkpoints, runtime).
- `crates/gpt-rs-cli`: model runner CLI (`generate` / `forward` / `trace`) with dump/profile hooks.
- `crates/gpt-rs-backend-*`: backend implementations (faer, ref-cpu, optional C backend).
- `crates/gpt-rs-backend-tests`: backend suite + Torch parity harness (via `tch` / libtorch).
- `scripts/`: Python baselines and exporters (`scripts/eval.py`, `scripts/export_gpt2.py`, ...).

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

## Quick Start

```bash
# build the workspace (requires Rust toolchain)
cargo build

# explore the CLI surface
cargo run -p gpt-rs-cli -- --help

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

Notes:
- Select a backend with `--backend` (or `GPTRS_BACKEND`, default: `faer`).
- `--profile` prints tables only when built with `-F gpt-rs/profiler`.
- Use `--dump-dir` / `--dump-mode` to capture PTIR programs (Rust CLI or Python runner).

## Testing

```bash
cargo test  # workspace unit + integration tests (no Torch)

cargo test -p gpt-rs-backend-faer --test backend_suite  # backend smoke tests
cargo test -p gpt-rs-backend-faer --features torch --test backend_suite -- --nocapture  # Torch parity + timings (libtorch via tch)
cargo test -p gpt-rs --features torch --test torch_parity  # smaller parity set (ref-cpu backend)
```

Torch parity tests live under `crates/gpt-rs-backend-tests/src/torch_parity/` and are wired into each backend via
`define_backend_tests!` (behind the `torch` feature). See [docs/testing.md](docs/testing.md).

## Status

Forward inference for GPT-2 generation and image classification models (ResNet34, MobileNetV2) is implemented, with
portable PTIR kernels and Torch baselines for correctness.

The functional layer exposes portable math (elementwise ops, matmul, normalization, attention, conv/pool) via the
`DeviceTensorOps` extension trait while still delegating to backend PTIR execution.
