# Testing Guide

This document explains how the gpt.rs workspace is tested, with a focus on the
"backend suite" + Torch parity framework that runs numerical checks across all
backends.

## Overview

There are three main categories of tests:

1. **Library/unit/integration tests** (in the `gpt-rs` crate)
2. **Backend suite tests** (one `backend_suite` per backend crate)
3. **Python validation scripts** (manual tools under `scripts/`)

Each category answers a different question:

- **"Does capture/validation behave correctly?"** -> `crates/gpt-rs/tests/`
- **"Do backend kernels produce the same numbers as Torch?"** -> backend suites
- **"Do full models match torchvision/HF end-to-end?"** -> Python scripts

## 1) gpt-rs crate tests (PTIR inspection)

Portable ops live in `crates/gpt-rs/src/ops/functional/` and are typically
implemented as:

- a validation phase that checks shapes/dtypes and plans the op
- a capture phase that emits PTIR for the backend to execute

Tests under `crates/gpt-rs/tests/` focus on:

- PTIR capture shape rules and error messages
- graph structure and invariants
- non-numerical behavior (rejecting invalid inputs consistently)

Run them with:

```bash
cargo test -p gpt-rs
```

## 2) Backend suites + Torch parity (numerical correctness)

### Where the tests live

The shared backend test harness is the crate:

- `crates/gpt-rs-backend-tests/`

It contains:

- `src/smoke.rs`: quick sanity tests (shapes, training updates, etc.)
- `src/torch_parity/`: numerical parity tests against Torch (via `tch`)
- `src/lib.rs`: the `define_backend_tests!` macro that wires everything together

Each backend crate (faer/cpu/...) then contains a single test entrypoint:

- `crates/<backend-crate>/tests/backend_suite.rs`

That file calls the macro and provides a constructor that returns a backend
instance, e.g.:

```rust
gpt_rs_backend_tests::define_backend_tests!(faer_backend_tests, || {
    std::sync::Arc::new(gpt_rs_backend_faer::FaerCpuBackend::create())
});
```

Because the tests are *generated inside the backend crate*, they run with
`cargo test -p <backend-crate> ...` and are automatically "populated" for every
backend that includes a `backend_suite.rs`.

### How Torch parity tests work

Torch parity tests live under:

- `crates/gpt-rs-backend-tests/src/torch_parity/`

They:

1. Generate deterministic random inputs (seeded RNG).
2. Compute a Torch reference output via `tch` (libtorch).
3. Compute a gpt.rs output by calling the portable op / layer against the
   provided backend.
4. Compare using a strict-ish tolerance (typically `assert_close` in
   `torch_parity/common.rs`).

Most tests take the form:

```rust
pub fn some_op_matches_torch<B: PortableBackend + 'static>(backend: &Arc<B>) {
    // build inputs -> torch -> gpt-rs -> assert_close
}
```

### How to run them

By default, backend suites run **smoke tests only**. Torch parity tests are
behind the `torch` feature (to avoid pulling `tch`/libtorch into every
`cargo test`).

Smoke tests (no Torch):

```bash
cargo test -p gpt-rs-backend-faer --test backend_suite
```

Torch parity (smoke + parity):

```bash
cargo test -p gpt-rs-backend-faer --features torch --test backend_suite
```

Run a single parity test by name filter:

```bash
cargo test -p gpt-rs-backend-faer --features torch --test backend_suite torch_vision_conv2d_nhwc_matches_torch
```

List all available backend suite tests:

```bash
cargo test -p gpt-rs-backend-faer --features torch --test backend_suite -- --list
```

There is also a smaller Torch parity entrypoint in the `gpt-rs` crate:

```bash
cargo test -p gpt-rs --features torch --test torch_parity
```

Without `--features torch`, the `torch_parity` integration test target exists
but runs 0 tests (this is expected).

#### Libtorch notes

The `torch` feature enables the optional `tch` dependency. By default this
workspace enables `tch`'s `download-libtorch` feature, so it may download a
libtorch distribution on first build. To use an existing install instead, set
`LIBTORCH` to your libtorch directory (and ensure the dynamic loader can find
its libraries).

### Timing output (Torch parity)

Each torch parity test runs twice:

- **baseline**: optimizer passes disabled
- **optimized**: optimizer passes enabled

Timings are printed as a compact table (one row per test) including baseline/optimized
times (gpt execution only; compile/optimizer time excluded), Torch reference time, and
speedup. Use `--nocapture` to see the output:

```bash
cargo test -p gpt-rs-backend-faer --features torch --test backend_suite -- --nocapture
```

### Torch parity tolerances

Tolerances default to the hardcoded `ATOL`/`RTOL` in
`crates/gpt-rs-backend-tests/src/torch_parity/common.rs`. Optional overrides
can be provided via `configs/torch_parity.json` (if the file does not exist,
defaults are used). Rules accept optional `backend` and `test` patterns using
`*` wildcards. Precedence is:

`default` < `backend` < `test` < `backend+test`

Later rules of the same specificity win.

Example:

```json
{
  "default": { "atol": 5e-4, "rtol": 1e-4 },
  "rules": [
    { "backend": "faer", "atol": 1.5e-3, "rtol": 2e-4 },
    { "test": "torch_feed_forward_*", "atol": 1.5e-3 },
    { "backend": "faer", "test": "torch_feed_forward_*", "atol": 2e-3, "rtol": 2e-4 }
  ]
}
```

### Adding a new Torch parity test (the "right" way)

1. Implement the test function in `crates/gpt-rs-backend-tests/src/torch_parity/`
   (either in an existing module or a new one exported by
   `torch_parity/mod.rs`).
2. Wire it into `define_backend_tests!` in
   `crates/gpt-rs-backend-tests/src/lib.rs` by calling your function from a
   `#[test]` body.

That is what makes the test run "normally" across all backends.

Avoid writing backend-specific parity test files like:

- `crates/gpt-rs-backend-*/tests/*.rs`

unless the test is truly backend-specific (e.g., testing a backend-only kernel
or feature flag). For cross-backend correctness, the shared harness is the
source of truth.

## 3) Python model validation scripts

For end-to-end model checks (e.g., torchvision ResNet/MobileNet), use the
scripts under `scripts/`. These are intentionally *not* part of `cargo test`,
because they depend on Python packages, the `gpt_rs` Python bindings, and external model downloads.

Prerequisites (once per checkout):

```bash
uv sync
uv pip install maturin
cd crates/gpt-rs-py && uv run maturin develop --release --features faer && cd ../..
```

The canonical entrypoint is `scripts/eval.py`, which runs both gpt-rs and a
Torch baseline for:

- `validate`: compare logits
- `trace`: compare intermediate activations (when the model adapter supports it)
- `bench`: compare throughput/latency

Examples:

```bash
uv run python scripts/eval.py --model resnet34 --workload validate
uv run python scripts/eval.py --model mobilenet_v2 --workload trace --stop-on-first-mismatch
uv run python scripts/eval.py --model gpt2 --workload bench --threads 1 4 --bench-tokens 1 64
uv run python scripts/eval.py --suite scripts/eval_suites/vision.json
```

### Dumps + profiling

- Dump PTIR programs while validating a vision model (requires rebuilding the Python extension after
  enabling the new debug helpers):
  ```bash
  uv run python scripts/eval.py --model resnet34 --workload validate --dump-dir dumps/resnet34_validate
  ```

- Print profiler tables from Python (requires `gpt_rs` built with profiler support):
  ```bash
  # Rebuild python extension with profiler enabled:
  #   cd crates/gpt-rs-py && uv run maturin develop --release --features faer,profiler
  uv run python scripts/eval.py --model resnet34 --workload run --profile
  ```

- Dump + profile a GPT-2 generation run (Python extension):
  ```bash
  # Rebuild python extension with profiler enabled:
  #   cd crates/gpt-rs-py && uv run maturin develop --release --features faer,profiler
  uv run python scripts/eval.py --model gpt2 --workload run \
    --prompt "Hi" --max-tokens 128 --temperature 0.8 \
    --checkpoint checkpoints/gpt2.bin --tokenizer configs/gpt2_tokenizer.json \
    --backend faer --kv-cache \
    --dump-dir dumps/gpt2_ptir_new_e --dump-mode all \
    --profile
  ```

Legacy wrapper scripts have been removed; use `scripts/eval.py` directly with
`--workload validate|trace|bench|run`.

## 4) Model-agnostic CLI runner (gpt-rs-cli)

`gpt-rs-cli` is a thin runner around `gpt-rs` that can execute different model kinds
with the same debugging hooks:

- `--dump-dir` / `--dump-mode` to dump PTIR programs
- `--profile` to print gpt-rs profiler tables

### GPT run (generation)

```bash
cargo run --release -p gpt-rs-cli -F gpt-rs/profiler -- run gpt \
  --prompt "Hi" \
  --max-tokens 128 \
  --temperature 0.8 \
  --checkpoint checkpoints/gpt2.bin \
  --tokenizer configs/gpt2_tokenizer.json \
  --kv-cache \
  --dump-dir dumps/gpt2_ptir \
  --dump-mode compile \
  --profile
```

### Vision run (ResNet34 / MobileNetV2)

First export weights from Torch into a portable tensor archive:

```bash
uv run python scripts/export_model_weights.py --model resnet34 --out checkpoints/resnet34.tensors
uv run python scripts/export_model_weights.py --model mobilenet_v2 --out checkpoints/mobilenet_v2.tensors
```

Then run the model (generates a deterministic random input unless you pass `--input`):

```bash
cargo run --release -p gpt-rs-cli -F gpt-rs/profiler -- run resnet34 --weights checkpoints/resnet34.tensors --profile
cargo run --release -p gpt-rs-cli -F gpt-rs/profiler -- run mobilenet_v2 --weights checkpoints/mobilenet_v2.tensors --profile
```

### Vision trace (layer-by-layer dumps)

```bash
cargo run --release -p gpt-rs-cli -- trace resnet34 \
  --weights checkpoints/resnet34.tensors \
  --out dumps/resnet34_trace.tensors
```
