# Testing Guide

This document explains how the gpt.rs workspace is tested, with a focus on the
"backend suite" + Torch parity framework that runs numerical checks across all
backends.

## Overview

The main categories of tests:

1. **Filetests and the law test** (in `gpt-rs-ir`, and the filetests of the C backend's passes in
   `gpt-rs-backend-c`)
2. **Contract tests** (in the crate that owns the API, mostly `gpt-rs`)
3. **Backend suite tests** (one `backend_suite` per backend crate)
4. **Python validation scripts** (manual tools under `scripts/`)

Each category answers a different question:

- **"Does a pass of the IR or of a backend, the text form or the verifier do the right thing?"**
  -> `filetests/`
- **"Does an op's trait hold for its semantics?"** -> `crates/gpt-rs-ir/tests/laws.rs`
- **"Does a public API keep its documented behavior?"** -> `crates/<crate>/tests/`
- **"Do backend kernels produce the same numbers as Torch?"** -> backend suites
- **"Do full models match Hugging Face end-to-end?"** -> Python scripts

## 1) Filetests and the law test

Filetests live under `filetests/` at the repository root:

- `filetests/tensor/<topic>/*.ptir` and `filetests/tile/<topic>/*.ptir`, for the IR passes and the
  text form;
- `filetests/backend/<backend>/*.ptir`, for the passes that a backend adds to its pipeline
  (`PortableBackend::pipeline()`).

One runner, the `gpt-rs-filetests` crate, runs them, and its crate doc gives the format. `gpt-rs-ir`
runs `filetests/tensor/` and `filetests/tile/` from its `tests/filetests.rs`. A backend crate with
passes of its own runs its directory from its `tests/filetests.rs`, with each pass of its pipeline
under its name and the whole pipeline as `default`.

The cases of a backend start from the IR that the functionals record. A run with `--dump-dir`
writes the module after each pass to `passes/`.

`crates/gpt-rs-ir/tests/laws.rs` evaluates every equation an op's traits claim (commutativity,
identities, ...) through the op's semantics, for every op of the roundtrip filetests.

```bash
cargo test -p gpt-rs-ir
cargo test -p gpt-rs-backend-c --test filetests
```

## 2) Contract tests

Tests under `crates/<crate>/tests/` check public behavior with an exact documented answer:
validation messages of functionals, checkpoint loading, tokenizer ids, program counts and cache
hits.

```bash
cargo test -p gpt-rs
```

## 3) Backend suites + Torch parity (numerical correctness)

### Where the tests live

The shared backend test harness is the crate:

- `crates/gpt-rs-backend-tests/`

It contains:

- `src/torch_parity/`: numerical parity tests against Torch (via `tch`)
- `src/lib.rs`: the `define_backend_tests!` macro that wires everything together

Each backend crate (ref-cpu, faer, c) then contains a single test entrypoint:

- `crates/<backend-crate>/tests/backend_suite.rs`

That file calls the macro and provides a constructor that returns a backend
instance, e.g.:

```rust
gpt_rs_backend_tests::define_backend_tests!(faer_backend_tests, || {
    std::sync::Arc::new(gpt_rs_backend_faer::FaerPortableBackend::new())
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

Backend suites hold Torch parity tests only. The `torch` feature gates them,
so a plain `cargo test` does not build `tch` or libtorch. Without the feature,
a backend suite runs 0 tests.

```bash
cargo test -p gpt-rs-backend-faer --features torch --test backend_suite
```

Run Torch parity across the workspace (default members) from the repo root:

```bash
cargo test --features torch
```

Run a single parity test by name filter:

```bash
cargo test -p gpt-rs-backend-faer --features torch --test backend_suite torch_vision_conv2d_nhwc_matches_torch
```

List all available backend suite tests:

```bash
cargo test -p gpt-rs-backend-faer --features torch --test backend_suite -- --list
```

#### Libtorch notes

The `torch` feature enables the optional `tch` dependency. By default this
workspace enables `tch`'s `download-libtorch` feature, so it may download a
libtorch distribution on first build. To use an existing install instead, set
`LIBTORCH` to your libtorch directory (and ensure the dynamic loader can find
its libraries).

### Timing output (Torch parity)

Each Torch parity test adds one row to a table: the Torch reference time and, for each backend,
the time its programs ran, which a trace sink measures. Recording and compiling are left out. The
harness prints the table when the last backend in `GPTRS_TORCH_PARITY_BACKENDS`
([development.md](development.md#environment-variables)) finishes its suite:

```bash
GPTRS_TORCH_PARITY_BACKENDS=faer cargo test -p gpt-rs-backend-faer --features torch --test backend_suite
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
    { "test": "torch_attention_kv_cache_*", "atol": 1.5e-3 },
    { "backend": "faer", "test": "torch_attention_kv_cache_*", "atol": 2e-3, "rtol": 2e-4 }
  ]
}
```

### Adding a new Torch parity test (the "right" way)

1. Implement the test function in `crates/gpt-rs-backend-tests/src/torch_parity/`
   (either in an existing module or a new one exported by
   `torch_parity/mod.rs`).
2. Wire it into `define_backend_tests!` in
   `crates/gpt-rs-backend-tests/src/lib.rs` with one line,
   `parity_test!(torch_<name>, <module>::<function>);`. A new module also joins the
   `use $crate::torch_parity::{...}` list at the top of the macro.

That is what makes the test run across all backends.

Torch parity never goes in a backend crate: the numbers a backend computes are checked in the
shared harness, and a backend's own passes are checked with filetests under
`filetests/backend/<backend>/`.

## 4) Python model validation scripts

For end-to-end model checks against Hugging Face `transformers`, use the
scripts under `scripts/`. These are intentionally *not* part of `cargo test`,
because they depend on Python packages, the `gpt_rs` Python bindings, and external model downloads.

Prerequisites (once per checkout):

```bash
uv sync
uv pip install maturin
uv run python scripts/rebuild_py.py --features faer
```

The canonical entrypoint is `scripts/eval.py`, which runs both gpt-rs and a Torch baseline on the
same Hugging Face checkpoint (`--hf-model`, a Hub id or a local directory) for:

- `validate`: compare logits. The projections take f32 inputs, so the logits match a float32
  reference to f32 precision. Outside validate, a bf16 checkpoint runs the projections in bf16 by
  default. Qwen3.5's `--matmul-input-dtype bf16` validates that path. The logits then move by up
  to bf16 rounding, about 3e-3 of their maximum on four layers, so that run needs a larger
  `--max-rel-diff`. The top-1 tokens must still agree.
- `bench`: compare throughput/latency
- `run`: run gpt-rs alone (generation, or a forward pass of an image model)
- `all`: every workload the case supports

Examples:

```bash
uv run python scripts/eval.py --model resnet34 --workload validate
uv run python scripts/eval.py --model gpt2 --workload bench --threads 1 4 --bench-tokens 1 64
uv run python scripts/eval.py --model ministral_3_3b_instruct_2512 --workload validate
uv run python scripts/eval.py --suite scripts/eval_suites/vision.json
```

Qwen3.5 models, for example `Qwen/Qwen3.8-27B`, mix Gated DeltaNet and gated attention layers.
`--num-layers N` runs only the first N decoder layers in gpt-rs and in the Torch baseline.
`--backend c` needs the extension built with `conversion-c`
(`uv run python scripts/rebuild_py.py --features faer,conversion-c`):

```bash
uv run python scripts/eval.py --model qwen3_8_27b --workload validate --backend c \
  --prompt-tokens 300 --generate-tokens 3
uv run python scripts/eval.py --model qwen3_8_27b --workload bench --backend c --threads 8 \
  --prompt-tokens 256 --bench-tokens 8
```

See `uv run python scripts/eval.py --help` for the remaining flags.

### Dumps + profiling

Notes:
- `--profile` prints tables only when built with `-F gpt-rs/profiler` (otherwise gpt-rs-cli warns).
- `--dump-dir` and `--dump-mode` work without the profiler feature.

- Dump the modules the backend runs while validating a vision model:
  ```bash
  uv run python scripts/eval.py --model resnet34 --workload validate --dump-dir dumps/resnet34_validate
  ```

- Print profiler tables from Python (requires `gpt_rs` built with profiler support):
  ```bash
  # Rebuild python extension with profiler enabled:
  #   uv run python scripts/rebuild_py.py --features faer,profiler
  uv run python scripts/eval.py --model resnet34 --workload run --profile
  ```

- Dump + profile a GPT-2 generation run (Python extension):
  ```bash
  # Rebuild python extension with profiler enabled:
  #   uv run python scripts/rebuild_py.py --features faer,profiler
  uv run python scripts/eval.py --model gpt2 --workload run \
    --prompt "Hi" --max-tokens 128 --temperature 0.8 \
    --backend faer \
    --dump-dir dumps/gpt2 --dump-mode all \
    --profile
  ```

For Ministral text runs, `scripts/eval.py --model ministral_3_3b_instruct_2512 --workload run`
tokenizes with the checkpoint's `transformers` `AutoTokenizer`, because `gpt_rs.Tokenizer` does not
support the `ignore_merges` option that its `tokenizer.json` sets.

### Debugging knobs (env vars)

See [development.md](development.md#environment-variables).

## 5) Model-agnostic CLI runner (gpt-rs-cli)

`gpt-rs-cli` is a thin runner around `gpt-rs` that can execute different model kinds
with the same debugging hooks:

- `--dump-dir` / `--dump-mode` to dump the module of each program the backend runs
- `--profile` to print gpt-rs profiler tables
- `--backend` (or `GPTRS_BACKEND`) to choose a backend (default: `faer`)

`--checkpoint` takes a local Hugging Face model directory, for example one fetched with
`uv run hf download gpt2 --local-dir checkpoints/gpt2 --include "*.json" --include "model*.safetensors"`.

### Text generation (any causal LM)

```bash
cargo run --release -p gpt-rs-cli -F gpt-rs/profiler -- generate \
  --prompt "Hi" \
  --max-tokens 128 \
  --temperature 0.8 \
  --checkpoint checkpoints/gpt2 \
  --dump-dir dumps/gpt2_ptir \
  --dump-mode compile \
  --profile
```

### Forward pass

Download the checkpoint and run the model. The CLI generates a deterministic random input unless
you pass `--input`, a safetensors file:

```bash
uv run hf download microsoft/resnet-34 --local-dir checkpoints/resnet-34 \
  --include "*.json" --include "model*.safetensors"
cargo run --release -p gpt-rs-cli -F gpt-rs/profiler -- forward --checkpoint checkpoints/resnet-34 --profile
```
