# Development notes

## Checks

`.pre-commit-config.yaml` holds every lint, format, and type check, and the Conventional Commits
message check. The Python tools run from the `uv.lock` environment. CI's `lint` job runs the same
hooks, except the commit-message check.

```bash
uv sync --extra dev
uv run --extra dev pre-commit install         # run the checks on every commit
uv run --extra dev pre-commit run --all-files # run them on the whole tree
```

The `cargo-fmt` hook runs the nightly rustfmt that it names, since only nightly applies the import
options of `rustfmt.toml`: it groups and merges imports. Install that nightly once, with the date
from the hook's `RUSTUP_TOOLCHAIN`:
`rustup toolchain install --profile minimal --component rustfmt nightly-<date>`. rustfmt groups a
bare module path with the external crates, so import a child module as `self::<module>::`.

## Environment variables

Common:

- `GPTRS_BACKEND`: the CLI backend when `--backend` is not given (default `faer`).
- `GPTRS_EAGER=1`: forces tensors created from recorded nodes to materialize immediately (useful for debugging fusion).
- `GPTRS_TRACE_CHECK=1`: records every call of a traced entry point, such as an image model's forward. A call fails when its graph differs from the one its trace replays ([ops.md](ops.md)).
- `GPTRS_PASS_STATS=1`: with `--dump-dir`, writes one line per pass to `passes.jsonl` of the dump directory: its options, whether it changed the module, the ops left, and the pass's own statistics.

x86 backend and targets:

- `GPTRS_TARGET`: the target when the CLI's `--target` or Python's `target=` is not given: a preset, `host` (the default) or a target file.
- `GPTRS_WINOGRAD=1`: as `--winograd`, off by default: 3x3 convolutions with stride 1 run as Winograd's F(2x2, 3x3), which rounds differently from the convolution.
- `GPTRS_X86_CACHE_DIR=/path`: the directory of compiled libraries (default `gpt-rs/x86` in the user's cache directory).
- `GPTRS_TARGET_CACHE_DIR=/path`: the directory of the probed `host` target, which the x86 backend and `gpt-rs-cli target` read (default `gpt-rs/targets` in the user's cache directory).

C backend only:

- `GPTRS_C_CACHE_DIR=/path`: the on-disk cache directory of the C backend (default `gpt_rs_c_backend` in the system's temporary directory). When it is unset, the Torch parity harness gives each run a directory of its own.
- `GPTRS_C_CACHE_DEBUG=1`: enables extra cache/debug logging in the C backend.
- `GPTRS_PROFILE_BACKEND=1`: enables C backend profiling counters (requires building with C backend support).

C and x86 backends:

- `OMP_NUM_THREADS=N`: the team size of the C backend's generated code (default: the number of physical cores, at most the number of CPUs the process can use). x86 does not read it: each parallel loop runs one thread per iteration, at most one per unit of its execution level.
- `CC`: C compiler used for generated modules (default `cc`). The x86 backend requires clang.
- `OMP_PLACES`, `OMP_PROC_BIND`, `KMP_AFFINITY`: where OpenMP puts its threads. When none is set, `gpt-rs-cli` sets `OMP_PLACES=cores OMP_PROC_BIND=close`, one thread on each core, in order.

Filetests:

- `GPTRS_FILETEST=<text>`: runs only the filetest files whose path contains `<text>`.

Torch parity harness:

- `GPTRS_TORCH_PARITY_BACKENDS=faer,c`: the backends whose rows the parity harness waits for before it prints its timing table. `expected_backends` in `crates/gpt-rs-backend-tests/src/torch_parity/harness.rs` holds the default.
