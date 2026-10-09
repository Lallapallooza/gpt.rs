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

## Environment variables

Common:

- `GPTRS_BACKEND`: the CLI backend when `--backend` is not given (default `faer`).
- `GPTRS_EAGER=1`: forces tensors created from recorded nodes to materialize immediately (useful for debugging fusion).
- `GPTRS_PASS_STATS=1`: with `--dump-dir`, writes one line per pass to `passes.jsonl` of the dump directory: its options, whether it changed the module, the ops left, and the pass's own statistics.

C backend only:

- `GPTRS_C_CACHE_DIR=/path`: override the on-disk cache directory used by the C backend.
- `GPTRS_C_CACHE_DEBUG=1`: enables extra cache/debug logging in the C backend.
- `GPTRS_PROFILE_BACKEND=1`: enables C backend profiling counters (requires building with C backend support).
- `OMP_NUM_THREADS=N`: OpenMP team size for generated modules (default: one thread per physical core
  that the process can use).
- `CC`: C compiler used for generated modules (default `cc`).

Torch parity harness:

- `GPTRS_TORCH_PARITY_BACKENDS=faer,c`: the backends whose rows the parity harness waits for before it prints its timing table. `expected_backends` in `crates/gpt-rs-backend-tests/src/torch_parity/harness.rs` holds the default.
