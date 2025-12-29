# Development notes

## Environment variables

Common:

- `GPTRS_BACKEND`: default backend for the CLI/Python runners (for example `faer`, `ref-cpu`, `c`).
- `GPTRS_EAGER=1`: forces tensors created from captured nodes to materialize immediately (useful for debugging fusion).
- `GPTRS_OPT_PRE_ITERS`: fixed-point iterations for the optimizer "pre" loop (default: 2).
- `GPTRS_OPT_POST_ITERS`: fixed-point iterations for the optimizer "post" loop (default: 4).
- `GPTRS_PASS_STATS=1`: emits per-pass optimizer stats into the trace sink (for example into `passes.jsonl` when using `--dump-dir`).

C backend only:

- `GPTRS_C_CACHE_DIR=/path`: override the on-disk cache directory used by the C backend.
- `GPTRS_C_CACHE_DEBUG=1`: enables extra cache/debug logging in the C backend.
- `GPTRS_PROFILE_BACKEND=1`: enables C backend profiling counters (requires building with C backend support).

Torch parity harness:

- `GPTRS_TORCH_PARITY_BACKENDS=...`: overrides which backends the parity harness runs against (see `crates/gpt-rs-backend-tests/src/torch_parity/harness.rs`).
