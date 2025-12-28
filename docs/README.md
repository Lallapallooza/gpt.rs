# Documentation

This directory contains the docs we consider worth keeping up to date.

Doc policy:
- If a doc stops matching the code, fix it immediately or delete it.
- Prefer linking to `--help` output / source paths over copying long CLI/API listings.

## Start here
- `testing.md`: how correctness/perf is validated (backend suites, Torch parity, Python baselines).
- `howto.md`: recipes for adding models/layers/functionals/backends (kept short and code-linked).

## Code orientation
- `architecture.md`: crate/module layout and data flow.
- `frontend.md`: models/layers/functionals/backends + runtime overrides.
- `code_map.md`: where the functional/capture plumbing lives.

## Reference
- `backend.md`: PTIR backend contract (ptir.v0.4). Treat as a spec.
- `ops.md`: ops/capture/graph execution architecture (implementation-oriented).
- `backend_optimizer.md`: PTIR rewrite/optimizer architecture (implementation-oriented).
