# Documentation

This directory contains current documentation for gpt.rs.

## Guides (current)
- `development.md`: developer workflow, CLI usage, profiling/debug hooks.
- `testing.md`: test structure (backend suites + Torch parity) and Python baselines.
- `gpt2_integration.md`: exporting + validating GPT-2 checkpoints.

## Architecture / design (current)
- `architecture.md`: crate/module layout and execution flow.
- `frontend.md`: models/layers/functionals/backends split and runtime overrides.
- `ops.md`: ops/graph/PTIR capture architecture.
- `backend.md`: PTIR backend contract (ptir.v0.4).
- `backend_optimizer.md`: PTIR rewrite system (patterns/driver/passes).
- `code_map.md`: quick map of the functional layer and capture flow.

## RFCs
- `rfcs/`: accepted/proposed design documents (implementation may lag).
  - `rfcs/0001-unified-module-api-and-parameter-streaming.md`
  - `rfcs/0002-ptir-pattern-matching.md`

## Drafts / notes / archive
- `drafts/`: proposals and sketches; may not match current code.
- `notes/`: research notes and background.
- `archive/`: historical one-off plans/requests kept for context.
