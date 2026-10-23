# Documentation

This directory contains the docs we consider worth keeping up to date.

Doc policy:
- If a doc stops matching the code, fix it immediately or delete it.
- Prefer linking to `--help` output / source paths over copying long CLI/API listings.

## Start here
- [testing.md](testing.md): how correctness/perf is validated (backend suites, Torch parity, Python baselines).
- [howto.md](howto.md): recipes for adding models/layers/functionals/backends (kept short and code-linked).

## Code orientation
- [architecture.md](architecture.md): crate/module layout and data flow.
- [runtime.md](runtime.md): checkpoint loading, capability dispatch.
- [frontend.md](frontend.md): models/layers/functionals/backends.
- [code_map.md](code_map.md): where the functional/recording plumbing lives.

## Reference
- [backends/README.md](backends/README.md): the backend interface, and how each backend executes PTIR (ref-cpu, faer, c, x86).
- [ptir/tensor.md](ptir/tensor.md): the `tensor` dialect, the PTIR spec, generated from the code.
- [ptir/arith.md](ptir/arith.md), [ptir/tile.md](ptir/tile.md), [ptir/loop.md](ptir/loop.md),
  [ptir/isa.md](ptir/isa.md) and [ptir/host.md](ptir/host.md): the scalar, tile, loop, vector and
  host dialects, generated from the code.
- [ops.md](ops.md): ops/recording/graph execution architecture (implementation-oriented).
- [ptir/passes.md](ptir/passes.md): the passes and the pipeline of each backend, generated from
  the code.
- [formats.md](formats.md): checkpoints (Hugging Face model directories) and safetensors tensor
  files.
- [development.md](development.md): env vars and local development notes.
- [perf.md](perf.md): timing and profiling the ops, layers and blocks of GPT-2, ResNet-18 and
  MobileNetV2, down to the lines of generated C and of Rust.
