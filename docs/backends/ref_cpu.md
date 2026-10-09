# Reference CPU backend (`gpt-rs-backend-ref-cpu`)

The reference backend runs every function on the IR's interpreter (`gpt_rs_ir::interp::run`), so its
results are the ops' `Eval` semantics by definition. It is for debugging and bring-up: correctness
first, performance last. It has no passes of its own and no kernels.

Source of truth: [../../crates/gpt-rs-backend-ref-cpu/src/lib.rs](../../crates/gpt-rs-backend-ref-cpu/src/lib.rs).

- `compile` keeps the module and the function's name.
- `run` calls the interpreter on the arguments.
- Its tensors are `HostTensor`s, so `upload` and `download` pass them through, views of a mapped
  checkpoint included.

## Backend name

The backend's name is `ref-cpu`, the name that `gpt_rs_backends::load_model`, the CLI's `--backend`
and Python's `load_model(..., backend=...)` take.
