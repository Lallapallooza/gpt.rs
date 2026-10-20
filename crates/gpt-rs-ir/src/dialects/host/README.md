# The host dialect

The `host` dialect is the program that runs dispatches. It has two forms.

- **The value form.** A function calls dispatches on tensors with `host.dispatch`: `tile.dispatch`
  symbols, or the `loop.func` symbols they lower to. Each call is pure: its results depend on its
  arguments alone.
- **The buffer form**, which `plan-memory` gives. Every value that a dispatch computes has a
  buffer, and `host.launch` runs a `loop.func` on buffers: `ins` it reads, `outs` it writes and
  `inouts` it updates in place.
  - Buffers are slices of the function's arena (`host.arena`, `host.slice`), memory of their own
    for the values that the function returns or that an op other than a launch reads
    (`host.alloc`), or the memory of a tensor: `host.to_buffer` reads one, and `host.take` takes
    one that the function reads no more.
  - An arena and memory of its own start at a multiple of the alignment they state, which
    `plan-memory` takes from the target: each buffer of the target starts at a multiple of that
    many bytes.
  - `host.copy` copies a buffer, and `host.to_tensor` gives the tensor of a buffer's elements.
  - Launches and copies run one after another, in the function's order: each starts after every
    launch and copy before it in the function has ended. So a value that no later launch or copy
    reads may give its bytes to the next value.
  - The function keeps tensor arguments and results, so a buffer-form function computes what its
    value form computes, which the filetests check through the interpreter.

## Conventions

- Each op is defined in `ir.rs`. Its type rule is in `verify.rs`, its semantics, its `Eval` impl,
  are in `interp.rs`, and `utils.rs` holds the buffer types the ops give and the buffer of each
  parameter of a launch.
- A call binds the callee's dimension symbols to the extents of its arguments, one extent per
  symbol across all of them.
- An `out<over = k>` parameter of a call or a launch may write the elements of the `in` parameter
  `k`, so a launch reads parameter `k` only at positions that it has not written yet. The
  interpreter fails a read through parameter `k` of an element that the `out` parameter wrote.
- The interpreter checks the memory plan as it runs it. Each slice holds its elements in cells of
  its own, and the arena remembers which slice wrote each of its bytes last. It fails:
  - a launch, a copy or a `host.to_tensor` that reads a slice no launch and no copy wrote, or one
    whose bytes another slice wrote since;
  - a write into the buffer of `host.to_buffer`;
  - one buffer for a parameter that a launch writes and another one, except an `out<over = k>`
    parameter with the `in` parameter `k`.
