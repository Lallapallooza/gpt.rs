<!-- Generated from the code by `cargo run -p gpt-rs-docs`. Do not edit: change the code and run it again. -->

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

## Ops

- `host.dispatch`
- `host.arena`
- `host.slice`
- `host.alloc`
- `host.to_buffer`
- `host.take`
- `host.launch`
- `host.copy`
- `host.to_tensor`

### `host.dispatch`

Calls the dispatch named `callee` on `args` and gives its results. The callee
is a `tile.dispatch`, or the `loop.func` it lowers to.

- The arguments have the callee's argument types, with each of its dimension symbols bound to
  one extent across all of them. The results have its result types under that binding.
- A `loop.func` takes each argument in an `in` or `inout` parameter, in order, and gives each
  `inout` or `out` parameter as a result, in order. An `inout` parameter starts as a copy of its
  argument.
- It is pure: a call computes its results from its arguments only.

Defined by `Dispatch` in [crates/gpt-rs-ir/src/dialects/host/ir.rs](../../crates/gpt-rs-ir/src/dialects/host/ir.rs).

| | |
|---|---|
| Operands | `args`: tensors |
| Results | `outs`: tensors |
| Attributes | `callee`: `SymbolRef` |
| Effects | none |
| Patterns | `LayOutInWalkOrder` (`walk-order-layouts`) |

Example, from `filetests/tile/roundtrip/block.ptir`:

```text
tile.dispatch @rms(%input: tensor<{T}x8xf32>) -> tensor<{T}xf32> attributes {grid = [d0], facts = {T = [16, 4096] % 16}} {
  %sum_of_squares = tile.block(%input) {iters = [par {T}, red 8], reads = [$0[d0, d1]], folds = [sum -> [d0]]} ({
  ^body(%element: f32):
    %square = arith.mul(%element, %element) : (f32, f32) -> f32
    tile.yield(%square) : (f32) -> ()
  }) : (tensor<{T}x8xf32>) -> tensor<{T}xf32>
  func.return(%sum_of_squares) : (tensor<{T}xf32>) -> ()
}
func.func @call(%input: tensor<{T}x8xf32>) -> (tensor<{T}xf32>, tensor<{T}x8xf32>) {
  %sum_of_squares = host.dispatch(%input) {callee = @rms} : (tensor<{T}x8xf32>) -> tensor<{T}xf32>
  %exponentials = tile.group() {grid = [d0]} ({
    %block_exponentials = tile.block(%input) {iters = [par {T}, par 8], reads = [$0[d0, d1]], folds = [none -> [d0, d1]]} ({
    ^body(%element: f32):
      %exponential = arith.exp(%element) : (f32) -> f32
      tile.yield(%exponential) : (f32) -> ()
    }) : (tensor<{T}x8xf32>) -> tensor<{T}x8xf32>
    tile.yield(%block_exponentials) : (tensor<{T}x8xf32>) -> ()
  }) : () -> tensor<{T}x8xf32>
  func.return(%sum_of_squares, %exponentials) : (tensor<{T}xf32>, tensor<{T}x8xf32>) -> ()
}
```

Rejected, as the verify filetests pin it:

- `host.dispatch: @double is not a tile.dispatch or a loop.func`
- `host.dispatch: no symbol @missing`
- `host.dispatch: argument 0: tensor<4xsi32> does not match tensor<{T}xf32>`

### `host.arena`

`bytes` bytes of memory that buffers of the function's values live in, which a
runtime keeps from one call to the next. They start at a multiple of `alignment` bytes, a power
of two, and their buffer is of the space `arena` with that alignment, which no other op gives.
No byte holds a value when a call starts.

Defined by `Arena` in [crates/gpt-rs-ir/src/dialects/host/ir.rs](../../crates/gpt-rs-ir/src/dialects/host/ir.rs).

| | |
|---|---|
| Operands | none |
| Results | `output`: buffer |
| Attributes | `bytes`: `u64`; `alignment`: `u64` |
| Effects | allocates |

Example, from `filetests/host/roundtrip/buffers.ptir`:

```text
loop.func @double(%input: !loop.buf<4xf32, global>, %output: !loop.buf<4xf32, global>) attributes {access = [in, out<over = 0>]} {
  %zero = loop.index() {expr = 0} : () -> index
  %count = loop.index() {expr = 4} : () -> index
  loop.for(%zero, %count) ({
  ^body(%position: index):
    %element = loop.load(%input, %position) : (!loop.buf<4xf32, global>, index) -> f32
    %doubled = arith.add(%element, %element) : (f32, f32) -> f32
    loop.store(%doubled, %output, %position) : (f32, !loop.buf<4xf32, global>, index) -> ()
    loop.yield() : () -> ()
  }) : (index, index) -> ()
  loop.return() : () -> ()
}
loop.func @accumulate(%totals: !loop.buf<4xf32, global>, %addend: !loop.buf<4xf32, global>) attributes {access = [inout, in]} {
  %zero = loop.index() {expr = 0} : () -> index
  %count = loop.index() {expr = 4} : () -> index
  loop.for(%zero, %count) ({
  ^body(%position: index):
    %term = loop.load(%addend, %position) : (!loop.buf<4xf32, global>, index) -> f32
    %so_far = loop.load(%totals, %position) : (!loop.buf<4xf32, global>, index) -> f32
    %sum = arith.add(%so_far, %term) : (f32, f32) -> f32
    loop.store(%sum, %totals, %position) : (f32, !loop.buf<4xf32, global>, index) -> ()
    loop.yield() : () -> ()
  }) : (index, index) -> ()
  loop.return() : () -> ()
}
func.func @main(%input: tensor<4xf32>, %totals: tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>) {
  %arena = host.arena() {bytes = 128, alignment = 64} : () -> !loop.buf<128xui8, arena<alignment = 64>>
  %doubled = host.slice(%arena) {offset = 0, shape = [4], dtype = f32} : (!loop.buf<128xui8, arena<alignment = 64>>) -> !loop.buf<4xf32, global>
  %kept = host.slice(%arena) {offset = 64, shape = [4], dtype = f32} : (!loop.buf<128xui8, arena<alignment = 64>>) -> !loop.buf<4xf32, global>
  %result = host.alloc() {shape = [4], dtype = f32, alignment = 64} : () -> !loop.buf<4xf32, global>
  %input_buffer = host.to_buffer(%input) : (tensor<4xf32>) -> !loop.buf<4xf32, read_only>
  %totals_buffer = host.take(%totals) : (tensor<4xf32>) -> !loop.buf<4xf32, global>
  host.launch(%input_buffer, %doubled) {callee = @double, operand_segments = [1, 1, 0]} : (!loop.buf<4xf32, read_only>, !loop.buf<4xf32, global>) -> ()
  host.copy(%doubled, %kept) : (!loop.buf<4xf32, global>, !loop.buf<4xf32, global>) -> ()
  host.launch(%doubled, %doubled) {callee = @double, operand_segments = [1, 1, 0]} : (!loop.buf<4xf32, global>, !loop.buf<4xf32, global>) -> ()
  host.launch(%doubled, %totals_buffer) {callee = @accumulate, operand_segments = [1, 0, 1]} : (!loop.buf<4xf32, global>, !loop.buf<4xf32, global>) -> ()
  host.launch(%kept, %result) {callee = @double, operand_segments = [1, 1, 0]} : (!loop.buf<4xf32, global>, !loop.buf<4xf32, global>) -> ()
  %output = host.to_tensor(%result) : (!loop.buf<4xf32, global>) -> tensor<4xf32>
  %accumulated = host.to_tensor(%totals_buffer) : (!loop.buf<4xf32, global>) -> tensor<4xf32>
  func.return(%output, %accumulated) : (tensor<4xf32>, tensor<4xf32>) -> ()
}
```

Rejected, as the verify filetests pin it:

- `host.arena: an alignment of 48 bytes is not a power of two`

### `host.slice`

The buffer of `dtype` with the extents `shape` at byte `offset` of `arena`, whose
bytes it fits in. The offset is a multiple of the element size and of the arena's alignment.
Two slices are two buffers, even of the same bytes. Slices whose bytes overlap hold values at
different times: a buffer holds its elements from the launch or the copy that writes it, until
another buffer writes one of its bytes.

Defined by `Slice` in [crates/gpt-rs-ir/src/dialects/host/ir.rs](../../crates/gpt-rs-ir/src/dialects/host/ir.rs).

| | |
|---|---|
| Operands | `arena`: buffer |
| Results | `output`: buffer |
| Attributes | `offset`: `u64`; `shape`: `[usize]`; `dtype`: `DType` |
| Effects | allocates |

Example, from `filetests/host/roundtrip/buffers.ptir`:

```text
loop.func @double(%input: !loop.buf<4xf32, global>, %output: !loop.buf<4xf32, global>) attributes {access = [in, out<over = 0>]} {
  %zero = loop.index() {expr = 0} : () -> index
  %count = loop.index() {expr = 4} : () -> index
  loop.for(%zero, %count) ({
  ^body(%position: index):
    %element = loop.load(%input, %position) : (!loop.buf<4xf32, global>, index) -> f32
    %doubled = arith.add(%element, %element) : (f32, f32) -> f32
    loop.store(%doubled, %output, %position) : (f32, !loop.buf<4xf32, global>, index) -> ()
    loop.yield() : () -> ()
  }) : (index, index) -> ()
  loop.return() : () -> ()
}
loop.func @accumulate(%totals: !loop.buf<4xf32, global>, %addend: !loop.buf<4xf32, global>) attributes {access = [inout, in]} {
  %zero = loop.index() {expr = 0} : () -> index
  %count = loop.index() {expr = 4} : () -> index
  loop.for(%zero, %count) ({
  ^body(%position: index):
    %term = loop.load(%addend, %position) : (!loop.buf<4xf32, global>, index) -> f32
    %so_far = loop.load(%totals, %position) : (!loop.buf<4xf32, global>, index) -> f32
    %sum = arith.add(%so_far, %term) : (f32, f32) -> f32
    loop.store(%sum, %totals, %position) : (f32, !loop.buf<4xf32, global>, index) -> ()
    loop.yield() : () -> ()
  }) : (index, index) -> ()
  loop.return() : () -> ()
}
func.func @main(%input: tensor<4xf32>, %totals: tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>) {
  %arena = host.arena() {bytes = 128, alignment = 64} : () -> !loop.buf<128xui8, arena<alignment = 64>>
  %doubled = host.slice(%arena) {offset = 0, shape = [4], dtype = f32} : (!loop.buf<128xui8, arena<alignment = 64>>) -> !loop.buf<4xf32, global>
  %kept = host.slice(%arena) {offset = 64, shape = [4], dtype = f32} : (!loop.buf<128xui8, arena<alignment = 64>>) -> !loop.buf<4xf32, global>
  %result = host.alloc() {shape = [4], dtype = f32, alignment = 64} : () -> !loop.buf<4xf32, global>
  %input_buffer = host.to_buffer(%input) : (tensor<4xf32>) -> !loop.buf<4xf32, read_only>
  %totals_buffer = host.take(%totals) : (tensor<4xf32>) -> !loop.buf<4xf32, global>
  host.launch(%input_buffer, %doubled) {callee = @double, operand_segments = [1, 1, 0]} : (!loop.buf<4xf32, read_only>, !loop.buf<4xf32, global>) -> ()
  host.copy(%doubled, %kept) : (!loop.buf<4xf32, global>, !loop.buf<4xf32, global>) -> ()
  host.launch(%doubled, %doubled) {callee = @double, operand_segments = [1, 1, 0]} : (!loop.buf<4xf32, global>, !loop.buf<4xf32, global>) -> ()
  host.launch(%doubled, %totals_buffer) {callee = @accumulate, operand_segments = [1, 0, 1]} : (!loop.buf<4xf32, global>, !loop.buf<4xf32, global>) -> ()
  host.launch(%kept, %result) {callee = @double, operand_segments = [1, 1, 0]} : (!loop.buf<4xf32, global>, !loop.buf<4xf32, global>) -> ()
  %output = host.to_tensor(%result) : (!loop.buf<4xf32, global>) -> tensor<4xf32>
  %accumulated = host.to_tensor(%totals_buffer) : (!loop.buf<4xf32, global>) -> tensor<4xf32>
  func.return(%output, %accumulated) : (tensor<4xf32>, tensor<4xf32>) -> ()
}
```

Rejected, as the verify filetests pin it:

- `host.slice: !loop.buf<4x8xf32, global> at byte 192 does not fit in the 256 bytes of the arena`
- `host.slice: byte 2 is not a multiple of the 4 bytes of a f32 element`
- `host.slice: byte 16 is not a multiple of the 64 bytes that the arena aligns to`
- `host.slice: !loop.buf<16xf32, global> is not an arena, the buffer of host.arena`
- `host.slice: !loop.buf<64xui8, global> is not an arena, the buffer of host.arena`
- `host.slice: !loop.buf<16xui8, read_only> is not an arena, the buffer of host.arena`

### `host.alloc`

A buffer of `dtype` with the extents `shape`, in memory of its own that starts at
a multiple of `alignment` bytes, a power of two, for a value that the function returns or that
an op other than a launch reads. No element holds a value until a launch or a copy writes it.

Defined by `Alloc` in [crates/gpt-rs-ir/src/dialects/host/ir.rs](../../crates/gpt-rs-ir/src/dialects/host/ir.rs).

| | |
|---|---|
| Operands | none |
| Results | `output`: buffer |
| Attributes | `shape`: `[usize]`; `dtype`: `DType`; `alignment`: `u64` |
| Effects | allocates |

Example, from `filetests/host/roundtrip/buffers.ptir`:

```text
loop.func @double(%input: !loop.buf<4xf32, global>, %output: !loop.buf<4xf32, global>) attributes {access = [in, out<over = 0>]} {
  %zero = loop.index() {expr = 0} : () -> index
  %count = loop.index() {expr = 4} : () -> index
  loop.for(%zero, %count) ({
  ^body(%position: index):
    %element = loop.load(%input, %position) : (!loop.buf<4xf32, global>, index) -> f32
    %doubled = arith.add(%element, %element) : (f32, f32) -> f32
    loop.store(%doubled, %output, %position) : (f32, !loop.buf<4xf32, global>, index) -> ()
    loop.yield() : () -> ()
  }) : (index, index) -> ()
  loop.return() : () -> ()
}
loop.func @accumulate(%totals: !loop.buf<4xf32, global>, %addend: !loop.buf<4xf32, global>) attributes {access = [inout, in]} {
  %zero = loop.index() {expr = 0} : () -> index
  %count = loop.index() {expr = 4} : () -> index
  loop.for(%zero, %count) ({
  ^body(%position: index):
    %term = loop.load(%addend, %position) : (!loop.buf<4xf32, global>, index) -> f32
    %so_far = loop.load(%totals, %position) : (!loop.buf<4xf32, global>, index) -> f32
    %sum = arith.add(%so_far, %term) : (f32, f32) -> f32
    loop.store(%sum, %totals, %position) : (f32, !loop.buf<4xf32, global>, index) -> ()
    loop.yield() : () -> ()
  }) : (index, index) -> ()
  loop.return() : () -> ()
}
func.func @main(%input: tensor<4xf32>, %totals: tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>) {
  %arena = host.arena() {bytes = 128, alignment = 64} : () -> !loop.buf<128xui8, arena<alignment = 64>>
  %doubled = host.slice(%arena) {offset = 0, shape = [4], dtype = f32} : (!loop.buf<128xui8, arena<alignment = 64>>) -> !loop.buf<4xf32, global>
  %kept = host.slice(%arena) {offset = 64, shape = [4], dtype = f32} : (!loop.buf<128xui8, arena<alignment = 64>>) -> !loop.buf<4xf32, global>
  %result = host.alloc() {shape = [4], dtype = f32, alignment = 64} : () -> !loop.buf<4xf32, global>
  %input_buffer = host.to_buffer(%input) : (tensor<4xf32>) -> !loop.buf<4xf32, read_only>
  %totals_buffer = host.take(%totals) : (tensor<4xf32>) -> !loop.buf<4xf32, global>
  host.launch(%input_buffer, %doubled) {callee = @double, operand_segments = [1, 1, 0]} : (!loop.buf<4xf32, read_only>, !loop.buf<4xf32, global>) -> ()
  host.copy(%doubled, %kept) : (!loop.buf<4xf32, global>, !loop.buf<4xf32, global>) -> ()
  host.launch(%doubled, %doubled) {callee = @double, operand_segments = [1, 1, 0]} : (!loop.buf<4xf32, global>, !loop.buf<4xf32, global>) -> ()
  host.launch(%doubled, %totals_buffer) {callee = @accumulate, operand_segments = [1, 0, 1]} : (!loop.buf<4xf32, global>, !loop.buf<4xf32, global>) -> ()
  host.launch(%kept, %result) {callee = @double, operand_segments = [1, 1, 0]} : (!loop.buf<4xf32, global>, !loop.buf<4xf32, global>) -> ()
  %output = host.to_tensor(%result) : (!loop.buf<4xf32, global>) -> tensor<4xf32>
  %accumulated = host.to_tensor(%totals_buffer) : (!loop.buf<4xf32, global>) -> tensor<4xf32>
  func.return(%output, %accumulated) : (tensor<4xf32>, tensor<4xf32>) -> ()
}
```

### `host.to_buffer`

A buffer that holds the elements of `tensor`, which no launch and no copy
writes.

Defined by `ToBuffer` in [crates/gpt-rs-ir/src/dialects/host/ir.rs](../../crates/gpt-rs-ir/src/dialects/host/ir.rs).

| | |
|---|---|
| Operands | `tensor`: tensor |
| Results | `output`: buffer |
| Effects | none |

Example, from `filetests/host/roundtrip/buffers.ptir`:

```text
loop.func @double(%input: !loop.buf<4xf32, global>, %output: !loop.buf<4xf32, global>) attributes {access = [in, out<over = 0>]} {
  %zero = loop.index() {expr = 0} : () -> index
  %count = loop.index() {expr = 4} : () -> index
  loop.for(%zero, %count) ({
  ^body(%position: index):
    %element = loop.load(%input, %position) : (!loop.buf<4xf32, global>, index) -> f32
    %doubled = arith.add(%element, %element) : (f32, f32) -> f32
    loop.store(%doubled, %output, %position) : (f32, !loop.buf<4xf32, global>, index) -> ()
    loop.yield() : () -> ()
  }) : (index, index) -> ()
  loop.return() : () -> ()
}
loop.func @accumulate(%totals: !loop.buf<4xf32, global>, %addend: !loop.buf<4xf32, global>) attributes {access = [inout, in]} {
  %zero = loop.index() {expr = 0} : () -> index
  %count = loop.index() {expr = 4} : () -> index
  loop.for(%zero, %count) ({
  ^body(%position: index):
    %term = loop.load(%addend, %position) : (!loop.buf<4xf32, global>, index) -> f32
    %so_far = loop.load(%totals, %position) : (!loop.buf<4xf32, global>, index) -> f32
    %sum = arith.add(%so_far, %term) : (f32, f32) -> f32
    loop.store(%sum, %totals, %position) : (f32, !loop.buf<4xf32, global>, index) -> ()
    loop.yield() : () -> ()
  }) : (index, index) -> ()
  loop.return() : () -> ()
}
func.func @main(%input: tensor<4xf32>, %totals: tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>) {
  %arena = host.arena() {bytes = 128, alignment = 64} : () -> !loop.buf<128xui8, arena<alignment = 64>>
  %doubled = host.slice(%arena) {offset = 0, shape = [4], dtype = f32} : (!loop.buf<128xui8, arena<alignment = 64>>) -> !loop.buf<4xf32, global>
  %kept = host.slice(%arena) {offset = 64, shape = [4], dtype = f32} : (!loop.buf<128xui8, arena<alignment = 64>>) -> !loop.buf<4xf32, global>
  %result = host.alloc() {shape = [4], dtype = f32, alignment = 64} : () -> !loop.buf<4xf32, global>
  %input_buffer = host.to_buffer(%input) : (tensor<4xf32>) -> !loop.buf<4xf32, read_only>
  %totals_buffer = host.take(%totals) : (tensor<4xf32>) -> !loop.buf<4xf32, global>
  host.launch(%input_buffer, %doubled) {callee = @double, operand_segments = [1, 1, 0]} : (!loop.buf<4xf32, read_only>, !loop.buf<4xf32, global>) -> ()
  host.copy(%doubled, %kept) : (!loop.buf<4xf32, global>, !loop.buf<4xf32, global>) -> ()
  host.launch(%doubled, %doubled) {callee = @double, operand_segments = [1, 1, 0]} : (!loop.buf<4xf32, global>, !loop.buf<4xf32, global>) -> ()
  host.launch(%doubled, %totals_buffer) {callee = @accumulate, operand_segments = [1, 0, 1]} : (!loop.buf<4xf32, global>, !loop.buf<4xf32, global>) -> ()
  host.launch(%kept, %result) {callee = @double, operand_segments = [1, 1, 0]} : (!loop.buf<4xf32, global>, !loop.buf<4xf32, global>) -> ()
  %output = host.to_tensor(%result) : (!loop.buf<4xf32, global>) -> tensor<4xf32>
  %accumulated = host.to_tensor(%totals_buffer) : (!loop.buf<4xf32, global>) -> tensor<4xf32>
  func.return(%output, %accumulated) : (tensor<4xf32>, tensor<4xf32>) -> ()
}
```

### `host.take`

A buffer that holds the elements of `tensor`, which launches and copies may write.
It is the tensor's only use, so a runtime may give the tensor's own memory when nothing else
holds the tensor, and gives a copy otherwise.

Defined by `Take` in [crates/gpt-rs-ir/src/dialects/host/ir.rs](../../crates/gpt-rs-ir/src/dialects/host/ir.rs).

| | |
|---|---|
| Operands | `tensor`: tensor |
| Results | `output`: buffer |
| Traits | only use |
| Effects | allocates |

Example, from `filetests/host/roundtrip/buffers.ptir`:

```text
loop.func @double(%input: !loop.buf<4xf32, global>, %output: !loop.buf<4xf32, global>) attributes {access = [in, out<over = 0>]} {
  %zero = loop.index() {expr = 0} : () -> index
  %count = loop.index() {expr = 4} : () -> index
  loop.for(%zero, %count) ({
  ^body(%position: index):
    %element = loop.load(%input, %position) : (!loop.buf<4xf32, global>, index) -> f32
    %doubled = arith.add(%element, %element) : (f32, f32) -> f32
    loop.store(%doubled, %output, %position) : (f32, !loop.buf<4xf32, global>, index) -> ()
    loop.yield() : () -> ()
  }) : (index, index) -> ()
  loop.return() : () -> ()
}
loop.func @accumulate(%totals: !loop.buf<4xf32, global>, %addend: !loop.buf<4xf32, global>) attributes {access = [inout, in]} {
  %zero = loop.index() {expr = 0} : () -> index
  %count = loop.index() {expr = 4} : () -> index
  loop.for(%zero, %count) ({
  ^body(%position: index):
    %term = loop.load(%addend, %position) : (!loop.buf<4xf32, global>, index) -> f32
    %so_far = loop.load(%totals, %position) : (!loop.buf<4xf32, global>, index) -> f32
    %sum = arith.add(%so_far, %term) : (f32, f32) -> f32
    loop.store(%sum, %totals, %position) : (f32, !loop.buf<4xf32, global>, index) -> ()
    loop.yield() : () -> ()
  }) : (index, index) -> ()
  loop.return() : () -> ()
}
func.func @main(%input: tensor<4xf32>, %totals: tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>) {
  %arena = host.arena() {bytes = 128, alignment = 64} : () -> !loop.buf<128xui8, arena<alignment = 64>>
  %doubled = host.slice(%arena) {offset = 0, shape = [4], dtype = f32} : (!loop.buf<128xui8, arena<alignment = 64>>) -> !loop.buf<4xf32, global>
  %kept = host.slice(%arena) {offset = 64, shape = [4], dtype = f32} : (!loop.buf<128xui8, arena<alignment = 64>>) -> !loop.buf<4xf32, global>
  %result = host.alloc() {shape = [4], dtype = f32, alignment = 64} : () -> !loop.buf<4xf32, global>
  %input_buffer = host.to_buffer(%input) : (tensor<4xf32>) -> !loop.buf<4xf32, read_only>
  %totals_buffer = host.take(%totals) : (tensor<4xf32>) -> !loop.buf<4xf32, global>
  host.launch(%input_buffer, %doubled) {callee = @double, operand_segments = [1, 1, 0]} : (!loop.buf<4xf32, read_only>, !loop.buf<4xf32, global>) -> ()
  host.copy(%doubled, %kept) : (!loop.buf<4xf32, global>, !loop.buf<4xf32, global>) -> ()
  host.launch(%doubled, %doubled) {callee = @double, operand_segments = [1, 1, 0]} : (!loop.buf<4xf32, global>, !loop.buf<4xf32, global>) -> ()
  host.launch(%doubled, %totals_buffer) {callee = @accumulate, operand_segments = [1, 0, 1]} : (!loop.buf<4xf32, global>, !loop.buf<4xf32, global>) -> ()
  host.launch(%kept, %result) {callee = @double, operand_segments = [1, 1, 0]} : (!loop.buf<4xf32, global>, !loop.buf<4xf32, global>) -> ()
  %output = host.to_tensor(%result) : (!loop.buf<4xf32, global>) -> tensor<4xf32>
  %accumulated = host.to_tensor(%totals_buffer) : (!loop.buf<4xf32, global>) -> tensor<4xf32>
  func.return(%output, %accumulated) : (tensor<4xf32>, tensor<4xf32>) -> ()
}
```

Rejected, as the verify filetests pin it:

- `host.take: operand 0 has another use, and this op must be its only one`

### `host.launch`

One launch of the `loop.func` named `callee`, with `ins` for its `in`
parameters, `outs` for its `out` parameters and `inouts` for its `inout` parameters, each in
order. It starts after every launch and copy before it in the function has ended.

- Each buffer has the type of its parameter, with each of the callee's dimension symbols bound
  to one extent across all of them.
- A buffer that the launch writes goes to no other parameter, except that an `out<over = k>`
  parameter may take the buffer of the `in` parameter at position `k`. No launch writes a
  buffer of `host.to_buffer`.

Defined by `Launch` in [crates/gpt-rs-ir/src/dialects/host/ir.rs](../../crates/gpt-rs-ir/src/dialects/host/ir.rs).

| | |
|---|---|
| Operands | `ins`: buffers; `outs`: buffers; `inouts`: buffers |
| Results | none |
| Attributes | `callee`: `SymbolRef` |
| Effects | reads the memory of `ins`, reads the memory of `inouts`, writes the memory of `outs`, writes the memory of `inouts` |

Example, from `filetests/host/roundtrip/buffers.ptir`:

```text
loop.func @double(%input: !loop.buf<4xf32, global>, %output: !loop.buf<4xf32, global>) attributes {access = [in, out<over = 0>]} {
  %zero = loop.index() {expr = 0} : () -> index
  %count = loop.index() {expr = 4} : () -> index
  loop.for(%zero, %count) ({
  ^body(%position: index):
    %element = loop.load(%input, %position) : (!loop.buf<4xf32, global>, index) -> f32
    %doubled = arith.add(%element, %element) : (f32, f32) -> f32
    loop.store(%doubled, %output, %position) : (f32, !loop.buf<4xf32, global>, index) -> ()
    loop.yield() : () -> ()
  }) : (index, index) -> ()
  loop.return() : () -> ()
}
loop.func @accumulate(%totals: !loop.buf<4xf32, global>, %addend: !loop.buf<4xf32, global>) attributes {access = [inout, in]} {
  %zero = loop.index() {expr = 0} : () -> index
  %count = loop.index() {expr = 4} : () -> index
  loop.for(%zero, %count) ({
  ^body(%position: index):
    %term = loop.load(%addend, %position) : (!loop.buf<4xf32, global>, index) -> f32
    %so_far = loop.load(%totals, %position) : (!loop.buf<4xf32, global>, index) -> f32
    %sum = arith.add(%so_far, %term) : (f32, f32) -> f32
    loop.store(%sum, %totals, %position) : (f32, !loop.buf<4xf32, global>, index) -> ()
    loop.yield() : () -> ()
  }) : (index, index) -> ()
  loop.return() : () -> ()
}
func.func @main(%input: tensor<4xf32>, %totals: tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>) {
  %arena = host.arena() {bytes = 128, alignment = 64} : () -> !loop.buf<128xui8, arena<alignment = 64>>
  %doubled = host.slice(%arena) {offset = 0, shape = [4], dtype = f32} : (!loop.buf<128xui8, arena<alignment = 64>>) -> !loop.buf<4xf32, global>
  %kept = host.slice(%arena) {offset = 64, shape = [4], dtype = f32} : (!loop.buf<128xui8, arena<alignment = 64>>) -> !loop.buf<4xf32, global>
  %result = host.alloc() {shape = [4], dtype = f32, alignment = 64} : () -> !loop.buf<4xf32, global>
  %input_buffer = host.to_buffer(%input) : (tensor<4xf32>) -> !loop.buf<4xf32, read_only>
  %totals_buffer = host.take(%totals) : (tensor<4xf32>) -> !loop.buf<4xf32, global>
  host.launch(%input_buffer, %doubled) {callee = @double, operand_segments = [1, 1, 0]} : (!loop.buf<4xf32, read_only>, !loop.buf<4xf32, global>) -> ()
  host.copy(%doubled, %kept) : (!loop.buf<4xf32, global>, !loop.buf<4xf32, global>) -> ()
  host.launch(%doubled, %doubled) {callee = @double, operand_segments = [1, 1, 0]} : (!loop.buf<4xf32, global>, !loop.buf<4xf32, global>) -> ()
  host.launch(%doubled, %totals_buffer) {callee = @accumulate, operand_segments = [1, 0, 1]} : (!loop.buf<4xf32, global>, !loop.buf<4xf32, global>) -> ()
  host.launch(%kept, %result) {callee = @double, operand_segments = [1, 1, 0]} : (!loop.buf<4xf32, global>, !loop.buf<4xf32, global>) -> ()
  %output = host.to_tensor(%result) : (!loop.buf<4xf32, global>) -> tensor<4xf32>
  %accumulated = host.to_tensor(%totals_buffer) : (!loop.buf<4xf32, global>) -> tensor<4xf32>
  func.return(%output, %accumulated) : (tensor<4xf32>, tensor<4xf32>) -> ()
}
```

Rejected, as the verify filetests pin it:

- `host.launch: @copy has 1 `out` parameters, got 2 buffers`
- `host.launch: parameter 0: tensor<4xf64> does not match tensor<4xf32>`
- `host.launch: @main is not a loop.func`
- `host.launch: parameter 1 gets !loop.buf<4xf32, read_only>, not a global buffer`

### `host.copy`

Writes the elements of `from` into `to`, a buffer of the same type. It starts after
every launch and copy before it in the function has ended.

Defined by `Copy` in [crates/gpt-rs-ir/src/dialects/host/ir.rs](../../crates/gpt-rs-ir/src/dialects/host/ir.rs).

| | |
|---|---|
| Operands | `from`: buffer; `to`: buffer |
| Results | none |
| Effects | reads the memory of `from`, writes the memory of `to` |

Example, from `filetests/host/roundtrip/buffers.ptir`:

```text
loop.func @double(%input: !loop.buf<4xf32, global>, %output: !loop.buf<4xf32, global>) attributes {access = [in, out<over = 0>]} {
  %zero = loop.index() {expr = 0} : () -> index
  %count = loop.index() {expr = 4} : () -> index
  loop.for(%zero, %count) ({
  ^body(%position: index):
    %element = loop.load(%input, %position) : (!loop.buf<4xf32, global>, index) -> f32
    %doubled = arith.add(%element, %element) : (f32, f32) -> f32
    loop.store(%doubled, %output, %position) : (f32, !loop.buf<4xf32, global>, index) -> ()
    loop.yield() : () -> ()
  }) : (index, index) -> ()
  loop.return() : () -> ()
}
loop.func @accumulate(%totals: !loop.buf<4xf32, global>, %addend: !loop.buf<4xf32, global>) attributes {access = [inout, in]} {
  %zero = loop.index() {expr = 0} : () -> index
  %count = loop.index() {expr = 4} : () -> index
  loop.for(%zero, %count) ({
  ^body(%position: index):
    %term = loop.load(%addend, %position) : (!loop.buf<4xf32, global>, index) -> f32
    %so_far = loop.load(%totals, %position) : (!loop.buf<4xf32, global>, index) -> f32
    %sum = arith.add(%so_far, %term) : (f32, f32) -> f32
    loop.store(%sum, %totals, %position) : (f32, !loop.buf<4xf32, global>, index) -> ()
    loop.yield() : () -> ()
  }) : (index, index) -> ()
  loop.return() : () -> ()
}
func.func @main(%input: tensor<4xf32>, %totals: tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>) {
  %arena = host.arena() {bytes = 128, alignment = 64} : () -> !loop.buf<128xui8, arena<alignment = 64>>
  %doubled = host.slice(%arena) {offset = 0, shape = [4], dtype = f32} : (!loop.buf<128xui8, arena<alignment = 64>>) -> !loop.buf<4xf32, global>
  %kept = host.slice(%arena) {offset = 64, shape = [4], dtype = f32} : (!loop.buf<128xui8, arena<alignment = 64>>) -> !loop.buf<4xf32, global>
  %result = host.alloc() {shape = [4], dtype = f32, alignment = 64} : () -> !loop.buf<4xf32, global>
  %input_buffer = host.to_buffer(%input) : (tensor<4xf32>) -> !loop.buf<4xf32, read_only>
  %totals_buffer = host.take(%totals) : (tensor<4xf32>) -> !loop.buf<4xf32, global>
  host.launch(%input_buffer, %doubled) {callee = @double, operand_segments = [1, 1, 0]} : (!loop.buf<4xf32, read_only>, !loop.buf<4xf32, global>) -> ()
  host.copy(%doubled, %kept) : (!loop.buf<4xf32, global>, !loop.buf<4xf32, global>) -> ()
  host.launch(%doubled, %doubled) {callee = @double, operand_segments = [1, 1, 0]} : (!loop.buf<4xf32, global>, !loop.buf<4xf32, global>) -> ()
  host.launch(%doubled, %totals_buffer) {callee = @accumulate, operand_segments = [1, 0, 1]} : (!loop.buf<4xf32, global>, !loop.buf<4xf32, global>) -> ()
  host.launch(%kept, %result) {callee = @double, operand_segments = [1, 1, 0]} : (!loop.buf<4xf32, global>, !loop.buf<4xf32, global>) -> ()
  %output = host.to_tensor(%result) : (!loop.buf<4xf32, global>) -> tensor<4xf32>
  %accumulated = host.to_tensor(%totals_buffer) : (!loop.buf<4xf32, global>) -> tensor<4xf32>
  func.return(%output, %accumulated) : (tensor<4xf32>, tensor<4xf32>) -> ()
}
```

Rejected, as the verify filetests pin it:

- `host.copy: it copies !loop.buf<4xf32, global> into !loop.buf<8xf32, global>, a buffer of another type`
- `host.copy: it copies into !loop.buf<4xf32, read_only>, which is not a global buffer`

### `host.to_tensor`

The tensor of the elements that `buffer` holds when it runs.

Defined by `ToTensor` in [crates/gpt-rs-ir/src/dialects/host/ir.rs](../../crates/gpt-rs-ir/src/dialects/host/ir.rs).

| | |
|---|---|
| Operands | `buffer`: buffer |
| Results | `output`: tensor |
| Effects | reads the memory of `buffer` |

Example, from `filetests/host/roundtrip/buffers.ptir`:

```text
loop.func @double(%input: !loop.buf<4xf32, global>, %output: !loop.buf<4xf32, global>) attributes {access = [in, out<over = 0>]} {
  %zero = loop.index() {expr = 0} : () -> index
  %count = loop.index() {expr = 4} : () -> index
  loop.for(%zero, %count) ({
  ^body(%position: index):
    %element = loop.load(%input, %position) : (!loop.buf<4xf32, global>, index) -> f32
    %doubled = arith.add(%element, %element) : (f32, f32) -> f32
    loop.store(%doubled, %output, %position) : (f32, !loop.buf<4xf32, global>, index) -> ()
    loop.yield() : () -> ()
  }) : (index, index) -> ()
  loop.return() : () -> ()
}
loop.func @accumulate(%totals: !loop.buf<4xf32, global>, %addend: !loop.buf<4xf32, global>) attributes {access = [inout, in]} {
  %zero = loop.index() {expr = 0} : () -> index
  %count = loop.index() {expr = 4} : () -> index
  loop.for(%zero, %count) ({
  ^body(%position: index):
    %term = loop.load(%addend, %position) : (!loop.buf<4xf32, global>, index) -> f32
    %so_far = loop.load(%totals, %position) : (!loop.buf<4xf32, global>, index) -> f32
    %sum = arith.add(%so_far, %term) : (f32, f32) -> f32
    loop.store(%sum, %totals, %position) : (f32, !loop.buf<4xf32, global>, index) -> ()
    loop.yield() : () -> ()
  }) : (index, index) -> ()
  loop.return() : () -> ()
}
func.func @main(%input: tensor<4xf32>, %totals: tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>) {
  %arena = host.arena() {bytes = 128, alignment = 64} : () -> !loop.buf<128xui8, arena<alignment = 64>>
  %doubled = host.slice(%arena) {offset = 0, shape = [4], dtype = f32} : (!loop.buf<128xui8, arena<alignment = 64>>) -> !loop.buf<4xf32, global>
  %kept = host.slice(%arena) {offset = 64, shape = [4], dtype = f32} : (!loop.buf<128xui8, arena<alignment = 64>>) -> !loop.buf<4xf32, global>
  %result = host.alloc() {shape = [4], dtype = f32, alignment = 64} : () -> !loop.buf<4xf32, global>
  %input_buffer = host.to_buffer(%input) : (tensor<4xf32>) -> !loop.buf<4xf32, read_only>
  %totals_buffer = host.take(%totals) : (tensor<4xf32>) -> !loop.buf<4xf32, global>
  host.launch(%input_buffer, %doubled) {callee = @double, operand_segments = [1, 1, 0]} : (!loop.buf<4xf32, read_only>, !loop.buf<4xf32, global>) -> ()
  host.copy(%doubled, %kept) : (!loop.buf<4xf32, global>, !loop.buf<4xf32, global>) -> ()
  host.launch(%doubled, %doubled) {callee = @double, operand_segments = [1, 1, 0]} : (!loop.buf<4xf32, global>, !loop.buf<4xf32, global>) -> ()
  host.launch(%doubled, %totals_buffer) {callee = @accumulate, operand_segments = [1, 0, 1]} : (!loop.buf<4xf32, global>, !loop.buf<4xf32, global>) -> ()
  host.launch(%kept, %result) {callee = @double, operand_segments = [1, 1, 0]} : (!loop.buf<4xf32, global>, !loop.buf<4xf32, global>) -> ()
  %output = host.to_tensor(%result) : (!loop.buf<4xf32, global>) -> tensor<4xf32>
  %accumulated = host.to_tensor(%totals_buffer) : (!loop.buf<4xf32, global>) -> tensor<4xf32>
  func.return(%output, %accumulated) : (tensor<4xf32>, tensor<4xf32>) -> ()
}
```

## Traits

A trait states a fact about the ops that have it. The folds and patterns it implies apply to every one of them; [passes.md](passes.md) says which pass runs each.

### `OnlyUse`

It is the only use of each of its operands, whose value it takes over.

Defined by `OnlyUse` in [crates/gpt-rs-ir/src/traits/structure.rs](../../crates/gpt-rs-ir/src/traits/structure.rs).

Ops: `host.take`.
