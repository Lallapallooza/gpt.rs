<!-- Generated from the code by `cargo run -p gpt-rs-docs`. Do not edit: change the code and run it again. -->

# The loop dialect

The `loop` dialect is one launch of a dispatch: loops over index values, buffers in memory spaces,
and loads and stores of their elements. `tile-to-loop` lowers each `tile.dispatch` to a
`loop.func`. In this dialect, values do not stand for tensors: a dispatch reads its arguments from
buffers and writes its results into buffers, one element at a time. Bodies compute in `arith`.

## Conventions

- Each op is defined in `ir.rs`. Its type rule is in `verify.rs`, and its semantics, its `Eval`
  impl, are in `interp.rs`.
- Index values have type `index`, an `i64`. `loop.index` and `loop.predicate` compute with the index
  language of `tile`, in which `di` is the op's operand `i`. Their expressions read no operand
  element: a load gives such an element, and `arith.to_index` makes it an index.
- Normative keywords: MUST / SHOULD / MAY.

## Buffers

`!loop.buf<4x{T}xf32, global>` holds elements of one dtype at the coordinates of its extents,
row-major, in one memory space:

- `global`: the parameters of a `loop.func`, which its caller owns. Their extents MAY be dimension
  symbols, which the caller's arguments bind.
- `read_only`: memory the caller owns and nothing writes. An `in` parameter may be bound to it.
- `private`: scratch of one unit of work. `loop.alloc` allocates it, with static extents, and it
  lives as long as the region that allocates it: an allocation in the body of a loop is a new
  buffer in each of its iterations.
- `reg`: scratch as `private` is, which `loop-to-isa` keeps in vector registers: the accumulators
  of a register tile. `loop-to-isa` accepts a `reg` buffer only when the vector code reaches the
  buffer in whole registers, at positions fixed before the run, with no pad. That is, each load
  and store of the buffer:
  - has the buffer's dtype;
  - uses registers of one lane count, which divides the last extent;
  - stays inside the buffer;
  - starts at a multiple of that lane count.

A load reads an element that holds a value: the caller's, in an `in` or `inout` parameter, or one a
store wrote. Reading an element that holds none, or a position outside the buffer, is an error,
unless the load gives a `pad` value for positions outside the buffer.

## Launches

A `loop.func` takes buffers, each with an access: an `in` parameter it only reads, an `out`
parameter whose every element it writes, and an `inout` parameter it updates in place, which holds
the argument's elements when the launch starts and the result's when it ends. An `out<over = k>`
parameter may share the buffer of the `in` parameter `k`. No other parameter may share that buffer,
and the launch MUST read its elements only at positions that it has not written yet. `host.dispatch`
calls it with tensors: each `in` or `inout` parameter holds one argument, in order, and each
`inout` or `out` parameter gives one result, in order. `host.launch` calls it with buffers: one for
each parameter, its `in` parameters first, then its `out` and its `inout` parameters, each in order.

## Register tiles

`loop.atom` runs its body once for each point of a static box, the register tile of a dispatch,
below its extents. `loop-to-isa` computes the points at once, in vectors along the tile's last
axis, for the target that the `loop.func` names. Each point is a unit of work: two points MUST NOT
access one element when one of them writes it, so every order of the points gives the same values.

## Units of work

`loop.forall` runs its body as units of work at an execution level, which may run at the same time
and in any order. Two units of one forall MUST NOT access one element of a buffer when one of them
writes it, so every order gives the same values. Accesses before a forall happen before each of its
units, and those after it happen after all of them.

## Reference semantics

The interpreter runs the ops of a region in order. `loop.for` runs its body once for each index
from its start up to its end, in increasing order, and `loop.if` runs its body when its condition
holds. `loop.forall` runs its units in order of their index, and `loop.atom` runs its points in
row-major order. Each unit runs to its end. The interpreter fails when two units race on one
element: each element remembers the unit of its last write and the units that read it since.

## Ops

- `loop.func`
- `loop.return`
- `loop.for`
- `loop.forall`
- `loop.if`
- `loop.atom`
- `loop.yield`
- `loop.index`
- `loop.predicate`
- `loop.alloc`
- `loop.pick`
- `loop.load`
- `loop.store`
- `loop.prefetch`

### `loop.func`

One launch of a dispatch. A top-level symbol, called by `host.dispatch` with
tensors and by `host.launch` with buffers.

- Its body's block arguments are its parameters, `global` buffers, and `access` says how it
  uses each.
- It gives no results: it writes them into its `out` and `inout` parameters. A call with tensors
  puts each argument into an `in` or `inout` parameter, in order, and gives the elements of each
  `inout` or `out` parameter as a result, in order.
- When it names a `target`, each `loop.forall` of its body runs at an execution level of the
  target, and each `loop.prefetch` asks a memory that the target prefetches into for a line.

Defined by `Func` in [crates/gpt-rs-ir/src/dialects/loops/ir.rs](../../crates/gpt-rs-ir/src/dialects/loops/ir.rs).

| | |
|---|---|
| Operands | none |
| Results | none |
| Regions | `body` |
| Attributes | `name`: `String`; `args`: `[Type]`; `access`: `[Access]`; `target`: `TargetRef`, optional; `contract`: `bool`, optional |
| Traits | isolated from above |
| Effects | none |

Example, from `filetests/isa/roundtrip/lanewise.ptir`:

```text
loop.func @square_plus(%input: !loop.buf<16xf32, global>, %output: !loop.buf<16xf32, global>) attributes {access = [in, out]} {
  %zero = loop.index() {expr = 0} : () -> index
  %lanes = isa.load(%input, %zero) {dims = [16]} : (!loop.buf<16xf32, global>, index) -> !isa.vec<16xf32>
  %square_plus = isa.fma(%lanes, %lanes, %lanes) : (!isa.vec<16xf32>, !isa.vec<16xf32>, !isa.vec<16xf32>) -> !isa.vec<16xf32>
  isa.store(%square_plus, %output, %zero) : (!isa.vec<16xf32>, !loop.buf<16xf32, global>, index) -> ()
  loop.return() : () -> ()
}
```

Rejected, as the verify filetests pin it:

- `loop.func: the access list has length 1, but the parameter list has length 2`
- `loop.func: parameter 0 is !loop.buf<4xf32, private>, not a global buffer`
- `loop.func: parameter 0 is tensor<4xf32>, not a global buffer`
- `loop.func: parameter 1 has the extent {N}, which no `in` or `inout` parameter binds`
- `loop.func: it stores into parameter 0, which is `in``
- `loop.func: it stores into parameter 0, which is `in``
- `loop.func: parameter 1 may share the buffer of parameter 0, which is not `in``
- `loop.func: parameter 1, !loop.buf<4xf64, global>, may share the buffer of parameter 0, !loop.buf<4xf32, global>, whose elements differ in size or position`
- `loop.func: parameters 1 and 2 may both share the buffer of parameter 0`
- `loop.func: a loop.forall runs on the units of cluster, which x86-64-v4 does not have`
- `loop.func: a loop.prefetch asks dram for lines, which is no memory that x86-64-v4 prefetches into`

### `loop.return`

Ends the body of a `loop.func`.

Defined by `Return` in [crates/gpt-rs-ir/src/dialects/loops/ir.rs](../../crates/gpt-rs-ir/src/dialects/loops/ir.rs).

| | |
|---|---|
| Operands | none |
| Results | none |
| Traits | terminator |
| Effects | none |

Example, from `filetests/isa/roundtrip/lanewise.ptir`:

```text
loop.func @square_plus(%input: !loop.buf<16xf32, global>, %output: !loop.buf<16xf32, global>) attributes {access = [in, out]} {
  %zero = loop.index() {expr = 0} : () -> index
  %lanes = isa.load(%input, %zero) {dims = [16]} : (!loop.buf<16xf32, global>, index) -> !isa.vec<16xf32>
  %square_plus = isa.fma(%lanes, %lanes, %lanes) : (!isa.vec<16xf32>, !isa.vec<16xf32>, !isa.vec<16xf32>) -> !isa.vec<16xf32>
  isa.store(%square_plus, %output, %zero) : (!isa.vec<16xf32>, !loop.buf<16xf32, global>, index) -> ()
  loop.return() : () -> ()
}
```

### `loop.for`

Runs its body once for each index from `start` up to, not including, `end`, in
increasing order. The body's block argument is the index. It runs no time when `end` is not
above `start`.

Defined by `For` in [crates/gpt-rs-ir/src/dialects/loops/ir.rs](../../crates/gpt-rs-ir/src/dialects/loops/ir.rs).

| | |
|---|---|
| Operands | `start`: index; `end`: index |
| Results | none |
| Regions | `body` |
| Effects | none |
| Patterns | `InlineSingleTrip` (`canonicalize`) |

Example, from `filetests/loop/roundtrip/func.ptir`:

```text
loop.func @scale_shift(%input: !loop.buf<8xf64, global>, %scale: !loop.buf<8xf64, global>, %shift: !loop.buf<8xf64, global>, %output: !loop.buf<8xf64, global>) attributes {access = [in, in, in, out], contract = true} {
  %zero = loop.index() {expr = 0} : () -> index
  %count = loop.index() {expr = 8} : () -> index
  loop.for(%zero, %count) ({
  ^body(%at: index):
    %element = loop.load(%input, %at) : (!loop.buf<8xf64, global>, index) -> f64
    %factor = loop.load(%scale, %at) : (!loop.buf<8xf64, global>, index) -> f64
    %offset = loop.load(%shift, %at) : (!loop.buf<8xf64, global>, index) -> f64
    %shifted = arith.fma(%element, %factor, %offset) : (f64, f64, f64) -> f64
    loop.store(%shifted, %output, %at) : (f64, !loop.buf<8xf64, global>, index) -> ()
    loop.yield() : () -> ()
  }) : (index, index) -> ()
  loop.return() : () -> ()
}
```

### `loop.forall`

Runs its body once for each index from `start` up to, not including, `end`, as
units of work at `level`, which may run at the same time and in any order. The body's block
argument is the index. Two units MUST NOT access one element of a buffer when one of them
writes it, so every order gives the same values.

Defined by `Forall` in [crates/gpt-rs-ir/src/dialects/loops/ir.rs](../../crates/gpt-rs-ir/src/dialects/loops/ir.rs).

| | |
|---|---|
| Operands | `start`: index; `end`: index |
| Results | none |
| Regions | `body` |
| Attributes | `level`: `Level` |
| Effects | none |

Example, from `filetests/loop/roundtrip/func.ptir`:

```text
loop.func @rows(%input: !loop.buf<4x2xf32, global>, %output: !loop.buf<4x2xf32, global>) attributes {access = [in, out]} {
  %zero = loop.index() {expr = 0} : () -> index
  %rows = loop.index() {expr = 4} : () -> index
  %columns = loop.index() {expr = 2} : () -> index
  loop.forall(%zero, %rows) {level = core} ({
  ^body(%row: index):
    loop.for(%zero, %columns) ({
    ^body(%column: index):
      %element = loop.load(%input, %row, %column) : (!loop.buf<4x2xf32, global>, index, index) -> f32
      loop.store(%element, %output, %row, %column) : (f32, !loop.buf<4x2xf32, global>, index, index) -> ()
      loop.yield() : () -> ()
    }) : (index, index) -> ()
    loop.yield() : () -> ()
  }) : (index, index) -> ()
  loop.return() : () -> ()
}
```

### `loop.if`

Runs its body when `condition`, an `i1`, holds.

Defined by `If` in [crates/gpt-rs-ir/src/dialects/loops/ir.rs](../../crates/gpt-rs-ir/src/dialects/loops/ir.rs).

| | |
|---|---|
| Operands | `condition`: scalar |
| Results | none |
| Regions | `then` |
| Effects | none |

Example, from `filetests/loop/roundtrip/func.ptir`:

```text
loop.func @shifted_gather(%table: !loop.buf<6x3xf32, global>, %which: !loop.buf<si32, global>, %output: !loop.buf<3xf32, global>) attributes {access = [in, in, out]} {
  %start = loop.index() {expr = 0} : () -> index
  %end = loop.index() {expr = 3} : () -> index
  %scratch = loop.alloc() {shape = [3], dtype = f32} : () -> !loop.buf<3xf32, private>
  %code = loop.load(%which) : (!loop.buf<si32, global>) -> si32
  %picked = arith.to_index(%code) : (si32) -> index
  loop.for(%start, %end) ({
  ^body(%column: index):
    %row = loop.index(%picked) {expr = d0 - 1} : (index) -> index
    %element = loop.load(%table, %row, %column) {pad = 0.0} : (!loop.buf<6x3xf32, global>, index, index) -> f32
    loop.store(%element, %scratch, %column) : (f32, !loop.buf<3xf32, private>, index) -> ()
    %first = loop.predicate(%column) {constraints = [d0 == 0]} : (index) -> i1
    loop.if(%first) ({
      %position = arith.from_index(%column) {dtype = f32} : (index) -> f32
      loop.store(%position, %scratch, %column) : (f32, !loop.buf<3xf32, private>, index) -> ()
      loop.yield() : () -> ()
    }) : (i1) -> ()
    %value = loop.load(%scratch, %column) : (!loop.buf<3xf32, private>, index) -> f32
    loop.store(%value, %output, %column) : (f32, !loop.buf<3xf32, global>, index) -> ()
    loop.yield() : () -> ()
  }) : (index, index) -> ()
  loop.return() : () -> ()
}
```

Rejected, as the verify filetests pin it:

- `loop.if: the condition is si32, not i1`

### `loop.atom`

Runs its body once for each point of the box `tile` whose coordinates are below
`extents`, one per axis, in row-major order. The points are those of a register tile, which
`loop-to-isa` computes at once in vectors of `lanes` elements along the last axis. The body's
block arguments are the point's coordinates. `extents` MUST NOT exceed `tile`. An extent of 0
runs no point. Each point is a unit of work: two points MUST NOT access one element of a buffer
when one of them writes it.

Defined by `Atom` in [crates/gpt-rs-ir/src/dialects/loops/ir.rs](../../crates/gpt-rs-ir/src/dialects/loops/ir.rs).

| | |
|---|---|
| Operands | `extents`: indices |
| Results | none |
| Regions | `body` |
| Attributes | `tile`: `[usize]`; `lanes`: `usize` |
| Effects | none |
| Patterns | `AtomToIsa` (`loop-to-isa`) |

Example, from `filetests/loop/roundtrip/func.ptir`:

```text
loop.func @double_in_registers(%input: !loop.buf<2x6xf32, global>, %output: !loop.buf<2x6xf32, global>) attributes {access = [in, out], target = "x86-64-v3@7d7ad928"} {
  %rows = loop.index() {expr = 2} : () -> index
  %columns = loop.index() {expr = 6} : () -> index
  %doubled = loop.alloc() {shape = [2, 8], dtype = f32, space = reg} : () -> !loop.buf<2x8xf32, reg>
  loop.atom(%rows, %columns) {tile = [2, 8], lanes = 4} ({
  ^body(%row: index, %column: index):
    %element = loop.load(%input, %row, %column) : (!loop.buf<2x6xf32, global>, index, index) -> f32
    %twice = arith.add(%element, %element) : (f32, f32) -> f32
    loop.store(%twice, %doubled, %row, %column) : (f32, !loop.buf<2x8xf32, reg>, index, index) -> ()
    loop.yield() : () -> ()
  }) : (index, index) -> ()
  loop.atom(%rows, %columns) {tile = [2, 8], lanes = 4} ({
  ^body(%row: index, %column: index):
    %element = loop.load(%doubled, %row, %column) : (!loop.buf<2x8xf32, reg>, index, index) -> f32
    loop.store(%element, %output, %row, %column) : (f32, !loop.buf<2x6xf32, global>, index, index) -> ()
    loop.yield() : () -> ()
  }) : (index, index) -> ()
  loop.return() : () -> ()
}
```

Rejected, as the verify filetests pin it:

- `loop.atom: a register tile has rank 1 to 3, not 4`
- `loop.atom: the extent count is 1, but the tile has rank 2`
- `loop.atom: vectors of 4 lanes do not fill the 6 points of the tile's last axis`

### `loop.yield`

Ends the body of a `loop.for`, a `loop.forall`, a `loop.if` or a `loop.atom`.

Defined by `Yield` in [crates/gpt-rs-ir/src/dialects/loops/ir.rs](../../crates/gpt-rs-ir/src/dialects/loops/ir.rs).

| | |
|---|---|
| Operands | none |
| Results | none |
| Traits | terminator |
| Effects | none |

Example, from `filetests/loop/roundtrip/func.ptir`:

```text
loop.func @scale_shift(%input: !loop.buf<8xf64, global>, %scale: !loop.buf<8xf64, global>, %shift: !loop.buf<8xf64, global>, %output: !loop.buf<8xf64, global>) attributes {access = [in, in, in, out], contract = true} {
  %zero = loop.index() {expr = 0} : () -> index
  %count = loop.index() {expr = 8} : () -> index
  loop.for(%zero, %count) ({
  ^body(%at: index):
    %element = loop.load(%input, %at) : (!loop.buf<8xf64, global>, index) -> f64
    %factor = loop.load(%scale, %at) : (!loop.buf<8xf64, global>, index) -> f64
    %offset = loop.load(%shift, %at) : (!loop.buf<8xf64, global>, index) -> f64
    %shifted = arith.fma(%element, %factor, %offset) : (f64, f64, f64) -> f64
    loop.store(%shifted, %output, %at) : (f64, !loop.buf<8xf64, global>, index) -> ()
    loop.yield() : () -> ()
  }) : (index, index) -> ()
  loop.return() : () -> ()
}
```

### `loop.index`

The value of `expr`, an expression of the index language in which `di` is
operand `i` and `{T}` the extent the launch binds to the symbol `T`.

Defined by `Index` in [crates/gpt-rs-ir/src/dialects/loops/ir.rs](../../crates/gpt-rs-ir/src/dialects/loops/ir.rs).

| | |
|---|---|
| Operands | `ins`: indices |
| Results | `output`: index |
| Attributes | `expr`: `IndexExpr` |
| Effects | none |
| Folds | An index whose expression is one of its operands is that operand. |
| Patterns | `ComposeIndices` (`canonicalize`) |

Example, from `filetests/isa/roundtrip/lanewise.ptir`:

```text
loop.func @square_plus(%input: !loop.buf<16xf32, global>, %output: !loop.buf<16xf32, global>) attributes {access = [in, out]} {
  %zero = loop.index() {expr = 0} : () -> index
  %lanes = isa.load(%input, %zero) {dims = [16]} : (!loop.buf<16xf32, global>, index) -> !isa.vec<16xf32>
  %square_plus = isa.fma(%lanes, %lanes, %lanes) : (!isa.vec<16xf32>, !isa.vec<16xf32>, !isa.vec<16xf32>) -> !isa.vec<16xf32>
  isa.store(%square_plus, %output, %zero) : (!isa.vec<16xf32>, !loop.buf<16xf32, global>, index) -> ()
  loop.return() : () -> ()
}
```

Rejected, as the verify filetests pin it:

- `loop.index: the expression names d1, but the op's operand count is 1`

### `loop.predicate`

Whether every one of `constraints` holds, as an `i1`. The constraints are
those of a `tile.block`, in which `di` is operand `i`, and read no operand element.

Defined by `Predicate` in [crates/gpt-rs-ir/src/dialects/loops/ir.rs](../../crates/gpt-rs-ir/src/dialects/loops/ir.rs).

| | |
|---|---|
| Operands | `ins`: indices |
| Results | `output`: scalar |
| Attributes | `constraints`: `[Constraint]` |
| Effects | none |

Example, from `filetests/loop/roundtrip/func.ptir`:

```text
loop.func @shifted_gather(%table: !loop.buf<6x3xf32, global>, %which: !loop.buf<si32, global>, %output: !loop.buf<3xf32, global>) attributes {access = [in, in, out]} {
  %start = loop.index() {expr = 0} : () -> index
  %end = loop.index() {expr = 3} : () -> index
  %scratch = loop.alloc() {shape = [3], dtype = f32} : () -> !loop.buf<3xf32, private>
  %code = loop.load(%which) : (!loop.buf<si32, global>) -> si32
  %picked = arith.to_index(%code) : (si32) -> index
  loop.for(%start, %end) ({
  ^body(%column: index):
    %row = loop.index(%picked) {expr = d0 - 1} : (index) -> index
    %element = loop.load(%table, %row, %column) {pad = 0.0} : (!loop.buf<6x3xf32, global>, index, index) -> f32
    loop.store(%element, %scratch, %column) : (f32, !loop.buf<3xf32, private>, index) -> ()
    %first = loop.predicate(%column) {constraints = [d0 == 0]} : (index) -> i1
    loop.if(%first) ({
      %position = arith.from_index(%column) {dtype = f32} : (index) -> f32
      loop.store(%position, %scratch, %column) : (f32, !loop.buf<3xf32, private>, index) -> ()
      loop.yield() : () -> ()
    }) : (i1) -> ()
    %value = loop.load(%scratch, %column) : (!loop.buf<3xf32, private>, index) -> f32
    loop.store(%value, %output, %column) : (f32, !loop.buf<3xf32, global>, index) -> ()
    loop.yield() : () -> ()
  }) : (index, index) -> ()
  loop.return() : () -> ()
}
```

Rejected, as the verify filetests pin it:

- `loop.predicate: a constraint reads an operand element`
- `loop.predicate: the expression names d2, but the op's operand count is 1`

### `loop.alloc`

A buffer of `dtype` with the static extents `shape`, in `private` scratch or in
`reg` vector registers, which lives as long as the region that allocates it. No element holds a
value until a store writes it.

Defined by `Alloc` in [crates/gpt-rs-ir/src/dialects/loops/ir.rs](../../crates/gpt-rs-ir/src/dialects/loops/ir.rs).

| | |
|---|---|
| Operands | none |
| Results | `output`: buffer |
| Attributes | `shape`: `[usize]`; `dtype`: `DType`; `space`: `Space`, optional |
| Effects | allocates |

Example, from `filetests/loop/roundtrip/func.ptir`:

```text
loop.func @double_in_registers(%input: !loop.buf<2x6xf32, global>, %output: !loop.buf<2x6xf32, global>) attributes {access = [in, out], target = "x86-64-v3@7d7ad928"} {
  %rows = loop.index() {expr = 2} : () -> index
  %columns = loop.index() {expr = 6} : () -> index
  %doubled = loop.alloc() {shape = [2, 8], dtype = f32, space = reg} : () -> !loop.buf<2x8xf32, reg>
  loop.atom(%rows, %columns) {tile = [2, 8], lanes = 4} ({
  ^body(%row: index, %column: index):
    %element = loop.load(%input, %row, %column) : (!loop.buf<2x6xf32, global>, index, index) -> f32
    %twice = arith.add(%element, %element) : (f32, f32) -> f32
    loop.store(%twice, %doubled, %row, %column) : (f32, !loop.buf<2x8xf32, reg>, index, index) -> ()
    loop.yield() : () -> ()
  }) : (index, index) -> ()
  loop.atom(%rows, %columns) {tile = [2, 8], lanes = 4} ({
  ^body(%row: index, %column: index):
    %element = loop.load(%doubled, %row, %column) : (!loop.buf<2x8xf32, reg>, index, index) -> f32
    loop.store(%element, %output, %row, %column) : (f32, !loop.buf<2x6xf32, global>, index, index) -> ()
    loop.yield() : () -> ()
  }) : (index, index) -> ()
  loop.return() : () -> ()
}
```

Rejected, as the verify filetests pin it:

- `loop.alloc: it allocates global memory, which only the caller of a launch owns`

### `loop.pick`

The buffer of `buffers` at position `index`, which MUST be one of them. The
buffers are of one type, and the result is the picked buffer itself: a store into it writes
that buffer.

Defined by `Pick` in [crates/gpt-rs-ir/src/dialects/loops/ir.rs](../../crates/gpt-rs-ir/src/dialects/loops/ir.rs).

| | |
|---|---|
| Operands | `index`: index; `buffers`: buffers |
| Results | `output`: buffer |
| Effects | none |

Example, from `filetests/loop/roundtrip/func.ptir`:

```text
loop.func @pick_table(%first: !loop.buf<2xf32, global>, %second: !loop.buf<2xf32, global>, %which: !loop.buf<si32, global>, %output: !loop.buf<2xf32, global>) attributes {access = [in, in, in, out]} {
  %zero = loop.index() {expr = 0} : () -> index
  %end = loop.index() {expr = 2} : () -> index
  %code = loop.load(%which) : (!loop.buf<si32, global>) -> si32
  %position = arith.to_index(%code) : (si32) -> index
  %table = loop.pick(%position, %first, %second) : (index, !loop.buf<2xf32, global>, !loop.buf<2xf32, global>) -> !loop.buf<2xf32, global>
  loop.for(%zero, %end) ({
  ^body(%column: index):
    %element = loop.load(%table, %column) : (!loop.buf<2xf32, global>, index) -> f32
    loop.store(%element, %output, %column) : (f32, !loop.buf<2xf32, global>, index) -> ()
    loop.yield() : () -> ()
  }) : (index, index) -> ()
  loop.return() : () -> ()
}
```

Rejected, as the verify filetests pin it:

- `loop.pick: it picks from no buffer`
- `loop.pick: it picks from buffers of two types, !loop.buf<2xf32, global> and !loop.buf<3xf32, global> at position 1`

### `loop.load`

The element of `buffer` at `indices`, one per axis. Without `pad`, the position
MUST be inside the buffer, and the element MUST hold a value: the caller's, in an `in` or
`inout` parameter, or one a store wrote. With `pad v`, a position outside the buffer gives `v`,
a value of the buffer's dtype.

Defined by `Load` in [crates/gpt-rs-ir/src/dialects/loops/ir.rs](../../crates/gpt-rs-ir/src/dialects/loops/ir.rs).

| | |
|---|---|
| Operands | `buffer`: buffer; `indices`: indices |
| Results | `output`: scalar |
| Attributes | `pad`: `F64`, optional |
| Effects | reads the memory of `buffer` |

Example, from `filetests/loop/roundtrip/func.ptir`:

```text
loop.func @scale_shift(%input: !loop.buf<8xf64, global>, %scale: !loop.buf<8xf64, global>, %shift: !loop.buf<8xf64, global>, %output: !loop.buf<8xf64, global>) attributes {access = [in, in, in, out], contract = true} {
  %zero = loop.index() {expr = 0} : () -> index
  %count = loop.index() {expr = 8} : () -> index
  loop.for(%zero, %count) ({
  ^body(%at: index):
    %element = loop.load(%input, %at) : (!loop.buf<8xf64, global>, index) -> f64
    %factor = loop.load(%scale, %at) : (!loop.buf<8xf64, global>, index) -> f64
    %offset = loop.load(%shift, %at) : (!loop.buf<8xf64, global>, index) -> f64
    %shifted = arith.fma(%element, %factor, %offset) : (f64, f64, f64) -> f64
    loop.store(%shifted, %output, %at) : (f64, !loop.buf<8xf64, global>, index) -> ()
    loop.yield() : () -> ()
  }) : (index, index) -> ()
  loop.return() : () -> ()
}
```

Rejected, as the verify filetests pin it:

- `loop.load: the index count is 1, but !loop.buf<4x4xf32, global> has rank 2`
- `loop.load: it pads with 0.5, which si32 cannot hold`

### `loop.store`

Writes `value` into `buffer` at `indices`, one per axis, which MUST be a position
inside the buffer.

Defined by `Store` in [crates/gpt-rs-ir/src/dialects/loops/ir.rs](../../crates/gpt-rs-ir/src/dialects/loops/ir.rs).

| | |
|---|---|
| Operands | `value`: scalar; `buffer`: buffer; `indices`: indices |
| Results | none |
| Effects | writes the memory of `buffer` |

Example, from `filetests/loop/roundtrip/func.ptir`:

```text
loop.func @scale_shift(%input: !loop.buf<8xf64, global>, %scale: !loop.buf<8xf64, global>, %shift: !loop.buf<8xf64, global>, %output: !loop.buf<8xf64, global>) attributes {access = [in, in, in, out], contract = true} {
  %zero = loop.index() {expr = 0} : () -> index
  %count = loop.index() {expr = 8} : () -> index
  loop.for(%zero, %count) ({
  ^body(%at: index):
    %element = loop.load(%input, %at) : (!loop.buf<8xf64, global>, index) -> f64
    %factor = loop.load(%scale, %at) : (!loop.buf<8xf64, global>, index) -> f64
    %offset = loop.load(%shift, %at) : (!loop.buf<8xf64, global>, index) -> f64
    %shifted = arith.fma(%element, %factor, %offset) : (f64, f64, f64) -> f64
    loop.store(%shifted, %output, %at) : (f64, !loop.buf<8xf64, global>, index) -> ()
    loop.yield() : () -> ()
  }) : (index, index) -> ()
  loop.return() : () -> ()
}
```

Rejected, as the verify filetests pin it:

- `loop.store: the index count is 0, but !loop.buf<4xf32, private> has rank 1`
- `loop.store: the value is f32, but !loop.buf<si32, private> holds si32`

### `loop.prefetch`

Asks the memory `level`, one that the target prefetches into, for the line that
holds the element of `buffer` at `indices`, one per axis, which a later load reads. A hint: it
computes and changes nothing, and a position outside the buffer reads nothing.

Defined by `Prefetch` in [crates/gpt-rs-ir/src/dialects/loops/ir.rs](../../crates/gpt-rs-ir/src/dialects/loops/ir.rs).

| | |
|---|---|
| Operands | `buffer`: buffer; `indices`: indices |
| Results | none |
| Attributes | `level`: `Level` |
| Effects | reads the memory of `buffer` |

Example, from `filetests/loop/roundtrip/func.ptir`:

```text
loop.func @add_bias(%input: !loop.buf<4x{N}xf32, global>, %bias: !loop.buf<{N}xf32, global>, %output: !loop.buf<4x{N}xf32, global>) attributes {access = [in, in, out]} {
  %zero = loop.index() {expr = 0} : () -> index
  %rows = loop.index() {expr = 4} : () -> index
  %columns = loop.index() {expr = {N}} : () -> index
  loop.for(%zero, %rows) ({
  ^body(%row: index):
    loop.for(%zero, %columns) ({
    ^body(%column: index):
      %next_row = loop.index(%row) {expr = d0 + 1} : (index) -> index
      loop.prefetch(%input, %next_row, %column) {level = l2} : (!loop.buf<4x{N}xf32, global>, index, index) -> ()
      %element = loop.load(%input, %row, %column) : (!loop.buf<4x{N}xf32, global>, index, index) -> f32
      %shift = loop.load(%bias, %column) : (!loop.buf<{N}xf32, global>, index) -> f32
      %sum = arith.add(%element, %shift) : (f32, f32) -> f32
      loop.store(%sum, %output, %row, %column) : (f32, !loop.buf<4x{N}xf32, global>, index, index) -> ()
      loop.yield() : () -> ()
    }) : (index, index) -> ()
    loop.yield() : () -> ()
  }) : (index, index) -> ()
  loop.return() : () -> ()
}
func.func @main(%rows_in: tensor<4x{N}xf32>, %bias_in: tensor<{N}xf32>) -> tensor<4x{N}xf32> {
  %sums = host.dispatch(%rows_in, %bias_in) {callee = @add_bias} : (tensor<4x{N}xf32>, tensor<{N}xf32>) -> tensor<4x{N}xf32>
  func.return(%sums) : (tensor<4x{N}xf32>) -> ()
}
```

Rejected, as the verify filetests pin it:

- `loop.prefetch: the index count is 3, but !loop.buf<4x4xf32, global> has rank 2`

## Traits

A trait states a fact about the ops that have it. The folds and patterns it implies apply to every one of them; [passes.md](passes.md) says which pass runs each.

### `IsolatedFromAbove`

Its regions read no value defined outside them.

Defined by `IsolatedFromAbove` in [crates/gpt-rs-ir/src/traits/structure.rs](../../crates/gpt-rs-ir/src/traits/structure.rs).

Ops: `loop.func`.

### `Terminator`

Ends a region: its operands are what the region yields, and it has no results.

Defined by `Terminator` in [crates/gpt-rs-ir/src/traits/structure.rs](../../crates/gpt-rs-ir/src/traits/structure.rs).

Ops: `loop.return`, `loop.yield`.
