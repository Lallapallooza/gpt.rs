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
- `private`: scratch of one unit of work. `loop.alloc` allocates it, with static extents, and it
  lives as long as the region that allocates it: an allocation in the body of a loop is a new
  buffer in each of its iterations.

A load reads an element that a store wrote before it. Reading an element that no store wrote, or a
position outside the buffer, is an error, unless the load gives a `pad` value for positions outside
the buffer.

## Launches

A `loop.func` takes buffers, each with an access: an `in` parameter it only reads, and an `out`
parameter whose every element it writes. `host.dispatch` calls it with tensors: each `in`
parameter holds one argument, in order, and each `out` parameter gives one result, in order.

## Units of work

`loop.forall` runs its body as units of work at an execution level, which may run at the same time
and in any order. Two units of one forall MUST NOT access one element of a buffer when one of them
writes it, so every order gives the same values. Accesses before a forall happen before each of its
units, and those after it happen after all of them.

## Reference semantics

The interpreter runs the ops of a region in order. `loop.for` runs its body once for each index
from its start up to its end, in increasing order, and `loop.if` runs its body when its condition
holds. `loop.forall` runs its units in order of their index, each to its end, and fails on two
accesses of its units that race: each element remembers the unit of its last write and the units
that read it since.

## Ops

- `loop.func`
- `loop.return`
- `loop.for`
- `loop.forall`
- `loop.if`
- `loop.yield`
- `loop.index`
- `loop.predicate`
- `loop.alloc`
- `loop.load`
- `loop.store`

### `loop.func`

One launch of a dispatch. A top-level symbol, called by `host.dispatch`.

- Its body's block arguments are its parameters, `global` buffers, and `access` says how it
  uses each.
- It gives no results: it writes them into its `out` parameters. A call with tensors puts each
  argument into an `in` parameter, in order, and gives the elements of each `out` parameter as a
  result, in order.

Defined by `Func` in [crates/gpt-rs-ir/src/dialects/loop/ir.rs](../../crates/gpt-rs-ir/src/dialects/loop/ir.rs).

| | |
|---|---|
| Operands | none |
| Results | none |
| Regions | `body` |
| Attributes | `name`: `String`; `args`: `[Type]`; `access`: `[Access]` |
| Traits | isolated from above |
| Effects | none |

| `access` | |
|---|---|
| `in` | It reads the parameter and never writes it. |
| `out` | It writes every element of the parameter, which holds nothing it reads before. |

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

Rejected, as the verify filetests pin it:

- `loop.func: the access list has length 1, but the parameter list has length 2`
- `loop.func: parameter 0 is !loop.buf<4xf32, private>, not a global buffer`
- `loop.func: parameter 0 is tensor<4xf32>, not a global buffer`
- `loop.func: parameter 1 has the extent {N}, which no `in` parameter binds`
- `loop.func: it stores into parameter 0, which is `in``

### `loop.return`

Ends the body of a `loop.func`.

Defined by `Return` in [crates/gpt-rs-ir/src/dialects/loop/ir.rs](../../crates/gpt-rs-ir/src/dialects/loop/ir.rs).

| | |
|---|---|
| Operands | none |
| Results | none |
| Traits | terminator |
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

### `loop.for`

Runs its body once for each index from `start` up to, not including, `end`, in
increasing order. The body's block argument is the index. It runs no time when `end` is not
above `start`.

Defined by `For` in [crates/gpt-rs-ir/src/dialects/loop/ir.rs](../../crates/gpt-rs-ir/src/dialects/loop/ir.rs).

| | |
|---|---|
| Operands | `start`: index; `end`: index |
| Results | none |
| Regions | `body` |
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

### `loop.forall`

Runs its body once for each index from `start` up to, not including, `end`, as
units of work at `level`, which may run at the same time and in any order. The body's block
argument is the index. Two units MUST NOT access one element of a buffer when one of them
writes it, so every order gives the same values.

Defined by `Forall` in [crates/gpt-rs-ir/src/dialects/loop/ir.rs](../../crates/gpt-rs-ir/src/dialects/loop/ir.rs).

| | |
|---|---|
| Operands | `start`: index; `end`: index |
| Results | none |
| Regions | `body` |
| Attributes | `level`: `Level` |
| Effects | none |

| `level` | |
|---|---|
| `l3` | The cores that share one last-level cache. |
| `core` | One core. |

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

Defined by `If` in [crates/gpt-rs-ir/src/dialects/loop/ir.rs](../../crates/gpt-rs-ir/src/dialects/loop/ir.rs).

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

### `loop.yield`

Ends the body of a `loop.for`, a `loop.forall` or a `loop.if`.

Defined by `Yield` in [crates/gpt-rs-ir/src/dialects/loop/ir.rs](../../crates/gpt-rs-ir/src/dialects/loop/ir.rs).

| | |
|---|---|
| Operands | none |
| Results | none |
| Traits | terminator |
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

### `loop.index`

The value of `expr`, an expression of the index language in which `di` is
operand `i` and `{T}` the extent the launch binds to the symbol `T`.

Defined by `Index` in [crates/gpt-rs-ir/src/dialects/loop/ir.rs](../../crates/gpt-rs-ir/src/dialects/loop/ir.rs).

| | |
|---|---|
| Operands | `ins`: indices |
| Results | `output`: index |
| Attributes | `expr`: `IndexExpr` |
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

Rejected, as the verify filetests pin it:

- `loop.index: the expression names d1, but the op's operand count is 1`

### `loop.predicate`

Whether every one of `constraints` holds, as an `i1`. The constraints are
those of a `tile.block`, in which `di` is operand `i`, and read no operand element.

Defined by `Predicate` in [crates/gpt-rs-ir/src/dialects/loop/ir.rs](../../crates/gpt-rs-ir/src/dialects/loop/ir.rs).

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

A `private` buffer of `dtype` with the static extents `shape`, which lives as long
as the region that allocates it. No element holds a value until a store writes it.

Defined by `Alloc` in [crates/gpt-rs-ir/src/dialects/loop/ir.rs](../../crates/gpt-rs-ir/src/dialects/loop/ir.rs).

| | |
|---|---|
| Operands | none |
| Results | `output`: buffer |
| Attributes | `shape`: `[usize]`; `dtype`: `DType` |
| Effects | allocates |

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

### `loop.load`

The element of `buffer` at `indices`, one per axis. Without `pad`, the position
MUST be inside the buffer, and a store MUST have written the element. With `pad v`, a position
outside the buffer gives `v`, a value of the buffer's dtype.

Defined by `Load` in [crates/gpt-rs-ir/src/dialects/loop/ir.rs](../../crates/gpt-rs-ir/src/dialects/loop/ir.rs).

| | |
|---|---|
| Operands | `buffer`: buffer; `indices`: indices |
| Results | `output`: scalar |
| Attributes | `pad`: `F64`, optional |
| Effects | reads the memory of `buffer` |

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

Rejected, as the verify filetests pin it:

- `loop.load: the index count is 1, but !loop.buf<4x4xf32, global> has rank 2`
- `loop.load: it pads with 0.5, which si32 cannot hold`

### `loop.store`

Writes `value` into `buffer` at `indices`, one per axis, which MUST be a position
inside the buffer.

Defined by `Store` in [crates/gpt-rs-ir/src/dialects/loop/ir.rs](../../crates/gpt-rs-ir/src/dialects/loop/ir.rs).

| | |
|---|---|
| Operands | `value`: scalar; `buffer`: buffer; `indices`: indices |
| Results | none |
| Effects | writes the memory of `buffer` |

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

Rejected, as the verify filetests pin it:

- `loop.store: the index count is 0, but !loop.buf<4xf32, private> has rank 1`
- `loop.store: the value is f32, but !loop.buf<si32, private> holds si32`

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
