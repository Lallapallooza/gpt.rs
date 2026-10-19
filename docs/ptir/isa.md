<!-- Generated from the code by `cargo run -p gpt-rs-docs`. Do not edit: change the code and run it again. -->

# The isa dialect

The `isa` dialect is vector code: the register tile of a dispatch, computed in vector registers
inside the loops of the `loop` dialect. `loop-to-isa` writes it from each `loop.atom`, with fixed
rules and no choice of its own. Its ops apply one operation to every lane of a group of registers,
and move lanes between registers and `loop` buffers.

## Conventions

- Each op is defined in `ir.rs`. Its type rule is in `verify.rs`, and its semantics, its `Eval`
  impl, are in `interp.rs`.
- A value of type `!isa.vec<6x4x16xf32>` is a group of 6x4 registers that each hold 16 lanes of
  `f32`. The last extent is the lanes of one register, and the extents before it, at most
  `MAX_GROUP_RANK` (3), count registers. Lane `l` of register `[i, j]` is element `[i, j, l]`. A
  vector of `i1` is a mask: one flag per lane.
- Lane-wise ops take operands of equal extents. All are of one type, except the mask of
  `isa.select`. Each applies the `arith` op of the same name to each lane, as the `tensor` op of
  that name applies it to each element: `isa.add` is `arith.add` per lane. `isa.expand` repeats a
  register along an axis of extent 1.
- Index values have type `index`, as in `loop`.
- Normative keywords: MUST / SHOULD / MAY.

## Memory

`isa.load`, `isa.broadcast` and `isa.store` move lanes between registers and a `loop` buffer. Each
names one base position in the buffer, and `steps`. `steps` has one entry per axis of the group.
Each entry gives, for each axis of the buffer, how far the position moves from one register to the
next along that group axis. Register `g` starts at `base + g[0] * steps[0] + ...`. `isa.load` and
`isa.store` take its lanes from there along the last axis of the buffer, and `isa.broadcast`
reads one element into every lane.

`isa.masked_load`, `isa.masked_broadcast` and `isa.masked_store` take a mask of the extents of
their registers first. A lane whose flag is clear is not accessed: a load gives it 0, and a store
leaves its element as it was. Every lane that is accessed MUST be inside the buffer, and a load
MUST read elements that hold a value: the caller's, in an `in` or `inout` parameter, or ones that
a store wrote.

A read with `pad v` gives `v` for a lane outside the buffer. Its `checked` attribute names, per
register, the axes on which the register's lanes may fall outside, so the code checks those
alone. An empty `checked` allows every axis. A lane outside on any other axis makes the read
wrong, and the interpreter rejects it. `loop-to-isa` writes `checked` from the ranges that the
loops around a read give its position. `loop-to-isa` gives no pad to a read whose lanes all stay
inside.

## Targets

A register holds its lanes times the bits of its dtype. These bits fill one of the target's
vector widths, or the low bits of one. A mask holds one flag per lane. The rules of a target's
architecture, `isa::Rules`, say for each op whether the target has code for it on the registers it
computes. They also say what that code needs: the extensions of the instruction set, the mask and
vector registers it holds besides its operands and its result, and the operands it can read from
memory.

## Reference semantics

The interpreter holds the lanes of a vector as a tensor of its extents, and computes each lane as
the `tensor` op of the same name computes an element. Memory ops read and write the buffers of
the `loop` interpreter, which checks every access.

## Ops

- `isa.neg`, `isa.abs`, `isa.exp`, `isa.log`, `isa.tanh`, `isa.erf`, `isa.rsqrt`, `isa.reciprocal`, `isa.sin`, `isa.cos`, `isa.not`
- `isa.add`, `isa.sub`, `isa.mul`, `isa.div`, `isa.maximum`, `isa.minimum`, `isa.and`, `isa.or`, `isa.xor`, `isa.shift_left`, `isa.shift_right_logical`, `isa.shift_right_arithmetic`
- `isa.less`, `isa.less_equal`, `isa.equal`, `isa.greater_equal`, `isa.greater`, `isa.not_equal`
- `isa.select`
- `isa.fma`
- `isa.cast`
- `isa.bitcast`
- `isa.splat`
- `isa.expand`
- `isa.tile_mask`
- `isa.load`, `isa.broadcast`
- `isa.masked_load`, `isa.masked_broadcast`
- `isa.store`
- `isa.masked_store`
- `isa.prefetch`

### `Unary`: `isa.neg`, `isa.abs`, `isa.exp`, `isa.log`, `isa.tanh`, `isa.erf`, `isa.rsqrt`, `isa.reciprocal`, `isa.sin`, `isa.cos`, `isa.not`

The `arith` op of the same name on each lane of a group of
registers.

Defined by `Unary` in [crates/gpt-rs-ir/src/dialects/isa/ir.rs](../../crates/gpt-rs-ir/src/dialects/isa/ir.rs).

| | |
|---|---|
| Operands | `input`: vector |
| Results | `output`: vector |
| Effects | none |

| Op | |
|---|---|
| `isa.neg` | `arith.neg` on each lane. |
| `isa.abs` | `arith.abs` on each lane. |
| `isa.exp` | `arith.exp` on each lane. |
| `isa.log` | `arith.log` on each lane. |
| `isa.tanh` | `arith.tanh` on each lane. |
| `isa.erf` | `arith.erf` on each lane. |
| `isa.rsqrt` | `arith.rsqrt` on each lane. |
| `isa.reciprocal` | `arith.reciprocal` on each lane. |
| `isa.sin` | `arith.sin` on each lane. |
| `isa.cos` | `arith.cos` on each lane. |
| `isa.not` | `arith.not` on each lane. |

Example, from `filetests/isa/roundtrip/lanewise.ptir`:

```text
loop.func @float_lanes(%input: !loop.buf<8xf32, global>, %output: !loop.buf<8xbf16, global>) attributes {access = [in, out]} {
  %zero = loop.index() {expr = 0} : () -> index
  %lanes = isa.load(%input, %zero) {dims = [8]} : (!loop.buf<8xf32, global>, index) -> !isa.vec<8xf32>
  %negated = isa.neg(%lanes) : (!isa.vec<8xf32>) -> !isa.vec<8xf32>
  %magnitudes = isa.abs(%negated) : (!isa.vec<8xf32>) -> !isa.vec<8xf32>
  %powers = isa.exp(%magnitudes) : (!isa.vec<8xf32>) -> !isa.vec<8xf32>
  %logs = isa.log(%powers) : (!isa.vec<8xf32>) -> !isa.vec<8xf32>
  %tanhs = isa.tanh(%logs) : (!isa.vec<8xf32>) -> !isa.vec<8xf32>
  %erfs = isa.erf(%tanhs) : (!isa.vec<8xf32>) -> !isa.vec<8xf32>
  %roots = isa.rsqrt(%erfs) : (!isa.vec<8xf32>) -> !isa.vec<8xf32>
  %inverses = isa.reciprocal(%roots) : (!isa.vec<8xf32>) -> !isa.vec<8xf32>
  %sines = isa.sin(%inverses) : (!isa.vec<8xf32>) -> !isa.vec<8xf32>
  %cosines = isa.cos(%sines) : (!isa.vec<8xf32>) -> !isa.vec<8xf32>
  %differences = isa.sub(%lanes, %cosines) : (!isa.vec<8xf32>, !isa.vec<8xf32>) -> !isa.vec<8xf32>
  %quotients = isa.div(%differences, %cosines) : (!isa.vec<8xf32>, !isa.vec<8xf32>) -> !isa.vec<8xf32>
  %smaller = isa.minimum(%quotients, %lanes) : (!isa.vec<8xf32>, !isa.vec<8xf32>) -> !isa.vec<8xf32>
  %below = isa.less(%smaller, %cosines) : (!isa.vec<8xf32>, !isa.vec<8xf32>) -> !isa.vec<8xi1>
  %at_most = isa.less_equal(%smaller, %cosines) : (!isa.vec<8xf32>, !isa.vec<8xf32>) -> !isa.vec<8xi1>
  %same = isa.equal(%smaller, %cosines) : (!isa.vec<8xf32>, !isa.vec<8xf32>) -> !isa.vec<8xi1>
  %at_least = isa.greater_equal(%smaller, %cosines) : (!isa.vec<8xf32>, !isa.vec<8xf32>) -> !isa.vec<8xi1>
  %above = isa.greater(%smaller, %cosines) : (!isa.vec<8xf32>, !isa.vec<8xf32>) -> !isa.vec<8xi1>
  %different = isa.not_equal(%smaller, %cosines) : (!isa.vec<8xf32>, !isa.vec<8xf32>) -> !isa.vec<8xi1>
  %chosen = isa.select(%below, %smaller, %cosines) : (!isa.vec<8xi1>, !isa.vec<8xf32>, !isa.vec<8xf32>) -> !isa.vec<8xf32>
  %narrow = isa.cast(%chosen) {to = bf16} : (!isa.vec<8xf32>) -> !isa.vec<8xbf16>
  isa.store(%narrow, %output, %zero) : (!isa.vec<8xbf16>, !loop.buf<8xbf16, global>, index) -> ()
  loop.return() : () -> ()
}
```

Rejected, as the verify filetests pin it:

- `isa.not: the input must be an i1 or integer dtype, got f32`

### `Binary`: `isa.add`, `isa.sub`, `isa.mul`, `isa.div`, `isa.maximum`, `isa.minimum`, `isa.and`, `isa.or`, `isa.xor`, `isa.shift_left`, `isa.shift_right_logical`, `isa.shift_right_arithmetic`

The `arith` op of the same name on each pair of lanes of two groups
of registers of one type.

Defined by `Binary` in [crates/gpt-rs-ir/src/dialects/isa/ir.rs](../../crates/gpt-rs-ir/src/dialects/isa/ir.rs).

| | |
|---|---|
| Operands | `lhs`: vector; `rhs`: vector |
| Results | `output`: vector |
| Effects | none |

| Op | |
|---|---|
| `isa.add` | `arith.add` on each pair of lanes. |
| `isa.sub` | `arith.sub` on each pair of lanes. |
| `isa.mul` | `arith.mul` on each pair of lanes. |
| `isa.div` | `arith.div` on each pair of lanes. |
| `isa.maximum` | `arith.maximum` on each pair of lanes. |
| `isa.minimum` | `arith.minimum` on each pair of lanes. |
| `isa.and` | `arith.and` on each pair of lanes. |
| `isa.or` | `arith.or` on each pair of lanes. |
| `isa.xor` | `arith.xor` on each pair of lanes. |
| `isa.shift_left` | `arith.shift_left` on each pair of lanes. |
| `isa.shift_right_logical` | `arith.shift_right_logical` on each pair of lanes. |
| `isa.shift_right_arithmetic` | `arith.shift_right_arithmetic` on each pair of lanes. |

Example, from `filetests/isa/roundtrip/lanewise.ptir`:

```text
loop.func @integer_lanes(%input: !loop.buf<16xsi32, global>, %output: !loop.buf<16xf32, global>) attributes {access = [in, out]} {
  %zero = loop.index() {expr = 0} : () -> index
  %lanes = isa.load(%input, %zero) {dims = [16]} : (!loop.buf<16xsi32, global>, index) -> !isa.vec<16xsi32>
  %inverted = isa.not(%lanes) : (!isa.vec<16xsi32>) -> !isa.vec<16xsi32>
  %both = isa.and(%lanes, %inverted) : (!isa.vec<16xsi32>, !isa.vec<16xsi32>) -> !isa.vec<16xsi32>
  %either = isa.or(%both, %lanes) : (!isa.vec<16xsi32>, !isa.vec<16xsi32>) -> !isa.vec<16xsi32>
  %all_ones = isa.xor(%either, %inverted) : (!isa.vec<16xsi32>, !isa.vec<16xsi32>) -> !isa.vec<16xsi32>
  %count = arith.constant() {value = 3 : si32} : () -> si32
  %counts = isa.splat(%count) {dims = [16]} : (si32) -> !isa.vec<16xsi32>
  %left = isa.shift_left(%all_ones, %counts) : (!isa.vec<16xsi32>, !isa.vec<16xsi32>) -> !isa.vec<16xsi32>
  %logical = isa.shift_right_logical(%left, %counts) : (!isa.vec<16xsi32>, !isa.vec<16xsi32>) -> !isa.vec<16xsi32>
  %shifted = isa.shift_right_arithmetic(%logical, %counts) : (!isa.vec<16xsi32>, !isa.vec<16xsi32>) -> !isa.vec<16xsi32>
  %floats = isa.bitcast(%shifted) {to = f32} : (!isa.vec<16xsi32>) -> !isa.vec<16xf32>
  isa.store(%floats, %output, %zero) : (!isa.vec<16xf32>, !loop.buf<16xf32, global>, index) -> ()
  loop.return() : () -> ()
}
```

Rejected, as the verify filetests pin it:

- `isa.add: operands must have equal types, got !isa.vec<16xf32> and !isa.vec<2x8xf32>`
- `isa.add: each operand must be an integer or float dtype, got i1`

### `Compare`: `isa.less`, `isa.less_equal`, `isa.equal`, `isa.greater_equal`, `isa.greater`, `isa.not_equal`

A mask whose flag says whether the relation of the `arith` op of
the same name holds between each pair of lanes of two groups of registers of one type.

Defined by `Compare` in [crates/gpt-rs-ir/src/dialects/isa/ir.rs](../../crates/gpt-rs-ir/src/dialects/isa/ir.rs).

| | |
|---|---|
| Operands | `lhs`: vector; `rhs`: vector |
| Results | `output`: vector |
| Effects | none |

| Op | |
|---|---|
| `isa.less` | `arith.less` on each pair of lanes. |
| `isa.less_equal` | `arith.less_equal` on each pair of lanes. |
| `isa.equal` | `arith.equal` on each pair of lanes. |
| `isa.greater_equal` | `arith.greater_equal` on each pair of lanes. |
| `isa.greater` | `arith.greater` on each pair of lanes. |
| `isa.not_equal` | `arith.not_equal` on each pair of lanes. |

Example, from `filetests/isa/roundtrip/lanewise.ptir`:

```text
loop.func @float_lanes(%input: !loop.buf<8xf32, global>, %output: !loop.buf<8xbf16, global>) attributes {access = [in, out]} {
  %zero = loop.index() {expr = 0} : () -> index
  %lanes = isa.load(%input, %zero) {dims = [8]} : (!loop.buf<8xf32, global>, index) -> !isa.vec<8xf32>
  %negated = isa.neg(%lanes) : (!isa.vec<8xf32>) -> !isa.vec<8xf32>
  %magnitudes = isa.abs(%negated) : (!isa.vec<8xf32>) -> !isa.vec<8xf32>
  %powers = isa.exp(%magnitudes) : (!isa.vec<8xf32>) -> !isa.vec<8xf32>
  %logs = isa.log(%powers) : (!isa.vec<8xf32>) -> !isa.vec<8xf32>
  %tanhs = isa.tanh(%logs) : (!isa.vec<8xf32>) -> !isa.vec<8xf32>
  %erfs = isa.erf(%tanhs) : (!isa.vec<8xf32>) -> !isa.vec<8xf32>
  %roots = isa.rsqrt(%erfs) : (!isa.vec<8xf32>) -> !isa.vec<8xf32>
  %inverses = isa.reciprocal(%roots) : (!isa.vec<8xf32>) -> !isa.vec<8xf32>
  %sines = isa.sin(%inverses) : (!isa.vec<8xf32>) -> !isa.vec<8xf32>
  %cosines = isa.cos(%sines) : (!isa.vec<8xf32>) -> !isa.vec<8xf32>
  %differences = isa.sub(%lanes, %cosines) : (!isa.vec<8xf32>, !isa.vec<8xf32>) -> !isa.vec<8xf32>
  %quotients = isa.div(%differences, %cosines) : (!isa.vec<8xf32>, !isa.vec<8xf32>) -> !isa.vec<8xf32>
  %smaller = isa.minimum(%quotients, %lanes) : (!isa.vec<8xf32>, !isa.vec<8xf32>) -> !isa.vec<8xf32>
  %below = isa.less(%smaller, %cosines) : (!isa.vec<8xf32>, !isa.vec<8xf32>) -> !isa.vec<8xi1>
  %at_most = isa.less_equal(%smaller, %cosines) : (!isa.vec<8xf32>, !isa.vec<8xf32>) -> !isa.vec<8xi1>
  %same = isa.equal(%smaller, %cosines) : (!isa.vec<8xf32>, !isa.vec<8xf32>) -> !isa.vec<8xi1>
  %at_least = isa.greater_equal(%smaller, %cosines) : (!isa.vec<8xf32>, !isa.vec<8xf32>) -> !isa.vec<8xi1>
  %above = isa.greater(%smaller, %cosines) : (!isa.vec<8xf32>, !isa.vec<8xf32>) -> !isa.vec<8xi1>
  %different = isa.not_equal(%smaller, %cosines) : (!isa.vec<8xf32>, !isa.vec<8xf32>) -> !isa.vec<8xi1>
  %chosen = isa.select(%below, %smaller, %cosines) : (!isa.vec<8xi1>, !isa.vec<8xf32>, !isa.vec<8xf32>) -> !isa.vec<8xf32>
  %narrow = isa.cast(%chosen) {to = bf16} : (!isa.vec<8xf32>) -> !isa.vec<8xbf16>
  isa.store(%narrow, %output, %zero) : (!isa.vec<8xbf16>, !loop.buf<8xbf16, global>, index) -> ()
  loop.return() : () -> ()
}
```

Rejected, as the verify filetests pin it:

- `isa.less: each operand must be an i1, integer or float dtype, got f8e4m3`

### `isa.select`

Per lane, `on_true` where the flag of `pred` is set, else `on_false`.

Defined by `Select` in [crates/gpt-rs-ir/src/dialects/isa/ir.rs](../../crates/gpt-rs-ir/src/dialects/isa/ir.rs).

| | |
|---|---|
| Operands | `pred`: vector; `on_true`: vector; `on_false`: vector |
| Results | `output`: vector |
| Effects | none |

Example, from `filetests/isa/roundtrip/lanewise.ptir`:

```text
loop.func @float_lanes(%input: !loop.buf<8xf32, global>, %output: !loop.buf<8xbf16, global>) attributes {access = [in, out]} {
  %zero = loop.index() {expr = 0} : () -> index
  %lanes = isa.load(%input, %zero) {dims = [8]} : (!loop.buf<8xf32, global>, index) -> !isa.vec<8xf32>
  %negated = isa.neg(%lanes) : (!isa.vec<8xf32>) -> !isa.vec<8xf32>
  %magnitudes = isa.abs(%negated) : (!isa.vec<8xf32>) -> !isa.vec<8xf32>
  %powers = isa.exp(%magnitudes) : (!isa.vec<8xf32>) -> !isa.vec<8xf32>
  %logs = isa.log(%powers) : (!isa.vec<8xf32>) -> !isa.vec<8xf32>
  %tanhs = isa.tanh(%logs) : (!isa.vec<8xf32>) -> !isa.vec<8xf32>
  %erfs = isa.erf(%tanhs) : (!isa.vec<8xf32>) -> !isa.vec<8xf32>
  %roots = isa.rsqrt(%erfs) : (!isa.vec<8xf32>) -> !isa.vec<8xf32>
  %inverses = isa.reciprocal(%roots) : (!isa.vec<8xf32>) -> !isa.vec<8xf32>
  %sines = isa.sin(%inverses) : (!isa.vec<8xf32>) -> !isa.vec<8xf32>
  %cosines = isa.cos(%sines) : (!isa.vec<8xf32>) -> !isa.vec<8xf32>
  %differences = isa.sub(%lanes, %cosines) : (!isa.vec<8xf32>, !isa.vec<8xf32>) -> !isa.vec<8xf32>
  %quotients = isa.div(%differences, %cosines) : (!isa.vec<8xf32>, !isa.vec<8xf32>) -> !isa.vec<8xf32>
  %smaller = isa.minimum(%quotients, %lanes) : (!isa.vec<8xf32>, !isa.vec<8xf32>) -> !isa.vec<8xf32>
  %below = isa.less(%smaller, %cosines) : (!isa.vec<8xf32>, !isa.vec<8xf32>) -> !isa.vec<8xi1>
  %at_most = isa.less_equal(%smaller, %cosines) : (!isa.vec<8xf32>, !isa.vec<8xf32>) -> !isa.vec<8xi1>
  %same = isa.equal(%smaller, %cosines) : (!isa.vec<8xf32>, !isa.vec<8xf32>) -> !isa.vec<8xi1>
  %at_least = isa.greater_equal(%smaller, %cosines) : (!isa.vec<8xf32>, !isa.vec<8xf32>) -> !isa.vec<8xi1>
  %above = isa.greater(%smaller, %cosines) : (!isa.vec<8xf32>, !isa.vec<8xf32>) -> !isa.vec<8xi1>
  %different = isa.not_equal(%smaller, %cosines) : (!isa.vec<8xf32>, !isa.vec<8xf32>) -> !isa.vec<8xi1>
  %chosen = isa.select(%below, %smaller, %cosines) : (!isa.vec<8xi1>, !isa.vec<8xf32>, !isa.vec<8xf32>) -> !isa.vec<8xf32>
  %narrow = isa.cast(%chosen) {to = bf16} : (!isa.vec<8xf32>) -> !isa.vec<8xbf16>
  isa.store(%narrow, %output, %zero) : (!isa.vec<8xbf16>, !loop.buf<8xbf16, global>, index) -> ()
  loop.return() : () -> ()
}
```

Rejected, as the verify filetests pin it:

- `isa.select: pred is !isa.vec<8xf32>, not a mask of the extents [8]`
- `isa.select: on_true and on_false must have equal types, got !isa.vec<8xf32> and !isa.vec<8xsi32>`

### `isa.fma`

`arith.fma` on each triple of lanes.

Defined by `Fma` in [crates/gpt-rs-ir/src/dialects/isa/ir.rs](../../crates/gpt-rs-ir/src/dialects/isa/ir.rs).

| | |
|---|---|
| Operands | `lhs`: vector; `rhs`: vector; `acc`: vector |
| Results | `output`: vector |
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

- `isa.fma: lhs and rhs must have equal types, got !isa.vec<16xf32> and !isa.vec<2x8xf32>`
- `isa.fma: lhs and acc must have equal types, got !isa.vec<16xf32> and !isa.vec<2x8xf32>`
- `isa.fma: fma computes on f32 and f64, not si32`

### `isa.cast`

Each lane converted to `to`, as `arith.cast` converts a scalar.

Defined by `Cast` in [crates/gpt-rs-ir/src/dialects/isa/ir.rs](../../crates/gpt-rs-ir/src/dialects/isa/ir.rs).

| | |
|---|---|
| Operands | `input`: vector |
| Results | `output`: vector |
| Attributes | `to`: `DType` |
| Effects | none |

Example, from `filetests/isa/roundtrip/lanewise.ptir`:

```text
loop.func @float_lanes(%input: !loop.buf<8xf32, global>, %output: !loop.buf<8xbf16, global>) attributes {access = [in, out]} {
  %zero = loop.index() {expr = 0} : () -> index
  %lanes = isa.load(%input, %zero) {dims = [8]} : (!loop.buf<8xf32, global>, index) -> !isa.vec<8xf32>
  %negated = isa.neg(%lanes) : (!isa.vec<8xf32>) -> !isa.vec<8xf32>
  %magnitudes = isa.abs(%negated) : (!isa.vec<8xf32>) -> !isa.vec<8xf32>
  %powers = isa.exp(%magnitudes) : (!isa.vec<8xf32>) -> !isa.vec<8xf32>
  %logs = isa.log(%powers) : (!isa.vec<8xf32>) -> !isa.vec<8xf32>
  %tanhs = isa.tanh(%logs) : (!isa.vec<8xf32>) -> !isa.vec<8xf32>
  %erfs = isa.erf(%tanhs) : (!isa.vec<8xf32>) -> !isa.vec<8xf32>
  %roots = isa.rsqrt(%erfs) : (!isa.vec<8xf32>) -> !isa.vec<8xf32>
  %inverses = isa.reciprocal(%roots) : (!isa.vec<8xf32>) -> !isa.vec<8xf32>
  %sines = isa.sin(%inverses) : (!isa.vec<8xf32>) -> !isa.vec<8xf32>
  %cosines = isa.cos(%sines) : (!isa.vec<8xf32>) -> !isa.vec<8xf32>
  %differences = isa.sub(%lanes, %cosines) : (!isa.vec<8xf32>, !isa.vec<8xf32>) -> !isa.vec<8xf32>
  %quotients = isa.div(%differences, %cosines) : (!isa.vec<8xf32>, !isa.vec<8xf32>) -> !isa.vec<8xf32>
  %smaller = isa.minimum(%quotients, %lanes) : (!isa.vec<8xf32>, !isa.vec<8xf32>) -> !isa.vec<8xf32>
  %below = isa.less(%smaller, %cosines) : (!isa.vec<8xf32>, !isa.vec<8xf32>) -> !isa.vec<8xi1>
  %at_most = isa.less_equal(%smaller, %cosines) : (!isa.vec<8xf32>, !isa.vec<8xf32>) -> !isa.vec<8xi1>
  %same = isa.equal(%smaller, %cosines) : (!isa.vec<8xf32>, !isa.vec<8xf32>) -> !isa.vec<8xi1>
  %at_least = isa.greater_equal(%smaller, %cosines) : (!isa.vec<8xf32>, !isa.vec<8xf32>) -> !isa.vec<8xi1>
  %above = isa.greater(%smaller, %cosines) : (!isa.vec<8xf32>, !isa.vec<8xf32>) -> !isa.vec<8xi1>
  %different = isa.not_equal(%smaller, %cosines) : (!isa.vec<8xf32>, !isa.vec<8xf32>) -> !isa.vec<8xi1>
  %chosen = isa.select(%below, %smaller, %cosines) : (!isa.vec<8xi1>, !isa.vec<8xf32>, !isa.vec<8xf32>) -> !isa.vec<8xf32>
  %narrow = isa.cast(%chosen) {to = bf16} : (!isa.vec<8xf32>) -> !isa.vec<8xbf16>
  isa.store(%narrow, %output, %zero) : (!isa.vec<8xbf16>, !loop.buf<8xbf16, global>, index) -> ()
  loop.return() : () -> ()
}
```

Rejected, as the verify filetests pin it:

- `isa.cast: to must be an i1, integer, float or narrow float dtype, got e8m0`

### `isa.bitcast`

The bits of each lane read as a value of `to`, of the same width, as
`arith.bitcast` reads a scalar.

Defined by `Bitcast` in [crates/gpt-rs-ir/src/dialects/isa/ir.rs](../../crates/gpt-rs-ir/src/dialects/isa/ir.rs).

| | |
|---|---|
| Operands | `input`: vector |
| Results | `output`: vector |
| Attributes | `to`: `DType` |
| Effects | none |

Example, from `filetests/isa/roundtrip/lanewise.ptir`:

```text
loop.func @integer_lanes(%input: !loop.buf<16xsi32, global>, %output: !loop.buf<16xf32, global>) attributes {access = [in, out]} {
  %zero = loop.index() {expr = 0} : () -> index
  %lanes = isa.load(%input, %zero) {dims = [16]} : (!loop.buf<16xsi32, global>, index) -> !isa.vec<16xsi32>
  %inverted = isa.not(%lanes) : (!isa.vec<16xsi32>) -> !isa.vec<16xsi32>
  %both = isa.and(%lanes, %inverted) : (!isa.vec<16xsi32>, !isa.vec<16xsi32>) -> !isa.vec<16xsi32>
  %either = isa.or(%both, %lanes) : (!isa.vec<16xsi32>, !isa.vec<16xsi32>) -> !isa.vec<16xsi32>
  %all_ones = isa.xor(%either, %inverted) : (!isa.vec<16xsi32>, !isa.vec<16xsi32>) -> !isa.vec<16xsi32>
  %count = arith.constant() {value = 3 : si32} : () -> si32
  %counts = isa.splat(%count) {dims = [16]} : (si32) -> !isa.vec<16xsi32>
  %left = isa.shift_left(%all_ones, %counts) : (!isa.vec<16xsi32>, !isa.vec<16xsi32>) -> !isa.vec<16xsi32>
  %logical = isa.shift_right_logical(%left, %counts) : (!isa.vec<16xsi32>, !isa.vec<16xsi32>) -> !isa.vec<16xsi32>
  %shifted = isa.shift_right_arithmetic(%logical, %counts) : (!isa.vec<16xsi32>, !isa.vec<16xsi32>) -> !isa.vec<16xsi32>
  %floats = isa.bitcast(%shifted) {to = f32} : (!isa.vec<16xsi32>) -> !isa.vec<16xf32>
  isa.store(%floats, %output, %zero) : (!isa.vec<16xf32>, !loop.buf<16xf32, global>, index) -> ()
  loop.return() : () -> ()
}
```

Rejected, as the verify filetests pin it:

- `isa.bitcast: a bitcast keeps the width: f32 has 32 bits and f64 has 64`

### `isa.splat`

The scalar in every lane of a group of registers of the extents `dims`, the lanes
last.

Defined by `Splat` in [crates/gpt-rs-ir/src/dialects/isa/ir.rs](../../crates/gpt-rs-ir/src/dialects/isa/ir.rs).

| | |
|---|---|
| Operands | `input`: scalar |
| Results | `output`: vector |
| Attributes | `dims`: `[usize]` |
| Effects | none |

Example, from `filetests/isa/roundtrip/memory.ptir`:

```text
loop.func @relu_tail(%input: !loop.buf<40xf32, global>, %scale: !loop.buf<f32, global>, %output: !loop.buf<40xf32, global>) attributes {access = [in, in, out]} {
  %start = loop.index() {expr = 32} : () -> index
  %valid = loop.index() {expr = 8} : () -> index
  %zero = arith.constant() {value = 0.0 : f32} : () -> f32
  %mask = isa.tile_mask(%valid) {dims = [1, 16]} : (index) -> !isa.vec<1x16xi1>
  %tail = isa.masked_load(%mask, %input, %start) {dims = [1, 16], steps = [[16]]} : (!isa.vec<1x16xi1>, !loop.buf<40xf32, global>, index) -> !isa.vec<1x16xf32>
  %factor = isa.masked_broadcast(%mask, %scale) {dims = [1, 16], steps = [[]]} : (!isa.vec<1x16xi1>, !loop.buf<f32, global>) -> !isa.vec<1x16xf32>
  %scaled = isa.mul(%tail, %factor) : (!isa.vec<1x16xf32>, !isa.vec<1x16xf32>) -> !isa.vec<1x16xf32>
  %zeros = isa.splat(%zero) {dims = [1, 16]} : (f32) -> !isa.vec<1x16xf32>
  %rectified = isa.maximum(%scaled, %zeros) : (!isa.vec<1x16xf32>, !isa.vec<1x16xf32>) -> !isa.vec<1x16xf32>
  isa.masked_store(%mask, %rectified, %output, %start) {steps = [[16]]} : (!isa.vec<1x16xi1>, !isa.vec<1x16xf32>, !loop.buf<40xf32, global>, index) -> ()
  loop.return() : () -> ()
}
```

Rejected, as the verify filetests pin it:

- `isa.splat: a vector has lanes and a group rank of at most 3, not the extents []`

### `isa.expand`

The group of registers with each axis of extent 1 repeated to its extent in
`dims`, which has the input's lanes and number of axes.

Defined by `Expand` in [crates/gpt-rs-ir/src/dialects/isa/ir.rs](../../crates/gpt-rs-ir/src/dialects/isa/ir.rs).

| | |
|---|---|
| Operands | `input`: vector |
| Results | `output`: vector |
| Attributes | `dims`: `[usize]` |
| Effects | none |

Example, from `filetests/isa/roundtrip/memory.ptir`:

```text
loop.func @scale_and_shift(%input: !loop.buf<2x16xf32, global>, %bias: !loop.buf<16xf32, global>, %scale: !loop.buf<f32, global>, %output: !loop.buf<2x16xf32, global>) attributes {access = [in, in, in, out]} {
  %zero = loop.index() {expr = 0} : () -> index
  %rows = isa.load(%input, %zero, %zero) {dims = [2, 1, 16], steps = [[1, 0], [0, 16]]} : (!loop.buf<2x16xf32, global>, index, index) -> !isa.vec<2x1x16xf32>
  %bias_row = isa.load(%bias, %zero) {dims = [1, 1, 16], steps = [[0], [16]]} : (!loop.buf<16xf32, global>, index) -> !isa.vec<1x1x16xf32>
  %bias_rows = isa.expand(%bias_row) {dims = [2, 1, 16]} : (!isa.vec<1x1x16xf32>) -> !isa.vec<2x1x16xf32>
  %factor = isa.broadcast(%scale) {dims = [2, 1, 16], steps = [[], []]} : (!loop.buf<f32, global>) -> !isa.vec<2x1x16xf32>
  %scaled = isa.mul(%rows, %factor) : (!isa.vec<2x1x16xf32>, !isa.vec<2x1x16xf32>) -> !isa.vec<2x1x16xf32>
  %shifted = isa.add(%scaled, %bias_rows) : (!isa.vec<2x1x16xf32>, !isa.vec<2x1x16xf32>) -> !isa.vec<2x1x16xf32>
  isa.store(%shifted, %output, %zero, %zero) {steps = [[1, 0], [0, 16]]} : (!isa.vec<2x1x16xf32>, !loop.buf<2x16xf32, global>, index, index) -> ()
  loop.return() : () -> ()
}
```

Rejected, as the verify filetests pin it:

- `isa.expand: !isa.vec<2x8xf32> repeats only axes of extent 1 to !isa.vec<4x8xf32>`
- `isa.expand: !isa.vec<2x8xf32> has other lanes or another number of axes than !isa.vec<2x16xf32>`
- `isa.expand: !isa.vec<2x8xf32> has other lanes or another number of axes than !isa.vec<3x2x8xf32>`

### `isa.tile_mask`

The mask of the points of a register tile that lie inside `extents`, one per
axis of the tile. The tile's axes are those of the group, the last one with its lanes, or the
lanes alone for a mask with no group axes: lane `l` of register `[i, j]` of a mask of extents
`[R, V, L]` is the point `[i, j * L + l]`. Its flag is set when each group axis but the last is
below its own extent, and the last one with the lanes below the last extent: here
`i < extents[0]` and `j * L + l < extents[1]`.

Defined by `TileMask` in [crates/gpt-rs-ir/src/dialects/isa/ir.rs](../../crates/gpt-rs-ir/src/dialects/isa/ir.rs).

| | |
|---|---|
| Operands | `extents`: indices |
| Results | `output`: vector |
| Attributes | `dims`: `[usize]` |
| Effects | none |

Example, from `filetests/isa/roundtrip/memory.ptir`:

```text
loop.func @relu_tail(%input: !loop.buf<40xf32, global>, %scale: !loop.buf<f32, global>, %output: !loop.buf<40xf32, global>) attributes {access = [in, in, out]} {
  %start = loop.index() {expr = 32} : () -> index
  %valid = loop.index() {expr = 8} : () -> index
  %zero = arith.constant() {value = 0.0 : f32} : () -> f32
  %mask = isa.tile_mask(%valid) {dims = [1, 16]} : (index) -> !isa.vec<1x16xi1>
  %tail = isa.masked_load(%mask, %input, %start) {dims = [1, 16], steps = [[16]]} : (!isa.vec<1x16xi1>, !loop.buf<40xf32, global>, index) -> !isa.vec<1x16xf32>
  %factor = isa.masked_broadcast(%mask, %scale) {dims = [1, 16], steps = [[]]} : (!isa.vec<1x16xi1>, !loop.buf<f32, global>) -> !isa.vec<1x16xf32>
  %scaled = isa.mul(%tail, %factor) : (!isa.vec<1x16xf32>, !isa.vec<1x16xf32>) -> !isa.vec<1x16xf32>
  %zeros = isa.splat(%zero) {dims = [1, 16]} : (f32) -> !isa.vec<1x16xf32>
  %rectified = isa.maximum(%scaled, %zeros) : (!isa.vec<1x16xf32>, !isa.vec<1x16xf32>) -> !isa.vec<1x16xf32>
  isa.masked_store(%mask, %rectified, %output, %start) {steps = [[16]]} : (!isa.vec<1x16xi1>, !isa.vec<1x16xf32>, !loop.buf<40xf32, global>, index) -> ()
  loop.return() : () -> ()
}
```

Rejected, as the verify filetests pin it:

- `isa.tile_mask: the extent count is 1, but the tile has rank 2`

### `Read`: `isa.load`, `isa.broadcast`

A group of registers of extents `dims` from `buffer`, register `g`
from the position `base + g[0] * steps[0] + ...`. Without `pad`, every position MUST be inside
the buffer. With `pad v`, a lane may fall outside the buffer on the axes that `checked` names
for its register, or on any axis when `checked` is empty. Such a lane is `v` and reads nothing.
On any other axis, its position MUST be inside.

Defined by `Read` in [crates/gpt-rs-ir/src/dialects/isa/ir.rs](../../crates/gpt-rs-ir/src/dialects/isa/ir.rs).

| | |
|---|---|
| Operands | `buffer`: buffer; `base`: indices |
| Results | `output`: vector |
| Attributes | `dims`: `[usize]`; `steps`: `[[i64]]`, optional; `pad`: `F64`, optional; `checked`: `[[usize]]`, optional |
| Effects | reads the memory of `buffer` |

| Op | |
|---|---|
| `isa.load` | Lane `l` of a register is the element `l` further along the last axis of the buffer than the register's position. |
| `isa.broadcast` | Every lane of a register is the element at the register's position. |

Example, from `filetests/isa/roundtrip/memory.ptir`:

```text
loop.func @scale_and_shift(%input: !loop.buf<2x16xf32, global>, %bias: !loop.buf<16xf32, global>, %scale: !loop.buf<f32, global>, %output: !loop.buf<2x16xf32, global>) attributes {access = [in, in, in, out]} {
  %zero = loop.index() {expr = 0} : () -> index
  %rows = isa.load(%input, %zero, %zero) {dims = [2, 1, 16], steps = [[1, 0], [0, 16]]} : (!loop.buf<2x16xf32, global>, index, index) -> !isa.vec<2x1x16xf32>
  %bias_row = isa.load(%bias, %zero) {dims = [1, 1, 16], steps = [[0], [16]]} : (!loop.buf<16xf32, global>, index) -> !isa.vec<1x1x16xf32>
  %bias_rows = isa.expand(%bias_row) {dims = [2, 1, 16]} : (!isa.vec<1x1x16xf32>) -> !isa.vec<2x1x16xf32>
  %factor = isa.broadcast(%scale) {dims = [2, 1, 16], steps = [[], []]} : (!loop.buf<f32, global>) -> !isa.vec<2x1x16xf32>
  %scaled = isa.mul(%rows, %factor) : (!isa.vec<2x1x16xf32>, !isa.vec<2x1x16xf32>) -> !isa.vec<2x1x16xf32>
  %shifted = isa.add(%scaled, %bias_rows) : (!isa.vec<2x1x16xf32>, !isa.vec<2x1x16xf32>) -> !isa.vec<2x1x16xf32>
  isa.store(%shifted, %output, %zero, %zero) {steps = [[1, 0], [0, 16]]} : (!isa.vec<2x1x16xf32>, !loop.buf<2x16xf32, global>, index, index) -> ()
  loop.return() : () -> ()
}
```

Rejected, as the verify filetests pin it:

- `isa.load: the base index count is 1, but !loop.buf<4x16xf32, global> has rank 2`
- `isa.load: the step count is 0, but the group rank of !isa.vec<4x16xf32> is 1`
- `isa.broadcast: a step's length is 1, but !loop.buf<4x16xf32, global> has rank 2`
- `isa.load: it pads with 0.5, which si32 cannot hold`
- `isa.load: it names checked axes but has no pad`
- `isa.load: `checked` has one entry per register, and holds 1 for the 2 registers of !isa.vec<2x16xf32>`
- `isa.broadcast: it checks axis 1 of !loop.buf<40xf32, global>, which has no axis 1`

### `MaskedRead`: `isa.masked_load`, `isa.masked_broadcast`

The read of the same name without `masked_` on the
lanes whose flag in `mask`, a mask of the extents `dims`, is set. The other lanes are 0, and
read nothing.

Defined by `MaskedRead` in [crates/gpt-rs-ir/src/dialects/isa/ir.rs](../../crates/gpt-rs-ir/src/dialects/isa/ir.rs).

| | |
|---|---|
| Operands | `mask`: vector; `buffer`: buffer; `base`: indices |
| Results | `output`: vector |
| Attributes | `dims`: `[usize]`; `steps`: `[[i64]]`, optional; `pad`: `F64`, optional; `checked`: `[[usize]]`, optional |
| Effects | reads the memory of `buffer` |

| Op | |
|---|---|
| `isa.masked_load` | `isa.load` of the lanes whose flag in `mask` is set. |
| `isa.masked_broadcast` | `isa.broadcast` into the lanes whose flag in `mask` is set. |

Example, from `filetests/isa/roundtrip/memory.ptir`:

```text
loop.func @relu_tail(%input: !loop.buf<40xf32, global>, %scale: !loop.buf<f32, global>, %output: !loop.buf<40xf32, global>) attributes {access = [in, in, out]} {
  %start = loop.index() {expr = 32} : () -> index
  %valid = loop.index() {expr = 8} : () -> index
  %zero = arith.constant() {value = 0.0 : f32} : () -> f32
  %mask = isa.tile_mask(%valid) {dims = [1, 16]} : (index) -> !isa.vec<1x16xi1>
  %tail = isa.masked_load(%mask, %input, %start) {dims = [1, 16], steps = [[16]]} : (!isa.vec<1x16xi1>, !loop.buf<40xf32, global>, index) -> !isa.vec<1x16xf32>
  %factor = isa.masked_broadcast(%mask, %scale) {dims = [1, 16], steps = [[]]} : (!isa.vec<1x16xi1>, !loop.buf<f32, global>) -> !isa.vec<1x16xf32>
  %scaled = isa.mul(%tail, %factor) : (!isa.vec<1x16xf32>, !isa.vec<1x16xf32>) -> !isa.vec<1x16xf32>
  %zeros = isa.splat(%zero) {dims = [1, 16]} : (f32) -> !isa.vec<1x16xf32>
  %rectified = isa.maximum(%scaled, %zeros) : (!isa.vec<1x16xf32>, !isa.vec<1x16xf32>) -> !isa.vec<1x16xf32>
  isa.masked_store(%mask, %rectified, %output, %start) {steps = [[16]]} : (!isa.vec<1x16xi1>, !isa.vec<1x16xf32>, !loop.buf<40xf32, global>, index) -> ()
  loop.return() : () -> ()
}
```

Rejected, as the verify filetests pin it:

- `isa.masked_load: the mask is !isa.vec<16xf32>, not a mask of the extents [16]`

### `isa.store`

Writes the lanes of `value` into `buffer` where `isa.load` would read them.

Defined by `Store` in [crates/gpt-rs-ir/src/dialects/isa/ir.rs](../../crates/gpt-rs-ir/src/dialects/isa/ir.rs).

| | |
|---|---|
| Operands | `value`: vector; `buffer`: buffer; `base`: indices |
| Results | none |
| Attributes | `steps`: `[[i64]]`, optional |
| Effects | writes the memory of `buffer` |

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

- `isa.store: the value is !isa.vec<16xf32>, but !loop.buf<16xsi32, global> holds si32`

### `isa.masked_store`

`isa.store` of the lanes whose flag in `mask` is set. The other elements
keep their values.

Defined by `MaskedStore` in [crates/gpt-rs-ir/src/dialects/isa/ir.rs](../../crates/gpt-rs-ir/src/dialects/isa/ir.rs).

| | |
|---|---|
| Operands | `mask`: vector; `value`: vector; `buffer`: buffer; `base`: indices |
| Results | none |
| Attributes | `steps`: `[[i64]]`, optional |
| Effects | writes the memory of `buffer` |

Example, from `filetests/isa/roundtrip/memory.ptir`:

```text
loop.func @relu_tail(%input: !loop.buf<40xf32, global>, %scale: !loop.buf<f32, global>, %output: !loop.buf<40xf32, global>) attributes {access = [in, in, out]} {
  %start = loop.index() {expr = 32} : () -> index
  %valid = loop.index() {expr = 8} : () -> index
  %zero = arith.constant() {value = 0.0 : f32} : () -> f32
  %mask = isa.tile_mask(%valid) {dims = [1, 16]} : (index) -> !isa.vec<1x16xi1>
  %tail = isa.masked_load(%mask, %input, %start) {dims = [1, 16], steps = [[16]]} : (!isa.vec<1x16xi1>, !loop.buf<40xf32, global>, index) -> !isa.vec<1x16xf32>
  %factor = isa.masked_broadcast(%mask, %scale) {dims = [1, 16], steps = [[]]} : (!isa.vec<1x16xi1>, !loop.buf<f32, global>) -> !isa.vec<1x16xf32>
  %scaled = isa.mul(%tail, %factor) : (!isa.vec<1x16xf32>, !isa.vec<1x16xf32>) -> !isa.vec<1x16xf32>
  %zeros = isa.splat(%zero) {dims = [1, 16]} : (f32) -> !isa.vec<1x16xf32>
  %rectified = isa.maximum(%scaled, %zeros) : (!isa.vec<1x16xf32>, !isa.vec<1x16xf32>) -> !isa.vec<1x16xf32>
  isa.masked_store(%mask, %rectified, %output, %start) {steps = [[16]]} : (!isa.vec<1x16xi1>, !isa.vec<1x16xf32>, !loop.buf<40xf32, global>, index) -> ()
  loop.return() : () -> ()
}
```

Rejected, as the verify filetests pin it:

- `isa.masked_store: the mask is !isa.vec<8xi1>, not a mask of the extents [16]`

### `isa.prefetch`

Asks the cache `level` for the element at the first lane of each register of
extents `dims` that `isa.load` of `steps` at `base` would read from `buffer`, which a later
load reads. A hint: it computes and changes nothing, and a position outside the buffer reads
nothing.

Defined by `Prefetch` in [crates/gpt-rs-ir/src/dialects/isa/ir.rs](../../crates/gpt-rs-ir/src/dialects/isa/ir.rs).

| | |
|---|---|
| Operands | `buffer`: buffer; `base`: indices |
| Results | none |
| Attributes | `dims`: `[usize]`; `steps`: `[[i64]]`, optional; `level`: `Level` |
| Effects | reads the memory of `buffer` |

Example, from `filetests/isa/roundtrip/memory.ptir`:

```text
loop.func @pairs(%input: !loop.buf<4x16xf32, global>, %larger_out: !loop.buf<2x16xf32, global>, %smaller_out: !loop.buf<2x16xf32, global>) attributes {access = [in, out, out]} {
  %zero = loop.index() {expr = 0} : () -> index
  %two = loop.index() {expr = 2} : () -> index
  %rows = isa.load(%input, %zero, %zero) {dims = [2, 1, 16], steps = [[1, 0], [0, 16]]} : (!loop.buf<4x16xf32, global>, index, index) -> !isa.vec<2x1x16xf32>
  isa.prefetch(%input, %two, %zero) {dims = [2, 1, 16], steps = [[1, 0], [0, 16]], level = l2} : (!loop.buf<4x16xf32, global>, index, index) -> ()
  %next_rows = isa.load(%input, %two, %zero) {dims = [2, 1, 16], steps = [[1, 0], [0, 16]]} : (!loop.buf<4x16xf32, global>, index, index) -> !isa.vec<2x1x16xf32>
  %larger = isa.maximum(%rows, %next_rows) : (!isa.vec<2x1x16xf32>, !isa.vec<2x1x16xf32>) -> !isa.vec<2x1x16xf32>
  %smaller = isa.minimum(%rows, %next_rows) : (!isa.vec<2x1x16xf32>, !isa.vec<2x1x16xf32>) -> !isa.vec<2x1x16xf32>
  isa.store(%larger, %larger_out, %zero, %zero) {steps = [[1, 0], [0, 16]]} : (!isa.vec<2x1x16xf32>, !loop.buf<2x16xf32, global>, index, index) -> ()
  isa.store(%smaller, %smaller_out, %zero, %zero) {steps = [[1, 0], [0, 16]]} : (!isa.vec<2x1x16xf32>, !loop.buf<2x16xf32, global>, index, index) -> ()
  loop.return() : () -> ()
}
```

## Traits

A trait states a fact about the ops that have it. The folds and patterns it implies apply to every one of them; [passes.md](passes.md) says which pass runs each.
