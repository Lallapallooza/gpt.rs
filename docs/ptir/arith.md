<!-- Generated from the code by `cargo run -p gpt-rs-docs`. Do not edit: change the code and run it again. -->

# The arith dialect

The `arith` dialect computes on scalars: the bodies of `tile.block` are written in it. Each op is
the per-element function of the `tensor` op of the same name, applied to one value, so the two
dialects agree bit for bit.

## Conventions

- Each op is defined in `ir.rs`, and its type rule is in `verify.rs`. The ops of a family
  (`Unary`, `Binary`, `Compare`, `Rng`) share the family's type rule and `Eval`.
- Each op's function is the one copy of its per-element semantics, which the `tensor` op of the
  same name applies to every element. `interp.rs` implements `UnaryMath`, `BinaryMath` and
  `CompareMath` of `crate::scalar`, and `tensor/interp/random.rs` implements `Draw`.
- Operands of one op have one dtype, as in `tensor`: there is no implicit promotion.
- bf16 and f16 compute in f32 and round once per op to their dtype, as the `tensor` ops do.
- `arith.constant` holds its value as a rank-0 constant, so constant queries, folds and trait
  rewrites see it as they see `tensor.constant`.
- Each op lists its traits, as the `tensor` op of the same name does.

## Ops

- `arith.constant`
- `arith.neg`, `arith.abs`, `arith.exp`, `arith.log`, `arith.tanh`, `arith.erf`, `arith.rsqrt`, `arith.reciprocal`, `arith.sin`, `arith.cos`, `arith.not`
- `arith.add`, `arith.sub`, `arith.mul`, `arith.div`, `arith.maximum`, `arith.minimum`, `arith.and`, `arith.or`, `arith.xor`, `arith.shift_left`, `arith.shift_right_logical`, `arith.shift_right_arithmetic`
- `arith.less`, `arith.less_equal`, `arith.equal`, `arith.greater_equal`, `arith.greater`, `arith.not_equal`
- `arith.select`
- `arith.cast`
- `arith.bitcast`
- `arith.rng_uniform`, `arith.rng_normal`

### `arith.constant`

The scalar `value`.

Defined by `Constant` in [crates/gpt-rs-ir/src/dialects/arith/ir.rs](../../crates/gpt-rs-ir/src/dialects/arith/ir.rs).

| | |
|---|---|
| Operands | none |
| Results | `output`: scalar |
| Attributes | `value`: `Literal` |
| Effects | none |

Example, from `filetests/tile/roundtrip/arith.ptir`:

```text
func.func @literals() -> (tensor<si64>, tensor<i1>, tensor<bf16>, tensor<f64>) {
  %0, %1, %2, %3 = tile.block() {iters = [], reads = [], folds = [none -> [], none -> [], none -> [], none -> []]} ({
    %0 = arith.constant() {value = -9223372036854775808 : si64} : () -> si64
    %1 = arith.constant() {value = 1 : i1} : () -> i1
    %2 = arith.constant() {value = -inf : bf16} : () -> bf16
    %3 = arith.constant() {value = 0.1 : f64} : () -> f64
    tile.yield(%0, %1, %2, %3) : (si64, i1, bf16, f64) -> ()
  }) : () -> (tensor<si64>, tensor<i1>, tensor<bf16>, tensor<f64>)
  func.return(%0, %1, %2, %3) : (tensor<si64>, tensor<i1>, tensor<bf16>, tensor<f64>) -> ()
}
```

### `Unary`: `arith.neg`, `arith.abs`, `arith.exp`, `arith.log`, `arith.tanh`, `arith.erf`, `arith.rsqrt`, `arith.reciprocal`, `arith.sin`, `arith.cos`, `arith.not`

An op on one scalar, as the `tensor` op of the same name applies
it to each element. [`UnaryMath`] is its function.

Defined by `Unary` in [crates/gpt-rs-ir/src/dialects/arith/ir.rs](../../crates/gpt-rs-ir/src/dialects/arith/ir.rs).

| | |
|---|---|
| Operands | `input`: scalar |
| Results | `output`: scalar |
| Effects | none |
| Patterns | `FoldIndexArith` (`canonicalize`, at `arith.neg`); `UnaryToTile` (`tensor-to-tile`) |

| Op | Traits | |
|---|---|---|
| `arith.neg` | involution | As `tensor.neg`, on one scalar. |
| `arith.abs` | idempotent | As `tensor.abs`, on one scalar. |
| `arith.exp` |  | As `tensor.exp`, on one scalar. |
| `arith.log` |  | As `tensor.log`, on one scalar. |
| `arith.tanh` |  | As `tensor.tanh`, on one scalar. |
| `arith.erf` |  | As `tensor.erf`, on one scalar. |
| `arith.rsqrt` |  | As `tensor.rsqrt`, on one scalar. |
| `arith.reciprocal` |  | As `tensor.reciprocal`, on one scalar. |
| `arith.sin` |  | As `tensor.sin`, on one scalar. |
| `arith.cos` |  | As `tensor.cos`, on one scalar. |
| `arith.not` | involution | As `tensor.not`, on one scalar. |

Example, from `filetests/tile/roundtrip/block.ptir`:

```text
func.func @logsumexp(%0: tensor<2x16xf32>) -> (tensor<2xf32>, tensor<2xf32>, tensor<2xf32>) {
  %1, %2 = tile.block(%0) {iters = [par 2, red 16], reads = [$0[d0, d1]], folds = [logsumexp -> [d0], [d0]]} ({
  ^body(%0: f32):
    tile.yield(%0) : (f32) -> ()
  }) : (tensor<2x16xf32>) -> (tensor<2xf32>, tensor<2xf32>)
  %3, %4 = tile.block(%1, %2) {iters = [par 2, red 2], reads = [$0[d0], $1[d0]], folds = [logsumexp_merge -> [d0], [d0]]} ({
  ^body(%0: f32, %1: f32):
    tile.yield(%0, %1) : (f32, f32) -> ()
  }) : (tensor<2xf32>, tensor<2xf32>) -> (tensor<2xf32>, tensor<2xf32>)
  %5 = tile.block(%3, %4) {iters = [par 2], reads = [$0[d0], $1[d0]], folds = [none -> [d0]]} ({
  ^body(%0: f32, %1: f32):
    %2 = arith.log(%1) : (f32) -> f32
    %3 = arith.add(%0, %2) : (f32, f32) -> f32
    tile.yield(%3) : (f32) -> ()
  }) : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
  func.return(%1, %2, %5) : (tensor<2xf32>, tensor<2xf32>, tensor<2xf32>) -> ()
}
```

### `Binary`: `arith.add`, `arith.sub`, `arith.mul`, `arith.div`, `arith.maximum`, `arith.minimum`, `arith.and`, `arith.or`, `arith.xor`, `arith.shift_left`, `arith.shift_right_logical`, `arith.shift_right_arithmetic`

An op on two scalars of one dtype, as the `tensor` op of the
same name applies it to each pair of elements. [`BinaryMath`] is its function.

Defined by `Binary` in [crates/gpt-rs-ir/src/dialects/arith/ir.rs](../../crates/gpt-rs-ir/src/dialects/arith/ir.rs).

| | |
|---|---|
| Operands | `lhs`: scalar; `rhs`: scalar |
| Results | `output`: scalar |
| Effects | none |
| Patterns | `FoldIndexArith` (`canonicalize`, at `arith.add`, `arith.sub`, `arith.mul`, `arith.maximum`, `arith.minimum`); `BinaryToTile` (`tensor-to-tile`); `ScaleIntoDot` (`dot-scale-fold`, at `arith.mul`) |

| Op | Traits | |
|---|---|---|
| `arith.add` | commutative; associative on integers; right identity -0.0 on floats, 0 on integers | As `tensor.add`, on two scalars. |
| `arith.sub` | right identity 0.0 on floats, 0 on integers | As `tensor.sub`, on two scalars. |
| `arith.mul` | commutative; associative on integers; right identity 1.0 on floats, 1 on integers; absorbing 0 on integers; distributes over `arith.add` on integers | As `tensor.mul`, on two scalars. |
| `arith.div` | right identity 1.0 on floats, 1 on integers | As `tensor.div`, on two scalars. |
| `arith.maximum` | commutative; associative; idempotent; right identity -inf on floats, min on integers; absorbing max on integers | As `tensor.maximum`, on two scalars. |
| `arith.minimum` | commutative; associative; idempotent; right identity inf on floats, max on integers; absorbing min on integers | As `tensor.minimum`, on two scalars. |
| `arith.and` | commutative; associative on integers; idempotent; right identity all ones on integers; absorbing 0 on integers; distributes over `arith.or` on integers | As `tensor.and`, on two scalars. |
| `arith.or` | commutative; associative on integers; idempotent; right identity 0 on integers; absorbing all ones on integers; distributes over `arith.and` on integers | As `tensor.or`, on two scalars. |
| `arith.xor` | commutative; associative on integers; right identity 0 on integers | As `tensor.xor`, on two scalars. |
| `arith.shift_left` | right identity 0 on integers | As `tensor.shift_left`, on two scalars. |
| `arith.shift_right_logical` | right identity 0 on integers | As `tensor.shift_right_logical`, on two scalars. |
| `arith.shift_right_arithmetic` | right identity 0 on integers | As `tensor.shift_right_arithmetic`, on two scalars. |

Example, from `filetests/tile/roundtrip/arith.ptir`:

```text
func.func @scalar_ops(%0: tensor<4xf32>, %1: tensor<4xsi32>, %2: tensor<2xui64>) -> (tensor<4xf32>, tensor<4xsi32>, tensor<4xi1>, tensor<4xbf16>) {
  %3, %4, %5, %6 = tile.block(%0, %1, %2) {iters = [par 4], reads = [$0[d0], $1[d0], $2[0], $2[1]], folds = [none -> [d0], none -> [d0], none -> [d0], none -> [d0]]} ({
  ^body(%0: f32, %1: si32, %2: ui64, %3: ui64):
    %4 = arith.neg(%0) : (f32) -> f32
    %5 = arith.abs(%4) : (f32) -> f32
    %6 = arith.exp(%5) : (f32) -> f32
    %7 = arith.log(%6) : (f32) -> f32
    %8 = arith.tanh(%7) : (f32) -> f32
    %9 = arith.erf(%8) : (f32) -> f32
    %10 = arith.rsqrt(%9) : (f32) -> f32
    %11 = arith.reciprocal(%10) : (f32) -> f32
    %12 = arith.sin(%11) : (f32) -> f32
    %13 = arith.cos(%12) : (f32) -> f32
    %14 = arith.constant() {value = 0.5 : f32} : () -> f32
    %15 = arith.add(%13, %14) : (f32, f32) -> f32
    %16 = arith.sub(%15, %14) : (f32, f32) -> f32
    %17 = arith.mul(%16, %14) : (f32, f32) -> f32
    %18 = arith.div(%17, %14) : (f32, f32) -> f32
    %19 = arith.maximum(%18, %14) : (f32, f32) -> f32
    %20 = arith.minimum(%19, %14) : (f32, f32) -> f32
    %21 = arith.constant() {value = 3 : si32} : () -> si32
    %22 = arith.and(%1, %21) : (si32, si32) -> si32
    %23 = arith.or(%22, %21) : (si32, si32) -> si32
    %24 = arith.xor(%23, %21) : (si32, si32) -> si32
    %25 = arith.shift_left(%24, %21) : (si32, si32) -> si32
    %26 = arith.shift_right_logical(%25, %21) : (si32, si32) -> si32
    %27 = arith.shift_right_arithmetic(%26, %21) : (si32, si32) -> si32
    %28 = arith.not(%27) : (si32) -> si32
    %29 = arith.less(%20, %14) : (f32, f32) -> i1
    %30 = arith.select(%29, %20, %14) : (i1, f32, f32) -> f32
    %31 = arith.cast(%30) {to = bf16} : (f32) -> bf16
    %32 = arith.bitcast(%30) {to = si32} : (f32) -> si32
    %33 = arith.add(%28, %32) : (si32, si32) -> si32
    %34 = tile.index() {expr = d0, dtype = ui64} : () -> ui64
    %35 = arith.rng_uniform(%2, %3, %34) {dtype = f32} : (ui64, ui64, ui64) -> f32
    %36 = arith.rng_normal(%2, %3, %34) {dtype = bf16} : (ui64, ui64, ui64) -> bf16
    tile.yield(%35, %33, %29, %36) : (f32, si32, i1, bf16) -> ()
  }) : (tensor<4xf32>, tensor<4xsi32>, tensor<2xui64>) -> (tensor<4xf32>, tensor<4xsi32>, tensor<4xi1>, tensor<4xbf16>)
  func.return(%3, %4, %5, %6) : (tensor<4xf32>, tensor<4xsi32>, tensor<4xi1>, tensor<4xbf16>) -> ()
}
```

Rejected, as the verify filetests pin it:

- `arith.add: operands must have equal types, got f32 and bf16`

### `Compare`: `arith.less`, `arith.less_equal`, `arith.equal`, `arith.greater_equal`, `arith.greater`, `arith.not_equal`

An `i1` that says whether the relation of the op holds
between two scalars of one dtype. `tensor.less`, ... apply it to each pair of elements.
[`CompareMath`] is its function.

Defined by `Compare` in [crates/gpt-rs-ir/src/dialects/arith/ir.rs](../../crates/gpt-rs-ir/src/dialects/arith/ir.rs).

| | |
|---|---|
| Operands | `lhs`: scalar; `rhs`: scalar |
| Results | `output`: scalar |
| Effects | none |
| Patterns | `CompareWithItself` (`canonicalize`); `CompareToTile` (`tensor-to-tile`) |

| Op | |
|---|---|
| `arith.less` | As `tensor.less`, on two scalars. |
| `arith.less_equal` | As `tensor.less_equal`, on two scalars. |
| `arith.equal` | As `tensor.equal`, on two scalars. |
| `arith.greater_equal` | As `tensor.greater_equal`, on two scalars. |
| `arith.greater` | As `tensor.greater`, on two scalars. |
| `arith.not_equal` | As `tensor.not_equal`, on two scalars. |

Example, from `filetests/tile/roundtrip/arith.ptir`:

```text
func.func @comparisons(%0: tensor<4xf32>, %1: tensor<4xf32>) -> tensor<4xi1> {
  %2 = tile.block(%0, %1) {iters = [par 4], reads = [$0[d0], $1[d0]], folds = [none -> [d0]]} ({
  ^body(%0: f32, %1: f32):
    %2 = arith.less(%0, %1) : (f32, f32) -> i1
    %3 = arith.less_equal(%0, %1) : (f32, f32) -> i1
    %4 = arith.equal(%0, %1) : (f32, f32) -> i1
    %5 = arith.greater_equal(%0, %1) : (f32, f32) -> i1
    %6 = arith.greater(%0, %1) : (f32, f32) -> i1
    %7 = arith.not_equal(%0, %1) : (f32, f32) -> i1
    %8 = arith.and(%2, %3) : (i1, i1) -> i1
    %9 = arith.or(%4, %5) : (i1, i1) -> i1
    %10 = arith.xor(%6, %7) : (i1, i1) -> i1
    %11 = arith.and(%8, %9) : (i1, i1) -> i1
    %12 = arith.or(%11, %10) : (i1, i1) -> i1
    tile.yield(%12) : (i1) -> ()
  }) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>
  func.return(%2) : (tensor<4xi1>) -> ()
}
```

### `arith.select`

`on_true` when `pred` is set, else `on_false`.

Defined by `Select` in [crates/gpt-rs-ir/src/dialects/arith/ir.rs](../../crates/gpt-rs-ir/src/dialects/arith/ir.rs).

| | |
|---|---|
| Operands | `pred`: scalar; `on_true`: scalar; `on_false`: scalar |
| Results | `output`: scalar |
| Effects | none |
| Folds | `select(pred, on_true, on_false)` is `on_true` when `pred` is all true or both branches are the same value, and `on_false` when `pred` is all false. |
| Patterns | `SelectToTile` (`tensor-to-tile`) |

Example, from `filetests/tile/roundtrip/arith.ptir`:

```text
func.func @scalar_ops(%0: tensor<4xf32>, %1: tensor<4xsi32>, %2: tensor<2xui64>) -> (tensor<4xf32>, tensor<4xsi32>, tensor<4xi1>, tensor<4xbf16>) {
  %3, %4, %5, %6 = tile.block(%0, %1, %2) {iters = [par 4], reads = [$0[d0], $1[d0], $2[0], $2[1]], folds = [none -> [d0], none -> [d0], none -> [d0], none -> [d0]]} ({
  ^body(%0: f32, %1: si32, %2: ui64, %3: ui64):
    %4 = arith.neg(%0) : (f32) -> f32
    %5 = arith.abs(%4) : (f32) -> f32
    %6 = arith.exp(%5) : (f32) -> f32
    %7 = arith.log(%6) : (f32) -> f32
    %8 = arith.tanh(%7) : (f32) -> f32
    %9 = arith.erf(%8) : (f32) -> f32
    %10 = arith.rsqrt(%9) : (f32) -> f32
    %11 = arith.reciprocal(%10) : (f32) -> f32
    %12 = arith.sin(%11) : (f32) -> f32
    %13 = arith.cos(%12) : (f32) -> f32
    %14 = arith.constant() {value = 0.5 : f32} : () -> f32
    %15 = arith.add(%13, %14) : (f32, f32) -> f32
    %16 = arith.sub(%15, %14) : (f32, f32) -> f32
    %17 = arith.mul(%16, %14) : (f32, f32) -> f32
    %18 = arith.div(%17, %14) : (f32, f32) -> f32
    %19 = arith.maximum(%18, %14) : (f32, f32) -> f32
    %20 = arith.minimum(%19, %14) : (f32, f32) -> f32
    %21 = arith.constant() {value = 3 : si32} : () -> si32
    %22 = arith.and(%1, %21) : (si32, si32) -> si32
    %23 = arith.or(%22, %21) : (si32, si32) -> si32
    %24 = arith.xor(%23, %21) : (si32, si32) -> si32
    %25 = arith.shift_left(%24, %21) : (si32, si32) -> si32
    %26 = arith.shift_right_logical(%25, %21) : (si32, si32) -> si32
    %27 = arith.shift_right_arithmetic(%26, %21) : (si32, si32) -> si32
    %28 = arith.not(%27) : (si32) -> si32
    %29 = arith.less(%20, %14) : (f32, f32) -> i1
    %30 = arith.select(%29, %20, %14) : (i1, f32, f32) -> f32
    %31 = arith.cast(%30) {to = bf16} : (f32) -> bf16
    %32 = arith.bitcast(%30) {to = si32} : (f32) -> si32
    %33 = arith.add(%28, %32) : (si32, si32) -> si32
    %34 = tile.index() {expr = d0, dtype = ui64} : () -> ui64
    %35 = arith.rng_uniform(%2, %3, %34) {dtype = f32} : (ui64, ui64, ui64) -> f32
    %36 = arith.rng_normal(%2, %3, %34) {dtype = bf16} : (ui64, ui64, ui64) -> bf16
    tile.yield(%35, %33, %29, %36) : (f32, si32, i1, bf16) -> ()
  }) : (tensor<4xf32>, tensor<4xsi32>, tensor<2xui64>) -> (tensor<4xf32>, tensor<4xsi32>, tensor<4xi1>, tensor<4xbf16>)
  func.return(%3, %4, %5, %6) : (tensor<4xf32>, tensor<4xsi32>, tensor<4xi1>, tensor<4xbf16>) -> ()
}
```

### `arith.cast`

The scalar converted to `to`, as `tensor.cast` converts each element.

Defined by `Cast` in [crates/gpt-rs-ir/src/dialects/arith/ir.rs](../../crates/gpt-rs-ir/src/dialects/arith/ir.rs).

| | |
|---|---|
| Operands | `input`: scalar |
| Results | `output`: scalar |
| Attributes | `to`: `dtype` |
| Effects | none |
| Folds | A cast to the input's own type is the input. |
| Patterns | `CastToTile` (`tensor-to-tile`) |

Example, from `filetests/tile/roundtrip/block.ptir`:

```text
func.func @linear_bias_gelu(%0: tensor<{M}x8xf32>, %1: tensor<6x8xbf16>, %2: tensor<6xf32>) -> tensor<{M}x6xf32> {
  %3 = tile.block(%0, %1) {iters = [par {M}, par 6, red 8], reads = [$0[d0, d2], $1[d1, d2]], folds = [sum -> [d0, d1]]} ({
  ^body(%0: f32, %1: bf16):
    %2 = arith.cast(%1) {to = f32} : (bf16) -> f32
    %3 = arith.mul(%0, %2) : (f32, f32) -> f32
    tile.yield(%3) : (f32) -> ()
  }) : (tensor<{M}x8xf32>, tensor<6x8xbf16>) -> tensor<{M}x6xf32>
  %4 = tile.block(%3, %2) {iters = [par {M}, par 6], reads = [$0[d0, d1], $1[d1]], folds = [none -> [d0, d1]]} ({
  ^body(%0: f32, %1: f32):
    %2 = arith.add(%0, %1) : (f32, f32) -> f32
    %3 = arith.constant() {value = 0.70710677 : f32} : () -> f32
    %4 = arith.mul(%2, %3) : (f32, f32) -> f32
    %5 = arith.erf(%4) : (f32) -> f32
    %6 = arith.constant() {value = 1.0 : f32} : () -> f32
    %7 = arith.add(%5, %6) : (f32, f32) -> f32
    %8 = arith.constant() {value = 0.5 : f32} : () -> f32
    %9 = arith.mul(%2, %8) : (f32, f32) -> f32
    %10 = arith.mul(%9, %7) : (f32, f32) -> f32
    tile.yield(%10) : (f32) -> ()
  }) : (tensor<{M}x6xf32>, tensor<6xf32>) -> tensor<{M}x6xf32>
  func.return(%4) : (tensor<{M}x6xf32>) -> ()
}
```

### `arith.bitcast`

The bits of the scalar read as a value of `to`, of the same width. Unpacking a
sub-byte code from a word is shifts and masks on integers, then a bitcast.

Defined by `Bitcast` in [crates/gpt-rs-ir/src/dialects/arith/ir.rs](../../crates/gpt-rs-ir/src/dialects/arith/ir.rs).

| | |
|---|---|
| Operands | `input`: scalar |
| Results | `output`: scalar |
| Attributes | `to`: `dtype` |
| Effects | none |
| Folds | A bitcast to the input's own type is the input. |
| Patterns | `BitcastToTile` (`tensor-to-tile`) |

Example, from `filetests/tile/roundtrip/arith.ptir`:

```text
func.func @scalar_ops(%0: tensor<4xf32>, %1: tensor<4xsi32>, %2: tensor<2xui64>) -> (tensor<4xf32>, tensor<4xsi32>, tensor<4xi1>, tensor<4xbf16>) {
  %3, %4, %5, %6 = tile.block(%0, %1, %2) {iters = [par 4], reads = [$0[d0], $1[d0], $2[0], $2[1]], folds = [none -> [d0], none -> [d0], none -> [d0], none -> [d0]]} ({
  ^body(%0: f32, %1: si32, %2: ui64, %3: ui64):
    %4 = arith.neg(%0) : (f32) -> f32
    %5 = arith.abs(%4) : (f32) -> f32
    %6 = arith.exp(%5) : (f32) -> f32
    %7 = arith.log(%6) : (f32) -> f32
    %8 = arith.tanh(%7) : (f32) -> f32
    %9 = arith.erf(%8) : (f32) -> f32
    %10 = arith.rsqrt(%9) : (f32) -> f32
    %11 = arith.reciprocal(%10) : (f32) -> f32
    %12 = arith.sin(%11) : (f32) -> f32
    %13 = arith.cos(%12) : (f32) -> f32
    %14 = arith.constant() {value = 0.5 : f32} : () -> f32
    %15 = arith.add(%13, %14) : (f32, f32) -> f32
    %16 = arith.sub(%15, %14) : (f32, f32) -> f32
    %17 = arith.mul(%16, %14) : (f32, f32) -> f32
    %18 = arith.div(%17, %14) : (f32, f32) -> f32
    %19 = arith.maximum(%18, %14) : (f32, f32) -> f32
    %20 = arith.minimum(%19, %14) : (f32, f32) -> f32
    %21 = arith.constant() {value = 3 : si32} : () -> si32
    %22 = arith.and(%1, %21) : (si32, si32) -> si32
    %23 = arith.or(%22, %21) : (si32, si32) -> si32
    %24 = arith.xor(%23, %21) : (si32, si32) -> si32
    %25 = arith.shift_left(%24, %21) : (si32, si32) -> si32
    %26 = arith.shift_right_logical(%25, %21) : (si32, si32) -> si32
    %27 = arith.shift_right_arithmetic(%26, %21) : (si32, si32) -> si32
    %28 = arith.not(%27) : (si32) -> si32
    %29 = arith.less(%20, %14) : (f32, f32) -> i1
    %30 = arith.select(%29, %20, %14) : (i1, f32, f32) -> f32
    %31 = arith.cast(%30) {to = bf16} : (f32) -> bf16
    %32 = arith.bitcast(%30) {to = si32} : (f32) -> si32
    %33 = arith.add(%28, %32) : (si32, si32) -> si32
    %34 = tile.index() {expr = d0, dtype = ui64} : () -> ui64
    %35 = arith.rng_uniform(%2, %3, %34) {dtype = f32} : (ui64, ui64, ui64) -> f32
    %36 = arith.rng_normal(%2, %3, %34) {dtype = bf16} : (ui64, ui64, ui64) -> bf16
    tile.yield(%35, %33, %29, %36) : (f32, si32, i1, bf16) -> ()
  }) : (tensor<4xf32>, tensor<4xsi32>, tensor<2xui64>) -> (tensor<4xf32>, tensor<4xsi32>, tensor<4xi1>, tensor<4xbf16>)
  func.return(%3, %4, %5, %6) : (tensor<4xf32>, tensor<4xsi32>, tensor<4xi1>, tensor<4xbf16>) -> ()
}
```

### `Rng`: `arith.rng_uniform`, `arith.rng_normal`

Element `index` of the stream that the `tensor` op of
the same name draws in `dtype` from the state `[key, counter]`, as that op computes it. `key`
and `counter` are `ui64`, and `index` an integer. [`Draw`] is its function.

Defined by `Rng` in [crates/gpt-rs-ir/src/dialects/arith/ir.rs](../../crates/gpt-rs-ir/src/dialects/arith/ir.rs).

| | |
|---|---|
| Operands | `key`: scalar; `counter`: scalar; `index`: scalar |
| Results | `output`: scalar |
| Attributes | `dtype`: `dtype` |
| Effects | none |
| Patterns | `RngToTile` (`tensor-to-tile`) |

| Op | |
|---|---|
| `arith.rng_uniform` | As `tensor.rng_uniform`, on one element. |
| `arith.rng_normal` | As `tensor.rng_normal`, on one element. |

Example, from `filetests/tile/roundtrip/arith.ptir`:

```text
func.func @scalar_ops(%0: tensor<4xf32>, %1: tensor<4xsi32>, %2: tensor<2xui64>) -> (tensor<4xf32>, tensor<4xsi32>, tensor<4xi1>, tensor<4xbf16>) {
  %3, %4, %5, %6 = tile.block(%0, %1, %2) {iters = [par 4], reads = [$0[d0], $1[d0], $2[0], $2[1]], folds = [none -> [d0], none -> [d0], none -> [d0], none -> [d0]]} ({
  ^body(%0: f32, %1: si32, %2: ui64, %3: ui64):
    %4 = arith.neg(%0) : (f32) -> f32
    %5 = arith.abs(%4) : (f32) -> f32
    %6 = arith.exp(%5) : (f32) -> f32
    %7 = arith.log(%6) : (f32) -> f32
    %8 = arith.tanh(%7) : (f32) -> f32
    %9 = arith.erf(%8) : (f32) -> f32
    %10 = arith.rsqrt(%9) : (f32) -> f32
    %11 = arith.reciprocal(%10) : (f32) -> f32
    %12 = arith.sin(%11) : (f32) -> f32
    %13 = arith.cos(%12) : (f32) -> f32
    %14 = arith.constant() {value = 0.5 : f32} : () -> f32
    %15 = arith.add(%13, %14) : (f32, f32) -> f32
    %16 = arith.sub(%15, %14) : (f32, f32) -> f32
    %17 = arith.mul(%16, %14) : (f32, f32) -> f32
    %18 = arith.div(%17, %14) : (f32, f32) -> f32
    %19 = arith.maximum(%18, %14) : (f32, f32) -> f32
    %20 = arith.minimum(%19, %14) : (f32, f32) -> f32
    %21 = arith.constant() {value = 3 : si32} : () -> si32
    %22 = arith.and(%1, %21) : (si32, si32) -> si32
    %23 = arith.or(%22, %21) : (si32, si32) -> si32
    %24 = arith.xor(%23, %21) : (si32, si32) -> si32
    %25 = arith.shift_left(%24, %21) : (si32, si32) -> si32
    %26 = arith.shift_right_logical(%25, %21) : (si32, si32) -> si32
    %27 = arith.shift_right_arithmetic(%26, %21) : (si32, si32) -> si32
    %28 = arith.not(%27) : (si32) -> si32
    %29 = arith.less(%20, %14) : (f32, f32) -> i1
    %30 = arith.select(%29, %20, %14) : (i1, f32, f32) -> f32
    %31 = arith.cast(%30) {to = bf16} : (f32) -> bf16
    %32 = arith.bitcast(%30) {to = si32} : (f32) -> si32
    %33 = arith.add(%28, %32) : (si32, si32) -> si32
    %34 = tile.index() {expr = d0, dtype = ui64} : () -> ui64
    %35 = arith.rng_uniform(%2, %3, %34) {dtype = f32} : (ui64, ui64, ui64) -> f32
    %36 = arith.rng_normal(%2, %3, %34) {dtype = bf16} : (ui64, ui64, ui64) -> bf16
    tile.yield(%35, %33, %29, %36) : (f32, si32, i1, bf16) -> ()
  }) : (tensor<4xf32>, tensor<4xsi32>, tensor<2xui64>) -> (tensor<4xf32>, tensor<4xsi32>, tensor<4xi1>, tensor<4xbf16>)
  func.return(%3, %4, %5, %6) : (tensor<4xf32>, tensor<4xsi32>, tensor<4xi1>, tensor<4xbf16>) -> ()
}
```

Rejected, as the verify filetests pin it:

- `arith.rng_normal: dtype must be a float dtype, got si32`

## Traits

A trait states a fact about the ops that have it. The folds and patterns it implies apply to every one of them; [passes.md](passes.md) says which pass runs each.

### `Absorbing`

`f(a, z) = z`, for the integers and `i1`.

Defined by `Absorbing` in [crates/gpt-rs-ir/src/traits/algebra.rs](../../crates/gpt-rs-ir/src/traits/algebra.rs).

Ops: `arith.mul`, `arith.maximum`, `arith.minimum`, `arith.and`, `arith.or`.

### `Associative`

`f(f(a, b), c) = f(a, f(b, c))`, for integers, and for floats when `float` is set: float
addition and multiplication round differently once regrouped.

Defined by `Associative` in [crates/gpt-rs-ir/src/traits/algebra.rs](../../crates/gpt-rs-ir/src/traits/algebra.rs).

Ops: `arith.add`, `arith.mul`, `arith.maximum`, `arith.minimum`, `arith.and`, `arith.or`, `arith.xor`.

Patterns rooted at it: `GroupStatic` (`canonicalize`).

### `Commutative`

`f(a, b) = f(b, a)`. Folding puts a run-time operand first and a constant last (see
`Module::known_before_run`), so the other folds, the patterns and CSE see a constant on the right.

Defined by `Commutative` in [crates/gpt-rs-ir/src/traits/algebra.rs](../../crates/gpt-rs-ir/src/traits/algebra.rs).

Ops: `arith.add`, `arith.mul`, `arith.maximum`, `arith.minimum`, `arith.and`, `arith.or`, `arith.xor`.

### `DistributesOver`

`f(a, g(b, c)) = g(f(a, b), f(a, c))` for the op `g` that `over` builds, for integers, and for
floats when `float` is set.

Defined by `DistributesOver` in [crates/gpt-rs-ir/src/traits/algebra.rs](../../crates/gpt-rs-ir/src/traits/algebra.rs).

Ops: `arith.mul`, `arith.and`, `arith.or`.

Patterns rooted at it: `Factor` (`canonicalize`).

### `Idempotent`

For a unary op `f(f(a)) = f(a)`, for a binary one `f(a, a) = a`.

Defined by `Idempotent` in [crates/gpt-rs-ir/src/traits/algebra.rs](../../crates/gpt-rs-ir/src/traits/algebra.rs).

Ops: `arith.abs`, `arith.maximum`, `arith.minimum`, `arith.and`, `arith.or`.

### `Involution`

`f(f(a)) = a`.

Defined by `Involution` in [crates/gpt-rs-ir/src/traits/algebra.rs](../../crates/gpt-rs-ir/src/traits/algebra.rs).

Ops: `arith.neg`, `arith.not`.

### `RightIdentity`

`f(a, e) = a`. A commutative op has `e` on the left too, since folding moves constants to the
right.

Defined by `RightIdentity` in [crates/gpt-rs-ir/src/traits/algebra.rs](../../crates/gpt-rs-ir/src/traits/algebra.rs).

Ops: `arith.add`, `arith.sub`, `arith.mul`, `arith.div`, `arith.maximum`, `arith.minimum`, `arith.and`, `arith.or`, `arith.xor`, `arith.shift_left`, `arith.shift_right_logical`, `arith.shift_right_arithmetic`.
