<!-- Generated from the code by `cargo run -p gpt-rs-docs`. Do not edit: change the code and run it again. -->

# The arith dialect

The `arith` dialect computes on scalars: the bodies of `tile.block` are written in it. Each op is
the per-element function of the `tensor` op of the same name, applied to one value, so the two
dialects agree bit for bit.

## Conventions

- Each op is defined in `ir.rs`, and its type rule is in `verify.rs`. The ops of a family
  (`Unary`, `Binary`, `Compare`, `Rng`) share the family's type rule and `Eval`.
- Each op's function is the one copy of its per-element semantics, which the `tensor` op of the
  same name applies to every element. `interp/math.rs` defines `UnaryMath`, `BinaryMath`,
  `CompareMath` and `Draw`. `interp/mod.rs` implements `UnaryMath`, `BinaryMath` and `CompareMath`
  for the arith ops, and `tensor/interp/random.rs` implements `Draw`.
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
- `arith.from_index`
- `arith.to_index`
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

| `value` | Text form |
|---|---|
| `LiteralText` | `` $value `:` $dtype `` |

Example, from `filetests/tile/roundtrip/arith.ptir`:

```text
func.func @literals() -> (tensor<si64>, tensor<i1>, tensor<bf16>, tensor<f64>) {
  %min, %true, %neg_inf, %tenth = tile.block() {iters = [], reads = [], folds = [none -> [], none -> [], none -> [], none -> []]} ({
    %min_scalar = arith.constant() {value = -9223372036854775808 : si64} : () -> si64
    %true_scalar = arith.constant() {value = 1 : i1} : () -> i1
    %neg_inf_scalar = arith.constant() {value = -inf : bf16} : () -> bf16
    %tenth_scalar = arith.constant() {value = 0.1 : f64} : () -> f64
    tile.yield(%min_scalar, %true_scalar, %neg_inf_scalar, %tenth_scalar) : (si64, i1, bf16, f64) -> ()
  }) : () -> (tensor<si64>, tensor<i1>, tensor<bf16>, tensor<f64>)
  func.return(%min, %true, %neg_inf, %tenth) : (tensor<si64>, tensor<i1>, tensor<bf16>, tensor<f64>) -> ()
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
| Patterns | `FoldIndexArith` (`canonicalize`, at `arith.neg`) |

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

Example, from `filetests/tile/roundtrip/arith.ptir`:

```text
func.func @scalar_ops(%input: tensor<4xf32>, %integers: tensor<4xsi32>, %state: tensor<2xui64>) -> (tensor<4xf32>, tensor<4xsi32>, tensor<4xi1>, tensor<4xbf16>) {
  %uniforms, %combined_bits, %less_flags, %normals = tile.block(%input, %integers, %state) {iters = [par 4], reads = [$0[d0], $1[d0], $2[0], $2[1]], folds = [none -> [d0], none -> [d0], none -> [d0], none -> [d0]]} ({
  ^body(%element: f32, %integer: si32, %key: ui64, %counter: ui64):
    %negated = arith.neg(%element) : (f32) -> f32
    %absolute = arith.abs(%negated) : (f32) -> f32
    %exponential = arith.exp(%absolute) : (f32) -> f32
    %logarithm = arith.log(%exponential) : (f32) -> f32
    %tanh = arith.tanh(%logarithm) : (f32) -> f32
    %erf = arith.erf(%tanh) : (f32) -> f32
    %rsqrt = arith.rsqrt(%erf) : (f32) -> f32
    %reciprocal = arith.reciprocal(%rsqrt) : (f32) -> f32
    %sine = arith.sin(%reciprocal) : (f32) -> f32
    %cosine = arith.cos(%sine) : (f32) -> f32
    %half = arith.constant() {value = 0.5 : f32} : () -> f32
    %sum = arith.add(%cosine, %half) : (f32, f32) -> f32
    %difference = arith.sub(%sum, %half) : (f32, f32) -> f32
    %product = arith.mul(%difference, %half) : (f32, f32) -> f32
    %quotient = arith.div(%product, %half) : (f32, f32) -> f32
    %maximum = arith.maximum(%quotient, %half) : (f32, f32) -> f32
    %minimum = arith.minimum(%maximum, %half) : (f32, f32) -> f32
    %three = arith.constant() {value = 3 : si32} : () -> si32
    %and_bits = arith.and(%integer, %three) : (si32, si32) -> si32
    %or_bits = arith.or(%and_bits, %three) : (si32, si32) -> si32
    %xor_bits = arith.xor(%or_bits, %three) : (si32, si32) -> si32
    %left_shifted = arith.shift_left(%xor_bits, %three) : (si32, si32) -> si32
    %logical_shifted = arith.shift_right_logical(%left_shifted, %three) : (si32, si32) -> si32
    %arithmetic_shifted = arith.shift_right_arithmetic(%logical_shifted, %three) : (si32, si32) -> si32
    %inverted = arith.not(%arithmetic_shifted) : (si32) -> si32
    %is_less = arith.less(%minimum, %half) : (f32, f32) -> i1
    %selected = arith.select(%is_less, %minimum, %half) : (i1, f32, f32) -> f32
    %narrowed = arith.cast(%selected) {to = bf16} : (f32) -> bf16
    %selected_bits = arith.bitcast(%selected) {to = si32} : (f32) -> si32
    %combined = arith.add(%inverted, %selected_bits) : (si32, si32) -> si32
    %index = tile.index() {expr = d0, dtype = ui64} : () -> ui64
    %uniform = arith.rng_uniform(%key, %counter, %index) {dtype = f32} : (ui64, ui64, ui64) -> f32
    %normal = arith.rng_normal(%key, %counter, %index) {dtype = bf16} : (ui64, ui64, ui64) -> bf16
    tile.yield(%uniform, %combined, %is_less, %normal) : (f32, si32, i1, bf16) -> ()
  }) : (tensor<4xf32>, tensor<4xsi32>, tensor<2xui64>) -> (tensor<4xf32>, tensor<4xsi32>, tensor<4xi1>, tensor<4xbf16>)
  func.return(%uniforms, %combined_bits, %less_flags, %normals) : (tensor<4xf32>, tensor<4xsi32>, tensor<4xi1>, tensor<4xbf16>) -> ()
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
| Patterns | `FoldIndexArith` (`canonicalize`, at `arith.add`, `arith.sub`, `arith.mul`, `arith.maximum`, `arith.minimum`) |

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
func.func @scalar_ops(%input: tensor<4xf32>, %integers: tensor<4xsi32>, %state: tensor<2xui64>) -> (tensor<4xf32>, tensor<4xsi32>, tensor<4xi1>, tensor<4xbf16>) {
  %uniforms, %combined_bits, %less_flags, %normals = tile.block(%input, %integers, %state) {iters = [par 4], reads = [$0[d0], $1[d0], $2[0], $2[1]], folds = [none -> [d0], none -> [d0], none -> [d0], none -> [d0]]} ({
  ^body(%element: f32, %integer: si32, %key: ui64, %counter: ui64):
    %negated = arith.neg(%element) : (f32) -> f32
    %absolute = arith.abs(%negated) : (f32) -> f32
    %exponential = arith.exp(%absolute) : (f32) -> f32
    %logarithm = arith.log(%exponential) : (f32) -> f32
    %tanh = arith.tanh(%logarithm) : (f32) -> f32
    %erf = arith.erf(%tanh) : (f32) -> f32
    %rsqrt = arith.rsqrt(%erf) : (f32) -> f32
    %reciprocal = arith.reciprocal(%rsqrt) : (f32) -> f32
    %sine = arith.sin(%reciprocal) : (f32) -> f32
    %cosine = arith.cos(%sine) : (f32) -> f32
    %half = arith.constant() {value = 0.5 : f32} : () -> f32
    %sum = arith.add(%cosine, %half) : (f32, f32) -> f32
    %difference = arith.sub(%sum, %half) : (f32, f32) -> f32
    %product = arith.mul(%difference, %half) : (f32, f32) -> f32
    %quotient = arith.div(%product, %half) : (f32, f32) -> f32
    %maximum = arith.maximum(%quotient, %half) : (f32, f32) -> f32
    %minimum = arith.minimum(%maximum, %half) : (f32, f32) -> f32
    %three = arith.constant() {value = 3 : si32} : () -> si32
    %and_bits = arith.and(%integer, %three) : (si32, si32) -> si32
    %or_bits = arith.or(%and_bits, %three) : (si32, si32) -> si32
    %xor_bits = arith.xor(%or_bits, %three) : (si32, si32) -> si32
    %left_shifted = arith.shift_left(%xor_bits, %three) : (si32, si32) -> si32
    %logical_shifted = arith.shift_right_logical(%left_shifted, %three) : (si32, si32) -> si32
    %arithmetic_shifted = arith.shift_right_arithmetic(%logical_shifted, %three) : (si32, si32) -> si32
    %inverted = arith.not(%arithmetic_shifted) : (si32) -> si32
    %is_less = arith.less(%minimum, %half) : (f32, f32) -> i1
    %selected = arith.select(%is_less, %minimum, %half) : (i1, f32, f32) -> f32
    %narrowed = arith.cast(%selected) {to = bf16} : (f32) -> bf16
    %selected_bits = arith.bitcast(%selected) {to = si32} : (f32) -> si32
    %combined = arith.add(%inverted, %selected_bits) : (si32, si32) -> si32
    %index = tile.index() {expr = d0, dtype = ui64} : () -> ui64
    %uniform = arith.rng_uniform(%key, %counter, %index) {dtype = f32} : (ui64, ui64, ui64) -> f32
    %normal = arith.rng_normal(%key, %counter, %index) {dtype = bf16} : (ui64, ui64, ui64) -> bf16
    tile.yield(%uniform, %combined, %is_less, %normal) : (f32, si32, i1, bf16) -> ()
  }) : (tensor<4xf32>, tensor<4xsi32>, tensor<2xui64>) -> (tensor<4xf32>, tensor<4xsi32>, tensor<4xi1>, tensor<4xbf16>)
  func.return(%uniforms, %combined_bits, %less_flags, %normals) : (tensor<4xf32>, tensor<4xsi32>, tensor<4xi1>, tensor<4xbf16>) -> ()
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
func.func @comparisons(%lhs: tensor<4xf32>, %rhs: tensor<4xf32>) -> tensor<4xi1> {
  %flags = tile.block(%lhs, %rhs) {iters = [par 4], reads = [$0[d0], $1[d0]], folds = [none -> [d0]]} ({
  ^body(%left: f32, %right: f32):
    %less = arith.less(%left, %right) : (f32, f32) -> i1
    %less_equal = arith.less_equal(%left, %right) : (f32, f32) -> i1
    %equal = arith.equal(%left, %right) : (f32, f32) -> i1
    %greater_equal = arith.greater_equal(%left, %right) : (f32, f32) -> i1
    %greater = arith.greater(%left, %right) : (f32, f32) -> i1
    %not_equal = arith.not_equal(%left, %right) : (f32, f32) -> i1
    %both_less = arith.and(%less, %less_equal) : (i1, i1) -> i1
    %either_equal = arith.or(%equal, %greater_equal) : (i1, i1) -> i1
    %greater_xor_unequal = arith.xor(%greater, %not_equal) : (i1, i1) -> i1
    %conjunction = arith.and(%both_less, %either_equal) : (i1, i1) -> i1
    %joined = arith.or(%conjunction, %greater_xor_unequal) : (i1, i1) -> i1
    tile.yield(%joined) : (i1) -> ()
  }) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>
  func.return(%flags) : (tensor<4xi1>) -> ()
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

Example, from `filetests/tile/roundtrip/arith.ptir`:

```text
func.func @scalar_ops(%input: tensor<4xf32>, %integers: tensor<4xsi32>, %state: tensor<2xui64>) -> (tensor<4xf32>, tensor<4xsi32>, tensor<4xi1>, tensor<4xbf16>) {
  %uniforms, %combined_bits, %less_flags, %normals = tile.block(%input, %integers, %state) {iters = [par 4], reads = [$0[d0], $1[d0], $2[0], $2[1]], folds = [none -> [d0], none -> [d0], none -> [d0], none -> [d0]]} ({
  ^body(%element: f32, %integer: si32, %key: ui64, %counter: ui64):
    %negated = arith.neg(%element) : (f32) -> f32
    %absolute = arith.abs(%negated) : (f32) -> f32
    %exponential = arith.exp(%absolute) : (f32) -> f32
    %logarithm = arith.log(%exponential) : (f32) -> f32
    %tanh = arith.tanh(%logarithm) : (f32) -> f32
    %erf = arith.erf(%tanh) : (f32) -> f32
    %rsqrt = arith.rsqrt(%erf) : (f32) -> f32
    %reciprocal = arith.reciprocal(%rsqrt) : (f32) -> f32
    %sine = arith.sin(%reciprocal) : (f32) -> f32
    %cosine = arith.cos(%sine) : (f32) -> f32
    %half = arith.constant() {value = 0.5 : f32} : () -> f32
    %sum = arith.add(%cosine, %half) : (f32, f32) -> f32
    %difference = arith.sub(%sum, %half) : (f32, f32) -> f32
    %product = arith.mul(%difference, %half) : (f32, f32) -> f32
    %quotient = arith.div(%product, %half) : (f32, f32) -> f32
    %maximum = arith.maximum(%quotient, %half) : (f32, f32) -> f32
    %minimum = arith.minimum(%maximum, %half) : (f32, f32) -> f32
    %three = arith.constant() {value = 3 : si32} : () -> si32
    %and_bits = arith.and(%integer, %three) : (si32, si32) -> si32
    %or_bits = arith.or(%and_bits, %three) : (si32, si32) -> si32
    %xor_bits = arith.xor(%or_bits, %three) : (si32, si32) -> si32
    %left_shifted = arith.shift_left(%xor_bits, %three) : (si32, si32) -> si32
    %logical_shifted = arith.shift_right_logical(%left_shifted, %three) : (si32, si32) -> si32
    %arithmetic_shifted = arith.shift_right_arithmetic(%logical_shifted, %three) : (si32, si32) -> si32
    %inverted = arith.not(%arithmetic_shifted) : (si32) -> si32
    %is_less = arith.less(%minimum, %half) : (f32, f32) -> i1
    %selected = arith.select(%is_less, %minimum, %half) : (i1, f32, f32) -> f32
    %narrowed = arith.cast(%selected) {to = bf16} : (f32) -> bf16
    %selected_bits = arith.bitcast(%selected) {to = si32} : (f32) -> si32
    %combined = arith.add(%inverted, %selected_bits) : (si32, si32) -> si32
    %index = tile.index() {expr = d0, dtype = ui64} : () -> ui64
    %uniform = arith.rng_uniform(%key, %counter, %index) {dtype = f32} : (ui64, ui64, ui64) -> f32
    %normal = arith.rng_normal(%key, %counter, %index) {dtype = bf16} : (ui64, ui64, ui64) -> bf16
    tile.yield(%uniform, %combined, %is_less, %normal) : (f32, si32, i1, bf16) -> ()
  }) : (tensor<4xf32>, tensor<4xsi32>, tensor<2xui64>) -> (tensor<4xf32>, tensor<4xsi32>, tensor<4xi1>, tensor<4xbf16>)
  func.return(%uniforms, %combined_bits, %less_flags, %normals) : (tensor<4xf32>, tensor<4xsi32>, tensor<4xi1>, tensor<4xbf16>) -> ()
}
```

### `arith.cast`

The scalar converted to `to`, as `tensor.cast` converts each element.

Defined by `Cast` in [crates/gpt-rs-ir/src/dialects/arith/ir.rs](../../crates/gpt-rs-ir/src/dialects/arith/ir.rs).

| | |
|---|---|
| Operands | `input`: scalar |
| Results | `output`: scalar |
| Attributes | `to`: `DType` |
| Effects | none |

Example, from `filetests/tile/roundtrip/block.ptir`:

```text
func.func @linear_bias_gelu(%input: tensor<{M}x8xf32>, %weight: tensor<6x8xbf16>, %bias: tensor<6xf32>) -> tensor<{M}x6xf32> {
  %projection = tile.block(%input, %weight) {iters = [par {M}, par 6, red 8], reads = [$0[d0, d2], $1[d1, d2]], folds = [sum -> [d0, d1]]} ({
  ^body(%activation: f32, %weight_element: bf16):
    %widened_weight = arith.cast(%weight_element) {to = f32} : (bf16) -> f32
    %product = arith.mul(%activation, %widened_weight) : (f32, f32) -> f32
    tile.yield(%product) : (f32) -> ()
  }) : (tensor<{M}x8xf32>, tensor<6x8xbf16>) -> tensor<{M}x6xf32>
  %activated = tile.block(%projection, %bias) {iters = [par {M}, par 6], reads = [$0[d0, d1], $1[d1]], folds = [none -> [d0, d1]]} ({
  ^body(%dot: f32, %bias_element: f32):
    %biased = arith.add(%dot, %bias_element) : (f32, f32) -> f32
    %inv_sqrt_two = arith.constant() {value = 0.70710677 : f32} : () -> f32
    %scaled = arith.mul(%biased, %inv_sqrt_two) : (f32, f32) -> f32
    %erf = arith.erf(%scaled) : (f32) -> f32
    %one = arith.constant() {value = 1.0 : f32} : () -> f32
    %erf_plus_one = arith.add(%erf, %one) : (f32, f32) -> f32
    %half = arith.constant() {value = 0.5 : f32} : () -> f32
    %half_biased = arith.mul(%biased, %half) : (f32, f32) -> f32
    %gelu = arith.mul(%half_biased, %erf_plus_one) : (f32, f32) -> f32
    tile.yield(%gelu) : (f32) -> ()
  }) : (tensor<{M}x6xf32>, tensor<6xf32>) -> tensor<{M}x6xf32>
  func.return(%activated) : (tensor<{M}x6xf32>) -> ()
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
| Attributes | `to`: `DType` |
| Effects | none |

Example, from `filetests/tile/roundtrip/arith.ptir`:

```text
func.func @scalar_ops(%input: tensor<4xf32>, %integers: tensor<4xsi32>, %state: tensor<2xui64>) -> (tensor<4xf32>, tensor<4xsi32>, tensor<4xi1>, tensor<4xbf16>) {
  %uniforms, %combined_bits, %less_flags, %normals = tile.block(%input, %integers, %state) {iters = [par 4], reads = [$0[d0], $1[d0], $2[0], $2[1]], folds = [none -> [d0], none -> [d0], none -> [d0], none -> [d0]]} ({
  ^body(%element: f32, %integer: si32, %key: ui64, %counter: ui64):
    %negated = arith.neg(%element) : (f32) -> f32
    %absolute = arith.abs(%negated) : (f32) -> f32
    %exponential = arith.exp(%absolute) : (f32) -> f32
    %logarithm = arith.log(%exponential) : (f32) -> f32
    %tanh = arith.tanh(%logarithm) : (f32) -> f32
    %erf = arith.erf(%tanh) : (f32) -> f32
    %rsqrt = arith.rsqrt(%erf) : (f32) -> f32
    %reciprocal = arith.reciprocal(%rsqrt) : (f32) -> f32
    %sine = arith.sin(%reciprocal) : (f32) -> f32
    %cosine = arith.cos(%sine) : (f32) -> f32
    %half = arith.constant() {value = 0.5 : f32} : () -> f32
    %sum = arith.add(%cosine, %half) : (f32, f32) -> f32
    %difference = arith.sub(%sum, %half) : (f32, f32) -> f32
    %product = arith.mul(%difference, %half) : (f32, f32) -> f32
    %quotient = arith.div(%product, %half) : (f32, f32) -> f32
    %maximum = arith.maximum(%quotient, %half) : (f32, f32) -> f32
    %minimum = arith.minimum(%maximum, %half) : (f32, f32) -> f32
    %three = arith.constant() {value = 3 : si32} : () -> si32
    %and_bits = arith.and(%integer, %three) : (si32, si32) -> si32
    %or_bits = arith.or(%and_bits, %three) : (si32, si32) -> si32
    %xor_bits = arith.xor(%or_bits, %three) : (si32, si32) -> si32
    %left_shifted = arith.shift_left(%xor_bits, %three) : (si32, si32) -> si32
    %logical_shifted = arith.shift_right_logical(%left_shifted, %three) : (si32, si32) -> si32
    %arithmetic_shifted = arith.shift_right_arithmetic(%logical_shifted, %three) : (si32, si32) -> si32
    %inverted = arith.not(%arithmetic_shifted) : (si32) -> si32
    %is_less = arith.less(%minimum, %half) : (f32, f32) -> i1
    %selected = arith.select(%is_less, %minimum, %half) : (i1, f32, f32) -> f32
    %narrowed = arith.cast(%selected) {to = bf16} : (f32) -> bf16
    %selected_bits = arith.bitcast(%selected) {to = si32} : (f32) -> si32
    %combined = arith.add(%inverted, %selected_bits) : (si32, si32) -> si32
    %index = tile.index() {expr = d0, dtype = ui64} : () -> ui64
    %uniform = arith.rng_uniform(%key, %counter, %index) {dtype = f32} : (ui64, ui64, ui64) -> f32
    %normal = arith.rng_normal(%key, %counter, %index) {dtype = bf16} : (ui64, ui64, ui64) -> bf16
    tile.yield(%uniform, %combined, %is_less, %normal) : (f32, si32, i1, bf16) -> ()
  }) : (tensor<4xf32>, tensor<4xsi32>, tensor<2xui64>) -> (tensor<4xf32>, tensor<4xsi32>, tensor<4xi1>, tensor<4xbf16>)
  func.return(%uniforms, %combined_bits, %less_flags, %normals) : (tensor<4xf32>, tensor<4xsi32>, tensor<4xi1>, tensor<4xbf16>) -> ()
}
```

### `arith.from_index`

The index as an element of `dtype`, converted as `tensor.iota` converts its
indices: as an si64 cast to `dtype`.

Defined by `FromIndex` in [crates/gpt-rs-ir/src/dialects/arith/ir.rs](../../crates/gpt-rs-ir/src/dialects/arith/ir.rs).

| | |
|---|---|
| Operands | `input`: index |
| Results | `output`: scalar |
| Attributes | `dtype`: `DType` |
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

- `arith.from_index: dtype must be an integer or float dtype, got i1`

### `arith.to_index`

The integer as an index. Its value MUST fit `i64`.

Defined by `ToIndex` in [crates/gpt-rs-ir/src/dialects/arith/ir.rs](../../crates/gpt-rs-ir/src/dialects/arith/ir.rs).

| | |
|---|---|
| Operands | `input`: scalar |
| Results | `output`: index |
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

- `arith.to_index: the input must be an integer dtype, got f32`

### `Rng`: `arith.rng_uniform`, `arith.rng_normal`

Element `index` of the stream that the `tensor` op of
the same name draws in `dtype` from the state `[key, counter]`, as that op computes it. `key`
and `counter` are `ui64`, and `index` an integer. [`Draw`] is its function.

Defined by `Rng` in [crates/gpt-rs-ir/src/dialects/arith/ir.rs](../../crates/gpt-rs-ir/src/dialects/arith/ir.rs).

| | |
|---|---|
| Operands | `key`: scalar; `counter`: scalar; `index`: scalar |
| Results | `output`: scalar |
| Attributes | `dtype`: `DType` |
| Effects | none |

| Op | |
|---|---|
| `arith.rng_uniform` | As `tensor.rng_uniform`, on one element. |
| `arith.rng_normal` | As `tensor.rng_normal`, on one element. |

Example, from `filetests/tile/roundtrip/arith.ptir`:

```text
func.func @scalar_ops(%input: tensor<4xf32>, %integers: tensor<4xsi32>, %state: tensor<2xui64>) -> (tensor<4xf32>, tensor<4xsi32>, tensor<4xi1>, tensor<4xbf16>) {
  %uniforms, %combined_bits, %less_flags, %normals = tile.block(%input, %integers, %state) {iters = [par 4], reads = [$0[d0], $1[d0], $2[0], $2[1]], folds = [none -> [d0], none -> [d0], none -> [d0], none -> [d0]]} ({
  ^body(%element: f32, %integer: si32, %key: ui64, %counter: ui64):
    %negated = arith.neg(%element) : (f32) -> f32
    %absolute = arith.abs(%negated) : (f32) -> f32
    %exponential = arith.exp(%absolute) : (f32) -> f32
    %logarithm = arith.log(%exponential) : (f32) -> f32
    %tanh = arith.tanh(%logarithm) : (f32) -> f32
    %erf = arith.erf(%tanh) : (f32) -> f32
    %rsqrt = arith.rsqrt(%erf) : (f32) -> f32
    %reciprocal = arith.reciprocal(%rsqrt) : (f32) -> f32
    %sine = arith.sin(%reciprocal) : (f32) -> f32
    %cosine = arith.cos(%sine) : (f32) -> f32
    %half = arith.constant() {value = 0.5 : f32} : () -> f32
    %sum = arith.add(%cosine, %half) : (f32, f32) -> f32
    %difference = arith.sub(%sum, %half) : (f32, f32) -> f32
    %product = arith.mul(%difference, %half) : (f32, f32) -> f32
    %quotient = arith.div(%product, %half) : (f32, f32) -> f32
    %maximum = arith.maximum(%quotient, %half) : (f32, f32) -> f32
    %minimum = arith.minimum(%maximum, %half) : (f32, f32) -> f32
    %three = arith.constant() {value = 3 : si32} : () -> si32
    %and_bits = arith.and(%integer, %three) : (si32, si32) -> si32
    %or_bits = arith.or(%and_bits, %three) : (si32, si32) -> si32
    %xor_bits = arith.xor(%or_bits, %three) : (si32, si32) -> si32
    %left_shifted = arith.shift_left(%xor_bits, %three) : (si32, si32) -> si32
    %logical_shifted = arith.shift_right_logical(%left_shifted, %three) : (si32, si32) -> si32
    %arithmetic_shifted = arith.shift_right_arithmetic(%logical_shifted, %three) : (si32, si32) -> si32
    %inverted = arith.not(%arithmetic_shifted) : (si32) -> si32
    %is_less = arith.less(%minimum, %half) : (f32, f32) -> i1
    %selected = arith.select(%is_less, %minimum, %half) : (i1, f32, f32) -> f32
    %narrowed = arith.cast(%selected) {to = bf16} : (f32) -> bf16
    %selected_bits = arith.bitcast(%selected) {to = si32} : (f32) -> si32
    %combined = arith.add(%inverted, %selected_bits) : (si32, si32) -> si32
    %index = tile.index() {expr = d0, dtype = ui64} : () -> ui64
    %uniform = arith.rng_uniform(%key, %counter, %index) {dtype = f32} : (ui64, ui64, ui64) -> f32
    %normal = arith.rng_normal(%key, %counter, %index) {dtype = bf16} : (ui64, ui64, ui64) -> bf16
    tile.yield(%uniform, %combined, %is_less, %normal) : (f32, si32, i1, bf16) -> ()
  }) : (tensor<4xf32>, tensor<4xsi32>, tensor<2xui64>) -> (tensor<4xf32>, tensor<4xsi32>, tensor<4xi1>, tensor<4xbf16>)
  func.return(%uniforms, %combined_bits, %less_flags, %normals) : (tensor<4xf32>, tensor<4xsi32>, tensor<4xi1>, tensor<4xbf16>) -> ()
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
