<!-- Generated from the code by `cargo run -p gpt-rs-docs`. Do not edit: change the code and run it again. -->

# The tensor dialect (ptir.v0.5)

The `tensor` dialect is the Portable Tensor IR (PTIR): whole-tensor ops with value semantics. The
ops of `DeviceTensor` and the functionals record it, and the passes and the backends consume it.
This page states the rules that every op follows, and each op's own documentation defines that op.
Together they are the normative contract between producers and consumers.

## Conventions

- Producers: the ops of `DeviceTensor` (`crates/gpt-rs/src/tensor/ops.rs`) and the functionals
  (`crates/gpt-rs/src/ops/functional/`), which record PTIR into lazy graphs.
- Consumers: the passes (`crates/gpt-rs-ir/src/passes/`, and the rewrites of `tensor` ops in this
  dialect's `passes/`) and the backend crates, such as `crates/gpt-rs-backend-ref-cpu`,
  `crates/gpt-rs-backend-faer` and `crates/gpt-rs-backend-c`.
- Normative keywords: MUST / SHOULD / MAY.
- Each op's attributes and type rule (`OpDef::result_types`) are in `ir/`. Its semantics are its
  `Eval` impl in `interp/`. The reference backend runs them, and every pass is checked against
  them.
- Each op's documentation states what the op means, independent of any specific backend
  implementation strategy. The dialect holds no op outside this specification. If an op's
  documentation and its code disagree, fix the documentation or fix the code.
- Notation: `T[shape]` means a tensor of dtype `T` and the given shape.
- Extension ops (`extract_patches`, the random ops, `top_k`, the segment ops and the quantization
  ops) go beyond the core ops, for broader model coverage. A backend MAY reject an extension op
  other than `extract_patches` as unimplemented.
- LLM attention is expressed as a **pattern** over core ops (no dedicated attention op), as
  "Canonical library forms" shows.
- Backend implementation notes (how each backend executes PTIR) are in `docs/backends/README.md`.

## Versioning

In the text form, a stored program, such as a dump, starts with the line
`spec_version = "ptir.v0.5"` (`Module::to_versioned_text`). The parser reads a module with that
line or without it, and rejects a module that names another version.

## Core invariants

1. Pure functional IR
   - Ops are referentially transparent.
   - No observable in-place mutation.

2. Numerical portability (relaxed)
   - Integer and `i1` ops MUST be bit-exact.
   - Floating point results are not required to be bit-identical across backends/hardware.
   - Backends MUST respect dtype contracts, including explicit `accum` / `out`.

3. No implicit dtype promotion
   - Elementwise arithmetic operands MUST have the same dtype (frontends insert `cast`).

4. Shapes and dynamic dims
   - Shapes are ordered tuples of non-negative integers.
   - Dims MAY be dynamic at compile time, written as a symbol in braces in the text form, such as
     `{T}`.
   - Runtime dims MUST be non-negative and satisfy op constraints.

## Tensor model

### Layout and indexing (row-major)

Unless stated otherwise, tensors are interpreted as dense, contiguous, **row-major** arrays:

- The last dimension is the fastest-varying (C-order).
- `reshape` never changes element order, only the logical view.
- Ops that compute indices (`slice`, `gather`, etc.) use this logical indexing model.

### Values and results

- Each op defines SSA values, one per result. Most ops have one result; an op with several
  results, such as `top_k` or `custom_call`, defines one value per result (`%a, %b = ...`).
- Operands are values. There are no tuple values and no literal operands: a constant is a
  `tensor.constant` op.

### Negative axes

Ops that accept an `axis` (or list of axes) MUST accept negative indices interpreted as
`axis + rank`.

## Broadcasting policy

PTIR uses **explicit** broadcasting:

- `broadcast_to` is the only operation that changes shapes by broadcasting.
- Elementwise ops (`add`, `mul`, `compare`, `select`, ...) require operands to have the **same
  shape** (after any explicit `broadcast_to`).

`broadcast_to` follows a standard right-aligned compatibility rule:

1. Left-pad the input shape with 1s to the rank of the target shape, which MUST NOT be lower.
2. Each dim is compatible if the sizes match or the input's size is 1.
3. Result dim is the target's.

If shapes are not broadcast-compatible, backends MUST reject the op as a spec violation.

## DTypes and casting

### DType set

The dtype universe is defined by `gpt_rs_ir::types::DType` and includes:

- booleans: `i1`
- integers: `si4/ui4/si8/ui8/si16/ui16/si32/ui32/si64/ui64`
- floats: `f4e2m1/f6e2m3/f6e3m2/f8e4m3/f8e5m2/f8e4m3fnuz/f8e5m2fnuz/bf16/f16/f32/f64`
- scale exponents: `e8m0`

The narrow floats include the element formats of the OCP Microscaling (MX) specification:

| Dtype | Max | Infinities | NaN | Encoding |
|---|---|---|---|---|
| `f4e2m1` | 6.0 | no | no | IEEE-style |
| `f6e2m3` | 7.5 | no | no | IEEE-style |
| `f6e3m2` | 28.0 | no | no | IEEE-style |
| `f8e4m3` | 448 | no | yes | E4M3FN |
| `f8e5m2` | 57344 | yes | yes | IEEE-style |
| `f8e4m3fnuz` | 240 | no | `0x80` | FNUZ, bias 8 |
| `f8e5m2fnuz` | 57344 | no | `0x80` | FNUZ, bias 16 |

The FNUZ formats are those of AMD's MI300. They have no negative zero, and their NaN takes the
code that `-0.0` would have. `e8m0` holds the power of two `2^(e - 127)` in its 8 bits, with
`0xFF` as NaN. Elements narrower than a byte pack densely in row-major order, starting at the
least significant bits of each byte. An `i1` element is stored as one byte.

Backends MAY support additional dtypes; if supported, they MUST obey the same rules.

### Dtype classes

The type rules group the dtypes into classes:

| Class | Dtypes |
|---|---|
| bool | `i1` |
| integer | `si4`, `ui4`, `si8`, `ui8`, `si16`, `ui16`, `si32`, `ui32`, `si64`, `ui64` |
| float | `bf16`, `f16`, `f32`, `f64`: the floats with arithmetic |
| narrow float | `f4e2m1`, `f6e2m3`, `f6e3m2`, `f8e4m3`, `f8e5m2`, `f8e4m3fnuz`, `f8e5m2fnuz` |
| scale | `e8m0` |

The classes each op takes:

| Ops | Classes |
|---|---|
| `constant`, `reshape`, `transpose`, `broadcast_to`, `slice`, `concat`, `take`, `gather`, `dynamic_slice`, `dynamic_update_slice`, `select`, `scatter_replace`, `stop_gradient`, `extract_patches`, `cond`, `while`, `scan`, `custom_call` | every dtype |
| `bitcast` | every dtype but `i1` |
| `neg`, `abs`, `add`, `sub`, `mul`, `div`, `maximum`, `minimum` | integer, float |
| `exp`, `log`, `tanh`, `erf`, `rsqrt`, `reciprocal`, `sin`, `cos` | float |
| `not`, `and`, `or`, `xor` | bool, integer |
| `shift_left`, `shift_right_logical`, `shift_right_arithmetic` | integer |
| `compare` | bool, integer, float |
| `cast` | every dtype in; every dtype but scale out |
| `reduce_{sum,max,min}`, `reduce_window_{sum,max,min}`, `segment_{sum,max,min}`, `scatter_{add,max,min}`, `argmax`, `top_k`, `iota` | integer, float |
| `reduce_{and,or}`, `reduce_window_{and,or}`, `segment_{and,or}` | bool, integer |
| `dot_general` | integer with integer; float or narrow float with float or narrow float |
| `rng_uniform` | integer, float; `rng_normal`: float |
| `quantize`, `quantize_dynamic`, `dequantize` | as "Quantization" states |

The `accum` and `out` dtypes of a reduction are of the input's class. A `dot_general` of floats or narrow floats
accumulates in a float and returns a float or a narrow float.

### `cast(x; to=dst)`

- Shape preserved.
- Float->float: round-to-nearest-even. A value beyond the range of a `bf16`, `f16` or `f32`
  destination becomes an infinity.
- Float->int: truncate toward 0; saturate on overflow; NaN maps to 0.
- Int->float: nearest representable, ties to even. An integer beyond the range of `f16` becomes an
  infinity.
- Int->int: saturate on overflow.
- Into a narrow float (`f8e4m3`, `f8e5m2`, `f8e4m3fnuz`, `f8e5m2fnuz`, `f6e2m3`, `f6e3m2`,
  `f4e2m1`), from a float or an integer: round-to-nearest-even and saturate to the largest finite
  value, as `tensor.quantize` does. Infinities saturate. NaN maps to NaN in a dtype that has one,
  and to 0 otherwise.
- Into `i1`: 1 for every value but zero, NaN included.
- Nothing casts into `e8m0`: `tensor.quantize_dynamic` computes e8m0 scales and states how it
  rounds them.

## Accumulator/output dtype policy

Some ops accept:

- `accum: dtype | None`
- `out: dtype | None`

Defaults:

- If input dtype is `f16`/`bf16`: default `accum = f32`
- If input dtype is `f32`: default `accum = f32`
- If input dtype is a narrower float (`f8e4m3`, `f8e5m2`, `f8e4m3fnuz`, `f8e5m2fnuz`, `f6e2m3`,
  `f6e3m2`, `f4e2m1`): default `accum = f32`
- If input dtype is `f64`, an integer or `i1`: default `accum = input dtype`
- Default `out = input dtype`

A reduction or contraction accumulates in `accum`: each operand element enters it as `cast`
converts it, each product and each partial result is rounded to `accum`, and the result is cast to
`out`. `segment_*` and `scatter_{add,max,min}` accumulate in the default accumulator of their dtype
and return their dtype.

## Quantization

Quantized tensors hold elements of a narrow dtype with scales, and offsets, that map them back to
real values. One scaling scheme covers, for training as well as inference:

- per-tensor, per-channel, per-token, per-group and block scales;
- static and dynamic scales;
- symmetric and asymmetric quantization;
- code tables;
- the MX and NVFP4 formats.

**Every format composes from core ops:**

- `bitcast` and the bitwise ops (`and`, `or`, `xor`, `not` and the shifts) unpack any packing,
  such as 3-bit codes spread over int32 words.
- `take` and `gather` map codes to values through any table, vector codebooks included.
- `cast` decodes any element dtype.
- `mul` and `add` apply scales and offsets, and a scale can itself be dequantized.

The ops `tensor.dequantize`, `tensor.quantize` and `tensor.quantize_dynamic` name the common
schemes, so that backends see their structure, and each is the composition its semantics states.
A new format starts as a composition and becomes an attribute of these ops when a backend needs
its structure.

**Scale layout.** The attribute `block` gives one extent per axis of `x`. Element `x[i]` uses
scale `scale[i / block]`, dividing per axis and rounding down, so the scale has the rank of `x` and
extent `ceil(x.shape[d] / block[d])` on each axis `d`:

- per tensor: `block = x.shape`, so the scale has one element;
- per channel along axis `a`: `block[a] = 1`, and every other extent is the whole axis;
- per group or block of `n` along axis `a` (MXFP4 and MXFP8: `n = 32`, NVFP4: `n = 16`):
  `block[a] = n`, and every other extent is 1;
- two-dimensional blocks, such as `128 x 128` weight blocks: two extents above 1.

An offset has the scale's shape and is read at the same index.

A second scale level, such as the per-tensor f32 scale of NVFP4 above its `e4m3` block scales, is
a `mul` by a scalar around these ops.

The quantized elements `Q` are integers or narrow floats. Scales `S` are floats, narrow floats or
`e8m0`; offsets are floats; a table holds floats or narrow floats along one axis. The real values
that the ops read and write are `f32`, `bf16` or `f16`. The optional operands come in the order
the ops list them, and the text form names which are present with `operand_segments`.

Requantizing is `tensor.quantize(tensor.dequantize(q, scale), new_scale)`; it has no op of its
own.

## Canonical library forms (attention-friendly)

### Stable softmax (last axis)

```text
m = reduce_max(z, axes=[-1], keepdims=true, accum=f32, out=f32)
e = exp(z - m)
s = reduce_sum(e, axes=[-1], keepdims=true, accum=f32, out=f32)
softmax = e / s
```

### Canonical attention (no explicit attention op)

For `q,k,v` in `[B,H,S,D]`:

1. `scores = dot_general(q, transpose(k,[0,1,3,2]), accum=f32, out=f32)`
2. `scores *= scale` (scalar f32 broadcast)
3. Add optional bias/mask (additive mask in score space: `0` or large negative / `-Inf`)
4. `p = softmax(scores)` (stable form above, f32)
5. `out = dot_general(p, v, accum=f32, out=desired)`

## Ops

- `tensor.constant`
- `tensor.neg`, `tensor.abs`, `tensor.exp`, `tensor.log`, `tensor.tanh`, `tensor.erf`, `tensor.rsqrt`, `tensor.reciprocal`, `tensor.sin`, `tensor.cos`, `tensor.not`
- `tensor.add`, `tensor.sub`, `tensor.mul`, `tensor.div`, `tensor.maximum`, `tensor.minimum`, `tensor.and`, `tensor.or`, `tensor.xor`, `tensor.shift_left`, `tensor.shift_right_logical`, `tensor.shift_right_arithmetic`
- `tensor.compare`
- `tensor.select`
- `tensor.cast`
- `tensor.dot_general`
- `tensor.reduce_sum`, `tensor.reduce_max`, `tensor.reduce_min`, `tensor.reduce_and`, `tensor.reduce_or`
- `tensor.reshape`
- `tensor.transpose`
- `tensor.broadcast_to`
- `tensor.slice`
- `tensor.bitcast`
- `tensor.concat`
- `tensor.iota`
- `tensor.take`
- `tensor.dynamic_slice`
- `tensor.dynamic_update_slice`
- `tensor.extract_patches`
- `tensor.reduce_window_sum`, `tensor.reduce_window_max`, `tensor.reduce_window_min`, `tensor.reduce_window_and`, `tensor.reduce_window_or`
- `tensor.custom_call`
- `tensor.stop_gradient`
- `tensor.argmax`
- `tensor.segment_sum`, `tensor.segment_max`, `tensor.segment_min`, `tensor.segment_and`, `tensor.segment_or`
- `tensor.top_k`
- `tensor.gather`
- `tensor.scatter_add`, `tensor.scatter_max`, `tensor.scatter_min`, `tensor.scatter_replace`
- `tensor.rng_uniform`, `tensor.rng_normal`
- `tensor.yield`
- `tensor.cond`
- `tensor.while`
- `tensor.scan`
- `tensor.dequantize`
- `tensor.quantize`
- `tensor.quantize_dynamic`

### `tensor.constant`

A tensor whose elements are an attribute.

- The result has the dtype and the static shape of its `value` attribute.
- The literal payload is interpreted as row-major, little-endian scalar storage, one byte per
  `i1` element. Backends that do not support a dtype MAY reject the program as unimplemented.
- In the text form, elements narrower than a byte are written in hex, packed as tensors store
  them.

Defined by `Constant` in [crates/gpt-rs-ir/src/dialects/tensor/ir/constant.rs](../../crates/gpt-rs-ir/src/dialects/tensor/ir/constant.rs).

| | |
|---|---|
| Operands | none |
| Results | `output`: tensor |
| Attributes | `value`: `dense` |
| Effects | none |

Example, from `filetests/tensor/roundtrip/elementwise.ptir`:

```text
func.func @constants() -> (tensor<2x2xf32>, tensor<3xsi32>, tensor<f32>, tensor<2xbf16>, tensor<2xf32>, tensor<1xf32>, tensor<2xi1>, tensor<2xf8e4m3>, tensor<0xf32>) {
  %0 = tensor.constant() {value = dense<[2, 2] x f32: [1.0, -0.5, 1e-7, inf]>} : () -> tensor<2x2xf32>
  %1 = tensor.constant() {value = dense<[3] x si32: 0>} : () -> tensor<3xsi32>
  %2 = tensor.constant() {value = dense<[] x f32: 0.5>} : () -> tensor<f32>
  %3 = tensor.constant() {value = dense<[2] x bf16: [1.0, 3.140625]>} : () -> tensor<2xbf16>
  %4 = tensor.constant() {value = dense<[2] x f32: [-0.0, -inf]>} : () -> tensor<2xf32>
  %5 = tensor.constant() {value = dense<[1] x f32: 0x0000c07f>} : () -> tensor<1xf32>
  %6 = tensor.constant() {value = dense<[2] x i1: [true, false]>} : () -> tensor<2xi1>
  %7 = tensor.constant() {value = dense<[2] x f8e4m3: 0x3840>} : () -> tensor<2xf8e4m3>
  %8 = tensor.constant() {value = dense<[0] x f32: []>} : () -> tensor<0xf32>
  func.return(%0, %1, %2, %3, %4, %5, %6, %7, %8) : (tensor<2x2xf32>, tensor<3xsi32>, tensor<f32>, tensor<2xbf16>, tensor<2xf32>, tensor<1xf32>, tensor<2xi1>, tensor<2xf8e4m3>, tensor<0xf32>) -> ()
}
```

Rejected, as the verify filetests pin it:

- `tensor.constant: only func.func may appear at the top level`

### `tensor.{kind}` (`Unary`)

Applies `kind` to every element.

- The result has the input's shape and dtype.

Defined by `Unary` in [crates/gpt-rs-ir/src/dialects/tensor/ir/elementwise.rs](../../crates/gpt-rs-ir/src/dialects/tensor/ir/elementwise.rs).

| | |
|---|---|
| Operands | `input`: tensor |
| Results | `output`: tensor |
| Traits | elementwise |
| Effects | none |

| Kind | Traits | |
|---|---|---|
| `neg` | involution | `-x`. Integers wrap, so the negation of the smallest signed integer is itself. |
| `abs` | idempotent | `abs(x)`. Integers wrap, so the absolute value of the smallest signed integer is itself. |
| `exp` |  | `e^x`. |
| `log` |  | The natural logarithm, `ln(x)`. |
| `tanh` |  | The hyperbolic tangent. |
| `erf` |  | The error function. |
| `rsqrt` |  | `1 / sqrt(x)`. |
| `reciprocal` |  | `1 / x`. |
| `sin` |  | The sine of `x` radians. |
| `cos` |  | The cosine of `x` radians. |
| `not` | involution | Flips every bit; on `i1`, logical not. |

Example, from `filetests/tensor/roundtrip/elementwise.ptir`:

```text
func.func @kinds(%0: tensor<4xf32>, %1: tensor<4xf32>) -> tensor<4xf32> {
  %2 = tensor.add(%0, %1) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  %3 = tensor.sub(%2, %1) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  %4 = tensor.mul(%3, %1) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  %5 = tensor.div(%4, %1) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  %6 = tensor.maximum(%5, %0) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  %7 = tensor.minimum(%6, %0) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  %8 = tensor.neg(%7) : (tensor<4xf32>) -> tensor<4xf32>
  %9 = tensor.abs(%8) : (tensor<4xf32>) -> tensor<4xf32>
  %10 = tensor.exp(%9) : (tensor<4xf32>) -> tensor<4xf32>
  %11 = tensor.log(%10) : (tensor<4xf32>) -> tensor<4xf32>
  %12 = tensor.tanh(%11) : (tensor<4xf32>) -> tensor<4xf32>
  %13 = tensor.erf(%12) : (tensor<4xf32>) -> tensor<4xf32>
  %14 = tensor.rsqrt(%13) : (tensor<4xf32>) -> tensor<4xf32>
  %15 = tensor.reciprocal(%14) : (tensor<4xf32>) -> tensor<4xf32>
  func.return(%15) : (tensor<4xf32>) -> ()
}
```

Rejected, as the verify filetests pin it:

- `tensor.exp: the input must be a float tensor, got tensor<4xsi32>`
- `tensor.not: the input must be an i1 or integer tensor, got tensor<4xf32>`
- `tensor.neg: the input must be an integer or float tensor, got tensor<4xi1>`

### `tensor.{kind}` (`Binary`)

Combines two tensors of equal type element by element:
`out[i] = kind(lhs[i], rhs[i])`.

- `lhs` and `rhs` MUST have the same dtype and the same shape, which the result has.
- Integer results wrap modulo `2^bits`, two's complement for the signed dtypes.
- A NaN result has no specified sign or payload.

Defined by `Binary` in [crates/gpt-rs-ir/src/dialects/tensor/ir/elementwise.rs](../../crates/gpt-rs-ir/src/dialects/tensor/ir/elementwise.rs).

| | |
|---|---|
| Operands | `lhs`: tensor; `rhs`: tensor |
| Results | `output`: tensor |
| Traits | elementwise |
| Effects | none |
| Patterns | `ScaleIntoDot` (`dot-scale-fold`) |

| Kind | Traits | |
|---|---|---|
| `add` | commutative; associative on integers; right identity -0.0 on floats, 0 on integers | `lhs + rhs`. |
| `sub` | right identity 0.0 on floats, 0 on integers | `lhs - rhs`. |
| `mul` | commutative; associative on integers; right identity 1.0 on floats, 1 on integers; absorbing 0 on integers; distributes over `tensor.add` on integers | `lhs * rhs`. |
| `div` | right identity 1.0 on floats, 1 on integers | `lhs / rhs`. Integer division truncates toward zero, and backends MUST reject an integer division by zero as a spec violation. Float division follows IEEE 754, so it can give Inf or NaN. |
| `maximum` | commutative; associative; idempotent; right identity -inf on floats, min on integers; absorbing max on integers | The larger operand. A float NaN operand gives NaN, and `-0.0` orders below `0.0`. |
| `minimum` | commutative; associative; idempotent; right identity inf on floats, max on integers; absorbing min on integers | The smaller operand. A float NaN operand gives NaN, and `-0.0` orders below `0.0`. |
| `and` | commutative; associative on integers; idempotent; right identity all ones on integers; absorbing 0 on integers; distributes over `tensor.or` on integers | Bitwise and; on `i1`, logical and. |
| `or` | commutative; associative on integers; idempotent; right identity 0 on integers; absorbing all ones on integers; distributes over `tensor.and` on integers | Bitwise or; on `i1`, logical or. |
| `xor` | commutative; associative on integers; right identity 0 on integers | Bitwise exclusive or; on `i1`, logical exclusive or. |
| `shift_left` | right identity 0 on integers | `lhs` shifted left by `rhs` bits, shifting in zeros. `rhs` reads as unsigned, and a shift by at least the bit width gives 0. |
| `shift_right_logical` | right identity 0 on integers | `lhs` shifted right by `rhs` bits, shifting in zeros, into a signed dtype too. `rhs` reads as unsigned, and a shift by at least the bit width gives 0. |
| `shift_right_arithmetic` | right identity 0 on integers | `lhs` shifted right by `rhs` bits, shifting in copies of the sign bit. `rhs` reads as unsigned, and a shift by at least the bit width gives -1 for a negative `lhs`, else 0. |

Example, from `filetests/tensor/roundtrip/elementwise.ptir`:

```text
func.func @bits(%0: tensor<4xui8>, %1: tensor<4xi1>, %2: tensor<4xf64>) -> (tensor<4xui8>, tensor<4xi1>, tensor<4xf64>) {
  %3 = tensor.and(%0, %0) : (tensor<4xui8>, tensor<4xui8>) -> tensor<4xui8>
  %4 = tensor.or(%3, %0) : (tensor<4xui8>, tensor<4xui8>) -> tensor<4xui8>
  %5 = tensor.xor(%4, %0) : (tensor<4xui8>, tensor<4xui8>) -> tensor<4xui8>
  %6 = tensor.shift_left(%5, %0) : (tensor<4xui8>, tensor<4xui8>) -> tensor<4xui8>
  %7 = tensor.shift_right_logical(%6, %0) : (tensor<4xui8>, tensor<4xui8>) -> tensor<4xui8>
  %8 = tensor.shift_right_arithmetic(%7, %0) : (tensor<4xui8>, tensor<4xui8>) -> tensor<4xui8>
  %9 = tensor.not(%8) : (tensor<4xui8>) -> tensor<4xui8>
  %10 = tensor.not(%1) : (tensor<4xi1>) -> tensor<4xi1>
  %11 = tensor.sin(%2) : (tensor<4xf64>) -> tensor<4xf64>
  %12 = tensor.cos(%11) : (tensor<4xf64>) -> tensor<4xf64>
  func.return(%9, %10, %12) : (tensor<4xui8>, tensor<4xi1>, tensor<4xf64>) -> ()
}
```

Rejected, as the verify filetests pin it:

- `tensor.add: operands must have equal types, got tensor<4xf32> and tensor<3xf32>`
- `tensor.add: each operand must be an integer or float tensor, got tensor<4xi1>`
- `tensor.shift_left: each operand must be an integer tensor, got tensor<4xf32>`
- `tensor.and: each operand must be an i1 or integer tensor, got tensor<4xbf16>`
- `tensor.add: each operand must be an integer or float tensor, got tensor<4xf8e4m3>`
- `tensor.add: result types are (tensor<8xf32>), but the op gives (tensor<4xf32>)`

### `tensor.compare`

Compares two tensors of equal type element by element into an `i1` tensor.

- `out[i]` is 1 where `kind` holds between `lhs[i]` and `rhs[i]`, and 0 elsewhere. The result
  has the operands' shape.
- `lhs` and `rhs` MUST have the same dtype and the same shape.
- Unsigned integers order as unsigned numbers, and `i1` orders false below true.
- Floats compare as IEEE does: `less`, `less_equal`, `equal`, `greater_equal` and `greater` are
  false if either operand is NaN, and `not_equal` is true if either operand is NaN.

Defined by `Compare` in [crates/gpt-rs-ir/src/dialects/tensor/ir/elementwise.rs](../../crates/gpt-rs-ir/src/dialects/tensor/ir/elementwise.rs).

| | |
|---|---|
| Operands | `lhs`: tensor; `rhs`: tensor |
| Results | `output`: tensor |
| Attributes | `kind`: `CompareKind` |
| Traits | elementwise |
| Effects | none |

| `kind` | |
|---|---|
| `less` | `lhs < rhs`. |
| `less_equal` | `lhs <= rhs`. |
| `equal` | `lhs == rhs`. |
| `greater_equal` | `lhs >= rhs`. |
| `greater` | `lhs > rhs`. |
| `not_equal` | `lhs != rhs`. |

Example, from `filetests/tensor/roundtrip/elementwise.ptir`:

```text
func.func @compare_si32(%0: tensor<2x{T}xsi32>, %1: tensor<2x{T}xsi32>) -> tensor<2x{T}xi1> {
  %2 = tensor.compare(%0, %1) {kind = greater_equal} : (tensor<2x{T}xsi32>, tensor<2x{T}xsi32>) -> tensor<2x{T}xi1>
  func.return(%2) : (tensor<2x{T}xi1>) -> ()
}
```

Rejected, as the verify filetests pin it:

- `tensor.compare: operands must have equal types, got tensor<4xf32> and tensor<4xsi32>`
- `tensor.compare: each operand must be an i1, integer or float tensor, got tensor<4xe8m0>`

### `tensor.select`

Picks `on_true` where `pred` is set and `on_false` elsewhere.

- For each element: `out[i] = pred[i] ? on_true[i] : on_false[i]`.
- `pred` is `i1`. All operands MUST have the same shape, and `on_true` and `on_false` MUST have
  the same dtype, which the result has.

Defined by `Select` in [crates/gpt-rs-ir/src/dialects/tensor/ir/elementwise.rs](../../crates/gpt-rs-ir/src/dialects/tensor/ir/elementwise.rs).

| | |
|---|---|
| Operands | `pred`: tensor; `on_true`: tensor; `on_false`: tensor |
| Results | `output`: tensor |
| Traits | elementwise |
| Effects | none |
| Folds | `select(pred, on_true, on_false)` is `on_true` when `pred` is all true or both branches are the same value, and `on_false` when `pred` is all false. |

Example, from `filetests/tensor/roundtrip/elementwise.ptir`:

```text
func.func @select(%0: tensor<2x{T}xi1>, %1: tensor<2x{T}xf32>, %2: tensor<2x{T}xf32>) -> tensor<2x{T}xf32> {
  %3 = tensor.select(%0, %1, %2) : (tensor<2x{T}xi1>, tensor<2x{T}xf32>, tensor<2x{T}xf32>) -> tensor<2x{T}xf32>
  func.return(%3) : (tensor<2x{T}xf32>) -> ()
}
```

Rejected, as the verify filetests pin it:

- `tensor.select: pred must be an i1 tensor, got tensor<4xf32>`
- `tensor.select: on_true and on_false must have equal types, got tensor<4xf32> and tensor<4xbf16>`
- `tensor.select: pred and the branches must have equal shapes, got tensor<1x4xi1> and tensor<4xf32>`

### `tensor.cast`

Converts every element to `to`.

- The shape is preserved.
- The conversion rules are those of `cast(x; to=dst)` in "DTypes and casting" of the
  dialect documentation.

Defined by `Cast` in [crates/gpt-rs-ir/src/dialects/tensor/ir/elementwise.rs](../../crates/gpt-rs-ir/src/dialects/tensor/ir/elementwise.rs).

| | |
|---|---|
| Operands | `input`: tensor |
| Results | `output`: tensor |
| Attributes | `to`: `dtype` |
| Traits | elementwise |
| Effects | none |
| Folds | A cast to the input's own type is the input. |

Example, from `filetests/tensor/roundtrip/elementwise.ptir`:

```text
func.func @cast(%0: tensor<2x3xbf16>, %1: tensor<4xsi32>) -> (tensor<2x3xf32>, tensor<4xi1>, tensor<4xsi32>) {
  %2 = tensor.cast(%0) {to = f32} : (tensor<2x3xbf16>) -> tensor<2x3xf32>
  %3 = tensor.cast(%1) {to = i1} : (tensor<4xsi32>) -> tensor<4xi1>
  %4 = tensor.cast(%1) {to = si32} : (tensor<4xsi32>) -> tensor<4xsi32>
  func.return(%2, %3, %4) : (tensor<2x3xf32>, tensor<4xi1>, tensor<4xsi32>) -> ()
}
```

Rejected, as the verify filetests pin it:

- `tensor.cast: to must be an i1, integer, float or narrow float dtype, got e8m0`

### `tensor.dot_general`

A contraction over `contract_*` axes, batched over `batch_*` axes, like
an einsum. Each result element sums the products of the `lhs` and `rhs` elements at its batch
and free coordinates, over every coordinate of the contracting axes.

- `batch_lhs` pairs with `batch_rhs`, and `contract_lhs` with `contract_rhs`; the axes of each
  pair MUST have matching extents. Every axis MUST be in range, a negative axis counting from
  the end, and an operand's axis MUST appear at most once in its batch and contracting axes.
- The result axes are the batch axes, in `batch_lhs` order, then the remaining non-contracting
  axes of `lhs`, then those of `rhs`, each in axis order.
- The operand dtypes MAY differ: integers with integers, and floats or narrow floats with each
  other. For example, FP8 training multiplies e5m2 gradients by e4m3 weights, and W4A8
  multiplies int4 weights by int8 activations. Then `accum` and `out` MUST be given.
- It accumulates the products in `accum` and casts the sums to `out`. The accumulator policy
  gives their defaults. A wider `accum` holds the products of narrower inputs exactly, for
  example bf16 x bf16 with f32 accumulation.

Defined by `DotGeneral` in [crates/gpt-rs-ir/src/dialects/tensor/ir/contraction.rs](../../crates/gpt-rs-ir/src/dialects/tensor/ir/contraction.rs).

| | |
|---|---|
| Operands | `lhs`: tensor; `rhs`: tensor |
| Results | `output`: tensor |
| Attributes | `batch_lhs`: `[i64]`, optional; `batch_rhs`: `[i64]`, optional; `contract_lhs`: `[i64]`; `contract_rhs`: `[i64]`; `accum`: `dtype`, optional; `out`: `dtype`, optional |
| Effects | none |
| Patterns | `PackGemmWeight` (`c-pack-weights`); `GroupedConvWeight` (`c-pack-weights`); `DropDefaultDotDtypes` (`canonicalize`); `NonNegativeDotAxes` (`canonicalize`); `MergeDotAxes` (`merge-contracted-axes`) |

Example, from `filetests/tensor/roundtrip/contraction.ptir`:

```text
func.func @matmul(%0: tensor<4x8xf32>, %1: tensor<16x8xf32>) -> tensor<4x16xf32> {
  %2 = tensor.dot_general(%0, %1) {contract_lhs = [1], contract_rhs = [1]} : (tensor<4x8xf32>, tensor<16x8xf32>) -> tensor<4x16xf32>
  func.return(%2) : (tensor<4x16xf32>) -> ()
}
```

Rejected, as the verify filetests pin it:

- `tensor.dot_general: operands of different dtypes, tensor<2x3xf32> and tensor<3x2xbf16>, need accum and out`
- `tensor.dot_general: lhs must be an integer, float or narrow float tensor, got tensor<2x3xi1>`
- `tensor.dot_general: accum must be a float dtype, got i1`
- `tensor.dot_general: batch_lhs [0] and batch_rhs [] differ in length`
- `tensor.dot_general: contract_lhs [1] and contract_rhs [0, 1] differ in length`
- `tensor.dot_general: contract_rhs axis 2 is out of range for rank 2`
- `tensor.dot_general: contract_lhs axis -3 is out of range for rank 2`
- `tensor.dot_general: lhs axis -1 is named twice`
- `tensor.dot_general: lhs axis 0 is named twice`
- `tensor.dot_general: rhs axis 0 is named twice`
- `tensor.dot_general: batch_lhs axis 0 has extent 2, but batch_rhs axis 0 has extent 3`
- `tensor.dot_general: contract_lhs axis 1 has extent {K}, but contract_rhs axis 0 has extent 8`
- `tensor.dot_general: rhs must be an integer tensor, got tensor<3x2xf32>`
- `tensor.dot_general: out must be a float or narrow float dtype, got i1`

### `tensor.reduce_{kind}` (`Reduce`)

Reduces the input over `axes` with `kind`.

- `axes` MUST be unique and in range. A negative axis counts from the end, and the order of
  `axes` does not matter.
- If `keepdims` is true, the reduced axes remain with extent 1; otherwise they are removed.
- It accumulates in `accum` and casts the result to `out`. The accumulator policy gives their
  defaults.
- When a reduced axis has extent 0, each result element is the identity of `kind`.

Defined by `Reduce` in [crates/gpt-rs-ir/src/dialects/tensor/ir/contraction.rs](../../crates/gpt-rs-ir/src/dialects/tensor/ir/contraction.rs).

| | |
|---|---|
| Operands | `input`: tensor |
| Results | `output`: tensor |
| Attributes | `axes`: `[i64]`; `keepdims`: `bool`, optional; `accum`: `dtype`, optional; `out`: `dtype`, optional |
| Effects | none |
| Patterns | `DropDefaultReduceDtypes` (`canonicalize`); `MergeReductions` (`canonicalize`); `ReduceOfBroadcast` (`canonicalize`); `ReduceOfConcat` (`canonicalize`); `SortedReduceAxes` (`canonicalize`) |

| Kind | |
|---|---|
| `sum` | Folds with `add`. The initial value is 0. |
| `max` | Folds with `maximum`. The initial value is `-inf` for floats and the smallest value of the dtype for integers. |
| `min` | Folds with `minimum`. The initial value is `+inf` for floats and the largest value of the dtype for integers. |
| `and` | Folds with `and`. The initial value has every bit set, true for `i1`, so on `i1` it is "all". |
| `or` | Folds with `or`. The initial value is 0, so on `i1` it is "any". |

Example, from `filetests/tensor/roundtrip/contraction.ptir`:

```text
func.func @reduce(%0: tensor<2x{T}x8xf32>) -> (tensor<2x{T}x1xf32>, tensor<{T}xf32>, tensor<2x1x1xf32>, tensor<2x{T}x8xf32>) {
  %1 = tensor.reduce_sum(%0) {axes = [2], keepdims = true} : (tensor<2x{T}x8xf32>) -> tensor<2x{T}x1xf32>
  %2 = tensor.reduce_max(%0) {axes = [0, 2]} : (tensor<2x{T}x8xf32>) -> tensor<{T}xf32>
  %3 = tensor.reduce_min(%0) {axes = [1, 2], keepdims = true, accum = f32, out = f32} : (tensor<2x{T}x8xf32>) -> tensor<2x1x1xf32>
  %4 = tensor.reduce_sum(%0) {axes = []} : (tensor<2x{T}x8xf32>) -> tensor<2x{T}x8xf32>
  func.return(%1, %2, %3, %4) : (tensor<2x{T}x1xf32>, tensor<{T}xf32>, tensor<2x1x1xf32>, tensor<2x{T}x8xf32>) -> ()
}
```

Rejected, as the verify filetests pin it:

- `tensor.reduce_sum: the input must be an integer or float tensor, got tensor<4xi1>`
- `tensor.reduce_max: accum must be a float dtype, got i1`
- `tensor.reduce_sum: axis 2 is out of range for rank 2`
- `tensor.reduce_sum: axis -4 is out of range for rank 3`
- `tensor.reduce_min: axis 1 is named twice`
- `tensor.reduce_max: axis -1 is named twice`
- `tensor.reduce_and: the input must be an i1 or integer tensor, got tensor<4xf32>`
- `tensor.reduce_sum: out must be a float dtype, got i1`

### `tensor.reshape`

The same elements in row-major order, with a new shape.

- `shape` gives every extent explicitly, and the element count MUST be preserved.

Defined by `Reshape` in [crates/gpt-rs-ir/src/dialects/tensor/ir/layout.rs](../../crates/gpt-rs-ir/src/dialects/tensor/ir/layout.rs).

| | |
|---|---|
| Operands | `input`: tensor |
| Results | `output`: tensor |
| Attributes | `shape`: `[dim]` |
| Traits | movement |
| Effects | none |
| Folds | A reshape to the input's own type is the input, and a reshape of a constant is that constant with the new extents, its elements in the same row-major order. |
| Patterns | `CollapseReshapeChain` (`canonicalize`) |

Example, from `filetests/tensor/roundtrip/layout.ptir`:

```text
func.func @reshape_scalar(%0: tensor<1x1xsi32>) -> tensor<si32> {
  %1 = tensor.reshape(%0) {shape = []} : (tensor<1x1xsi32>) -> tensor<si32>
  func.return(%1) : (tensor<si32>) -> ()
}
```

Rejected, as the verify filetests pin it:

- `tensor.reshape: shape [4, 2] has 8 elements, but tensor<2x3xf32> has 6`
- `tensor.reshape: shape [{T}, 2] has 2*{T} elements, but tensor<{B}x2xf32> has 2*{B}`
- `tensor.reshape: the element count of tensor<4294967296x4294967296xf32> overflows`
- `tensor.reshape: the element count of shape [4294967296, 4294967296] overflows`

### `tensor.transpose`

Permutes the axes. Output axis `i` is input axis `perm[i]`.

- `perm` MUST be a permutation of `[0, rank)`.

Defined by `Transpose` in [crates/gpt-rs-ir/src/dialects/tensor/ir/layout.rs](../../crates/gpt-rs-ir/src/dialects/tensor/ir/layout.rs).

| | |
|---|---|
| Operands | `input`: tensor |
| Results | `output`: tensor |
| Attributes | `perm`: `[usize]` |
| Traits | movement |
| Effects | none |
| Folds | A transpose by the identity `perm` is the input. |
| Patterns | `CollapseTransposeChain` (`canonicalize`) |

Example, from `filetests/tensor/roundtrip/layout.ptir`:

```text
func.func @transpose(%0: tensor<2x3x4xf32>) -> tensor<4x2x3xf32> {
  %1 = tensor.transpose(%0) {perm = [2, 0, 1]} : (tensor<2x3x4xf32>) -> tensor<4x2x3xf32>
  func.return(%1) : (tensor<4x2x3xf32>) -> ()
}
```

Rejected, as the verify filetests pin it:

- `tensor.transpose: perm [0, 0] is not a permutation of rank 2`

### `tensor.broadcast_to`

Aligns the input with the trailing axes of `shape` and repeats it along
every axis where the input has extent 1 or no axis.

- The result has the extents of `shape`.
- Compatibility follows the right-aligned rule of "Broadcasting policy" in the dialect
  documentation: `shape` has at least the input's rank, and each input extent is 1 or the
  extent of `shape` on its axis. Backends MUST reject shapes that are not broadcast-compatible
  as a spec violation.

Defined by `BroadcastTo` in [crates/gpt-rs-ir/src/dialects/tensor/ir/layout.rs](../../crates/gpt-rs-ir/src/dialects/tensor/ir/layout.rs).

| | |
|---|---|
| Operands | `input`: tensor |
| Results | `output`: tensor |
| Attributes | `shape`: `[dim]` |
| Traits | movement |
| Effects | none |
| Folds | A broadcast to the input's own type is the input. |
| Patterns | `CollapseBroadcastChain` (`canonicalize`) |

Example, from `filetests/tensor/roundtrip/layout.ptir`:

```text
func.func @broadcast_to_dynamic(%0: tensor<1x768xf32>, %1: tensor<{B}x1xf32>) -> (tensor<{B}x768xf32>, tensor<2x{B}x4xf32>) {
  %2 = tensor.broadcast_to(%0) {shape = [{B}, 768]} : (tensor<1x768xf32>) -> tensor<{B}x768xf32>
  %3 = tensor.broadcast_to(%1) {shape = [2, {B}, 4]} : (tensor<{B}x1xf32>) -> tensor<2x{B}x4xf32>
  func.return(%2, %3) : (tensor<{B}x768xf32>, tensor<2x{B}x4xf32>) -> ()
}
```

Rejected, as the verify filetests pin it:

- `tensor.broadcast_to: shape [3] has a lower rank than tensor<2x3xf32>`
- `tensor.broadcast_to: input axis 0 has extent 2, which does not broadcast to 4`
- `tensor.broadcast_to: input axis 0 has extent {B}, which does not broadcast to {T}`

### `tensor.slice`

The block of extents `sizes` starting at `starts`, taking every `strides[i]`-th
element along axis `i`.

- Along axis `i`, result element `k` is input element `starts[i] + k * strides[i]`. Without
  `strides`, the window is contiguous: unit stride.
- `starts` and `sizes` have one entry per axis of the input, and `strides` is empty or has one
  positive entry per axis.
- The window MUST lie in the input: on every axis `i`, `starts[i] + span <= in_shape[i]`, where
  `span` is `(sizes[i] - 1) * strides[i] + 1`, or 0 when `sizes[i]` is 0. With unit stride,
  that is `starts[i] + sizes[i] <= in_shape[i]`.

Defined by `Slice` in [crates/gpt-rs-ir/src/dialects/tensor/ir/layout.rs](../../crates/gpt-rs-ir/src/dialects/tensor/ir/layout.rs).

| | |
|---|---|
| Operands | `input`: tensor |
| Results | `output`: tensor |
| Attributes | `starts`: `[usize]`; `sizes`: `[usize]`; `strides`: `[usize]`, optional |
| Traits | movement |
| Effects | none |
| Folds | A slice with the input's extents covers the whole input, so it is the input. |
| Patterns | `CollapseSliceChain` (`canonicalize`); `UnitStrides` (`canonicalize`) |

Example, from `filetests/tensor/roundtrip/layout.ptir`:

```text
func.func @slice(%0: tensor<4x6xf32>, %1: tensor<{B}x6xf32>) -> (tensor<2x3xf32>, tensor<1x4xf32>) {
  %2 = tensor.slice(%0) {starts = [1, 2], sizes = [2, 3]} : (tensor<4x6xf32>) -> tensor<2x3xf32>
  %3 = tensor.slice(%1) {starts = [0, 2], sizes = [1, 4]} : (tensor<{B}x6xf32>) -> tensor<1x4xf32>
  func.return(%2, %3) : (tensor<2x3xf32>, tensor<1x4xf32>) -> ()
}
```

Rejected, as the verify filetests pin it:

- `tensor.slice: starts [0] and sizes [1] do not match rank 2`
- `tensor.slice: start 3 + size 2 exceeds extent 4 on axis 0`
- `tensor.slice: strides [2] do not match rank 2`
- `tensor.slice: strides [1, 0] must all be positive`
- `tensor.slice: start 1 + size 7 exceeds extent 6 on axis 1`

### `tensor.bitcast`

The bits of the input read as elements of `to`.

- Each row of the last axis is one stream of bits: its elements in order, each from its least
  significant bit up, as tensors store them. The last axis holds the same bits, so its extent
  scales by `bits(input) / bits(to)`, which MUST give a whole number; the other axes stay.
  Dtypes of equal width keep the shape.
- `i1` takes no part: its elements are bytes holding 0 or 1, not bits.
- It reads published packings in place: GPTQ and AWQ pack eight 4-bit codes into each int32,
  and gpt-oss stores MXFP4 blocks and their e8m0 scales as bytes.

Defined by `Bitcast` in [crates/gpt-rs-ir/src/dialects/tensor/ir/layout.rs](../../crates/gpt-rs-ir/src/dialects/tensor/ir/layout.rs).

| | |
|---|---|
| Operands | `input`: tensor |
| Results | `output`: tensor |
| Attributes | `to`: `dtype` |
| Effects | none |
| Folds | A bitcast to the input's own type is the input. |

Example, from `filetests/tensor/roundtrip/layout.ptir`:

```text
func.func @bitcast(%0: tensor<4x2xsi32>, %1: tensor<4x3xui8>, %2: tensor<4xbf16>) -> (tensor<4x16xui4>, tensor<4x4xf6e2m3>, tensor<4xsi16>) {
  %3 = tensor.bitcast(%0) {to = ui4} : (tensor<4x2xsi32>) -> tensor<4x16xui4>
  %4 = tensor.bitcast(%1) {to = f6e2m3} : (tensor<4x3xui8>) -> tensor<4x4xf6e2m3>
  %5 = tensor.bitcast(%2) {to = si16} : (tensor<4xbf16>) -> tensor<4xsi16>
  func.return(%3, %4, %5) : (tensor<4x16xui4>, tensor<4x4xf6e2m3>, tensor<4xsi16>) -> ()
}
```

Rejected, as the verify filetests pin it:

- `tensor.bitcast: i1 elements have no bits to reinterpret`
- `tensor.bitcast: the 16 bits of each row of tensor<4x2xui8> are no whole number of f6e2m3 elements`
- `tensor.bitcast: reinterpreting tensor<{B}xsi32> as ui4 needs a static last axis`

### `tensor.concat`

Joins the inputs along `axis`.

- Concatenates one or more inputs along `axis`, a negative axis counting from the end. The
  result's extent on `axis` is the sum of the inputs' extents.
- All dtypes MUST match, and all shapes MUST match on the axes other than `axis`.

Defined by `Concat` in [crates/gpt-rs-ir/src/dialects/tensor/ir/layout.rs](../../crates/gpt-rs-ir/src/dialects/tensor/ir/layout.rs).

| | |
|---|---|
| Operands | `inputs`: tensors |
| Results | `output`: tensor |
| Attributes | `axis`: `i64` |
| Traits | movement |
| Effects | none |
| Folds | A concat of one input is that input. |
| Patterns | `NonNegativeAxis` (`canonicalize`) |

Example, from `filetests/tensor/roundtrip/layout.ptir`:

```text
func.func @concat_dynamic(%0: tensor<{B}x3xf32>, %1: tensor<{B}x5xf32>, %2: tensor<{T}x2xf32>) -> (tensor<{B}x8xf32>, tensor<{T}x2xf32>) {
  %3 = tensor.concat(%0, %1) {axis = 1} : (tensor<{B}x3xf32>, tensor<{B}x5xf32>) -> tensor<{B}x8xf32>
  %4 = tensor.concat(%2) {axis = 0} : (tensor<{T}x2xf32>) -> tensor<{T}x2xf32>
  func.return(%3, %4) : (tensor<{B}x8xf32>, tensor<{T}x2xf32>) -> ()
}
```

Rejected, as the verify filetests pin it:

- `tensor.concat: needs at least one input`
- `tensor.concat: axis 2 is out of range for rank 2`
- `tensor.concat: inputs must have equal dtypes, got tensor<2xf32> and tensor<2xsi32>`
- `tensor.concat: inputs must have equal ranks, got tensor<2x3xf32> and tensor<6xf32>`
- `tensor.concat: inputs must have equal extents off axis 0, got tensor<2x3xf32> and tensor<2x4xf32>`
- `tensor.concat: axis 0 joins to {B} + 2, which is not one dimension`
- `tensor.concat: the joined extent on axis 0 overflows`

### `tensor.iota`

A tensor of `shape` whose elements count up along `axis`.

- The value increases along `axis`, a negative axis counting from the end, from 0 to
  `shape[axis] - 1`, and repeats along every other axis.
- Each value is its integer index converted to `dtype` as `tensor.cast` converts an integer: a
  float rounds it once, to nearest-even, and a narrow integer saturates it.

Defined by `Iota` in [crates/gpt-rs-ir/src/dialects/tensor/ir/indexing.rs](../../crates/gpt-rs-ir/src/dialects/tensor/ir/indexing.rs).

| | |
|---|---|
| Operands | none |
| Results | `output`: tensor |
| Attributes | `shape`: `[dim]`; `dtype`: `dtype`; `axis`: `i64` |
| Effects | none |
| Patterns | `NonNegativeAxis` (`canonicalize`) |

Example, from `filetests/tensor/roundtrip/indexing.ptir`:

```text
func.func @iota() -> (tensor<2x3xsi32>, tensor<{T}xf32>) {
  %0 = tensor.iota() {shape = [2, 3], dtype = si32, axis = 1} : () -> tensor<2x3xsi32>
  %1 = tensor.iota() {shape = [{T}], dtype = f32, axis = 0} : () -> tensor<{T}xf32>
  func.return(%0, %1) : (tensor<2x3xsi32>, tensor<{T}xf32>) -> ()
}
```

Rejected, as the verify filetests pin it:

- `tensor.iota: axis 2 is out of range for rank 2`
- `tensor.iota: dtype must be an integer or float dtype, got i1`

### `tensor.take`

The rows of `table` at `indices`. The result shape is the indices' shape
followed by the table's trailing axes.

- Equivalent to `index_select` on axis 0, the common embedding primitive.
- `indices` are `si32` or `si64`, and each MUST satisfy `0 <= index < N`, where `N` is the
  extent of the table's leading axis.

Defined by `Take` in [crates/gpt-rs-ir/src/dialects/tensor/ir/indexing.rs](../../crates/gpt-rs-ir/src/dialects/tensor/ir/indexing.rs).

| | |
|---|---|
| Operands | `table`: tensor; `indices`: tensor |
| Results | `output`: tensor |
| Traits | movement of operand 0 |
| Effects | none |

Example, from `filetests/tensor/roundtrip/indexing.ptir`:

```text
func.func @take(%0: tensor<50x8xf32> {param = 3}, %1: tensor<2x3xsi32>, %2: tensor<50x8xbf16>, %3: tensor<{B}xsi32>) -> (tensor<2x3x8xf32>, tensor<{B}x8xbf16>) {
  %4 = tensor.take(%0, %1) : (tensor<50x8xf32>, tensor<2x3xsi32>) -> tensor<2x3x8xf32>
  %5 = tensor.take(%2, %3) : (tensor<50x8xbf16>, tensor<{B}xsi32>) -> tensor<{B}x8xbf16>
  func.return(%4, %5) : (tensor<2x3x8xf32>, tensor<{B}x8xbf16>) -> ()
}
```

Rejected, as the verify filetests pin it:

- `tensor.take: the table must have rank at least 1, got tensor<f32>`
- `tensor.take: indices must be si32 or si64, got tensor<3xf32>`

### `tensor.dynamic_slice`

The block of extents `sizes` starting at the run-time `starts`, with
unit stride.

- `sizes` has one extent per axis, and `starts` holds one `si32` or `si64` start per axis, of
  shape `[rank]`.
- On each axis, the start is clamped into `[0, extent - size]`, so the block lies inside the
  input.
- If a size exceeds its axis's extent at run time, backends MUST reject the op as a spec
  violation.

Defined by `DynamicSlice` in [crates/gpt-rs-ir/src/dialects/tensor/ir/indexing.rs](../../crates/gpt-rs-ir/src/dialects/tensor/ir/indexing.rs).

| | |
|---|---|
| Operands | `input`: tensor; `starts`: tensor |
| Results | `output`: tensor |
| Attributes | `sizes`: `[usize]` |
| Traits | movement of operand 0 |
| Effects | none |
| Folds | A slice as large as its input starts at 0 whatever `starts` holds, so it is the input. |

Example, from `filetests/tensor/roundtrip/indexing.ptir`:

```text
func.func @dynamic_slice(%0: tensor<4x8xf32>, %1: tensor<{S}x8xsi32>, %2: tensor<2xsi32>) -> (tensor<1x8xf32>, tensor<2x8xsi32>) {
  %3 = tensor.dynamic_slice(%0, %2) {sizes = [1, 8]} : (tensor<4x8xf32>, tensor<2xsi32>) -> tensor<1x8xf32>
  %4 = tensor.dynamic_slice(%1, %2) {sizes = [2, 8]} : (tensor<{S}x8xsi32>, tensor<2xsi32>) -> tensor<2x8xsi32>
  func.return(%3, %4) : (tensor<1x8xf32>, tensor<2x8xsi32>) -> ()
}
```

Rejected, as the verify filetests pin it:

- `tensor.dynamic_slice: sizes [1] do not match rank 2`
- `tensor.dynamic_slice: starts must be si32 or si64 of shape [2], got tensor<2xf32>`
- `tensor.dynamic_slice: starts must be si32 or si64 of shape [2], got tensor<{N}xsi32>`
- `tensor.dynamic_slice: size 5 exceeds extent 4 on axis 0`

### `tensor.dynamic_update_slice`

`base` with the block at the run-time `starts` replaced by
`update`.

- The block's extents are the static extents of `update`, which has the dtype and rank of
  `base`.
- `starts` holds one `si32` or `si64` start per axis, of shape `[rank]`, clamped as in
  `tensor.dynamic_slice`: into `[0, extent - size]`.

Defined by `DynamicUpdateSlice` in [crates/gpt-rs-ir/src/dialects/tensor/ir/indexing.rs](../../crates/gpt-rs-ir/src/dialects/tensor/ir/indexing.rs).

| | |
|---|---|
| Operands | `base`: tensor; `update`: tensor; `starts`: tensor |
| Results | `output`: tensor |
| Traits | movement of operands 0 to 1 |
| Effects | none |
| Folds | An update as large as the base replaces all of it, so the result is the update. |

Example, from `filetests/tensor/roundtrip/indexing.ptir`:

```text
func.func @dynamic_update_slice(%0: tensor<2x16x4xf32>, %1: tensor<2x1x4xf32>, %2: tensor<3xsi32>, %3: tensor<2x{S}x4xf32>) -> (tensor<2x16x4xf32>, tensor<2x{S}x4xf32>) {
  %4 = tensor.dynamic_update_slice(%0, %1, %2) : (tensor<2x16x4xf32>, tensor<2x1x4xf32>, tensor<3xsi32>) -> tensor<2x16x4xf32>
  %5 = tensor.dynamic_update_slice(%3, %1, %2) : (tensor<2x{S}x4xf32>, tensor<2x1x4xf32>, tensor<3xsi32>) -> tensor<2x{S}x4xf32>
  func.return(%4, %5) : (tensor<2x16x4xf32>, tensor<2x{S}x4xf32>) -> ()
}
```

Rejected, as the verify filetests pin it:

- `tensor.dynamic_update_slice: update must have rank 2, got tensor<1xf32>`
- `tensor.dynamic_update_slice: size 5 exceeds extent 4 on axis 0`
- `tensor.dynamic_update_slice: update must have dtype f32, got tensor<1x8xsi32>`
- `tensor.dynamic_update_slice: update must have static extents, got tensor<1x{N}xf32>`

### `tensor.extract_patches`

Gathers every `window` patch of the padded input into its own row
(im2col, unfold).

- The input is channels-last, `[N, spatial..., C]`, and the result is
  `[N, positions..., prod(window) * C]`.
- `window`, `strides`, `dilation` and `padding` have one entry per spatial axis. The positions
  along each spatial axis are those of `tensor.reduce_window_*` with a base dilation of 1.
- A patch holds the elements of its window in row-major order of the window, `C` channels each.
  Elements in the padding are `pad_value`, which MUST be a value of the input's dtype.
- Backends SHOULD fuse `extract_patches` + `dot_general` into convolution kernels.
- An extension op.

Defined by `ExtractPatches` in [crates/gpt-rs-ir/src/dialects/tensor/ir/window.rs](../../crates/gpt-rs-ir/src/dialects/tensor/ir/window.rs).

| | |
|---|---|
| Operands | `input`: tensor |
| Results | `output`: tensor |
| Attributes | `window`: `[usize]`; `strides`: `[usize]`; `dilation`: `[usize]`; `padding`: `[(usize, usize)]`; `pad_value`: `f64` |
| Effects | none |

Example, from `filetests/tensor/roundtrip/window.ptir`:

```text
func.func @conv_patches(%0: tensor<1x5x5x3xf32>) -> tensor<1x3x3x27xf32> {
  %1 = tensor.extract_patches(%0) {window = [3, 3], strides = [2, 2], dilation = [1, 1], padding = [(1, 1), (1, 1)], pad_value = 0.0} : (tensor<1x5x5x3xf32>) -> tensor<1x3x3x27xf32>
  func.return(%1) : (tensor<1x3x3x27xf32>) -> ()
}
```

Rejected, as the verify filetests pin it:

- `tensor.extract_patches: tensor<{N}x4x4x3xf32> has a dynamic dimension`
- `tensor.extract_patches: input must be [N, spatial..., C], got tensor<4x4xf32>`
- `tensor.extract_patches: window [3] has rank 1, but the input has 2 spatial axes`
- `tensor.extract_patches: strides [1, 1, 1] has rank 3, but the input has 2 spatial axes`
- `tensor.extract_patches: dilation [1] has rank 1, but the input has 2 spatial axes`
- `tensor.extract_patches: padding [(0, 0)] has rank 1, but the input has 2 spatial axes`
- `tensor.extract_patches: input tensor<1x4x4x0xf32> has an empty spatial or channel axis`
- `tensor.extract_patches: window [0, 3] must all be positive`
- `tensor.extract_patches: strides [1, 0] must all be positive`
- `tensor.extract_patches: dilation [0, 1] must all be positive`
- `tensor.extract_patches: patch size overflows`
- `tensor.extract_patches: pad_value 0.5 is not representable in si32`

### `tensor.reduce_window_{kind}` (`ReduceWindow`)

Reduces every window of the
padded, dilated input, the pooling primitive.

- Each attribute has one entry per axis: `window_dims` sizes the window, `strides` steps it,
  `padding` pads the input with `(low, high)` positions, and `base_dilation` and
  `window_dilation` space the input and the window elements. All but `padding` are positive.
- Along an axis of extent `n`, the dilated, padded input spans
  `padded = (n - 1) * base_dilation + 1 + low + high` positions, the dilated window
  `span = (window - 1) * window_dilation + 1`, and the result has
  `(padded - span) / stride + 1` positions.
- Padding and the holes between dilated elements add nothing to a window.
- It reduces with `kind` and accumulates in `accum` as `tensor.reduce_*` does, by the
  accumulator policy; the result dtype is `out`, by default the input's.

Defined by `ReduceWindow` in [crates/gpt-rs-ir/src/dialects/tensor/ir/window.rs](../../crates/gpt-rs-ir/src/dialects/tensor/ir/window.rs).

| | |
|---|---|
| Operands | `input`: tensor |
| Results | `output`: tensor |
| Attributes | `window_dims`: `[usize]`; `strides`: `[usize]`; `padding`: `[(usize, usize)]`; `base_dilation`: `[usize]`; `window_dilation`: `[usize]`; `accum`: `dtype`, optional; `out`: `dtype`, optional |
| Effects | none |
| Patterns | `DropDefaultWindowAccum` (`canonicalize`) |

| Kind | |
|---|---|
| `sum` | Folds with `add`. The initial value is 0. |
| `max` | Folds with `maximum`. The initial value is `-inf` for floats and the smallest value of the dtype for integers. |
| `min` | Folds with `minimum`. The initial value is `+inf` for floats and the largest value of the dtype for integers. |
| `and` | Folds with `and`. The initial value has every bit set, true for `i1`, so on `i1` it is "all". |
| `or` | Folds with `or`. The initial value is 0, so on `i1` it is "any". |

Example, from `filetests/tensor/roundtrip/window.ptir`:

```text
func.func @window_kinds(%0: tensor<6xi1>, %1: tensor<6xbf16>) -> (tensor<5xi1>, tensor<5xi1>, tensor<5xf32>) {
  %2 = tensor.reduce_window_and(%0) {window_dims = [2], strides = [1], padding = [(0, 0)], base_dilation = [1], window_dilation = [1]} : (tensor<6xi1>) -> tensor<5xi1>
  %3 = tensor.reduce_window_or(%0) {window_dims = [2], strides = [1], padding = [(0, 0)], base_dilation = [1], window_dilation = [1]} : (tensor<6xi1>) -> tensor<5xi1>
  %4 = tensor.reduce_window_sum(%1) {window_dims = [2], strides = [1], padding = [(0, 0)], base_dilation = [1], window_dilation = [1], out = f32} : (tensor<6xbf16>) -> tensor<5xf32>
  func.return(%2, %3, %4) : (tensor<5xi1>, tensor<5xi1>, tensor<5xf32>) -> ()
}
```

Rejected, as the verify filetests pin it:

- `tensor.reduce_window_max: tensor<{N}x4xf32> has a dynamic dimension`
- `tensor.reduce_window_max: window_dims [2] has rank 1, but the input has rank 2`
- `tensor.reduce_window_max: strides [1] has rank 1, but the input has rank 2`
- `tensor.reduce_window_max: padding [(0, 0), (0, 0), (0, 0)] has rank 3, but the input has rank 2`
- `tensor.reduce_window_max: base_dilation [] has rank 0, but the input has rank 2`
- `tensor.reduce_window_max: window_dilation [1] has rank 1, but the input has rank 2`
- `tensor.reduce_window_sum: window_dims [2, 0] must all be positive`
- `tensor.reduce_window_sum: strides [0, 1] must all be positive`
- `tensor.reduce_window_sum: base_dilation [1, 0] must all be positive`
- `tensor.reduce_window_sum: window_dilation [0, 1] must all be positive`
- `tensor.reduce_window_sum: accum must be a float dtype, got i1`
- `tensor.reduce_window_max: the input must be an integer or float tensor, got tensor<4xi1>`
- `tensor.reduce_window_min: the dilated input overflows on axis 0`
- `tensor.reduce_window_min: the padded input overflows on axis 0`
- `tensor.reduce_window_min: the dilated window overflows on axis 0`
- `tensor.reduce_window_min: window extent 3 exceeds padded input extent 2 on axis 0`

### `tensor.custom_call`

Runs the backend kernel `target` on operands that the target defines, with
one result per entry of `results`. The result types are attributes, since only the rewrite that
creates the call knows them.

- Effects: a custom call has effects that the IR does not describe. Passes MUST NOT merge,
  precompute or erase a custom call, even when nothing reads its results. Backends MAY cache
  internal compiled kernels, packed weights, or plans.
- Portability: if a backend does not recognize `target`, it MUST reject it as unimplemented
  (no silent fallback to a portable decomposition). The interpreter has no semantics for it.
- Typing: `results` is authoritative. Backends MAY validate that operand dtypes and shapes and
  `attrs` agree with it.
- Stable naming: `target` strings MUST be namespaced and versioned, such as
  `<namespace>.conv2d.nhwc.f32.v1`.
- Attributes: `attrs` is a small serializable map of primitives (integers, floats, bools and
  strings) and arrays of primitives. Backends SHOULD reject unknown or malformed attributes as
  a spec violation.

Defined by `CustomCall` in [crates/gpt-rs-ir/src/dialects/tensor/ir/custom_call.rs](../../crates/gpt-rs-ir/src/dialects/tensor/ir/custom_call.rs).

| | |
|---|---|
| Operands | `inputs`: tensors |
| Results | `outputs`: tensors |
| Attributes | `target`: `String`; `attrs`: `BTreeMap<String,CustomAttr>`, optional; `results`: `[tensor type]` |
| Effects | opaque |

Example, from `filetests/tensor/roundtrip/custom_call.ptir`:

```text
func.func @call(%0: tensor<8x16xf32>, %1: tensor<16xf32>) -> tensor<8x16xf32> {
  %2 = tensor.custom_call(%0, %1) {target = "gpt_rs.c.conv2d.nhwc.f32.v1", attrs = {alpha = 1.5, "conv.stride" = array<i64: 1, 1>, fused = true, name = "conv"}, results = [tensor<8x16xf32>]} : (tensor<8x16xf32>, tensor<16xf32>) -> tensor<8x16xf32>
  func.return(%2) : (tensor<8x16xf32>) -> ()
}
```

Rejected, as the verify filetests pin it:

- `tensor.custom_call: the target is empty`

### `tensor.stop_gradient`

Its input, unchanged, as a derivative barrier.

- The forward value is identical to the input.
- Autodiff MUST treat it as a derivative barrier.
- Passes MUST keep it until autodiff has run, including through lowering to lower levels when
  autodiff runs there, so no pass folds it away. Backends that run no autodiff MAY treat it as a
  pure no-op.

Defined by `StopGradient` in [crates/gpt-rs-ir/src/dialects/tensor/ir/elementwise.rs](../../crates/gpt-rs-ir/src/dialects/tensor/ir/elementwise.rs).

| | |
|---|---|
| Operands | `input`: tensor |
| Results | `output`: tensor |
| Effects | none |

Example, from `filetests/tensor/roundtrip/elementwise.ptir`:

```text
func.func @stop_gradient(%0: tensor<2x{T}xf32>) -> tensor<2x{T}xf32> {
  %1 = tensor.stop_gradient(%0) : (tensor<2x{T}xf32>) -> tensor<2x{T}xf32>
  func.return(%1) : (tensor<2x{T}xf32>) -> ()
}
```

### `tensor.argmax`

The index of the largest element along `axis`, a negative axis counting from
the end.

- Ties break toward the smallest index. NaN is larger than every number, as in Torch.
- If `keepdims` is true, `axis` remains with extent 1; otherwise it is removed. The indices
  have the dtype `out`, `si32` or `si64`.
- If `axis` has extent 0, backends MUST reject the op as a spec violation.

Defined by `ArgMax` in [crates/gpt-rs-ir/src/dialects/tensor/ir/contraction.rs](../../crates/gpt-rs-ir/src/dialects/tensor/ir/contraction.rs).

| | |
|---|---|
| Operands | `input`: tensor |
| Results | `output`: tensor |
| Attributes | `axis`: `i64`; `keepdims`: `bool`, optional; `out`: `dtype` |
| Effects | none |
| Patterns | `NonNegativeAxis` (`canonicalize`) |

Example, from `filetests/tensor/roundtrip/contraction.ptir`:

```text
func.func @argmax(%0: tensor<2x5xf32>, %1: tensor<4x{T}xsi32>) -> (tensor<2xsi32>, tensor<4x1xsi64>) {
  %2 = tensor.argmax(%0) {axis = 1, out = si32} : (tensor<2x5xf32>) -> tensor<2xsi32>
  %3 = tensor.argmax(%1) {axis = 1, keepdims = true, out = si64} : (tensor<4x{T}xsi32>) -> tensor<4x1xsi64>
  func.return(%2, %3) : (tensor<2xsi32>, tensor<4x1xsi64>) -> ()
}
```

Rejected, as the verify filetests pin it:

- `tensor.argmax: axis 2 is out of range for rank 2`
- `tensor.argmax: indices must be si32 or si64, got f32`
- `tensor.argmax: axis 1 of tensor<2x0xf32> is empty`

### `tensor.segment_{kind}` (`Segment`)

Reduces the rows of `data` into
`num_segments` rows, row `i` into row `segment_ids[i]`.

- `segment_ids` holds one `si32` or `si64` id per row of `data`, along its leading axis. The
  result has the shape of `data` with `num_segments` rows: `[num_segments, ...]`.
- Each id MUST lie in `[0, num_segments)`. A segment that no row reaches holds the initial
  value of `kind`, as `ReduceKind` states.
- It accumulates in the default accumulator of the dtype of `data` and returns that dtype.
- `indices_are_sorted` and `unique` state facts about the ids that MUST hold when set, so
  kernels MAY rely on them.
- An extension op: a backend MAY reject it as unimplemented.

Defined by `Segment` in [crates/gpt-rs-ir/src/dialects/tensor/ir/contraction.rs](../../crates/gpt-rs-ir/src/dialects/tensor/ir/contraction.rs).

| | |
|---|---|
| Operands | `data`: tensor; `segment_ids`: tensor |
| Results | `output`: tensor |
| Attributes | `num_segments`: `usize`; `indices_are_sorted`: `bool`, optional; `unique`: `bool`, optional |
| Effects | none |

| Kind | |
|---|---|
| `sum` | Folds with `add`. The initial value is 0. |
| `max` | Folds with `maximum`. The initial value is `-inf` for floats and the smallest value of the dtype for integers. |
| `min` | Folds with `minimum`. The initial value is `+inf` for floats and the largest value of the dtype for integers. |
| `and` | Folds with `and`. The initial value has every bit set, true for `i1`, so on `i1` it is "all". |
| `or` | Folds with `or`. The initial value is 0, so on `i1` it is "any". |

Example, from `filetests/tensor/roundtrip/contraction.ptir`:

```text
func.func @segment(%0: tensor<6x4xf32>, %1: tensor<6xsi32>) -> (tensor<3x4xf32>, tensor<3x4xf32>, tensor<8x4xf32>) {
  %2 = tensor.segment_sum(%0, %1) {num_segments = 3} : (tensor<6x4xf32>, tensor<6xsi32>) -> tensor<3x4xf32>
  %3 = tensor.segment_max(%0, %1) {num_segments = 3, indices_are_sorted = true} : (tensor<6x4xf32>, tensor<6xsi32>) -> tensor<3x4xf32>
  %4 = tensor.segment_min(%0, %1) {num_segments = 8, indices_are_sorted = true, unique = true} : (tensor<6x4xf32>, tensor<6xsi32>) -> tensor<8x4xf32>
  func.return(%2, %3, %4) : (tensor<3x4xf32>, tensor<3x4xf32>, tensor<8x4xf32>) -> ()
}
```

Rejected, as the verify filetests pin it:

- `tensor.segment_sum: segment ids must be si32 or si64 with one id per row of tensor<6x4xf32>, got tensor<5xsi32>`
- `tensor.segment_sum: data must have rank at least 1, got tensor<f32>`

### `tensor.top_k`

The top-k values along `axis` and their indices.

- With `largest`, the `k` largest elements along `axis`, largest first; otherwise the `k`
  smallest, smallest first. A negative axis counts from the end.
- Of equal elements, the one with the smaller index comes first, and NaN is larger than every
  number.
- The values have the input's shape with extent `k` on `axis`, and their indices along `axis`
  have the same shape and the dtype that the attribute `indices` names, `si32` or `si64`.
- An extension op: a backend MAY reject it as unimplemented.

Defined by `TopK` in [crates/gpt-rs-ir/src/dialects/tensor/ir/contraction.rs](../../crates/gpt-rs-ir/src/dialects/tensor/ir/contraction.rs).

| | |
|---|---|
| Operands | `input`: tensor |
| Results | `values`: tensor; `indices`: tensor |
| Attributes | `k`: `usize`; `axis`: `i64`; `largest`: `bool`; `indices`: `dtype` |
| Effects | none |
| Patterns | `NonNegativeAxis` (`canonicalize`) |

Example, from `filetests/tensor/roundtrip/contraction.ptir`:

```text
func.func @top_k(%0: tensor<2x5xf32>) -> (tensor<2x3xf32>, tensor<2x3xsi32>, tensor<1x5xf32>, tensor<1x5xsi64>) {
  %1, %2 = tensor.top_k(%0) {k = 3, axis = 1, largest = true, indices = si32} : (tensor<2x5xf32>) -> (tensor<2x3xf32>, tensor<2x3xsi32>)
  %3, %4 = tensor.top_k(%0) {k = 1, axis = 0, largest = false, indices = si64} : (tensor<2x5xf32>) -> (tensor<1x5xf32>, tensor<1x5xsi64>)
  func.return(%1, %2, %3, %4) : (tensor<2x3xf32>, tensor<2x3xsi32>, tensor<1x5xf32>, tensor<1x5xsi64>) -> ()
}
```

Rejected, as the verify filetests pin it:

- `tensor.top_k: k = 6 exceeds axis 1 of tensor<2x5xf32>`
- `tensor.top_k: indices must be si32 or si64, got f32`

### `tensor.gather`

`torch.gather`. Element `i` of the result is element `i` of `input` with its
coordinate on `axis` replaced by `indices[i]`.

- `indices` are `si32` or `si64` and MUST have the rank of the input and its extents on every
  axis but `axis`, a negative axis counting from the end; along `axis` they MAY differ.
- The result has the shape of `indices`.
- Each index MUST satisfy `0 <= index < in_shape[axis]`.

Defined by `Gather` in [crates/gpt-rs-ir/src/dialects/tensor/ir/indexing.rs](../../crates/gpt-rs-ir/src/dialects/tensor/ir/indexing.rs).

| | |
|---|---|
| Operands | `input`: tensor; `indices`: tensor |
| Results | `output`: tensor |
| Attributes | `axis`: `i64` |
| Effects | none |
| Patterns | `NonNegativeAxis` (`canonicalize`) |

Example, from `filetests/tensor/roundtrip/indexing.ptir`:

```text
func.func @gather(%0: tensor<3x4xf32>, %1: tensor<3x2xsi32>, %2: tensor<5x4xsi32>) -> (tensor<3x2xf32>, tensor<5x4xf32>) {
  %3 = tensor.gather(%0, %1) {axis = 1} : (tensor<3x4xf32>, tensor<3x2xsi32>) -> tensor<3x2xf32>
  %4 = tensor.gather(%0, %2) {axis = 0} : (tensor<3x4xf32>, tensor<5x4xsi32>) -> tensor<5x4xf32>
  func.return(%3, %4) : (tensor<3x2xf32>, tensor<5x4xf32>) -> ()
}
```

Rejected, as the verify filetests pin it:

- `tensor.gather: indices must be si32 or si64, got tensor<3x2xf32>`
- `tensor.gather: indices tensor<2x2xsi32> must match tensor<3x4xf32> on every axis but 1`
- `tensor.gather: axis 2 is out of range for rank 2`

### `tensor.scatter_{kind}` (`Scatter`)

`torch.scatter_reduce` of the same-rank form.

- The result is `input` with element `i` of `updates` combined by `kind` into the element whose
  coordinate on `axis` is `indices[i]`, in row-major order of `updates`.
- `indices` are `si32` or `si64`, and `updates` MUST have their shape and the input's dtype.
  Both MUST have the input's rank and its extents on every axis but `axis`, a negative axis
  counting from the end; along `axis` they MAY differ.
- Each index MUST be in range.
- `scatter_add`, `scatter_max` and `scatter_min` accumulate in the default accumulator of the
  dtype, then return the dtype.

Defined by `Scatter` in [crates/gpt-rs-ir/src/dialects/tensor/ir/indexing.rs](../../crates/gpt-rs-ir/src/dialects/tensor/ir/indexing.rs).

| | |
|---|---|
| Operands | `input`: tensor; `indices`: tensor; `updates`: tensor |
| Results | `output`: tensor |
| Attributes | `axis`: `i64` |
| Effects | none |
| Patterns | `NonNegativeAxis` (`canonicalize`) |

| Kind | |
|---|---|
| `add` | Accumulates the sum of the element and its updates. |
| `max` | Accumulates the maximum of the element and its updates. |
| `min` | Accumulates the minimum of the element and its updates. |
| `replace` | Replaces the element: of several updates to one element, the last in row-major order of `updates` wins. |

Example, from `filetests/tensor/roundtrip/indexing.ptir`:

```text
func.func @scatter(%0: tensor<3x4xf32>, %1: tensor<3x2xsi32>, %2: tensor<3x2xf32>) -> (tensor<3x4xf32>, tensor<3x4xf32>, tensor<3x4xf32>, tensor<3x4xf32>) {
  %3 = tensor.scatter_add(%0, %1, %2) {axis = 1} : (tensor<3x4xf32>, tensor<3x2xsi32>, tensor<3x2xf32>) -> tensor<3x4xf32>
  %4 = tensor.scatter_max(%0, %1, %2) {axis = 1} : (tensor<3x4xf32>, tensor<3x2xsi32>, tensor<3x2xf32>) -> tensor<3x4xf32>
  %5 = tensor.scatter_min(%0, %1, %2) {axis = 1} : (tensor<3x4xf32>, tensor<3x2xsi32>, tensor<3x2xf32>) -> tensor<3x4xf32>
  %6 = tensor.scatter_replace(%0, %1, %2) {axis = 1} : (tensor<3x4xf32>, tensor<3x2xsi32>, tensor<3x2xf32>) -> tensor<3x4xf32>
  func.return(%3, %4, %5, %6) : (tensor<3x4xf32>, tensor<3x4xf32>, tensor<3x4xf32>, tensor<3x4xf32>) -> ()
}
```

Rejected, as the verify filetests pin it:

- `tensor.scatter_add: updates must have the dtype of tensor<3x4xf32> and the shape of tensor<3x2xsi32>, got tensor<3x3xf32>`

### `tensor.rng_{distribution}` (`Rng`)

A tensor of `shape` whose elements are random, and
the state after them.

- The ops are pure: one state gives the same values on every backend.
- A state, `ui64[2]`, is a key and a counter: `[key, counter]`. Its stream is the output of
  Philox4x32-10 (Random123). Counter block `j` is the 128-bit number `counter + j`, split into
  four 32-bit words from the least significant one up. The key splits into two words the same
  way. Each block gives four 32-bit words. An op reads the first words of its stream and
  returns `[key, counter + blocks read]`, the counter wrapping modulo `2^64`.
- Each element takes the next `ceil(bits / 32)` words, the first one least significant, where
  `bits` is the width of `dtype`, or the significand of a uniform float.
- A uniform integer is the low `bits` of its words: each value of the dtype equally likely.
- A uniform float with a significand of `p` bits (`bf16`: 8, `f16`: 11, `f32`: 24, `f64`: 53)
  is the top `p` bits of its words times `2^-p`: on `[0, 1)`, and exact in its dtype.
- A normal float takes two uniforms `u1` and `u2` of its dtype's precision, in that order, and
  is `sqrt(-2 ln(1 - u1)) * cos(2 pi u2)`, rounded to its dtype.
- Uniform `ui32` values are the draws of stochastic rounding (`tensor.quantize`).
- An extension op: a backend MAY reject it as unimplemented.

Defined by `Rng` in [crates/gpt-rs-ir/src/dialects/tensor/ir/random.rs](../../crates/gpt-rs-ir/src/dialects/tensor/ir/random.rs).

| | |
|---|---|
| Operands | `state`: tensor |
| Results | `new_state`: tensor; `output`: tensor |
| Attributes | `shape`: `[dim]`; `dtype`: `dtype` |
| Effects | none |

| Kind | |
|---|---|
| `uniform` | Each value of an integer dtype equally likely, or a float on `[0, 1)`. |
| `normal` | The standard normal distribution, of a float dtype. |

Example, from `filetests/tensor/roundtrip/random.ptir`:

```text
func.func @rng(%0: tensor<2xui64>) -> (tensor<2x3xf32>, tensor<4xbf16>, tensor<2xui64>, tensor<4xui32>) {
  %1, %2 = tensor.rng_uniform(%0) {shape = [2, 3], dtype = f32} : (tensor<2xui64>) -> (tensor<2xui64>, tensor<2x3xf32>)
  %3, %4 = tensor.rng_normal(%1) {shape = [4], dtype = bf16} : (tensor<2xui64>) -> (tensor<2xui64>, tensor<4xbf16>)
  %5, %6 = tensor.rng_uniform(%3) {shape = [4], dtype = ui32} : (tensor<2xui64>) -> (tensor<2xui64>, tensor<4xui32>)
  func.return(%2, %4, %5, %6) : (tensor<2x3xf32>, tensor<4xbf16>, tensor<2xui64>, tensor<4xui32>) -> ()
}
```

Rejected, as the verify filetests pin it:

- `tensor.rng_normal: dtype must be a float dtype, got si32`
- `tensor.rng_uniform: dtype must be an integer or float dtype, got i1`
- `tensor.rng_uniform: the state must be tensor<2xui64>, got tensor<2xsi64>`

### `tensor.yield`

Ends a region of a control-flow op. Its operands are the region's results.

- Each region of a control-flow op has its own arguments and ends with a `tensor.yield`. The
  ops compute on whole tensor values, like the rest of the dialect.

Defined by `Yield` in [crates/gpt-rs-ir/src/dialects/tensor/ir/control.rs](../../crates/gpt-rs-ir/src/dialects/tensor/ir/control.rs).

| | |
|---|---|
| Operands | `values`: tensors |
| Results | none |
| Traits | terminator |
| Effects | none |

Example, from `filetests/tensor/roundtrip/control.ptir`:

```text
func.func @scan(%0: tensor<4xf32>, %1: tensor<6x4xf32>) -> (tensor<4xf32>, tensor<6x4xf32>) {
  %2, %3 = tensor.scan(%0, %1) {carry_count = 1, outputs = [tensor<4xf32>]} ({
  ^body(%4: tensor<4xf32>, %5: tensor<4xf32>):
    %6 = tensor.add(%4, %5) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
    tensor.yield(%6, %6) : (tensor<4xf32>, tensor<4xf32>) -> ()
  }) : (tensor<4xf32>, tensor<6x4xf32>) -> (tensor<4xf32>, tensor<6x4xf32>)
  func.return(%2, %3) : (tensor<4xf32>, tensor<6x4xf32>) -> ()
}
```

### `tensor.cond`

Runs `then_branch` on `args` when `pred` holds and `else_branch` otherwise,
and gives the results that the branch yields, of the types `results`.

- `pred` is an `i1` scalar, `i1[]`.
- Both branches take arguments of the types of `args` and MUST yield the types `results`.

Defined by `Cond` in [crates/gpt-rs-ir/src/dialects/tensor/ir/control.rs](../../crates/gpt-rs-ir/src/dialects/tensor/ir/control.rs).

| | |
|---|---|
| Operands | `pred`: tensor; `args`: tensors |
| Results | `outputs`: tensors |
| Regions | `then_branch`, `else_branch` |
| Attributes | `results`: `[tensor type]` |
| Effects | none |

Example, from `filetests/tensor/roundtrip/control.ptir`:

```text
func.func @cond(%0: tensor<i1>, %1: tensor<4xf32>) -> tensor<4xf32> {
  %2 = tensor.cond(%0, %1) {results = [tensor<4xf32>]} ({
  ^then_branch(%3: tensor<4xf32>):
    %4 = tensor.neg(%3) : (tensor<4xf32>) -> tensor<4xf32>
    tensor.yield(%4) : (tensor<4xf32>) -> ()
  }, {
  ^else_branch(%5: tensor<4xf32>):
    tensor.yield(%5) : (tensor<4xf32>) -> ()
  }) : (tensor<i1>, tensor<4xf32>) -> tensor<4xf32>
  func.return(%2) : (tensor<4xf32>) -> ()
}
```

Rejected, as the verify filetests pin it:

- `tensor.cond: the predicate must be an i1 scalar, got tensor<2xi1>`
- `tensor.cond: region 1 yields (tensor<2x2xf32>), expected (tensor<4xf32>)`

### `tensor.while`

A loop over the carried values `init`.

- `cond` takes the carried values and yields an `i1` scalar and new carried values:
  `cond(carry...) -> (i1[], carry...)`.
- If the scalar is true, `body` runs on the carried values `cond` yielded, `body(carry...) ->
  carry...`, and the loop repeats with the values `body` yields.
- If the scalar is false, the results are the carried values `cond` yielded.

Defined by `While` in [crates/gpt-rs-ir/src/dialects/tensor/ir/control.rs](../../crates/gpt-rs-ir/src/dialects/tensor/ir/control.rs).

| | |
|---|---|
| Operands | `init`: tensors |
| Results | `outputs`: tensors |
| Regions | `cond`, `body` |
| Effects | none |

Example, from `filetests/tensor/roundtrip/control.ptir`:

```text
func.func @while(%0: tensor<si32>, %1: tensor<4xf32>) -> (tensor<si32>, tensor<4xf32>) {
  %2, %3 = tensor.while(%0, %1) ({
  ^cond(%4: tensor<si32>, %5: tensor<4xf32>):
    %6 = tensor.constant() {value = dense<[] x si32: 10>} : () -> tensor<si32>
    %7 = tensor.compare(%4, %6) {kind = less} : (tensor<si32>, tensor<si32>) -> tensor<i1>
    tensor.yield(%7, %4, %5) : (tensor<i1>, tensor<si32>, tensor<4xf32>) -> ()
  }, {
  ^body(%8: tensor<si32>, %9: tensor<4xf32>):
    %10 = tensor.constant() {value = dense<[] x si32: 1>} : () -> tensor<si32>
    %11 = tensor.add(%8, %10) : (tensor<si32>, tensor<si32>) -> tensor<si32>
    %12 = tensor.add(%9, %9) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
    tensor.yield(%11, %12) : (tensor<si32>, tensor<4xf32>) -> ()
  }) : (tensor<si32>, tensor<4xf32>) -> (tensor<si32>, tensor<4xf32>)
  func.return(%2, %3) : (tensor<si32>, tensor<4xf32>) -> ()
}
```

Rejected, as the verify filetests pin it:

- `tensor.while: region 0 yields (tensor<4xf32>), expected (tensor<i1>, tensor<4xf32>)`

### `tensor.scan`

A loop over the leading axis of the scanned inputs, like a functional for-loop.

- Its operands are `carry_count` carried values, then the scanned inputs, each `[T, ...]`:
  at least one, all with the same leading extent `T`.
- Step `t` runs `body` on the carried values and slice `t` of each scanned input, and `body`
  yields the next carried values, then one output of each type in `outputs`.
- The results are the final carried values, then each output stacked along a new leading axis
  over the steps, `[T, ...]`.

Defined by `Scan` in [crates/gpt-rs-ir/src/dialects/tensor/ir/control.rs](../../crates/gpt-rs-ir/src/dialects/tensor/ir/control.rs).

| | |
|---|---|
| Operands | `inputs`: tensors |
| Results | `outputs`: tensors |
| Regions | `body` |
| Attributes | `carry_count`: `usize`; `outputs`: `[tensor type]` |
| Effects | none |

Example, from `filetests/tensor/roundtrip/control.ptir`:

```text
func.func @scan(%0: tensor<4xf32>, %1: tensor<6x4xf32>) -> (tensor<4xf32>, tensor<6x4xf32>) {
  %2, %3 = tensor.scan(%0, %1) {carry_count = 1, outputs = [tensor<4xf32>]} ({
  ^body(%4: tensor<4xf32>, %5: tensor<4xf32>):
    %6 = tensor.add(%4, %5) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
    tensor.yield(%6, %6) : (tensor<4xf32>, tensor<4xf32>) -> ()
  }) : (tensor<4xf32>, tensor<6x4xf32>) -> (tensor<4xf32>, tensor<6x4xf32>)
  func.return(%2, %3) : (tensor<4xf32>, tensor<6x4xf32>) -> ()
}
```

Rejected, as the verify filetests pin it:

- `tensor.scan: the scanned inputs must share their leading extent, got tensor<5x4xf32>`
- `tensor.scan: carry_count 2 exceeds the 1 operands`

### `tensor.dequantize`

`out[i] = to(value(q[i]) * scale[i / block] + offset[i / block])`,
computed in f32, then rounded to nearest-even in `to`.

- `to` is `f32`, `bf16` or `f16`. The scale and the offset follow the scale layout of
  "Quantization" in the dialect documentation.
- `value(q)` is `table[q]` with a table, whose codes `q` are integers in `[0, n)`, and the value
  of `q` otherwise. Without an offset, the sum adds nothing.
- The offset carries asymmetric quantization: the zero points of GPTQ and AWQ as
  `offset = -zero_point * scale`, computed once at load time, and the block minimums of GGUF.
- A backend SHOULD read `q` and `scale` in place inside the op that consumes the result, such
  as a `dot_general` with a quantized weight.
- Autodiff MUST differentiate it as multiplication by the scale, so that
  `dequantize(quantize(x))` passes gradients through unchanged.
- An extension op: a backend MAY reject it as unimplemented.

Defined by `Dequantize` in [crates/gpt-rs-ir/src/dialects/tensor/ir/quant.rs](../../crates/gpt-rs-ir/src/dialects/tensor/ir/quant.rs).

| | |
|---|---|
| Operands | `q`: tensor; `scale`: tensor; `offset`: optional tensor; `table`: optional tensor |
| Results | `output`: tensor |
| Attributes | `block`: `[usize]`; `to`: `dtype` |
| Effects | none |

Example, from `filetests/tensor/roundtrip/quant.ptir`:

```text
func.func @mxfp4(%0: tensor<4x64xf4e2m1>, %1: tensor<4x2xe8m0>) -> tensor<4x64xbf16> {
  %2 = tensor.dequantize(%0, %1) {block = [1, 32], to = bf16} : (tensor<4x64xf4e2m1>, tensor<4x2xe8m0>) -> tensor<4x64xbf16>
  func.return(%2) : (tensor<4x64xbf16>) -> ()
}
```

Rejected, as the verify filetests pin it:

- `tensor.dequantize: the scale must have the extents [4, 2] for block [1, 32], got tensor<4x3xe8m0>`
- `tensor.dequantize: q must be an integer or narrow float tensor, got tensor<4x64xf32>`
- `tensor.dequantize: codes with a table must be an integer dtype, got f4e2m1`
- `tensor.dequantize: the table must have rank 1, got tensor<4x4xf32>`
- `tensor.dequantize: the offset must have the scale's extents, got tensor<4x1xf16> and tensor<4x2xf16>`
- `tensor.dequantize: the offset must be a float tensor, got tensor<4x2xe8m0>`

### `tensor.quantize`

Quantizes `x` into `to` with given scales and offsets, or into the indices
of a table.

- `x` is `f32`, `bf16` or `f16`, and `scale` holds static scales, for example parameters. The
  scale and the offset follow the scale layout of "Quantization" in the dialect documentation.
- `y[i] = (x[i] - offset[i / block]) / scale[i / block]`, computed in f32. A zero scale gives
  `y[i] = 0`, so without a table `q[i] = 0`.
- Without a table, `q[i] = to(y[i])`, rounded as `rounding` says and saturated to the largest
  finite value of `to`. Infinities saturate. NaN maps to NaN in a dtype that has one, and to 0
  otherwise.
- With a table, `q[i]` is the index of the table value nearest to `y[i]`, the lowest index on
  a tie, and 0 for NaN. `to` MUST be an integer dtype and `rounding` `nearest_even`. NF4 weights
  and 8-bit optimizer states quantize this way.
- `random` is present exactly when `rounding` is `stochastic`, and holds one `ui32` draw per
  element of `x`. The draws come from `tensor.rng_uniform` with `dtype = ui32`, so a state
  reproduces the rounding.
- Autodiff MUST differentiate it as division by the scale, ignoring the rounding: the
  straight-through estimator of quantization-aware training.
- An extension op: a backend MAY reject it as unimplemented.

Defined by `Quantize` in [crates/gpt-rs-ir/src/dialects/tensor/ir/quant.rs](../../crates/gpt-rs-ir/src/dialects/tensor/ir/quant.rs).

| | |
|---|---|
| Operands | `x`: tensor; `scale`: tensor; `offset`: optional tensor; `table`: optional tensor; `random`: optional tensor |
| Results | `output`: tensor |
| Attributes | `block`: `[usize]`; `to`: `dtype`; `rounding`: `Rounding`, optional |
| Effects | none |

| `rounding` | |
|---|---|
| `nearest_even` | The nearest value of the quantized dtype, the even one on a tie. |
| `stochastic` | Rounds `abs(y)` to one of its two neighbors `lo` and `lo + q` in the quantized dtype, where `q` is their spacing. It picks `lo + q` when the element's draw `r` from the `random` operand satisfies `r < floor((abs(y) - lo) / q * 2^32)`, else `lo`, and keeps the sign of `y`. So rounding picks `lo + q` with probability `(abs(y) - lo) / q`. FP8 and FP4 training round gradients this way. |

Example, from `filetests/tensor/roundtrip/quant.ptir`:

```text
func.func @quantize_options(%0: tensor<8x16xf32>, %1: tensor<8x1xf32>, %2: tensor<8x16xui32>, %3: tensor<16xf32>) -> (tensor<8x16xf8e4m3>, tensor<8x16xui8>, tensor<8x16xui4>) {
  %4 = tensor.quantize(%0, %1, %2) {block = [1, 16], to = f8e4m3, rounding = stochastic, operand_segments = [1, 1, 0, 0, 1]} : (tensor<8x16xf32>, tensor<8x1xf32>, tensor<8x16xui32>) -> tensor<8x16xf8e4m3>
  %5 = tensor.quantize(%0, %1, %1) {block = [1, 16], to = ui8, operand_segments = [1, 1, 1, 0, 0]} : (tensor<8x16xf32>, tensor<8x1xf32>, tensor<8x1xf32>) -> tensor<8x16xui8>
  %6 = tensor.quantize(%0, %1, %3) {block = [1, 16], to = ui4, operand_segments = [1, 1, 0, 1, 0]} : (tensor<8x16xf32>, tensor<8x1xf32>, tensor<16xf32>) -> tensor<8x16xui4>
  func.return(%4, %5, %6) : (tensor<8x16xf8e4m3>, tensor<8x16xui8>, tensor<8x16xui4>) -> ()
}
```

Rejected, as the verify filetests pin it:

- `tensor.quantize: block [32] must have one extent per axis of rank 2`
- `tensor.quantize: to must be an integer or narrow float dtype, got bf16`
- `tensor.quantize: stochastic rounding needs a random operand`
- `tensor.quantize: only stochastic rounding reads random bits`
- `tensor.quantize: the random bits must be tensor<8x16xui32>, got tensor<8x16xsi32>`
- `tensor.quantize: a table quantizes to the nearest value`

### `tensor.quantize_dynamic`

Computes each block's scale, and with `asymmetric` its offset, from
the block's values, then quantizes as `tensor.quantize` does.

- `x` is `f32`, `bf16` or `f16`. The blocks, and the shape of the scales and offsets, follow
  the scale layout of "Quantization" in the dialect documentation.
- It quantizes `x` as `tensor.quantize` does with the scales and offsets as stored, in
  `scale_dtype`. The scales and offsets ignore NaN elements.
- A symmetric scale comes from the block's largest magnitude `amax`, as `scale_rule` says.
- An asymmetric scale maps the block's range `[lo, hi]`, widened to hold 0, onto the range of
  `to`: `scale = (hi - lo) / (max(to) - min(to))`. The integer zero point is
  `zero_point = round(min(to) - lo / scale)`, rounded to nearest-even and clamped into the range
  of `to`, so that 0 quantizes exactly. The offset is `offset = -zero_point * scale`. `max(to)`
  and `min(to)` are the largest and smallest finite values of `to`. `scale_dtype` is a float,
  and `scale_rule` is `amax`.
- Its results are the quantized tensor `q`, of dtype `to`, the scales, and with `asymmetric`
  the offsets, both of dtype `scale_dtype`.
- `rounding` and `random` are those of `tensor.quantize`.
- An extension op: a backend MAY reject it as unimplemented.

Defined by `QuantizeDynamic` in [crates/gpt-rs-ir/src/dialects/tensor/ir/quant.rs](../../crates/gpt-rs-ir/src/dialects/tensor/ir/quant.rs).

| | |
|---|---|
| Operands | `x`: tensor; `random`: optional tensor |
| Results | `q`: tensor; `scale`: tensor; `offset`: optional tensor |
| Attributes | `block`: `[usize]`; `to`: `dtype`; `scale_dtype`: `dtype`; `scale_rule`: `ScaleRule`, optional; `asymmetric`: `bool`, optional; `rounding`: `Rounding`, optional |
| Effects | none |

| `scale_rule` | |
|---|---|
| `amax` | `scale_dtype(amax / max(to))`, rounded to nearest-even, so `amax` maps to the largest value of `to`; a block whose `amax` is 0 gets the scale 0. NVFP4 takes its `f8e4m3` block scales this way, and per-tensor, per-channel and per-token scales are `f32` or `bf16`. |
| `pow2_floor` | `2^(floor(log2(amax)) - emax)`, the microscaling (MX) rule, where `emax = floor(log2(max(to)))` is the exponent of the largest power of two `to` holds (`f4e2m1`: 2, `f6e2m3`: 2, `f6e3m2`: 4, `f8e4m3`: 8, `f8e5m2`: 15). Values above `max(to)` times the scale saturate. |
| `pow2_ceil` | `2^ceil(log2(amax / max(to)))`, the smallest power of two that maps `amax` into `to`, as DeepSeek's UE8M0 scales are. |

| `rounding` | |
|---|---|
| `nearest_even` | The nearest value of the quantized dtype, the even one on a tie. |
| `stochastic` | Rounds `abs(y)` to one of its two neighbors `lo` and `lo + q` in the quantized dtype, where `q` is their spacing. It picks `lo + q` when the element's draw `r` from the `random` operand satisfies `r < floor((abs(y) - lo) / q * 2^32)`, else `lo`, and keeps the sign of `y`. So rounding picks `lo + q` with probability `(abs(y) - lo) / q`. FP8 and FP4 training round gradients this way. |

Example, from `filetests/tensor/roundtrip/quant.ptir`:

```text
func.func @dynamic(%0: tensor<{T}x64xf32>) -> (tensor<{T}x64xf4e2m1>, tensor<{T}x4xf8e4m3>, tensor<{T}x64xf6e3m2>, tensor<{T}x2xe8m0>) {
  %1, %2 = tensor.quantize_dynamic(%0) {block = [1, 16], to = f4e2m1, scale_dtype = f8e4m3} : (tensor<{T}x64xf32>) -> (tensor<{T}x64xf4e2m1>, tensor<{T}x4xf8e4m3>)
  %3, %4 = tensor.quantize_dynamic(%0) {block = [1, 32], to = f6e3m2, scale_dtype = e8m0, scale_rule = pow2_floor} : (tensor<{T}x64xf32>) -> (tensor<{T}x64xf6e3m2>, tensor<{T}x2xe8m0>)
  func.return(%1, %2, %3, %4) : (tensor<{T}x64xf4e2m1>, tensor<{T}x4xf8e4m3>, tensor<{T}x64xf6e3m2>, tensor<{T}x2xe8m0>) -> ()
}
```

Rejected, as the verify filetests pin it:

- `tensor.quantize_dynamic: e8m0 scales are powers of two: they need scale_rule pow2_floor or pow2_ceil`
- `tensor.quantize_dynamic: a block of 2 needs a static extent, got {T}`
- `tensor.quantize_dynamic: an asymmetric scale_dtype must be a float dtype, got e8m0`
- `tensor.quantize_dynamic: an asymmetric scale maps the block's range, not a power of two`

## Traits

A trait states a fact about the ops that have it. The folds and patterns it implies apply to every one of them; [passes.md](passes.md) says which pass runs each.

### `Absorbing`

`f(a, z) = z`, for the integers and `i1`.

Defined by `Absorbing` in [crates/gpt-rs-ir/src/traits/algebra.rs](../../crates/gpt-rs-ir/src/traits/algebra.rs).

Ops: `tensor.mul`, `tensor.maximum`, `tensor.minimum`, `tensor.and`, `tensor.or`.

### `Associative`

`f(f(a, b), c) = f(a, f(b, c))`, for integers, and for floats when `float` is set: float
addition and multiplication round differently once regrouped.

Defined by `Associative` in [crates/gpt-rs-ir/src/traits/algebra.rs](../../crates/gpt-rs-ir/src/traits/algebra.rs).

Ops: `tensor.add`, `tensor.mul`, `tensor.maximum`, `tensor.minimum`, `tensor.and`, `tensor.or`, `tensor.xor`.

Patterns rooted at it: `GroupStatic` (`canonicalize`).

### `Commutative`

`f(a, b) = f(b, a)`. Folding puts a run-time operand first and a constant last (see
`Module::is_static`), so the other folds, the patterns and CSE see a constant on the right.

Defined by `Commutative` in [crates/gpt-rs-ir/src/traits/algebra.rs](../../crates/gpt-rs-ir/src/traits/algebra.rs).

Ops: `tensor.add`, `tensor.mul`, `tensor.maximum`, `tensor.minimum`, `tensor.and`, `tensor.or`, `tensor.xor`.

### `DistributesOver`

`f(a, g(b, c)) = g(f(a, b), f(a, c))` for the op `g` that `over` builds, for integers, and for
floats when `float` is set.

Defined by `DistributesOver` in [crates/gpt-rs-ir/src/traits/algebra.rs](../../crates/gpt-rs-ir/src/traits/algebra.rs).

Ops: `tensor.mul`, `tensor.and`, `tensor.or`.

Patterns rooted at it: `Factor` (`canonicalize`).

### `Elementwise`

Each result element depends only on the operand elements at its own position, so every tensor
operand and result has one shape.

Defined by `Elementwise` in [crates/gpt-rs-ir/src/traits/structure.rs](../../crates/gpt-rs-ir/src/traits/structure.rs).

Ops: `tensor.neg`, `tensor.abs`, `tensor.exp`, `tensor.log`, `tensor.tanh`, `tensor.erf`, `tensor.rsqrt`, `tensor.reciprocal`, `tensor.sin`, `tensor.cos`, `tensor.not`, `tensor.add`, `tensor.sub`, `tensor.mul`, `tensor.div`, `tensor.maximum`, `tensor.minimum`, `tensor.and`, `tensor.or`, `tensor.xor`, `tensor.shift_left`, `tensor.shift_right_logical`, `tensor.shift_right_arithmetic`, `tensor.compare`, `tensor.select`, `tensor.cast`.

Patterns rooted at it: `HoistBroadcast` (`canonicalize`).

### `Idempotent`

For a unary op `f(f(a)) = f(a)`, for a binary one `f(a, a) = a`.

Defined by `Idempotent` in [crates/gpt-rs-ir/src/traits/algebra.rs](../../crates/gpt-rs-ir/src/traits/algebra.rs).

Ops: `tensor.abs`, `tensor.maximum`, `tensor.minimum`, `tensor.and`, `tensor.or`.

### `Involution`

`f(f(a)) = a`.

Defined by `Involution` in [crates/gpt-rs-ir/src/traits/algebra.rs](../../crates/gpt-rs-ir/src/traits/algebra.rs).

Ops: `tensor.neg`, `tensor.not`.

### `Movement`

Each result element is an element of one of the data operands: the first `data` operands, or
all of them when `data` is `None`. The op moves elements without computing on them, so a
splat stays a splat. The fold builds that splat only when it holds no more elements than the
operands: a constant stores every element, so a broadcast splat stays a broadcast.

Defined by `Movement` in [crates/gpt-rs-ir/src/traits/structure.rs](../../crates/gpt-rs-ir/src/traits/structure.rs).

Ops: `tensor.reshape`, `tensor.transpose`, `tensor.broadcast_to`, `tensor.slice`, `tensor.concat`, `tensor.take`, `tensor.dynamic_slice`, `tensor.dynamic_update_slice`.

### `RightIdentity`

`f(a, e) = a`. A commutative op has `e` on the left too, since folding moves constants to the
right.

Defined by `RightIdentity` in [crates/gpt-rs-ir/src/traits/algebra.rs](../../crates/gpt-rs-ir/src/traits/algebra.rs).

Ops: `tensor.add`, `tensor.sub`, `tensor.mul`, `tensor.div`, `tensor.maximum`, `tensor.minimum`, `tensor.and`, `tensor.or`, `tensor.xor`, `tensor.shift_left`, `tensor.shift_right_logical`, `tensor.shift_right_arithmetic`.

### `Terminator`

Ends a region: its operands are what the region yields, and it has no results.

Defined by `Terminator` in [crates/gpt-rs-ir/src/traits/structure.rs](../../crates/gpt-rs-ir/src/traits/structure.rs).

Ops: `tensor.yield`.
