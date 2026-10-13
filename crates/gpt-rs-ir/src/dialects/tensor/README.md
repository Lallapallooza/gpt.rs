# The tensor dialect (ptir.v0.6)

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
- Each op is defined in `ir.rs`: its struct, its attributes and its groups. Its type rule
  (`OpDef::result_types`) is in `verify/`. Its semantics, its `Eval` impl, are in `interp/`. Its
  folder is in `fold/`, and its cost is in `cost/`. Each is in the file named after the op's section
  of `ir.rs`. Helpers are in `utils/`. The reference backend runs the semantics, and every pass
  is checked against them.
- Each op's documentation states what the op means, independent of any specific backend
  implementation strategy. The dialect holds no op outside this specification. If an op's
  documentation and its code disagree, fix the documentation or fix the code.
- Notation: `T[shape]` means a tensor of dtype `T` and the given shape.
- Extension ops (`extract_patches`, the random ops, `top_k`, `bottom_k`, the segment ops and the
  quantization ops) go beyond the core ops, for broader model coverage. A backend MAY reject an
  extension op other than `extract_patches` as unimplemented.
- LLM attention is expressed as a **pattern** over core ops (no dedicated attention op), as
  "Canonical library forms" shows.
- Backend implementation notes (how each backend executes PTIR) are in `docs/backends/README.md`.

## Versioning

In the text form, a stored program, such as a dump, starts with the line
`spec_version = "ptir.v0.6"` (`Module::to_versioned_text`). The parser reads a module with that
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
   - A tensor MUST have at most 8 axes. The text form rejects a type of higher rank, and an op
     whose result would have a higher rank is rejected.
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
- Elementwise ops (`add`, `mul`, `less`, `select`, ...) require operands to have the **same
  shape** (after any explicit `broadcast_to`).

`broadcast_to` follows a standard right-aligned compatibility rule:

1. Left-pad the input shape with 1s to the rank of the target shape, which MUST NOT be lower.
2. Each dim is compatible if the sizes match or the input's size is 1.
3. Result dim is the target's.

If shapes are not broadcast-compatible, backends MUST reject the op as a spec violation.

## DTypes and casting

### DType set

The dtype universe is defined by `gpt_rs_ir::ir::DType` and includes:

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
| `less`, `less_equal`, `equal`, `greater_equal`, `greater`, `not_equal` | bool, integer, float |
| `cast` | every dtype in; every dtype but scale out |
| `reduce_{sum,max,min}`, `reduce_window_{sum,max,min}`, `segment_{sum,max,min}`, `scatter_{add,max,min}`, `argmax`, `top_k`, `bottom_k`, `iota` | integer, float |
| `reduce_{and,or}`, `reduce_window_{and,or}`, `segment_{and,or}` | bool, integer |
| `dot_general` | integer with integer; float or narrow float with float or narrow float |
| `rng_uniform` | integer, float; `rng_normal`: float |
| `quantize`, `quantize_stochastic`, `quantize_table`, `dequantize`, `block_scale_*`, `block_scale_range` | as "Quantization" states |

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
- Nothing casts into `e8m0`: the power-of-two `tensor.block_scale_*` ops compute e8m0 scales and
  state how they round them.

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

The quantization ops name the common schemes, so that backends see their structure, and each is
the composition its semantics states:

- `tensor.quantize`, `tensor.quantize_stochastic` and `tensor.quantize_table` quantize with given
  scales, rounding to the nearest value, rounding at random, or to the nearest entry of a code
  table;
- `tensor.dequantize` maps quantized elements, or the values `take` read from a code table, back
  to real values;
- the `tensor.block_scale_*` ops and `tensor.block_scale_range` compute the scales, and offsets,
  of the blocks of a tensor: dynamic quantization is one of them followed by a quantize op.

Each op computes one thing: a rounding or a scale formula is an op of its own, never an attribute
that switches what an op computes. A new format starts as a composition and becomes an op when a
backend needs its structure.

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
that the ops read and write are `f32`, `bf16` or `f16`. The offset is the one optional operand.

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
