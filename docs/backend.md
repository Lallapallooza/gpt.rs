# PTIR Backend Contract (ptir.v0.4)

This document defines the **Portable Tensor IR (PTIR)** contract used between:

- Producers: `crates/gpt-rs/src/ops/ptir/*`, `crates/gpt-rs/src/ops/functional/*`
- Consumers: `crates/gpt-rs/src/backend/spec.rs` + backend crates (e.g. `crates/gpt-rs-backend-ref-cpu`, `crates/gpt-rs-backend-faer`)

Normative keywords: MUST / SHOULD / MAY.

PTIR is designed to be optimization-friendly: LLM attention is expressed as a **pattern** over core ops (no dedicated attention op).

## Versioning

- Serialized PTIR programs MUST set `spec_version = ptir.v0.4`.
- Deserialization rejects other versions.

## Core invariants

1. Pure functional IR
   - Ops are referentially transparent.
   - No observable in-place mutation.

2. Numerical portability (relaxed)
   - Integer and `i1` ops MUST be bit-exact.
   - Floating point results are not required to be bit-identical across backends/hardware.
   - Backends MUST respect dtype contracts, including explicit `accum_dtype` / `out_dtype`.

3. No implicit dtype promotion
   - Elementwise arithmetic operands MUST have the same dtype (frontends insert `cast`).

4. Shapes and dynamic dims
   - Shapes are ordered tuples of non-negative integers.
   - Dims MAY be dynamic (`?X`) at compile time; runtime dims MUST be non-negative and satisfy op constraints.

## Tensor model

### Layout and indexing (row-major)

Unless stated otherwise, tensors are interpreted as dense, contiguous, **row-major** arrays:

- The last dimension is the fastest-varying (C-order).
- `reshape` never changes element order, only the logical view.
- Ops that compute indices (`slice`, `gather`, etc.) use this logical indexing model.

### Values, tuples, and literals

- Each instruction defines one SSA value id (`ValueId`).
- Instruction outputs are typed as either:
  - `tensor<dtype x shape>`, or
  - `tuple<...>` (multiple results).
- Operands may be:
  - `Operand::Value(%id)` for a prior SSA value,
  - `Operand::TupleElement(%tuple[id])` to select a tuple element, or
  - `Operand::Literal(TensorLiteral)` (an inline constant operand).

Note: some backends (including the current reference CPU backend) only support tensor outputs
and do not execute tuple-typed values yet.

### Negative axes

Ops that accept an `axis` (or list of axes) MUST accept negative indices interpreted as `axis + rank`.

## Broadcasting policy

PTIR uses **explicit** broadcasting:

- `broadcast_to` is the only operation that changes shapes by broadcasting.
- Elementwise ops (`add`, `mul`, `compare`, `select`, ...) require operands to have the **same shape**
  (after any explicit `broadcast_to`).

`broadcast_to` follows a standard right-aligned compatibility rule:

1. Left-pad the lower-rank shape with 1s.
2. Each dim is compatible if sizes match or one is 1.
3. Result dim is the max of the two.

If shapes are not broadcast-compatible, backends MUST reject the op as a spec violation.

## DTypes and casting

### DType set

The dtype universe is defined by `crates/gpt-rs/src/backend/spec.rs::DType` and includes:

- booleans: `i1`
- integers: `si4/ui4/si8/ui8/si16/ui16/si32/ui32/si64/ui64`
- floats: `fp8_e4m3/fp8_e5m2/bf16/f16/f32/f64`
- complex: `cf32/cf64`

Backends MAY support additional dtypes; if supported, they MUST obey the same rules.

### `cast(x; dtype=dst)`

- Shape preserved.
- Float->float: round-to-nearest-even.
- Float->int: truncate toward 0; saturate on overflow; NaN maps to 0.
- Int->float: nearest representable.
- Int->int: saturate on overflow.

## Accumulator/output dtype policy

Some ops accept:

- `accum_dtype: dtype | None`
- `out_dtype: dtype | None`

Defaults:

- If input dtype is `f16`/`bf16`: default `accum_dtype = f32`
- If input dtype is `f32`: default `accum_dtype = f32`
- Default `out_dtype = input dtype`

## Operation reference

This section describes what each `Operation` variant means, independent of any specific backend
implementation strategy.

Notation: `T[shape]` means a tensor of dtype `T` and the given shape.

### `constant` and inline literals

#### `Constant(TensorLiteral)`

Operands: none

Output: `T[shape]`

Semantics:

- Produces a dense tensor literal.
- The output dtype/shape MUST match `TensorLiteral.spec`.
- The literal payload is interpreted as row-major, little-endian scalar storage.
  (Backends that do not support a dtype may reject the program as unimplemented.)

#### `Operand::Literal(TensorLiteral)`

Semantics:

- An inline literal operand is equivalent to referencing a `Constant` op with that literal.
- This exists to make rewrite patterns simpler (they can inject small constants without also
  inserting a separate `Constant` instruction).

### Elementwise ops (no implicit broadcasting)

#### `ElementwiseUnary(op)`

Operands: `x: T[shape]`

Output: `T[shape]`

Semantics:

- Applies `op` independently to every element in `x`.
- Input and output shapes are identical.

Supported unary ops are enumerated by `ElementwiseUnaryOp`:
`neg`, `abs`, `exp`, `log`, `tanh`, `erf`, `rsqrt`, `reciprocal`.

#### `ElementwiseBinary(op)`

Operands: `lhs: T[shape]`, `rhs: T[shape]`

Output: `T[shape]`

Semantics:

- Applies `op` independently per element: `out[i] = f(lhs[i], rhs[i])`.
- `lhs` and `rhs` MUST have the same dtype and the same shape.

Supported binary ops are enumerated by `ElementwiseBinaryOp`:
`add`, `sub`, `mul`, `div`, `maximum`, `minimum`.

Notes:

- Integer division by zero MUST be rejected as a spec violation.
- Floating division follows IEEE (Inf/NaN allowed).

### Comparisons and selection

#### `Compare(CompareSpec { op })`

Operands: `lhs: T[shape]`, `rhs: T[shape]`

Output: `i1[shape]`

Semantics:

- Compares elementwise using `op in {<, <=, ==, >=, >, !=}`.
- `lhs` and `rhs` MUST have the same dtype and the same shape.

NaN rules for floating point comparisons (IEEE-style):

- `<, <=, >, >=, ==` are false if either operand is NaN.
- `!=` is true if either operand is NaN.

#### `Select`

Operands: `pred: i1[shape]`, `on_true: T[shape]`, `on_false: T[shape]`

Output: `T[shape]`

Semantics:

- For each element: `out[i] = pred[i] ? on_true[i] : on_false[i]`.
- All operands MUST have the same shape.
- `on_true` and `on_false` MUST have the same dtype.

### `cast`

#### `Cast(CastSpec { dtype })`

Operands: `x: Tin[shape]`

Output: `Tout[shape]`

Semantics:

- Performs elementwise dtype conversion to `dtype`.
- Shape is preserved.
- Conversion rules are defined in the "DTypes and casting" section.

### `stop_gradient`

#### `StopGradient`

Operands: `x: T[shape]`

Output: `T[shape]`

Semantics:

- Forward value is identical to `x`.
- Autograd (when implemented) MUST treat this as a gradient barrier.
- Backends that do not implement autograd MAY treat it as a pure no-op.

### Reductions and argmax

#### `Reduce(ReduceSpec { kind, axes, keepdims, accum_dtype, out_dtype })`

Operands: `x: T[in_shape]`

Output: `Tout[out_shape]`

Semantics:

- Reduces `x` across `axes`.
- `axes` MUST be unique, in-range, and normalized to `[0, rank)`.
- If `keepdims = true`, reduced axes remain with size 1; otherwise they are removed.
- Accumulation happens in `accum_dtype` (or the default policy), then cast to `out_dtype`.

Empty reduction identities (when a reduced axis has extent 0):

- `sum`: identity 0
- `max`: float identity `-Inf`, integer identity `dtype_min`
- `min`: float identity `+Inf`, integer identity `dtype_max`

#### `ArgMax(ArgMaxSpec { axis, keepdims, output_dtype })`

Operands: `x: T[in_shape]`

Output: `I[out_shape]` where `I in {si32, si64}` as specified by `output_dtype`

Semantics:

- Returns the index of a maximum value along `axis`.
- Ties break toward the smallest index.
- `axis` accepts negative indexing.
- If the selected axis has extent 0, backends MUST reject as a spec violation.

### Linear algebra

#### `DotGeneral(DotGeneralSpec { ... })`

Operands: `lhs: T[A]`, `rhs: T[B]`

Output: `Tout[out_shape]`

Semantics:

- A generalized contraction (like XLA dot_general / einsum).
- `batch_lhs` pairs with `batch_rhs` and MUST have matching extents.
- `contract_lhs` pairs with `contract_rhs` and MUST have matching extents.
- Dims in each list MUST be unique and in-range.

Output dim order:

1. batch dims (lhs order)
2. lhs remaining non-contract dims (in axis order)
3. rhs remaining non-contract dims (in axis order)

Numeric:

- Multiply in input dtype, accumulate in `accum_dtype` (or default), cast to `out_dtype` (or default).

### Shape and layout ops

#### `Reshape(ReshapeSpec { new_shape })`

Operands: `x: T[in_shape]`

Output: `T[out_shape]`

Semantics:

- Changes the logical shape without changing element order.
- Element count MUST be preserved.
- `new_shape` entries are either explicit dimensions or a single inferred dimension.

#### `Transpose(TransposeSpec { perm })`

Operands: `x: T[in_shape]`

Output: `T[out_shape]`

Semantics:

- Permutes axes according to `perm`, which MUST be a permutation of `[0..rank-1]`.

#### `BroadcastTo(BroadcastToSpec { result_shape })`

Operands: `x: T[in_shape]`

Output: `T[out_shape]`

Semantics:

- Broadcasts `x` to `result_shape` by repeating elements along axes where `in_shape` has size 1.
- Compatibility is checked by the "Broadcasting policy" rule.

#### `Slice(SliceSpec { starts, sizes })`

Operands: `x: T[in_shape]`

Output: `T[sizes]`

Semantics:

- Extracts a contiguous window with unit stride.
- `starts.len == sizes.len == rank(x)`.
- For every axis `i`: `starts[i] + sizes[i] <= in_shape[i]`.

#### `Concat(ConcatSpec { axis })`

Operands: `xs: [T[s0], T[s1], ...]` (one or more tensors)

Output: `T[out_shape]`

Semantics:

- Concatenates inputs along `axis` (supports negative axis).
- All dtypes MUST match.
- All shapes MUST match on non-concatenated axes.

#### `Pad(PadSpec { low, high, interior, pad_value })`

Operands: `x: T[in_shape]`

Output: `T[out_shape]`

Semantics:

- Pads `x` with `pad_value` and optional interior padding.
- `low/high/interior` lengths MUST equal `rank(x)`.
- Output axis size:
  `out[i] = low[i] + in[i] + high[i] + (in[i].saturating_sub(1) * interior[i])`.

#### `Tile(TileSpec { repeats })`

Operands: `x: T[in_shape]`

Output: `T[out_shape]`

Semantics:

- Repeats `x` along each axis `i` exactly `repeats[i]` times.
- `repeats.len == rank(x)`.
- Output axis size: `out[i] = in[i] * repeats[i]`.

#### `Iota(IotaSpec { shape, dtype, axis })`

Operands: none

Output: `T[shape]`

Semantics:

- Produces a tensor where the value increases along `axis` from `0..shape[axis]-1`.
- All other axes are broadcast/repeated.

### Indexing ops

#### `Take`

Operands: `x: T[N, ...]`, `indices: si32|si64[...]`

Output: `T[indices.shape..., ...]`

Semantics:

- Equivalent to `index_select` on axis 0 (common embedding primitive).
- Each index MUST satisfy `0 <= index < N`.

#### `Gather(GatherSpec { axis })`

Operands: `x: T[in_shape]`, `indices: si32|si64[index_shape]`

Output: `T[index_shape]`

Semantics:

- Equivalent to `torch.gather`.
- `indices` MUST have the same rank as `x`.
- For all dims `d != axis`: `indices.shape[d] == x.shape[d]` (the gather axis may differ).
- For each output element at index `i`, read from `x` where `i[axis]` is replaced by
  `indices[i]`.
- Each index MUST satisfy `0 <= index < x.shape[axis]`.

#### `ScatterAdd(ScatterSpec { ... })`

Operands: backend-defined (XLA-style scatter interface)

Semantics:

- Legacy scatter form used for experimentation; uses XLA-style dimension-number attributes.
- Prefer `scatter_reduce` for new work.

#### `ScatterReduce(ScatterReduceSpec { axis, reduce })`

Operands: `x: T[in_shape]`, `indices: si32|si64[update_shape]`, `updates: T[update_shape]`

Output: `T[in_shape]`

Semantics:

- Equivalent to `torch.scatter_reduce` (same-rank form).
- `indices` and `updates` MUST have the same rank and shape.
- For all dims `d != axis`: `updates.shape[d] == x.shape[d]` (the scatter axis may differ).
- For each element position `i`, update `x` at `i` with `axis` replaced by `indices[i]`.
- Indices MUST be in range.
- Reduction kinds:
  - `add`: accumulate sums
  - `max`: accumulate maximum
  - `min`: accumulate minimum
  - `replace`: last writer wins (update order may be backend-defined)

### Dynamic slice/update (decode-friendly)

#### `DynamicSlice(DynamicSliceSpec { sizes })`

Operands: `x: T[in_shape]`, `start: si32|si64[rank(x)]`

Output: `T[sizes]`

Semantics:

- Runtime-start slice with unit stride.
- `sizes.len == rank(x)`.
- For each dim `d`, the start is clamped into `[0, x[d] - sizes[d]]`.
- If `sizes[d] > x[d]` at runtime, backends MUST reject as a spec violation.

#### `DynamicUpdateSlice(DynamicUpdateSliceSpec { sizes })`

Operands: `x: T[in_shape]`, `update: T[sizes]`, `start: si32|si64[rank(x)]`

Output: `T[in_shape]`

Semantics:

- Returns a new tensor equal to `x` except the window starting at `start` is overwritten by `update`.
- Start clamping matches `dynamic_slice` (clamp into `[0, x[d] - update[d]]`).

### Fused and convenience ops

#### `LayerNorm { epsilon, feature_axis }`

Operands: `x: T[in_shape]`, `gamma: T[feature]`, `beta: T[feature]`

Output: `T[in_shape]`

Semantics:

- Equivalent to the canonical layer norm computation over `feature_axis`:
  compute mean/variance along `feature_axis`, normalize, then apply affine scale/shift.
- `feature_axis` is an axis index (supports negative axis by convention in frontends).

#### `Clamp`

Operands: `x: T[shape]`, `lo: T[shape]`, `hi: T[shape]`

Output: `T[shape]`

Semantics:

- Elementwise clamp: `out = min(max(x, lo), hi)`.
- Frontends often lower `clamp` to `maximum` + `minimum` for portability.

### Control flow (regions)

Control-flow ops reference `RegionId` entries in `Program.regions`. Regions have their own
parameter list and results list.

#### `Cond(CondSpec { true_region, false_region })`

Operands: `pred: i1[]`, then the region arguments

Output: the region result(s)

Semantics:

- If `pred` is true, executes `true_region`, else executes `false_region`.
- Region parameter/result types MUST match.

#### `While(WhileSpec { cond_region, body_region })`

Operands: initial loop-carried values

Output: final loop-carried values

Semantics:

- Repeatedly runs `cond_region(carry...) -> (i1[], carry...)`.
- If predicate is true, runs `body_region(carry...) -> carry...` and repeats.
- If predicate is false, returns the carry values.

#### `Scan(ScanSpec { body_region, carry_count, scan_output_count })`

Operands: `carry_count` carry values, then scanned inputs

Output: `carry_count` final carry values, then `scan_output_count` stacked scan outputs

Semantics:

- Like a functional for-loop over a leading time axis of the scanned inputs.
- `body_region` consumes carry values and one time-slice of scanned inputs, producing updated
  carry values and per-step outputs that are stacked.

### Windowed ops

#### `ReduceWindow(ReduceWindowSpec { ... })`

Operands: `x: T[in_shape]`

Output: `Tout[out_shape]`

Semantics:

- Sliding-window reduction (pooling-style primitive).
- Attributes define window size, stride, padding, and dilation.
- Reduction kind is `sum/max/min` with accumulator dtype policy like `reduce`.

### Patch extraction (extension)

#### `ExtractPatches(ExtractPatchesSpec { ... })`

Operands: `x: T[N, spatial..., C]`

Output: `T[N, out_spatial..., (prod(window) * C)]`

Semantics:

- Extracts sliding windows (im2col / unfold) from a channels-last input.
- Backends SHOULD fuse `extract_patches + dot_general` into convolution kernels.

### Custom calls (backend-provided fused kernels)

#### `CustomCall(CustomCallSpec { target, attrs })`

Operands: backend-defined (but MUST be tensors or literals expressible as operands)

Output: backend-defined (typed by `Instruction.output`, which is authoritative)

Semantics:

- Represents an explicit backend-provided fused kernel.
- **Purity:** Custom calls MUST be referentially transparent. Backends MAY cache internal
  compiled kernels, packed weights, or plans, but MUST NOT perform observable side effects.
- **Portability:** If a backend does not recognize `target`, it MUST return `Unimplemented`
  (no silent fallback to a portable decomposition).
- **Typing:** The `ValueType` stored in the instruction output is authoritative. Backends MAY
  validate that operand dtypes/shapes and `attrs` agree with the declared output type.
- **Stable naming:** `target` strings MUST be namespaced and versioned, e.g.
  `gpt_rs.faer.conv2d.nhwc.f32.v1`.
- **Attributes:** `attrs` is a small serializable map of primitives (ints/floats/bools/strings)
  and arrays of primitives. Backends SHOULD reject unknown or malformed attributes as a
  spec violation (`InvalidAttributeValue`).
- **Multi-output:** Multi-output custom calls MAY be represented by tuple-typed outputs.
  Some backends do not execute tuple outputs yet; producers SHOULD avoid tuple custom calls
  until the tuple execution contract is widely supported.

### Randomness, top-k, segments, quantization (extensions)

These ops exist for broader model coverage and experimentation. Many backends may choose to
return `Unimplemented` unless explicitly needed.

#### `RngUniform(RngUniformSpec { shape, dtype })`, `RngNormal(RngNormalSpec { shape, dtype })`

Semantics:

- Produce random values with the requested shape and dtype.
- Unless a backend documents stronger guarantees, RNG algorithm and determinism are backend-defined.

#### `TopK(TopKSpec { k, axis, largest, indices_dtype })`

Semantics:

- Returns the top-k values and their indices along `axis`.
- Output is typically a tuple: `(values, indices)`.

#### `SegmentReduce(SegmentReduceSpec { ... })`

Semantics:

- Reduces values into `num_segments` buckets using segment ids.
- Supports common constraints flags (`indices_are_sorted`, `unique`) to enable faster kernels.

#### `Quantize(QuantizeSpec { output_dtype, axis })`

Semantics:

- Quantizes a float tensor using a per-tensor or per-axis scale/zero-point convention.
- Exact quantization scheme is currently backend-defined; treat as an extension point.

#### `Dequantize(DequantizeSpec { axis, output_dtype })`, `Requantize(RequantizeSpec { output_dtype, axis })`

Semantics:

- Convert quantized tensors back to float (dequantize) or between quantized formats (requantize).
- Exact conventions are backend-defined unless a specific backend documents otherwise.

## Canonical library forms (attention-friendly)

### Stable softmax (last axis)

```
m = reduce_max(z, axes=[-1], keepdims=true, accum_dtype=f32, out_dtype=f32)
e = exp(z - m)
s = reduce_sum(e, axes=[-1], keepdims=true, accum_dtype=f32, out_dtype=f32)
softmax = e / s
```

### Canonical attention (no explicit attention op)

For `q,k,v` in `[B,H,S,D]`:

1. `scores = dot_general(q, transpose(k,[0,1,3,2]), accum_dtype=f32, out_dtype=f32)`
2. `scores *= scale` (scalar f32 broadcast)
3. Add optional bias/mask (additive mask in score space: `0` or large negative / `-Inf`)
4. `p = softmax(scores)` (stable form above, f32)
5. `out = dot_general(p, v, accum_dtype=f32, out_dtype=desired)`

## Conversion infrastructure (IR-to-IR)

The `backend::conversion` module provides shared plumbing for PTIR conversion targets:

- Global registry for conversion targets (`register_conversion_target`, `get_conversion_target`).
- Deterministic entrypoint naming helpers.
- Legality checks (`check_program_legality`) with structured diagnostics.
- Buffer planning (`plan_buffers`) for preflight sizing and ABI scaffolding.
- Program walker (`walk_program`) for deterministic traversal.

Conversion targets consume optimized PTIR and emit a target IR module. Code generation and runtime
packaging are intentionally out of scope for the conversion layer.
