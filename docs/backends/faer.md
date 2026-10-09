# Faer backend (`gpt-rs-backend-faer`)

The faer backend runs a function op by op on host memory. An op that faer has a kernel for runs on
it, and every other op runs on the IR's interpreter, whose results define the op.

Source of truth:
- [../../crates/gpt-rs-backend-faer/src/lib.rs](../../crates/gpt-rs-backend-faer/src/lib.rs)
- [../../crates/gpt-rs-backend-faer/src/select.rs](../../crates/gpt-rs-backend-faer/src/select.rs)

## Data flow

```text
recorded module (`tensor` dialect)
  -> the default `PortableBackend::pipeline` (docs/ptir/passes.md lists it)
  -> FaerPortableBackend::compile: the function's ops as steps, where a step is one op or a
     sequence of ops that one kernel computes, and the values each step reads for the last time
  -> FaerPortableBackend::run: each step on a faer kernel, or on `interp::eval_inst`
```

## Kernel selection

faer runs the default `PortableBackend::pipeline`. `compile` chooses the kernels
([select.rs](../../crates/gpt-rs-backend-faer/src/select.rs)):
- Convolution: a `dot_general` that contracts the last axis of `extract_patches` with a
  `[KH * KW * C_in, C_out]` weight, which `merge-contracted-axes` writes, runs on faer's conv
  kernel. So does the depthwise convolution that `conv2d` records for one input and one output
  channel per group. The kernel also covers a bias add on the channel axis that follows it.
- Linear layers with weights stored in a narrower float dtype:
  `gpt_rs::ops::functional::linear_weight` matches the `dot_general` and the casts of its weight,
  and the linear kernel reads the weight in place.
- relu6, `minimum(maximum(x, broadcast_to(0)), broadcast_to(6))`, runs in one loop.

An op that a kernel covers, such as the cast of a weight that the linear kernel reads in place,
runs only when another step reads it.

## Kernels

faer has kernels for:
- `dot_general` in the f32 matmul layouts, through faer's matmul;
- f32 `add`, `mul`, `maximum` and `minimum` of equal shapes;
- `broadcast_to`, `transpose`, `slice`, `dynamic_slice`, `dynamic_update_slice` and `concat`, which
  copy contiguous runs of any dtype ([layout.rs](../../crates/gpt-rs-backend-faer/src/layout.rs));
- f32 `reduce` ([reduce.rs](../../crates/gpt-rs-backend-faer/src/reduce.rs)) and max
  `reduce_window`;
- the convolutions, linear layers and relu6 above.

Each kernel computes what the op's `Eval` impl computes. An f32 sum folds its terms in the same
order, so it rounds the same way.
