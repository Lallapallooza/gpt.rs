//! The faer CPU backend. It runs a function op by op on host memory: an op that faer has a kernel
//! for runs on it, and every other op on the IR's interpreter.

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{anyhow, ensure, Context, Result};
use faer::linalg::matmul::matmul;
use faer::mat::{MatMut, MatRef};
use faer::{Accum, Par};
use gpt_rs::backend::{ParamStore, PortableBackend};
use gpt_rs_ir::dialects::tensor::{
    BinaryKind, DotGeneral, ExtractPatches, ReduceKind, ReduceWindow, TensorOp,
};
use gpt_rs_ir::host::{Buffer, HostTensor, HostValue, TensorData};
use gpt_rs_ir::interp;
use gpt_rs_ir::module::{Inst, Module, Value};
use gpt_rs_ir::traits::Terminator;
use gpt_rs_ir::types::{DType, TensorType};
use gpt_rs_ir::Op;

mod layout;
mod linear;
mod reduce;
mod select;

use select::{Step, StepKind};

thread_local! {
    static CONV2D_IM2COL_SCRATCH_F32: std::cell::RefCell<Vec<f32>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

fn faer_parallelism() -> Par {
    let par = faer::get_global_parallelism();
    if par.degree() == 1 {
        Par::Seq
    } else {
        par
    }
}

#[derive(Default)]
pub struct FaerPortableBackend {
    params: ParamStore<HostTensor>,
}

impl FaerPortableBackend {
    pub fn new() -> Self {
        Self::default()
    }
}

/// A compiled function: the steps that compute its results from its arguments.
pub struct Function {
    module: Arc<Module>,
    args: Vec<Value>,
    steps: Vec<Step>,
    results: Vec<Value>,
}

impl PortableBackend for FaerPortableBackend {
    type TensorHandle = HostTensor;
    type Executable = Function;

    fn backend_name(&self) -> &str {
        "faer"
    }

    fn params(&self) -> &ParamStore<HostTensor> {
        &self.params
    }

    fn compile(&self, module: &Arc<Module>, func: &str) -> Result<Function> {
        let inst = module
            .func(func)
            .ok_or_else(|| anyhow!("the module has no function @{func}"))?;
        let body = module.regions(inst)[0];
        let (insts, terminator): (Vec<Inst>, Vec<Inst>) = module
            .insts(body)
            .partition(|&inst| !module.op(inst).has_trait::<Terminator>());
        let results = terminator
            .first()
            .map(|&inst| module.operands(inst).to_vec())
            .unwrap_or_default();
        let steps = select::steps(module, &insts, &results);
        Ok(Function {
            module: Arc::clone(module),
            args: module.args(body).to_vec(),
            steps,
            results,
        })
    }

    fn run(&self, exe: &Function, args: &[HostTensor]) -> Result<Vec<HostTensor>> {
        ensure!(
            args.len() == exe.args.len(),
            "the function takes {} arguments, got {}",
            exe.args.len(),
            args.len()
        );
        let module = exe.module.as_ref();
        let mut env: HashMap<Value, HostTensor> =
            exe.args.iter().copied().zip(args.iter().cloned()).collect();
        let get = |env: &HashMap<Value, HostTensor>, value: Value| {
            env.get(&value)
                .cloned()
                .ok_or_else(|| anyhow!("{value:?} has no value"))
        };
        for step in &exe.steps {
            match step.kind {
                StepKind::Op { inst, label } => {
                    let inputs = module
                        .operands(inst)
                        .iter()
                        .map(|&value| get(&env, value))
                        .collect::<Result<Vec<_>>>()?;
                    let outputs = run_op(module, inst, label, &inputs)
                        .with_context(|| module.op(inst).name().into_owned())?;
                    env.extend(module.results(inst).iter().copied().zip(outputs));
                }
                StepKind::Relu6 { input, output } => {
                    let input = get(&env, input)?;
                    env.insert(output, relu6_f32(&input)?);
                }
                StepKind::Conv2d(ref conv) => {
                    let bias = conv.bias.map(|bias| get(&env, bias)).transpose()?;
                    let input = get(&env, conv.input)?;
                    let weight = get(&env, conv.weight)?;
                    let output = select_output_type(module, conv.output)?;
                    let out = run_conv2d(conv, &input, &weight, bias.as_ref(), &output)?;
                    env.insert(conv.output, out);
                }
                StepKind::Linear(ref projection) => {
                    let scale = projection.scale.map(|scale| get(&env, scale)).transpose()?;
                    let x = get(&env, projection.x)?;
                    let weight = get(&env, projection.weight)?;
                    let output = select_output_type(module, projection.output)?;
                    let out = linear::run(&x, &weight, scale.as_ref(), projection.layout, &output)?;
                    env.insert(projection.output, out);
                }
            }
            for value in &step.last_uses {
                env.remove(value);
            }
        }
        exe.results.iter().map(|&value| get(&env, value)).collect()
    }

    fn upload(&self, host: HostTensor) -> Result<HostTensor> {
        Ok(host)
    }

    fn download(&self, tensor: &HostTensor) -> Result<HostTensor> {
        Ok(tensor.clone())
    }
}

/// Runs the op of `inst` on `inputs`: on faer's kernel for it when there is one, else on the
/// interpreter.
fn run_op(
    module: &Module,
    inst: Inst,
    label: &'static str,
    inputs: &[HostTensor],
) -> Result<Vec<HostTensor>> {
    let Op::Tensor(op) = module.op(inst) else {
        return Err(anyhow!("faer runs only tensor ops"));
    };
    let outputs = module
        .result_types(inst)
        .into_iter()
        .map(|ty| {
            ty.as_tensor()
                .cloned()
                .ok_or_else(|| anyhow!("a result of type {ty} is not a tensor"))
        })
        .collect::<Result<Vec<_>>>()?;
    let _scope = gpt_rs::profiling::backend_scope_with_meta(label, || {
        gpt_rs::profiling::signature_id(&outputs[0].to_string())
            .map(gpt_rs::profiling::ScopeMeta::signature)
            .unwrap_or_default()
    });
    if let Some(result) = kernel(op, inputs, &outputs) {
        return result;
    }
    let values: Vec<HostValue> = inputs.iter().cloned().map(HostValue::Tensor).collect();
    let refs: Vec<&HostValue> = values.iter().collect();
    interp::eval_inst(module, inst, &refs)?
        .into_iter()
        .map(|value| match value {
            HostValue::Tensor(tensor) => Ok(tensor),
            other => Err(anyhow!("the op gave a {}, not a tensor", other.ty())),
        })
        .collect()
}

/// faer's kernel for `op` on `inputs`, with results of the types `outputs`, when it has one.
fn kernel(
    op: &TensorOp,
    inputs: &[HostTensor],
    outputs: &[TensorType],
) -> Option<Result<Vec<HostTensor>>> {
    let one = |result: Result<HostTensor>| Some(result.map(|tensor| vec![tensor]));
    match (op, inputs, outputs) {
        (TensorOp::DotGeneral(dot), ..) => try_dot_general(inputs, outputs, dot),
        (TensorOp::Binary(binary), ..) => try_elementwise_binary(inputs, outputs, binary.kind),
        (TensorOp::ReduceWindow(window), ..) => try_reduce_window(inputs, outputs, window),
        (TensorOp::Reduce(op), [input], [out]) => {
            reduce::reduce(op, input, out).map(|result| result.map(|tensor| vec![tensor]))
        }
        (TensorOp::BroadcastTo(op), [input], [out]) => one(layout::broadcast_to(op, input, out)),
        (TensorOp::Transpose(op), [input], [out]) => one(layout::transpose(op, input, out)),
        (TensorOp::Slice(op), [input], [out]) => one(layout::slice(op, input, out)),
        (TensorOp::DynamicSlice(op), [input, starts], [out]) => {
            one(layout::dynamic_slice(op, input, starts, out))
        }
        (TensorOp::DynamicUpdateSlice(_), [base, update, starts], _) => {
            one(layout::dynamic_update_slice(base, update, starts))
        }
        (TensorOp::Concat(op), _, [out]) => one(layout::concat(op, inputs, out)),
        _ => None,
    }
}

/// `minimum(maximum(v, 0.0), 6.0)` of each element, as two selects that vectorize. `v <= 0.0`
/// also maps -0.0 to +0.0, and a NaN fails both comparisons and stays.
fn relu6_f32(input: &HostTensor) -> Result<HostTensor> {
    let _scope = gpt_rs::profiling::backend_scope("backend.fused_relu6");
    let values = input.as_f32()?;
    let out = values
        .iter()
        .map(|&v| {
            let low = if v <= 0.0 { 0.0 } else { v };
            if low >= 6.0 {
                6.0
            } else {
                low
            }
        })
        .collect();
    HostTensor::f32(input.dims.clone(), out)
}

/// The extents of `ty`, which must be static.
pub(crate) fn dims_of(ty: &TensorType) -> Result<Vec<usize>> {
    ty.static_dims()
        .map(|dims| dims.into_vec())
        .ok_or_else(|| anyhow!("{ty} has a dynamic dimension"))
}

/// A tensor of type `ty` holding `data`.
pub(crate) fn output_tensor(ty: &TensorType, data: TensorData) -> Result<HostTensor> {
    ensure!(
        data.dtype() == ty.dtype,
        "a {} result is not a {ty}",
        data.dtype()
    );
    HostTensor::new(dims_of(ty)?, data)
}

fn conv2d_nhwc_im2col_tiled_f32(
    input: &HostTensor,
    weight: &HostTensor,
    bias: Option<&HostTensor>,
    spec: &ExtractPatches,
    output_spec: &TensorType,
) -> Result<HostTensor> {
    let input_dims = input.dims.clone();
    let weight_dims = weight.dims.clone();
    let out_dims = dims_of(output_spec)?;

    if input.dtype() != DType::F32
        || weight.dtype() != DType::F32
        || output_spec.dtype != DType::F32
    {
        return Err(anyhow!("conv2d fusion only supports f32"));
    }
    if input_dims.len() != 4 || out_dims.len() != 4 || weight_dims.len() != 2 {
        return Err(anyhow!("conv2d fusion shape rank mismatch"));
    }
    let (n, h, w, c_in) = (input_dims[0], input_dims[1], input_dims[2], input_dims[3]);
    let (out_n, out_h, out_w, c_out) = (out_dims[0], out_dims[1], out_dims[2], out_dims[3]);
    if out_n != n {
        return Err(anyhow!("conv2d fusion batch mismatch"));
    }
    if spec.window.len() != 2
        || spec.strides.len() != 2
        || spec.dilation.len() != 2
        || spec.padding.len() != 2
    {
        return Err(anyhow!(
            "conv2d fusion expects 2D window/stride/dilation/padding",
        ));
    }

    let (k_h, k_w) = (spec.window[0], spec.window[1]);
    let (s_h, s_w) = (spec.strides[0], spec.strides[1]);
    let (d_h, d_w) = (spec.dilation[0], spec.dilation[1]);
    let pad_top = spec.padding[0].0;
    let pad_left = spec.padding[1].0;

    let k = k_h
        .checked_mul(k_w)
        .and_then(|v| v.checked_mul(c_in))
        .ok_or_else(|| anyhow!("conv2d fusion kernel size overflow"))?;

    if weight_dims[0] != k || weight_dims[1] != c_out {
        return Err(anyhow!("conv2d fusion weight shape does not match kernel",));
    }

    if let Some(bias) = bias {
        let bias_dims = bias.dims.clone();
        if bias.dtype() != DType::F32 || bias_dims.iter().product::<usize>() != c_out {
            return Err(anyhow!("conv2d fusion bias shape mismatch"));
        }
    }

    let input_values = match &input.data {
        TensorData::F32(values) => values.as_ref(),
        _ => return Err(anyhow!("conv2d fusion expects f32 input")),
    };
    let weight_values = match &weight.data {
        TensorData::F32(values) => values.as_ref(),
        _ => return Err(anyhow!("conv2d fusion expects f32 weight")),
    };
    let bias_values = match bias {
        Some(bias) => match &bias.data {
            TensorData::F32(values) => Some(values.as_ref()),
            _ => return Err(anyhow!("conv2d fusion expects f32 bias")),
        },
        None => None,
    };

    let m = out_n
        .checked_mul(out_h)
        .and_then(|v| v.checked_mul(out_w))
        .ok_or_else(|| anyhow!("conv2d fusion output size overflow"))?;

    if m == 0 || c_out == 0 {
        return output_tensor(
            output_spec,
            TensorData::F32(Buffer::from(Vec::<f32>::new())),
        );
    }

    let output_len = m
        .checked_mul(c_out)
        .ok_or_else(|| anyhow!("conv2d fusion output overflow"))?;
    let mut output_uninit = vec![std::mem::MaybeUninit::<f32>::uninit(); output_len];
    // Safety: `MaybeUninit<f32>` has the same memory layout as `f32`, and `process_parallel`
    // (via `matmul` with `Accum::Replace`) fully initializes all `output_len` elements on the
    // success path.
    let output_slice = unsafe {
        std::slice::from_raw_parts_mut(output_uninit.as_mut_ptr().cast::<f32>(), output_len)
    };

    let rhs_view = MatRef::from_row_major_slice(weight_values, k, c_out);
    let rhs_t = rhs_view.transpose();

    let max_scratch_floats = 1_048_576usize; // 4 MiB
    let mut block_rows_cap = m.min(8192);
    while block_rows_cap > 1 && block_rows_cap.saturating_mul(k) > max_scratch_floats {
        block_rows_cap /= 2;
    }
    block_rows_cap = block_rows_cap.max(1);

    let out_hw = out_h
        .checked_mul(out_w)
        .ok_or_else(|| anyhow!("conv2d fusion out_hw overflow"))?;

    let in_hw = h
        .checked_mul(w)
        .ok_or_else(|| anyhow!("conv2d fusion in_hw overflow"))?;
    let in_stride_n = in_hw
        .checked_mul(c_in)
        .ok_or_else(|| anyhow!("conv2d fusion in_stride overflow"))?;
    let in_stride_h = w
        .checked_mul(c_in)
        .ok_or_else(|| anyhow!("conv2d fusion in_stride overflow"))?;

    let expected_input_len = n
        .checked_mul(in_stride_n)
        .ok_or_else(|| anyhow!("conv2d fusion input size overflow"))?;
    if input_values.len() != expected_input_len {
        return Err(anyhow!("conv2d fusion input length mismatch",));
    }

    let expected_weight_len = k
        .checked_mul(c_out)
        .ok_or_else(|| anyhow!("conv2d fusion weight size overflow"))?;
    if weight_values.len() != expected_weight_len {
        return Err(anyhow!("conv2d fusion weight length mismatch",));
    }

    if let Some(bias) = bias_values {
        if bias.len() != c_out {
            return Err(anyhow!("conv2d fusion bias length mismatch",));
        }
    }

    let s_h_isize = s_h as isize;
    let s_w_isize = s_w as isize;
    let d_h_isize = d_h as isize;
    let d_w_isize = d_w as isize;
    let pad_top_isize = pad_top as isize;
    let pad_left_isize = pad_left as isize;
    let h_isize = h as isize;
    let w_isize = w as isize;
    let par = faer_parallelism();

    #[derive(Clone, Copy)]
    struct Conv2dIm2colContext<'a> {
        out_w: usize,
        out_hw: usize,
        in_stride_n: usize,
        in_stride_h: usize,
        k_h: usize,
        k_w: usize,
        c_in: usize,
        k: usize,
        c_out: usize,
        block_rows_cap: usize,
        s_h_isize: isize,
        s_w_isize: isize,
        d_h_isize: isize,
        d_w_isize: isize,
        pad_top_isize: isize,
        pad_left_isize: isize,
        h_isize: isize,
        w_isize: isize,
        input_values: &'a [f32],
        rhs_t: MatRef<'a, f32>,
        bias_values: Option<&'a [f32]>,
    }

    fn process_range(
        ctx: Conv2dIm2colContext<'_>,
        row_start: usize,
        row_end: usize,
        out_slice: &mut [f32],
        matmul_par: Par,
    ) -> Result<()> {
        let range_rows = row_end - row_start;
        if range_rows == 0 {
            return Ok(());
        }

        let out_h = ctx.out_hw / ctx.out_w;
        let step_h = (ctx.d_h_isize as usize)
            .checked_mul(ctx.in_stride_h)
            .ok_or_else(|| anyhow!("conv2d fusion h offset overflow"))?;
        let step_w = (ctx.d_w_isize as usize)
            .checked_mul(ctx.c_in)
            .ok_or_else(|| anyhow!("conv2d fusion w offset overflow"))?;
        let k_h_span = (ctx.k_h.saturating_sub(1) as isize)
            .checked_mul(ctx.d_h_isize)
            .ok_or_else(|| anyhow!("conv2d fusion h offset overflow"))?;
        let k_w_span = (ctx.k_w.saturating_sub(1) as isize)
            .checked_mul(ctx.d_w_isize)
            .ok_or_else(|| anyhow!("conv2d fusion w offset overflow"))?;
        let input_ptr = ctx.input_values.as_ptr();

        let scratch_rows = range_rows.min(ctx.block_rows_cap);
        let scratch_capacity = scratch_rows
            .checked_mul(ctx.k)
            .ok_or_else(|| anyhow!("conv2d fusion scratch overflow"))?;
        CONV2D_IM2COL_SCRATCH_F32.with(|scratch_cell| {
            {
                let mut scratch = scratch_cell.borrow_mut();
                if scratch.len() < scratch_capacity {
                    scratch.resize(scratch_capacity, 0.0);
                }
            }

            for block_offset in (0..range_rows).step_by(ctx.block_rows_cap) {
                let cur_rows = (range_rows - block_offset).min(ctx.block_rows_cap);
                let scratch_len = cur_rows
                    .checked_mul(ctx.k)
                    .ok_or_else(|| anyhow!("conv2d fusion scratch overflow"))?;
                let out_offset = block_offset
                    .checked_mul(ctx.c_out)
                    .ok_or_else(|| anyhow!("conv2d fusion output overflow"))?;
                let out_chunk = &mut out_slice[out_offset..out_offset + cur_rows * ctx.c_out];

                let flat0 = row_start + block_offset;
                let n_idx = flat0 / ctx.out_hw;
                let rem = flat0 - n_idx * ctx.out_hw;
                let mut oh = rem / ctx.out_w;
                let mut ow = rem - oh * ctx.out_w;
                let mut src_n_base = n_idx * ctx.in_stride_n;

                let mut fill_scratch = |scratch_buf: &mut [f32]| -> Result<()> {
                    for row in 0..cur_rows {
                        let base_h = oh as isize * ctx.s_h_isize - ctx.pad_top_isize;
                        let base_w = ow as isize * ctx.s_w_isize - ctx.pad_left_isize;

                        let dst_row = &mut scratch_buf[row * ctx.k..(row + 1) * ctx.k];
                        let dst_ptr = dst_row.as_mut_ptr();

                        let max_h = base_h + k_h_span;
                        let max_w = base_w + k_w_span;
                        let interior = base_h >= 0
                            && base_w >= 0
                            && max_h < ctx.h_isize
                            && max_w < ctx.w_isize;

                        if interior {
                            let mut dst_off = 0usize;
                            let mut src_row_base = src_n_base
                                + (base_h as usize) * ctx.in_stride_h
                                + (base_w as usize) * ctx.c_in;
                            for _kh in 0..ctx.k_h {
                                let mut src = src_row_base;
                                for _kw in 0..ctx.k_w {
                                    // Safety: `input_values`/`dst_row` lengths are validated by the
                                    // caller, and the `interior` bounds guarantee each patch element
                                    // points at a valid `[C_in]` slice.
                                    unsafe {
                                        std::ptr::copy_nonoverlapping(
                                            input_ptr.add(src),
                                            dst_ptr.add(dst_off),
                                            ctx.c_in,
                                        );
                                    }
                                    dst_off += ctx.c_in;
                                    src += step_w;
                                }
                                src_row_base += step_h;
                            }
                        } else {
                            dst_row.fill(0.0);
                            for kh in 0..ctx.k_h {
                                let ih = base_h + kh as isize * ctx.d_h_isize;
                                if ih < 0 || ih >= ctx.h_isize {
                                    continue;
                                }
                                let src_h_base = src_n_base + (ih as usize) * ctx.in_stride_h;
                                for kw in 0..ctx.k_w {
                                    let iw = base_w + kw as isize * ctx.d_w_isize;
                                    if iw < 0 || iw >= ctx.w_isize {
                                        continue;
                                    }
                                    let dst_off = (kh * ctx.k_w + kw) * ctx.c_in;
                                    let src = src_h_base + (iw as usize) * ctx.c_in;
                                    // Safety: bounds checks above guarantee `ih/iw` are in range.
                                    unsafe {
                                        std::ptr::copy_nonoverlapping(
                                            input_ptr.add(src),
                                            dst_ptr.add(dst_off),
                                            ctx.c_in,
                                        );
                                    }
                                }
                            }
                        }

                        ow += 1;
                        if ow == ctx.out_w {
                            ow = 0;
                            oh += 1;
                            if oh == out_h {
                                oh = 0;
                                src_n_base += ctx.in_stride_n;
                            }
                        }
                    }

                    Ok(())
                };

                if matmul_par.degree() > 1 {
                    let scratch_ptr = {
                        let mut scratch = scratch_cell.borrow_mut();
                        let scratch_buf = &mut scratch[..scratch_len];
                        let _fill_scope = gpt_rs::profiling::backend_scope_with_meta(
                            "backend.faer.conv2d.im2col_fill",
                            || {
                                let bytes = (scratch_len as u64).saturating_mul(4);
                                gpt_rs::profiling::ScopeMeta::default().with_work(
                                    gpt_rs::profiling::WorkStats {
                                        elements: scratch_len as u64,
                                        bytes_read: bytes,
                                        bytes_written: bytes,
                                        flops: 0,
                                        alloc_bytes: 0,
                                        alloc_count: 0,
                                    },
                                )
                            },
                        );
                        fill_scratch(scratch_buf)?;
                        scratch_buf.as_ptr()
                    };

                    // Safety: `scratch_ptr` points into the thread-local scratch buffer for the
                    // current thread, which stays alive for the duration of this call. We do not
                    // borrow the scratch buffer again until after the parallel matmul completes.
                    let scratch_view =
                        unsafe { std::slice::from_raw_parts(scratch_ptr, scratch_len) };
                    let lhs_view = MatRef::from_row_major_slice(scratch_view, cur_rows, ctx.k);
                    let lhs_t = lhs_view.transpose();

                    let out_view =
                        MatMut::from_column_major_slice_mut(out_chunk, ctx.c_out, cur_rows);
                    let _matmul_scope = gpt_rs::profiling::backend_scope_with_meta(
                        "backend.faer.conv2d.matmul",
                        || {
                            let out_elems = (cur_rows as u64).saturating_mul(ctx.c_out as u64);
                            let k = ctx.k as u64;
                            let bytes_per_elem = 4u64;
                            let rhs_bytes = (ctx.k as u64)
                                .saturating_mul(ctx.c_out as u64)
                                .saturating_mul(bytes_per_elem);
                            let lhs_bytes = (cur_rows as u64)
                                .saturating_mul(ctx.k as u64)
                                .saturating_mul(bytes_per_elem);
                            let out_bytes = out_elems.saturating_mul(bytes_per_elem);
                            let flops = (out_elems as u128)
                                .saturating_mul(k as u128)
                                .saturating_mul(2)
                                .min(u64::MAX as u128)
                                as u64;
                            gpt_rs::profiling::ScopeMeta::default().with_work(
                                gpt_rs::profiling::WorkStats {
                                    elements: out_elems,
                                    bytes_read: rhs_bytes.saturating_add(lhs_bytes),
                                    bytes_written: out_bytes,
                                    flops,
                                    alloc_bytes: 0,
                                    alloc_count: 0,
                                },
                            )
                        },
                    );
                    matmul_tiled_parallel(matmul_par, out_view, ctx.rhs_t, lhs_t)?;
                } else {
                    let mut scratch = scratch_cell.borrow_mut();
                    let scratch_buf = &mut scratch[..scratch_len];
                    {
                        let _fill_scope = gpt_rs::profiling::backend_scope_with_meta(
                            "backend.faer.conv2d.im2col_fill",
                            || {
                                let bytes = (scratch_len as u64).saturating_mul(4);
                                gpt_rs::profiling::ScopeMeta::default().with_work(
                                    gpt_rs::profiling::WorkStats {
                                        elements: scratch_len as u64,
                                        bytes_read: bytes,
                                        bytes_written: bytes,
                                        flops: 0,
                                        alloc_bytes: 0,
                                        alloc_count: 0,
                                    },
                                )
                            },
                        );
                        fill_scratch(scratch_buf)?;
                    }
                    let lhs_view = MatRef::from_row_major_slice(scratch_buf, cur_rows, ctx.k);
                    let lhs_t = lhs_view.transpose();

                    let mut out_view =
                        MatMut::from_column_major_slice_mut(out_chunk, ctx.c_out, cur_rows);
                    {
                        let _matmul_scope = gpt_rs::profiling::backend_scope_with_meta(
                            "backend.faer.conv2d.matmul",
                            || {
                                let out_elems = (cur_rows as u64).saturating_mul(ctx.c_out as u64);
                                let k = ctx.k as u64;
                                let bytes_per_elem = 4u64;
                                let rhs_bytes = (ctx.k as u64)
                                    .saturating_mul(ctx.c_out as u64)
                                    .saturating_mul(bytes_per_elem);
                                let lhs_bytes = (cur_rows as u64)
                                    .saturating_mul(ctx.k as u64)
                                    .saturating_mul(bytes_per_elem);
                                let out_bytes = out_elems.saturating_mul(bytes_per_elem);
                                let flops = (out_elems as u128)
                                    .saturating_mul(k as u128)
                                    .saturating_mul(2)
                                    .min(u64::MAX as u128)
                                    as u64;
                                gpt_rs::profiling::ScopeMeta::default().with_work(
                                    gpt_rs::profiling::WorkStats {
                                        elements: out_elems,
                                        bytes_read: rhs_bytes.saturating_add(lhs_bytes),
                                        bytes_written: out_bytes,
                                        flops,
                                        alloc_bytes: 0,
                                        alloc_count: 0,
                                    },
                                )
                            },
                        );
                        matmul(
                            &mut out_view,
                            Accum::Replace,
                            ctx.rhs_t,
                            lhs_t,
                            1.0f32,
                            Par::Seq,
                        );
                    }
                }

                if let Some(bias) = ctx.bias_values {
                    let _bias_scope = gpt_rs::profiling::backend_scope_with_meta(
                        "backend.faer.conv2d.bias_add",
                        || {
                            let out_elems = (cur_rows as u64).saturating_mul(ctx.c_out as u64);
                            let bytes_per_elem = 4u64;
                            let out_bytes = out_elems.saturating_mul(bytes_per_elem);
                            let bias_bytes = (ctx.c_out as u64).saturating_mul(bytes_per_elem);
                            gpt_rs::profiling::ScopeMeta::default().with_work(
                                gpt_rs::profiling::WorkStats {
                                    elements: out_elems,
                                    bytes_read: out_bytes.saturating_add(bias_bytes),
                                    bytes_written: out_bytes,
                                    flops: out_elems,
                                    alloc_bytes: 0,
                                    alloc_count: 0,
                                },
                            )
                        },
                    );
                    for row in out_chunk.chunks_exact_mut(ctx.c_out) {
                        for (slot, b) in row.iter_mut().zip(bias.iter()) {
                            *slot += *b;
                        }
                    }
                }
            }

            Ok(())
        })
    }

    fn matmul_tiled_parallel(
        par: Par,
        mut out: MatMut<'_, f32>,
        rhs_t: MatRef<'_, f32>,
        lhs_t: MatRef<'_, f32>,
    ) -> Result<()> {
        const MIN_ROWS: usize = 64;
        const MIN_COLS: usize = 16;

        let nrows = out.nrows();
        let ncols = out.ncols();
        if par.degree() <= 1 || nrows == 0 || ncols == 0 {
            matmul(&mut out, Accum::Replace, rhs_t, lhs_t, 1.0f32, Par::Seq);
            return Ok(());
        }

        let row_splits = nrows / MIN_ROWS;
        let col_splits = ncols / MIN_COLS;
        let split_rows = nrows >= 2 * MIN_ROWS && row_splits >= col_splits;

        let mut left_res: Result<()> = Ok(());
        let mut right_res: Result<()> = Ok(());

        if split_rows {
            let mid = nrows / 2;
            let (out_left, out_right) = out.split_at_row_mut(mid);
            let (rhs_left, rhs_right) = rhs_t.split_at_row(mid);

            faer::utils::thread::join_raw(
                |par_left| {
                    left_res = matmul_tiled_parallel(par_left, out_left, rhs_left, lhs_t);
                },
                |par_right| {
                    right_res = matmul_tiled_parallel(par_right, out_right, rhs_right, lhs_t);
                },
                par,
            );
        } else if ncols >= 2 * MIN_COLS {
            let mid = ncols / 2;
            let (out_left, out_right) = out.split_at_col_mut(mid);
            let (lhs_left, lhs_right) = lhs_t.split_at_col(mid);

            faer::utils::thread::join_raw(
                |par_left| {
                    left_res = matmul_tiled_parallel(par_left, out_left, rhs_t, lhs_left);
                },
                |par_right| {
                    right_res = matmul_tiled_parallel(par_right, out_right, rhs_t, lhs_right);
                },
                par,
            );
        } else {
            matmul(&mut out, Accum::Replace, rhs_t, lhs_t, 1.0f32, Par::Seq);
            return Ok(());
        }

        left_res?;
        right_res?;
        Ok(())
    }

    fn process_parallel<F>(
        par: Par,
        row_start: usize,
        row_end: usize,
        c_out: usize,
        out_slice: &mut [f32],
        process: &F,
    ) -> Result<()>
    where
        F: Fn(usize, usize, &mut [f32]) -> Result<()> + Send + Sync,
    {
        let rows = row_end - row_start;
        if rows == 0 {
            return Ok(());
        }

        if par.degree() <= 1 || rows == 1 {
            return process(row_start, row_end, out_slice);
        }

        let mid = row_start + rows / 2;
        let left_rows = mid - row_start;
        let (out_left, out_right) = out_slice.split_at_mut(left_rows * c_out);
        let mut left_res: Result<()> = Ok(());
        let mut right_res: Result<()> = Ok(());

        faer::utils::thread::join_raw(
            |par_left| {
                left_res = process_parallel(par_left, row_start, mid, c_out, out_left, process)
            },
            |par_right| {
                right_res = process_parallel(par_right, mid, row_end, c_out, out_right, process)
            },
            par,
        );

        left_res?;
        right_res?;
        Ok(())
    }

    let ctx = Conv2dIm2colContext {
        out_w,
        out_hw,
        in_stride_n,
        in_stride_h,
        k_h,
        k_w,
        c_in,
        k,
        c_out,
        block_rows_cap,
        s_h_isize,
        s_w_isize,
        d_h_isize,
        d_w_isize,
        pad_top_isize,
        pad_left_isize,
        h_isize,
        w_isize,
        input_values,
        rhs_t,
        bias_values,
    };

    let use_channel_parallel = par.degree() > 1 && m <= 64 && c_out >= 256;
    if use_channel_parallel {
        process_range(ctx, 0, m, output_slice, par)?;
    } else {
        let process = |row_start: usize, row_end: usize, out_slice: &mut [f32]| -> Result<()> {
            process_range(ctx, row_start, row_end, out_slice, Par::Seq)
        };
        process_parallel(par, 0, m, c_out, output_slice, &process)?;
    }

    let output = unsafe {
        let ptr = output_uninit.as_mut_ptr().cast::<f32>();
        let len = output_uninit.len();
        let cap = output_uninit.capacity();
        std::mem::forget(output_uninit);
        Vec::from_raw_parts(ptr, len, cap)
    };

    output_tensor(output_spec, TensorData::F32(Buffer::from(output)))
}

fn depthwise_conv2d_nhwc_direct_f32(
    input: &HostTensor,
    weight: &HostTensor,
    bias: Option<&HostTensor>,
    spec: &ExtractPatches,
    output_spec: &TensorType,
) -> Result<HostTensor> {
    let input_dims = input.dims.clone();
    let weight_dims = weight.dims.clone();
    let out_dims = dims_of(output_spec)?;

    if input.dtype() != DType::F32
        || weight.dtype() != DType::F32
        || output_spec.dtype != DType::F32
    {
        return Err(anyhow!("depthwise conv fusion only supports f32",));
    }
    if input_dims.len() != 4 || out_dims.len() != 4 {
        return Err(anyhow!("depthwise conv fusion expects rank-4 NHWC",));
    }

    if spec.window.len() != 2
        || spec.strides.len() != 2
        || spec.dilation.len() != 2
        || spec.padding.len() != 2
    {
        return Err(anyhow!(
            "depthwise conv fusion expects 2D window/stride/dilation/padding",
        ));
    }

    let (n, h, w, c) = (input_dims[0], input_dims[1], input_dims[2], input_dims[3]);
    let (out_n, out_h, out_w, out_c) = (out_dims[0], out_dims[1], out_dims[2], out_dims[3]);
    if out_n != n || out_c != c {
        return Err(anyhow!("depthwise conv fusion output shape mismatch",));
    }

    let (k_h, k_w) = (spec.window[0], spec.window[1]);
    let (s_h, s_w) = (spec.strides[0], spec.strides[1]);
    let (d_h, d_w) = (spec.dilation[0], spec.dilation[1]);
    let pad_top = spec.padding[0].0;
    let pad_left = spec.padding[1].0;

    let khkw = k_h
        .checked_mul(k_w)
        .ok_or_else(|| anyhow!("depthwise conv fusion kernel overflow"))?;

    if weight_dims[..] != [c, khkw] {
        return Err(anyhow!(
            "the depthwise kernel expects a weight [C, KH * KW]"
        ));
    }

    if let Some(bias) = bias {
        let bias_dims = bias.dims.clone();
        if bias.dtype() != DType::F32 || bias_dims.iter().product::<usize>() != c {
            return Err(anyhow!("depthwise conv fusion bias shape mismatch",));
        }
    }

    let input_values = match &input.data {
        TensorData::F32(values) => values.as_ref(),
        _ => return Err(anyhow!("depthwise conv fusion expects f32 input",)),
    };
    let weight_values = match &weight.data {
        TensorData::F32(values) => values.as_ref(),
        _ => return Err(anyhow!("depthwise conv fusion expects f32 weight",)),
    };
    let bias_values = match bias {
        Some(bias) => match &bias.data {
            TensorData::F32(values) => Some(values.as_ref()),
            _ => return Err(anyhow!("depthwise conv fusion expects f32 bias",)),
        },
        None => None,
    };

    let out_len = out_dims
        .iter()
        .try_fold(1usize, |acc, &d| acc.checked_mul(d))
        .ok_or_else(|| anyhow!("depthwise conv fusion output size overflow"))?;
    let mut out = vec![0.0f32; out_len];

    let in_hw = h
        .checked_mul(w)
        .ok_or_else(|| anyhow!("depthwise conv fusion in_hw overflow"))?;
    let in_stride_n = in_hw
        .checked_mul(c)
        .ok_or_else(|| anyhow!("depthwise conv fusion in_stride overflow"))?;
    let in_stride_h = w
        .checked_mul(c)
        .ok_or_else(|| anyhow!("depthwise conv fusion in_stride overflow"))?;

    let out_hw = out_h
        .checked_mul(out_w)
        .ok_or_else(|| anyhow!("depthwise conv fusion out_hw overflow"))?;

    let s_h_isize = s_h as isize;
    let s_w_isize = s_w as isize;
    let d_h_isize = d_h as isize;
    let d_w_isize = d_w as isize;
    let pad_top_isize = pad_top as isize;
    let pad_left_isize = pad_left as isize;
    let h_isize = h as isize;
    let w_isize = w as isize;

    for n_idx in 0..n {
        let in_n_base = n_idx * in_stride_n;
        let out_n_base = n_idx * out_hw * c;
        for oh in 0..out_h {
            let base_h = oh as isize * s_h_isize - pad_top_isize;
            let out_h_base = out_n_base + oh * out_w * c;
            for ow in 0..out_w {
                let base_w = ow as isize * s_w_isize - pad_left_isize;
                let out_base = out_h_base + ow * c;

                // Initialize with bias.
                if let Some(bias) = bias_values {
                    out[out_base..out_base + c].copy_from_slice(bias);
                }

                for kh in 0..k_h {
                    let ih = base_h + kh as isize * d_h_isize;
                    if ih < 0 || ih >= h_isize {
                        continue;
                    }
                    let in_h_base = in_n_base
                        + (ih as usize)
                            .checked_mul(in_stride_h)
                            .ok_or_else(|| anyhow!("depthwise conv fusion offset overflow"))?;
                    for kw in 0..k_w {
                        let iw = base_w + kw as isize * d_w_isize;
                        if iw < 0 || iw >= w_isize {
                            continue;
                        }
                        let in_base = in_h_base
                            + (iw as usize)
                                .checked_mul(c)
                                .ok_or_else(|| anyhow!("depthwise conv fusion offset overflow"))?;

                        let khkw_idx = kh * k_w + kw;
                        // For each channel, accumulate input * weight[channel, khkw].
                        for ch in 0..c {
                            let w_idx = ch * khkw + khkw_idx;
                            out[out_base + ch] += input_values[in_base + ch] * weight_values[w_idx];
                        }
                    }
                }
            }
        }
    }

    output_tensor(output_spec, TensorData::F32(Buffer::from(out)))
}

/// IEEE 754-2019 maximum, as the IR interpreter defines `tensor.maximum`: NaN when either operand
/// is NaN, and -0.0 below +0.0. Equal operands differ at most in the sign of a zero, and ANDing
/// their bits keeps +0.0. Written as selects, so loops over it vectorize.
#[inline]
pub(crate) fn maximum(lhs: f32, rhs: f32) -> f32 {
    let larger = if lhs < rhs { rhs } else { lhs };
    let equal = f32::from_bits(lhs.to_bits() & rhs.to_bits());
    let result = if lhs == rhs { equal } else { larger };
    if rhs.is_nan() {
        rhs
    } else {
        result
    }
}

/// IEEE 754-2019 minimum, as the IR interpreter defines `tensor.minimum`: NaN when either operand
/// is NaN, and -0.0 below +0.0. Equal operands differ at most in the sign of a zero, and ORing
/// their bits keeps -0.0. Written as selects, so loops over it vectorize.
#[inline]
pub(crate) fn minimum(lhs: f32, rhs: f32) -> f32 {
    let smaller = if lhs > rhs { rhs } else { lhs };
    let equal = f32::from_bits(lhs.to_bits() | rhs.to_bits());
    let result = if lhs == rhs { equal } else { smaller };
    if rhs.is_nan() {
        rhs
    } else {
        result
    }
}

fn try_elementwise_binary(
    inputs: &[HostTensor],
    outputs: &[TensorType],
    op: BinaryKind,
) -> Option<Result<Vec<HostTensor>>> {
    if inputs.len() != 2 || outputs.len() != 1 {
        return None;
    }
    let lhs = &inputs[0];
    let rhs = &inputs[1];
    let output_spec = &outputs[0];

    let (lhs_values, rhs_values) = match (&lhs.data, &rhs.data) {
        (TensorData::F32(a), TensorData::F32(b)) => (a.as_ref(), b.as_ref()),
        _ => return None,
    };

    if lhs.dtype() != DType::F32 || rhs.dtype() != DType::F32 || output_spec.dtype != DType::F32 {
        return None;
    }
    if lhs_values.len() != rhs_values.len() {
        return Some(Err(anyhow!("elementwise size mismatch")));
    }

    match op {
        BinaryKind::Add | BinaryKind::Maximum | BinaryKind::Minimum | BinaryKind::Mul => {}
        _ => return None,
    }

    let len = lhs_values.len();
    let mut out = vec![0.0f32; len];
    match op {
        BinaryKind::Add => {
            for i in 0..len {
                out[i] = lhs_values[i] + rhs_values[i];
            }
        }
        BinaryKind::Maximum => {
            for i in 0..len {
                out[i] = maximum(lhs_values[i], rhs_values[i]);
            }
        }
        BinaryKind::Minimum => {
            for i in 0..len {
                out[i] = minimum(lhs_values[i], rhs_values[i]);
            }
        }
        BinaryKind::Mul => {
            for i in 0..len {
                out[i] = lhs_values[i] * rhs_values[i];
            }
        }
        _ => unreachable!("filtered above"),
    }

    Some(output_tensor(output_spec, TensorData::F32(Buffer::from(out))).map(|tensor| vec![tensor]))
}

fn try_reduce_window(
    inputs: &[HostTensor],
    outputs: &[TensorType],
    spec: &ReduceWindow,
) -> Option<Result<Vec<HostTensor>>> {
    if inputs.len() != 1 || outputs.len() != 1 {
        return None;
    }
    if spec.kind != ReduceKind::Max {
        return None;
    }
    let input = &inputs[0];
    let output_spec = &outputs[0];
    if input.dtype() != DType::F32 || output_spec.dtype != DType::F32 {
        return None;
    }
    let (
        &[1, k_h, k_w, 1],
        &[1, s_h, s_w, 1],
        &[1, 1, 1, 1],
        &[1, 1, 1, 1],
        &[(0, 0), (p_h, _), (p_w, _), (0, 0)],
    ) = (
        spec.window_dims.as_slice(),
        spec.strides.as_slice(),
        spec.base_dilation.as_slice(),
        spec.window_dilation.as_slice(),
        spec.padding.as_slice(),
    )
    else {
        return None;
    };

    let input_dims = input.dims.clone();
    let out_dims = match dims_of(output_spec) {
        Ok(dims) => dims,
        Err(err) => return Some(Err(err)),
    };
    if input_dims.len() != 4 || out_dims.len() != 4 {
        return None;
    }

    let (n, h, w, c) = (input_dims[0], input_dims[1], input_dims[2], input_dims[3]);
    let (out_n, out_h, out_w, out_c) = (out_dims[0], out_dims[1], out_dims[2], out_dims[3]);
    if out_n != n || out_c != c {
        return Some(Err(anyhow!(
            "reduce_window expects batch/channels preserved",
        )));
    }

    let values = match &input.data {
        TensorData::F32(values) => values.as_ref(),
        _ => return None,
    };

    let out_len = match out_n
        .checked_mul(out_h)
        .and_then(|v| v.checked_mul(out_w))
        .and_then(|v| v.checked_mul(out_c))
    {
        Some(v) => v,
        None => return Some(Err(anyhow!("reduce_window size overflow"))),
    };
    let mut out = vec![f32::NEG_INFINITY; out_len];

    let in_hw = match h.checked_mul(w) {
        Some(v) => v,
        None => return Some(Err(anyhow!("reduce_window in_hw overflow"))),
    };
    let in_stride_n = match in_hw.checked_mul(c) {
        Some(v) => v,
        None => return Some(Err(anyhow!("reduce_window stride overflow",))),
    };
    let in_stride_h = match w.checked_mul(c) {
        Some(v) => v,
        None => return Some(Err(anyhow!("reduce_window stride overflow",))),
    };

    for n_idx in 0..n {
        let in_n_base = n_idx * in_stride_n;
        for oh in 0..out_h {
            let ih0 = (oh * s_h) as isize - p_h as isize;
            for ow in 0..out_w {
                let iw0 = (ow * s_w) as isize - p_w as isize;
                let dst_offset = ((n_idx * out_h + oh) * out_w + ow) * c;
                let dst = &mut out[dst_offset..dst_offset + c];

                for kh in 0..k_h as isize {
                    let ih = ih0 + kh;
                    if ih < 0 || ih >= h as isize {
                        continue;
                    }
                    let ih = ih as usize;
                    let in_h_base = in_n_base + ih * in_stride_h;
                    for kw in 0..k_w as isize {
                        let iw = iw0 + kw;
                        if iw < 0 || iw >= w as isize {
                            continue;
                        }
                        let iw = iw as usize;
                        let src_offset = in_h_base + iw * c;
                        let src = &values[src_offset..src_offset + c];
                        for i in 0..c {
                            dst[i] = maximum(dst[i], src[i]);
                        }
                    }
                }
            }
        }
    }

    Some(output_tensor(output_spec, TensorData::F32(Buffer::from(out))).map(|tensor| vec![tensor]))
}

/// The type of the tensor `value`, which a kernel step gives.
fn select_output_type(module: &Module, value: Value) -> Result<TensorType> {
    let ty = module.ty(value);
    ty.as_tensor()
        .cloned()
        .ok_or_else(|| anyhow!("a kernel result of type {ty} is not a tensor"))
}

/// Runs a convolution step on its tensors.
fn run_conv2d(
    conv: &select::Conv2d,
    input: &HostTensor,
    weight: &HostTensor,
    bias: Option<&HostTensor>,
    output: &TensorType,
) -> Result<HostTensor> {
    let patches = &conv.patches;
    let label = if conv.depthwise {
        "backend.faer.depthwise_conv2d"
    } else {
        "backend.faer.conv2d"
    };
    let _scope = gpt_rs::profiling::backend_scope_with_meta(label, || {
        let (window, strides, dilation, padding) = (
            &patches.window,
            &patches.strides,
            &patches.dilation,
            &patches.padding,
        );
        let signature = format!(
            "in={} weight={} out={output} window={window:?} strides={strides:?} dilation={dilation:?} padding={padding:?}",
            input.tensor_type(),
            weight.tensor_type(),
        );
        let bytes = |dims: &[usize]| {
            dims.iter()
                .fold(4u64, |acc, &v| acc.saturating_mul(v as u64))
        };
        let out_dims = dims_of(output).unwrap_or_default();
        let out_bytes = bytes(&out_dims);
        // Each result element sums one window over the input channels of its group.
        let group_channels = if conv.depthwise {
            1
        } else {
            input.dims.last().copied().unwrap_or(0)
        };
        let k = patches.window.iter().product::<usize>() * group_channels;
        let flops = out_dims
            .iter()
            .chain([&k, &2])
            .fold(1u64, |acc, &v| acc.saturating_mul(v as u64));
        let bias_bytes = bias.map_or(0, |bias| bytes(&bias.dims));
        let work = gpt_rs::profiling::WorkStats {
            elements: out_bytes / 4,
            bytes_read: bytes(&input.dims)
                .saturating_add(bytes(&weight.dims))
                .saturating_add(bias_bytes),
            bytes_written: out_bytes,
            flops,
            alloc_bytes: out_bytes,
            alloc_count: 1,
        };
        gpt_rs::profiling::signature_id(&signature)
            .map(gpt_rs::profiling::ScopeMeta::signature)
            .unwrap_or_default()
            .with_work(work)
    });
    if !conv.depthwise {
        return conv2d_nhwc_im2col_tiled_f32(input, weight, bias, patches, output);
    }
    // The weight [C, 1, 1, KH, KW] holds [C, KH * KW] in the same order.
    let &[c, 1, 1, k_h, k_w] = weight.dims.as_slice() else {
        return Err(anyhow!("a depthwise weight of shape {:?}", weight.dims));
    };
    let weight = HostTensor::new(vec![c, k_h * k_w], weight.data.clone())?;
    depthwise_conv2d_nhwc_direct_f32(input, &weight, bias, patches, output)
}

fn try_dot_general(
    inputs: &[HostTensor],
    outputs: &[TensorType],
    spec: &DotGeneral,
) -> Option<Result<Vec<HostTensor>>> {
    if inputs.len() != 2 || outputs.len() != 1 {
        return None;
    }

    let lhs = &inputs[0];
    let rhs = &inputs[1];
    let output_spec = &outputs[0];

    let (lhs_values, rhs_values) = match (&lhs.data, &rhs.data) {
        (TensorData::F32(lhs_values), TensorData::F32(rhs_values)) => {
            (lhs_values.as_ref(), rhs_values.as_ref())
        }
        _ => return None,
    };

    if lhs.dtype() != DType::F32 || rhs.dtype() != DType::F32 || output_spec.dtype != DType::F32 {
        return None;
    }

    if let Some(accum) = spec.accum {
        if accum != DType::F32 {
            return None;
        }
    }

    let lhs_dims = lhs.dims.clone();
    let rhs_dims = rhs.dims.clone();
    let out_dims = match dims_of(output_spec) {
        Ok(dims) => dims,
        Err(err) => return Some(Err(err)),
    };

    if lhs_dims.len() == 2
        && rhs_dims.len() == 2
        && spec.batch_lhs.is_empty()
        && spec.batch_rhs.is_empty()
        && spec.contract_lhs.as_slice() == [1]
        && spec.contract_rhs.as_slice() == [0]
    {
        return Some(dot_general_2d(
            lhs_values,
            rhs_values,
            &lhs_dims,
            &rhs_dims,
            &out_dims,
            output_spec,
        ));
    }

    // Linear layout `x [M, K] . w [N, K]`: a single-batch rhs-transposed matmul.
    if lhs_dims.len() == 2
        && rhs_dims.len() == 2
        && out_dims.len() == 2
        && spec.batch_lhs.is_empty()
        && spec.batch_rhs.is_empty()
        && spec.contract_lhs.as_slice() == [1]
        && spec.contract_rhs.as_slice() == [1]
    {
        return Some(dot_general_batched_rhs_transposed(
            lhs_values,
            rhs_values,
            &[1, lhs_dims[0], lhs_dims[1]],
            &[1, rhs_dims[0], rhs_dims[1]],
            &[1, out_dims[0], out_dims[1]],
            output_spec,
        ));
    }

    if lhs_dims.len() == 3
        && rhs_dims.len() == 3
        && spec.batch_lhs.as_slice() == [0]
        && spec.batch_rhs.as_slice() == [0]
        && spec.contract_lhs.as_slice() == [2]
        && spec.contract_rhs.as_slice() == [1]
    {
        return Some(dot_general_batched(
            lhs_values,
            rhs_values,
            &lhs_dims,
            &rhs_dims,
            &out_dims,
            output_spec,
        ));
    }

    // Batched matmul where rhs is laid out as [B, N, K] (contracting rhs axis 2).
    if lhs_dims.len() == 3
        && rhs_dims.len() == 3
        && spec.batch_lhs.as_slice() == [0]
        && spec.batch_rhs.as_slice() == [0]
        && spec.contract_lhs.as_slice() == [2]
        && spec.contract_rhs.as_slice() == [2]
    {
        return Some(dot_general_batched_rhs_transposed(
            lhs_values,
            rhs_values,
            &lhs_dims,
            &rhs_dims,
            &out_dims,
            output_spec,
        ));
    }

    None
}

fn dot_general_2d(
    lhs: &[f32],
    rhs: &[f32],
    lhs_dims: &[usize],
    rhs_dims: &[usize],
    out_dims: &[usize],
    output_spec: &TensorType,
) -> Result<Vec<HostTensor>> {
    let (m, k_lhs) = (lhs_dims[0], lhs_dims[1]);
    let (k_rhs, n) = (rhs_dims[0], rhs_dims[1]);

    if k_lhs != k_rhs {
        return Err(anyhow!("dot_general lhs/rhs contract mismatch",));
    }

    if out_dims.len() != 2 || out_dims[0] != m || out_dims[1] != n {
        return Err(anyhow!("dot_general output shape mismatch"));
    }

    if lhs.len() != m * k_lhs || rhs.len() != k_rhs * n {
        return Err(anyhow!("dot_general operand size mismatch"));
    }

    if m == 0 || n == 0 || k_lhs == 0 {
        return output_tensor(
            output_spec,
            TensorData::F32(Buffer::from(vec![0.0f32; m * n])),
        )
        .map(|tensor| vec![tensor]);
    }

    let lhs_view = MatRef::from_row_major_slice(lhs, m, k_lhs);
    let rhs_view = MatRef::from_row_major_slice(rhs, k_rhs, n);

    // faer prefers column-major output. To avoid an explicit output transpose/copy, compute
    // C^T = B^T * A^T into a column-major (n x m) matrix. The underlying buffer layout matches
    // row-major (m x n) for C, so we can return it directly.
    let a_t = lhs_view.transpose();
    let b_t = rhs_view.transpose();
    let mut row_major = vec![0.0f32; m * n];
    let mut out_view = MatMut::from_column_major_slice_mut(row_major.as_mut_slice(), n, m);
    let par = faer_parallelism();
    matmul(&mut out_view, Accum::Replace, b_t, a_t, 1.0f32, par);

    output_tensor(output_spec, TensorData::F32(Buffer::from(row_major))).map(|tensor| vec![tensor])
}

fn dot_general_batched(
    lhs: &[f32],
    rhs: &[f32],
    lhs_dims: &[usize],
    rhs_dims: &[usize],
    out_dims: &[usize],
    output_spec: &TensorType,
) -> Result<Vec<HostTensor>> {
    let (batch, m, k_lhs) = (lhs_dims[0], lhs_dims[1], lhs_dims[2]);
    let (batch_rhs, k_rhs, n) = (rhs_dims[0], rhs_dims[1], rhs_dims[2]);

    if batch != batch_rhs {
        return Err(anyhow!("dot_general batch mismatch"));
    }

    if k_lhs != k_rhs {
        return Err(anyhow!("dot_general lhs/rhs contract mismatch",));
    }

    if out_dims.len() != 3 || out_dims[0] != batch || out_dims[1] != m || out_dims[2] != n {
        return Err(anyhow!("dot_general output shape mismatch"));
    }

    if lhs.len() != batch * m * k_lhs || rhs.len() != batch * k_rhs * n {
        return Err(anyhow!("dot_general operand size mismatch"));
    }

    if batch == 0 || m == 0 || n == 0 || k_lhs == 0 {
        return output_tensor(
            output_spec,
            TensorData::F32(Buffer::from(vec![0.0f32; batch * m * n])),
        )
        .map(|tensor| vec![tensor]);
    }

    let lhs_batch_stride = m * k_lhs;
    let rhs_batch_stride = k_rhs * n;
    let out_batch_stride = m * n;

    let mut row_output = vec![0.0f32; batch * out_batch_stride];
    let par = faer_parallelism();

    for b in 0..batch {
        let lhs_slice = &lhs[b * lhs_batch_stride..(b + 1) * lhs_batch_stride];
        let rhs_slice = &rhs[b * rhs_batch_stride..(b + 1) * rhs_batch_stride];

        let lhs_view = MatRef::from_row_major_slice(lhs_slice, m, k_lhs);
        let rhs_view = MatRef::from_row_major_slice(rhs_slice, k_rhs, n);

        let a_t = lhs_view.transpose();
        let b_t = rhs_view.transpose();
        let dst = &mut row_output[b * out_batch_stride..(b + 1) * out_batch_stride];
        let mut out_view = MatMut::from_column_major_slice_mut(dst, n, m);
        matmul(&mut out_view, Accum::Replace, b_t, a_t, 1.0f32, par);
    }

    output_tensor(output_spec, TensorData::F32(Buffer::from(row_output))).map(|tensor| vec![tensor])
}

fn dot_general_batched_rhs_transposed(
    lhs: &[f32],
    rhs: &[f32],
    lhs_dims: &[usize],
    rhs_dims: &[usize],
    out_dims: &[usize],
    output_spec: &TensorType,
) -> Result<Vec<HostTensor>> {
    let (batch, m, k_lhs) = (lhs_dims[0], lhs_dims[1], lhs_dims[2]);
    let (batch_rhs, n, k_rhs) = (rhs_dims[0], rhs_dims[1], rhs_dims[2]);

    if batch != batch_rhs {
        return Err(anyhow!("dot_general batch mismatch"));
    }

    if k_lhs != k_rhs {
        return Err(anyhow!("dot_general lhs/rhs contract mismatch",));
    }

    if out_dims.len() != 3 || out_dims[0] != batch || out_dims[1] != m || out_dims[2] != n {
        return Err(anyhow!("dot_general output shape mismatch"));
    }

    if lhs.len() != batch * m * k_lhs || rhs.len() != batch * n * k_rhs {
        return Err(anyhow!("dot_general operand size mismatch"));
    }

    if batch == 0 || m == 0 || n == 0 || k_lhs == 0 {
        return output_tensor(
            output_spec,
            TensorData::F32(Buffer::from(vec![0.0f32; batch * m * n])),
        )
        .map(|tensor| vec![tensor]);
    }

    let lhs_batch_stride = m * k_lhs;
    let rhs_batch_stride = n * k_rhs;
    let out_batch_stride = m * n;

    let mut row_output = vec![0.0f32; batch * out_batch_stride];
    let par = faer_parallelism();

    for b in 0..batch {
        let lhs_slice = &lhs[b * lhs_batch_stride..(b + 1) * lhs_batch_stride];
        let rhs_slice = &rhs[b * rhs_batch_stride..(b + 1) * rhs_batch_stride];

        let lhs_view = MatRef::from_row_major_slice(lhs_slice, m, k_lhs);
        let rhs_view = MatRef::from_row_major_slice(rhs_slice, n, k_rhs);

        // Compute C^T = B * A^T into a column-major (n x m) matrix.
        // The underlying buffer layout matches row-major (m x n) for C.
        let a_t = lhs_view.transpose();
        let dst = &mut row_output[b * out_batch_stride..(b + 1) * out_batch_stride];
        let mut out_view = MatMut::from_column_major_slice_mut(dst, n, m);
        matmul(&mut out_view, Accum::Replace, rhs_view, a_t, 1.0f32, par);
    }

    output_tensor(output_spec, TensorData::F32(Buffer::from(row_output))).map(|tensor| vec![tensor])
}
