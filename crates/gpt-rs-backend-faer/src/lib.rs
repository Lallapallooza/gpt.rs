//! The faer CPU backend. It runs a function op by op on host memory: an op that faer has a kernel
//! for runs on it, and every other op on the IR's interpreter.

use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{anyhow, ensure, Context, Result};
use faer::linalg::matmul::matmul;
use faer::mat::{MatMut, MatRef};
use faer::{Accum, Par};
use gpt_rs::backend::{ParamStore, PortableBackend};
use gpt_rs_ir::dialects::tensor::{DotGeneral, ExtractPatches, ReduceWindowAttrs, TensorOps};
use gpt_rs_ir::host::{Buffer, HostTensor, TensorData};
use gpt_rs_ir::interp::HostValue;
use gpt_rs_ir::ir::{DType, Inst, Module, TensorType, Value};
use gpt_rs_ir::traits::Terminator;
use gpt_rs_ir::{interp, Op};

mod layout;
mod linear;
mod reduce;
mod select;

use self::select::{Step, StepKind};

thread_local! {
    static CONV2D_IM2COL_SCRATCH_F32: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
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
        let function = module
            .func(func)
            .ok_or_else(|| anyhow!("the module has no function @{func}"))?;
        let body = module.regions(function)[0];
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

    fn run(&self, exe: &Function, args: Vec<HostTensor>) -> Result<Vec<HostTensor>> {
        ensure!(
            args.len() == exe.args.len(),
            "the function takes {} arguments, got {}",
            exe.args.len(),
            args.len()
        );
        let module = exe.module.as_ref();
        let mut env: HashMap<Value, HostTensor> = exe.args.iter().copied().zip(args).collect();
        let get = |bindings: &HashMap<Value, HostTensor>, value: Value| {
            bindings
                .get(&value)
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
                    let tensor = get(&env, input)?;
                    env.insert(output, relu6_f32(&tensor)?);
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

/// faer's kernel for `tensor_op` on `inputs`, with results of the types `outputs`, when it has
/// one.
fn kernel(
    tensor_op: &TensorOps,
    inputs: &[HostTensor],
    outputs: &[TensorType],
) -> Option<Result<Vec<HostTensor>>> {
    let one = |result: Result<HostTensor>| Some(result.map(|tensor| vec![tensor]));
    let one_of = |result: Option<Result<HostTensor>>| {
        result.map(|computed| computed.map(|tensor| vec![tensor]))
    };
    match (tensor_op, inputs, outputs) {
        (TensorOps::DotGeneral(dot), ..) => try_dot_general(inputs, outputs, dot),
        (TensorOps::Add(_), ..) => try_elementwise_binary(inputs, outputs, |lhs, rhs| lhs + rhs),
        (TensorOps::Mul(_), ..) => try_elementwise_binary(inputs, outputs, |lhs, rhs| lhs * rhs),
        (TensorOps::Maximum(_), ..) => try_elementwise_binary(inputs, outputs, maximum),
        (TensorOps::Minimum(_), ..) => try_elementwise_binary(inputs, outputs, minimum),
        (TensorOps::ReduceWindowMax(window), ..) => try_reduce_window(inputs, outputs, window),
        (TensorOps::ReduceSum(op), [input], [out]) => {
            one_of(reduce::reduce(op, input, out, 0.0, |lhs, rhs| lhs + rhs))
        }
        (TensorOps::ReduceMax(op), [input], [out]) => {
            one_of(reduce::reduce(op, input, out, f32::NEG_INFINITY, maximum))
        }
        (TensorOps::ReduceMin(op), [input], [out]) => {
            one_of(reduce::reduce(op, input, out, f32::INFINITY, minimum))
        }
        (TensorOps::BroadcastTo(op), [input], [out]) => one(layout::broadcast_to(op, input, out)),
        (TensorOps::Transpose(op), [input], [out]) => one(layout::transpose(op, input, out)),
        (TensorOps::Slice(op), [input], [out]) => one(layout::slice(op, input, out)),
        (TensorOps::DynamicSlice(op), [input, starts], [out]) => {
            one(layout::dynamic_slice(op, input, starts, out))
        }
        (TensorOps::DynamicUpdateSlice(_), [base, update, starts], _) => {
            one(layout::dynamic_update_slice(base, update, starts))
        }
        (TensorOps::Concat(op), _, [out]) => one(layout::concat(op, inputs, out)),
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
        .map(|&value| {
            let low = if value <= 0.0 { 0.0 } else { value };
            if low >= 6.0 {
                6.0
            } else {
                low
            }
        })
        .collect();
    HostTensor::f32(input.dims, out)
}

/// The extents of `ty`, which must be static.
pub(crate) fn dims_of(ty: &TensorType) -> Result<Vec<usize>> {
    ty.static_dims()
        .map(|dims| dims.to_vec())
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

/// An f32 NHWC convolution with the weight `[KH * KW * C_in, C_out]` as im2col: blocks of output
/// rows, one per output pixel, gather their patches into a scratch buffer, which a matmul with
/// the weight turns into the rows' results.
fn conv2d_nhwc_im2col_tiled_f32(
    input: &HostTensor,
    weight: &HostTensor,
    bias: Option<&HostTensor>,
    spec: &ExtractPatches,
    output_spec: &TensorType,
) -> Result<HostTensor> {
    let (shape, patch_len) = conv2d_im2col_shape(input, weight, bias, spec, output_spec)?;
    let input_values = f32_values(input, "conv2d fusion expects f32 input")?;
    let weight_values = f32_values(weight, "conv2d fusion expects f32 weight")?;
    let bias_values = bias
        .map(|tensor| f32_values(tensor, "conv2d fusion expects f32 bias"))
        .transpose()?;

    let output_rows = shape
        .batch
        .checked_mul(shape.out_h)
        .and_then(|rows| rows.checked_mul(shape.out_w))
        .ok_or_else(|| anyhow!("conv2d fusion output size overflow"))?;

    if output_rows == 0 || shape.out_c == 0 {
        return output_tensor(
            output_spec,
            TensorData::F32(Buffer::from(Vec::<f32>::new())),
        );
    }

    let output_len = output_rows
        .checked_mul(shape.out_c)
        .ok_or_else(|| anyhow!("conv2d fusion output overflow"))?;
    let mut output_uninit = vec![std::mem::MaybeUninit::<f32>::uninit(); output_len];
    // Safety: `MaybeUninit<f32>` has the same memory layout as `f32`, and `process_parallel`
    // (via `matmul` with `Accum::Replace`) fully initializes all `output_len` elements on the
    // success path.
    let output_slice = unsafe {
        std::slice::from_raw_parts_mut(output_uninit.as_mut_ptr().cast::<f32>(), output_len)
    };

    let ctx = Conv2dIm2colContext::new(
        &shape,
        patch_len,
        output_rows,
        input_values,
        weight_values,
        bias_values,
    )?;
    let par = faer_parallelism();
    let use_channel_parallel = par.degree() > 1 && output_rows <= 64 && shape.out_c >= 256;
    if use_channel_parallel {
        process_range(ctx, 0, output_rows, output_slice, par)?;
    } else {
        let process = |row_start: usize, row_end: usize, out_slice: &mut [f32]| -> Result<()> {
            process_range(ctx, row_start, row_end, out_slice, Par::Seq)
        };
        process_parallel(par, 0, output_rows, shape.out_c, output_slice, &process)?;
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

/// The values of the f32 tensor `tensor`, or the error `message` for any other dtype.
fn f32_values<'a>(tensor: &'a HostTensor, message: &'static str) -> Result<&'a [f32]> {
    match &tensor.data {
        TensorData::F32(values) => Ok(values.as_ref()),
        _ => Err(anyhow!(message)),
    }
}

/// The extents of a 2-D NHWC convolution and of the window it slides over its input.
#[derive(Clone, Copy)]
struct Conv2dShape {
    batch: usize,
    in_h: usize,
    in_w: usize,
    in_c: usize,
    out_h: usize,
    out_w: usize,
    out_c: usize,
    k_h: usize,
    k_w: usize,
    s_h: usize,
    s_w: usize,
    d_h: usize,
    d_w: usize,
    pad_top: usize,
    pad_left: usize,
}

impl Conv2dShape {
    /// The shape of a convolution of the NHWC extents `input` to the NHWC extents `output`, with
    /// the two-axis window, strides, dilation and padding of `spec`.
    fn new(input: &[usize], output: &[usize], spec: &ExtractPatches) -> Self {
        Self {
            batch: input[0],
            in_h: input[1],
            in_w: input[2],
            in_c: input[3],
            out_h: output[1],
            out_w: output[2],
            out_c: output[3],
            k_h: spec.window[0],
            k_w: spec.window[1],
            s_h: spec.strides[0],
            s_w: spec.strides[1],
            d_h: spec.dilation[0],
            d_w: spec.dilation[1],
            pad_top: spec.padding[0].0,
            pad_left: spec.padding[1].0,
        }
    }
}

/// The shape of an im2col convolution, checked against its tensors, and the length of each
/// patch, `KH * KW * C_in`.
fn conv2d_im2col_shape(
    input: &HostTensor,
    weight: &HostTensor,
    bias: Option<&HostTensor>,
    spec: &ExtractPatches,
    output_spec: &TensorType,
) -> Result<(Conv2dShape, usize)> {
    let input_dims = input.dims;
    let weight_dims = weight.dims;
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
    if out_dims[0] != input_dims[0] {
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
    let shape = Conv2dShape::new(&input_dims, &out_dims, spec);

    let patch_len = shape
        .k_h
        .checked_mul(shape.k_w)
        .and_then(|len| len.checked_mul(shape.in_c))
        .ok_or_else(|| anyhow!("conv2d fusion kernel size overflow"))?;

    if weight_dims[0] != patch_len || weight_dims[1] != shape.out_c {
        return Err(anyhow!("conv2d fusion weight shape does not match kernel",));
    }

    if let Some(bias_tensor) = bias {
        let bias_dims = bias_tensor.dims;
        if bias_tensor.dtype() != DType::F32 || bias_dims.iter().product::<usize>() != shape.out_c {
            return Err(anyhow!("conv2d fusion bias shape mismatch"));
        }
    }
    Ok((shape, patch_len))
}

/// What an im2col convolution reads as it fills and multiplies its blocks of rows: the extents
/// and strides of its input, the values of its input and bias, and its weight transposed.
#[derive(Clone, Copy)]
struct Conv2dIm2colContext<'a> {
    out_w: usize,
    out_hw: usize,
    in_stride_n: usize,
    in_stride_h: usize,
    k_h: usize,
    k_w: usize,
    c_in: usize,
    patch_len: usize,
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

impl<'a> Conv2dIm2colContext<'a> {
    /// The context of a convolution of `shape` with `output_rows` output pixels, whose patches
    /// hold `patch_len` values. Fails when the values do not have the extents of the shape.
    fn new(
        shape: &Conv2dShape,
        patch_len: usize,
        output_rows: usize,
        input_values: &'a [f32],
        weight_values: &'a [f32],
        bias_values: Option<&'a [f32]>,
    ) -> Result<Self> {
        let rhs_view = MatRef::from_row_major_slice(weight_values, patch_len, shape.out_c);
        let rhs_t = rhs_view.transpose();

        let max_scratch_floats = 1_048_576usize; // 4 MiB
        let mut block_rows_cap = output_rows.min(8192);
        while block_rows_cap > 1 && block_rows_cap.saturating_mul(patch_len) > max_scratch_floats {
            block_rows_cap /= 2;
        }
        block_rows_cap = block_rows_cap.max(1);

        let out_hw = shape
            .out_h
            .checked_mul(shape.out_w)
            .ok_or_else(|| anyhow!("conv2d fusion out_hw overflow"))?;

        let in_hw = shape
            .in_h
            .checked_mul(shape.in_w)
            .ok_or_else(|| anyhow!("conv2d fusion in_hw overflow"))?;
        let in_stride_n = in_hw
            .checked_mul(shape.in_c)
            .ok_or_else(|| anyhow!("conv2d fusion in_stride overflow"))?;
        let in_stride_h = shape
            .in_w
            .checked_mul(shape.in_c)
            .ok_or_else(|| anyhow!("conv2d fusion in_stride overflow"))?;

        let expected_input_len = shape
            .batch
            .checked_mul(in_stride_n)
            .ok_or_else(|| anyhow!("conv2d fusion input size overflow"))?;
        if input_values.len() != expected_input_len {
            return Err(anyhow!("conv2d fusion input length mismatch",));
        }

        let expected_weight_len = patch_len
            .checked_mul(shape.out_c)
            .ok_or_else(|| anyhow!("conv2d fusion weight size overflow"))?;
        if weight_values.len() != expected_weight_len {
            return Err(anyhow!("conv2d fusion weight length mismatch",));
        }

        if let Some(bias) = bias_values {
            if bias.len() != shape.out_c {
                return Err(anyhow!("conv2d fusion bias length mismatch",));
            }
        }

        Ok(Self {
            out_w: shape.out_w,
            out_hw,
            in_stride_n,
            in_stride_h,
            k_h: shape.k_h,
            k_w: shape.k_w,
            c_in: shape.in_c,
            patch_len,
            c_out: shape.out_c,
            block_rows_cap,
            s_h_isize: shape.s_h as isize,
            s_w_isize: shape.s_w as isize,
            d_h_isize: shape.d_h as isize,
            d_w_isize: shape.d_w as isize,
            pad_top_isize: shape.pad_top as isize,
            pad_left_isize: shape.pad_left as isize,
            h_isize: shape.in_h as isize,
            w_isize: shape.in_w as isize,
            input_values,
            rhs_t,
            bias_values,
        })
    }
}

/// How the im2col fill walks the input: the output rows of an image, the input offsets between
/// the window's rows and between its columns, and how far past its first row and column a
/// window reaches.
#[derive(Clone, Copy)]
struct PatchWalk {
    out_h: usize,
    step_h: usize,
    step_w: usize,
    k_h_span: isize,
    k_w_span: isize,
}

impl PatchWalk {
    /// The walk of the patches that `ctx` reads.
    fn new(ctx: &Conv2dIm2colContext) -> Result<Self> {
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
        Ok(Self {
            out_h,
            step_h,
            step_w,
            k_h_span,
            k_w_span,
        })
    }
}

/// A block of the rows of an im2col convolution: the output rows `first_row..first_row + rows`,
/// whose patches take `scratch_len` values of scratch.
#[derive(Clone, Copy)]
struct Im2colBlock {
    first_row: usize,
    rows: usize,
    scratch_len: usize,
}

/// Computes the output rows `row_start..row_end` of an im2col convolution into `out_slice`, a
/// block of rows at a time. The block's patches fill this thread's scratch, and a matmul on the
/// threads of `matmul_par` multiplies them by the weight. Then it adds the bias to the block's
/// results.
fn process_range(
    ctx: Conv2dIm2colContext,
    row_start: usize,
    row_end: usize,
    out_slice: &mut [f32],
    matmul_par: Par,
) -> Result<()> {
    let range_rows = row_end - row_start;
    if range_rows == 0 {
        return Ok(());
    }

    let walk = PatchWalk::new(&ctx)?;
    let scratch_rows = range_rows.min(ctx.block_rows_cap);
    let scratch_capacity = scratch_rows
        .checked_mul(ctx.patch_len)
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
                .checked_mul(ctx.patch_len)
                .ok_or_else(|| anyhow!("conv2d fusion scratch overflow"))?;
            let out_offset = block_offset
                .checked_mul(ctx.c_out)
                .ok_or_else(|| anyhow!("conv2d fusion output overflow"))?;
            let out_chunk = &mut out_slice[out_offset..out_offset + cur_rows * ctx.c_out];
            let block = Im2colBlock {
                first_row: row_start + block_offset,
                rows: cur_rows,
                scratch_len,
            };

            if matmul_par.degree() > 1 {
                im2col_block_parallel(&ctx, &walk, scratch_cell, block, out_chunk, matmul_par)?;
            } else {
                im2col_block_sequential(&ctx, &walk, scratch_cell, block, out_chunk);
            }

            if let Some(bias) = ctx.bias_values {
                add_conv2d_bias(bias, ctx.c_out, cur_rows, out_chunk);
            }
        }

        Ok(())
    })
}

/// Fills the patches of `block` into this thread's scratch, then multiplies them by the weight
/// into `out_chunk` on the threads of `matmul_par`.
fn im2col_block_parallel(
    ctx: &Conv2dIm2colContext,
    walk: &PatchWalk,
    scratch_cell: &RefCell<Vec<f32>>,
    block: Im2colBlock,
    out_chunk: &mut [f32],
    matmul_par: Par,
) -> Result<()> {
    let scratch_ptr = {
        let mut scratch = scratch_cell.borrow_mut();
        let scratch_buf = &mut scratch[..block.scratch_len];
        let _fill_scope = im2col_fill_scope(block.scratch_len);
        fill_im2col_rows(ctx, walk, block.first_row, block.rows, scratch_buf);
        scratch_buf.as_ptr()
    };

    // Safety: `scratch_ptr` points into the thread-local scratch buffer for the
    // current thread, which stays alive for the duration of this call. We do not
    // borrow the scratch buffer again until after the parallel matmul completes.
    let scratch_view = unsafe { std::slice::from_raw_parts(scratch_ptr, block.scratch_len) };
    let lhs_view = MatRef::from_row_major_slice(scratch_view, block.rows, ctx.patch_len);
    let lhs_t = lhs_view.transpose();

    let out_view = MatMut::from_column_major_slice_mut(out_chunk, ctx.c_out, block.rows);
    let _matmul_scope = im2col_matmul_scope(ctx, block.rows);
    matmul_tiled_parallel(matmul_par, out_view, ctx.rhs_t, lhs_t)
}

/// Fills the patches of `block` into this thread's scratch, then multiplies them by the weight
/// into `out_chunk` on this thread.
fn im2col_block_sequential(
    ctx: &Conv2dIm2colContext,
    walk: &PatchWalk,
    scratch_cell: &RefCell<Vec<f32>>,
    block: Im2colBlock,
    out_chunk: &mut [f32],
) {
    let mut scratch = scratch_cell.borrow_mut();
    let scratch_buf = &mut scratch[..block.scratch_len];
    {
        let _fill_scope = im2col_fill_scope(block.scratch_len);
        fill_im2col_rows(ctx, walk, block.first_row, block.rows, scratch_buf);
    }
    let lhs_view = MatRef::from_row_major_slice(scratch_buf, block.rows, ctx.patch_len);
    let lhs_t = lhs_view.transpose();

    let mut out_view = MatMut::from_column_major_slice_mut(out_chunk, ctx.c_out, block.rows);
    let _matmul_scope = im2col_matmul_scope(ctx, block.rows);
    matmul(
        &mut out_view,
        Accum::Replace,
        ctx.rhs_t,
        lhs_t,
        1.0f32,
        Par::Seq,
    );
}

/// Writes the patches of the output rows `first_row..first_row + rows` into `scratch_buf`, one
/// row of `patch_len` values each, with zeros where a window reaches past the input.
fn fill_im2col_rows(
    ctx: &Conv2dIm2colContext,
    walk: &PatchWalk,
    first_row: usize,
    rows: usize,
    scratch_buf: &mut [f32],
) {
    let n_idx = first_row / ctx.out_hw;
    let rem = first_row - n_idx * ctx.out_hw;
    let mut oh = rem / ctx.out_w;
    let mut ow = rem - oh * ctx.out_w;
    let mut src_n_base = n_idx * ctx.in_stride_n;

    for row in 0..rows {
        let base_h = oh as isize * ctx.s_h_isize - ctx.pad_top_isize;
        let base_w = ow as isize * ctx.s_w_isize - ctx.pad_left_isize;

        let dst_row = &mut scratch_buf[row * ctx.patch_len..(row + 1) * ctx.patch_len];

        let max_h = base_h + walk.k_h_span;
        let max_w = base_w + walk.k_w_span;
        let interior = base_h >= 0 && base_w >= 0 && max_h < ctx.h_isize && max_w < ctx.w_isize;

        if interior {
            let src_row_base =
                src_n_base + (base_h as usize) * ctx.in_stride_h + (base_w as usize) * ctx.c_in;
            copy_interior_patch(ctx, walk, src_row_base, dst_row);
        } else {
            copy_border_patch(ctx, src_n_base, base_h, base_w, dst_row);
        }

        ow += 1;
        if ow == ctx.out_w {
            ow = 0;
            oh += 1;
            if oh == walk.out_h {
                oh = 0;
                src_n_base += ctx.in_stride_n;
            }
        }
    }
}

/// Copies into `dst_row` a patch that lies inside the input, whose first input row starts at
/// `src_row_base`: a run of `c_in` values per window position.
fn copy_interior_patch(
    ctx: &Conv2dIm2colContext,
    walk: &PatchWalk,
    mut src_row_base: usize,
    dst_row: &mut [f32],
) {
    let input_ptr = ctx.input_values.as_ptr();
    let dst_ptr = dst_row.as_mut_ptr();
    let mut dst_off = 0usize;
    for _kh in 0..ctx.k_h {
        let mut src = src_row_base;
        for _kw in 0..ctx.k_w {
            // Safety: `input_values`/`dst_row` lengths are validated by the
            // caller, and the `interior` bounds guarantee each patch element
            // points at a valid `[C_in]` slice.
            unsafe {
                std::ptr::copy_nonoverlapping(input_ptr.add(src), dst_ptr.add(dst_off), ctx.c_in);
            }
            dst_off += ctx.c_in;
            src += walk.step_w;
        }
        src_row_base += walk.step_h;
    }
}

/// Copies into `dst_row` a patch that reaches past the input, whose window starts at input row
/// `base_h` and column `base_w` of the image at `src_n_base`, with zeros at the window positions
/// outside the input.
fn copy_border_patch(
    ctx: &Conv2dIm2colContext,
    src_n_base: usize,
    base_h: isize,
    base_w: isize,
    dst_row: &mut [f32],
) {
    dst_row.fill(0.0);
    let input_ptr = ctx.input_values.as_ptr();
    let dst_ptr = dst_row.as_mut_ptr();
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
                std::ptr::copy_nonoverlapping(input_ptr.add(src), dst_ptr.add(dst_off), ctx.c_in);
            }
        }
    }
}

/// The profiler scope of filling `scratch_len` values of im2col patches.
fn im2col_fill_scope(scratch_len: usize) -> gpt_rs::profiling::ScopeGuard {
    gpt_rs::profiling::backend_scope_with_meta("backend.faer.conv2d.im2col_fill", || {
        let bytes = (scratch_len as u64).saturating_mul(4);
        gpt_rs::profiling::ScopeMeta::default().with_work(gpt_rs::profiling::WorkStats {
            elements: scratch_len as u64,
            bytes_read: bytes,
            bytes_written: bytes,
            flops: 0,
            alloc_bytes: 0,
            alloc_count: 0,
        })
    })
}

/// The profiler scope of the matmul of the patches of `rows` output rows with the weight.
fn im2col_matmul_scope(ctx: &Conv2dIm2colContext, rows: usize) -> gpt_rs::profiling::ScopeGuard {
    gpt_rs::profiling::backend_scope_with_meta("backend.faer.conv2d.matmul", || {
        let out_elems = (rows as u64).saturating_mul(ctx.c_out as u64);
        let patch_len = ctx.patch_len as u64;
        let bytes_per_elem = 4u64;
        let rhs_bytes = (ctx.patch_len as u64)
            .saturating_mul(ctx.c_out as u64)
            .saturating_mul(bytes_per_elem);
        let lhs_bytes = (rows as u64)
            .saturating_mul(ctx.patch_len as u64)
            .saturating_mul(bytes_per_elem);
        let out_bytes = out_elems.saturating_mul(bytes_per_elem);
        let flops = (out_elems as u128)
            .saturating_mul(patch_len as u128)
            .saturating_mul(2)
            .min(u64::MAX as u128) as u64;
        gpt_rs::profiling::ScopeMeta::default().with_work(gpt_rs::profiling::WorkStats {
            elements: out_elems,
            bytes_read: rhs_bytes.saturating_add(lhs_bytes),
            bytes_written: out_bytes,
            flops,
            alloc_bytes: 0,
            alloc_count: 0,
        })
    })
}

/// Adds `bias` to each of the `rows` rows of `c_out` results in `out_chunk`.
fn add_conv2d_bias(bias: &[f32], c_out: usize, rows: usize, out_chunk: &mut [f32]) {
    let _bias_scope =
        gpt_rs::profiling::backend_scope_with_meta("backend.faer.conv2d.bias_add", || {
            let out_elems = (rows as u64).saturating_mul(c_out as u64);
            let bytes_per_elem = 4u64;
            let out_bytes = out_elems.saturating_mul(bytes_per_elem);
            let bias_bytes = (c_out as u64).saturating_mul(bytes_per_elem);
            gpt_rs::profiling::ScopeMeta::default().with_work(gpt_rs::profiling::WorkStats {
                elements: out_elems,
                bytes_read: out_bytes.saturating_add(bias_bytes),
                bytes_written: out_bytes,
                flops: out_elems,
                alloc_bytes: 0,
                alloc_count: 0,
            })
        });
    for row in out_chunk.chunks_exact_mut(c_out) {
        for (slot, bias_value) in row.iter_mut().zip(bias.iter()) {
            *slot += *bias_value;
        }
    }
}

/// `out = rhs_t * lhs_t` on the threads of `par`, which split `out` in halves: by rows, into
/// halves of at least `MIN_ROWS` rows, when the rows allow as many splits as the columns, else by
/// columns, into halves of at least `MIN_COLS` columns.
fn matmul_tiled_parallel(
    par: Par,
    mut out: MatMut<f32>,
    rhs_t: MatRef<f32>,
    lhs_t: MatRef<f32>,
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

/// Runs `process` on the rows `row_start..row_end` of `c_out` results in `out_slice`, split in
/// halves on the threads of `par` while it has more than one thread and the range has more than
/// one row.
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
        |par_left| left_res = process_parallel(par_left, row_start, mid, c_out, out_left, process),
        |par_right| {
            right_res = process_parallel(par_right, mid, row_end, c_out, out_right, process)
        },
        par,
    );

    left_res?;
    right_res?;
    Ok(())
}

/// A depthwise f32 NHWC convolution with the weight `[C, KH * KW]`, as a direct loop over the
/// output pixels.
fn depthwise_conv2d_nhwc_direct_f32(
    input: &HostTensor,
    weight: &HostTensor,
    bias: Option<&HostTensor>,
    spec: &ExtractPatches,
    output_spec: &TensorType,
) -> Result<HostTensor> {
    let (shape, khkw) = depthwise_conv2d_shape(input, weight, bias, spec, output_spec)?;
    let input_values = f32_values(input, "depthwise conv fusion expects f32 input")?;
    let weight_values = f32_values(weight, "depthwise conv fusion expects f32 weight")?;
    let bias_values = bias
        .map(|tensor| f32_values(tensor, "depthwise conv fusion expects f32 bias"))
        .transpose()?;

    let out_len = [shape.batch, shape.out_h, shape.out_w, shape.out_c]
        .iter()
        .try_fold(1usize, |acc, &extent| acc.checked_mul(extent))
        .ok_or_else(|| anyhow!("depthwise conv fusion output size overflow"))?;
    let mut out = vec![0.0f32; out_len];

    let channels = shape.in_c;
    let in_hw = shape
        .in_h
        .checked_mul(shape.in_w)
        .ok_or_else(|| anyhow!("depthwise conv fusion in_hw overflow"))?;
    let in_stride_n = in_hw
        .checked_mul(channels)
        .ok_or_else(|| anyhow!("depthwise conv fusion in_stride overflow"))?;
    let in_stride_h = shape
        .in_w
        .checked_mul(channels)
        .ok_or_else(|| anyhow!("depthwise conv fusion in_stride overflow"))?;

    let out_hw = shape
        .out_h
        .checked_mul(shape.out_w)
        .ok_or_else(|| anyhow!("depthwise conv fusion out_hw overflow"))?;

    let window = DepthwiseWindow {
        shape,
        khkw,
        in_stride_h,
        input_values,
        weight_values,
    };
    let s_h_isize = shape.s_h as isize;
    let s_w_isize = shape.s_w as isize;
    let pad_top_isize = shape.pad_top as isize;
    let pad_left_isize = shape.pad_left as isize;

    for n_idx in 0..shape.batch {
        let in_n_base = n_idx * in_stride_n;
        let out_n_base = n_idx * out_hw * channels;
        for oh in 0..shape.out_h {
            let base_h = oh as isize * s_h_isize - pad_top_isize;
            let out_h_base = out_n_base + oh * shape.out_w * channels;
            for ow in 0..shape.out_w {
                let base_w = ow as isize * s_w_isize - pad_left_isize;
                let out_base = out_h_base + ow * channels;

                // Initialize with bias.
                if let Some(channel_biases) = bias_values {
                    out[out_base..out_base + channels].copy_from_slice(channel_biases);
                }

                window.accumulate(in_n_base, (base_h, base_w), &mut out, out_base)?;
            }
        }
    }

    output_tensor(output_spec, TensorData::F32(Buffer::from(out)))
}

/// The shape of a depthwise convolution, checked against its tensors, and the length of each
/// channel's window, `KH * KW`.
fn depthwise_conv2d_shape(
    input: &HostTensor,
    weight: &HostTensor,
    bias: Option<&HostTensor>,
    spec: &ExtractPatches,
    output_spec: &TensorType,
) -> Result<(Conv2dShape, usize)> {
    let input_dims = input.dims;
    let weight_dims = weight.dims;
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

    if out_dims[0] != input_dims[0] || out_dims[3] != input_dims[3] {
        return Err(anyhow!("depthwise conv fusion output shape mismatch",));
    }
    let shape = Conv2dShape::new(&input_dims, &out_dims, spec);

    let khkw = shape
        .k_h
        .checked_mul(shape.k_w)
        .ok_or_else(|| anyhow!("depthwise conv fusion kernel overflow"))?;

    if weight_dims[..] != [shape.in_c, khkw] {
        return Err(anyhow!(
            "the depthwise kernel expects a weight [C, KH * KW]"
        ));
    }

    if let Some(bias_tensor) = bias {
        let bias_dims = bias_tensor.dims;
        if bias_tensor.dtype() != DType::F32 || bias_dims.iter().product::<usize>() != shape.in_c {
            return Err(anyhow!("depthwise conv fusion bias shape mismatch",));
        }
    }
    Ok((shape, khkw))
}

/// What the direct depthwise loop reads at each output pixel: the convolution's shape, the
/// input, the weight `[C, KH * KW]`, and the input's row stride.
struct DepthwiseWindow<'a> {
    shape: Conv2dShape,
    khkw: usize,
    in_stride_h: usize,
    input_values: &'a [f32],
    weight_values: &'a [f32],
}

impl DepthwiseWindow<'_> {
    /// Adds to the `C` results at `out_base` of `out`, for every channel, the channel's window of
    /// the image at `in_n_base` times its weights. The window starts at input row and column
    /// `base`.
    fn accumulate(
        &self,
        in_n_base: usize,
        base: (isize, isize),
        out: &mut [f32],
        out_base: usize,
    ) -> Result<()> {
        let (base_h, base_w) = base;
        let (k_h, k_w, channels) = (self.shape.k_h, self.shape.k_w, self.shape.in_c);
        let d_h_isize = self.shape.d_h as isize;
        let d_w_isize = self.shape.d_w as isize;
        let h_isize = self.shape.in_h as isize;
        let w_isize = self.shape.in_w as isize;
        let (input_values, weight_values) = (self.input_values, self.weight_values);
        let khkw = self.khkw;

        for kh in 0..k_h {
            let ih = base_h + kh as isize * d_h_isize;
            if ih < 0 || ih >= h_isize {
                continue;
            }
            let in_h_base = in_n_base
                + (ih as usize)
                    .checked_mul(self.in_stride_h)
                    .ok_or_else(|| anyhow!("depthwise conv fusion offset overflow"))?;
            for kw in 0..k_w {
                let iw = base_w + kw as isize * d_w_isize;
                if iw < 0 || iw >= w_isize {
                    continue;
                }
                let in_base = in_h_base
                    + (iw as usize)
                        .checked_mul(channels)
                        .ok_or_else(|| anyhow!("depthwise conv fusion offset overflow"))?;

                let khkw_idx = kh * k_w + kw;
                // For each channel, accumulate input * weight[channel, khkw].
                for ch in 0..channels {
                    let w_idx = ch * khkw + khkw_idx;
                    out[out_base + ch] += input_values[in_base + ch] * weight_values[w_idx];
                }
            }
        }
        Ok(())
    }
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

/// An f32 elementwise op that computes `op` on each pair of elements.
fn try_elementwise_binary(
    inputs: &[HostTensor],
    outputs: &[TensorType],
    op: impl Fn(f32, f32) -> f32,
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
    if lhs_values.len() != rhs_values.len() {
        return Some(Err(anyhow!("elementwise size mismatch")));
    }

    let out: Vec<f32> = lhs_values
        .iter()
        .zip(rhs_values)
        .map(|(&lhs_value, &rhs_value)| op(lhs_value, rhs_value))
        .collect();

    Some(output_tensor(output_spec, TensorData::F32(Buffer::from(out))).map(|tensor| vec![tensor]))
}

fn try_reduce_window(
    inputs: &[HostTensor],
    outputs: &[TensorType],
    spec: &ReduceWindowAttrs,
) -> Option<Result<Vec<HostTensor>>> {
    if inputs.len() != 1 || outputs.len() != 1 {
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

    let input_dims = input.dims;
    let out_dims = match dims_of(output_spec) {
        Ok(dims) => dims,
        Err(err) => return Some(Err(err)),
    };
    if input_dims.len() != 4 || out_dims.len() != 4 {
        return None;
    }

    let (batch, channels) = (input_dims[0], input_dims[3]);
    if out_dims[0] != batch || out_dims[3] != channels {
        return Some(Err(anyhow!(
            "reduce_window expects batch/channels preserved",
        )));
    }

    let TensorData::F32(values) = &input.data else {
        return None;
    };
    let pool = MaxPool2d {
        batch,
        in_h: input_dims[1],
        in_w: input_dims[2],
        channels,
        out_h: out_dims[1],
        out_w: out_dims[2],
        window: (k_h, k_w),
        strides: (s_h, s_w),
        padding: (p_h, p_w),
    };
    Some(
        pool.run(values.as_ref(), output_spec)
            .map(|tensor| vec![tensor]),
    )
}

/// A 2-D max pool over f32 NHWC values without dilation: each result is the maximum, from
/// `-inf`, of the input positions of its window that lie inside the unpadded input.
#[derive(Clone, Copy)]
struct MaxPool2d {
    batch: usize,
    in_h: usize,
    in_w: usize,
    channels: usize,
    out_h: usize,
    out_w: usize,
    window: (usize, usize),
    strides: (usize, usize),
    /// The padding before the first row and the first column.
    padding: (usize, usize),
}

impl MaxPool2d {
    /// The pool of `values`, as a tensor of type `output_spec`.
    fn run(&self, values: &[f32], output_spec: &TensorType) -> Result<HostTensor> {
        let Self {
            batch,
            in_h,
            in_w,
            channels,
            out_h,
            out_w,
            ..
        } = *self;
        let (s_h, s_w) = self.strides;
        let (p_h, p_w) = self.padding;
        let out_len = batch
            .checked_mul(out_h)
            .and_then(|len| len.checked_mul(out_w))
            .and_then(|len| len.checked_mul(channels))
            .ok_or_else(|| anyhow!("reduce_window size overflow"))?;
        let mut out = vec![f32::NEG_INFINITY; out_len];

        let in_hw = in_h
            .checked_mul(in_w)
            .ok_or_else(|| anyhow!("reduce_window in_hw overflow"))?;
        let in_stride_n = in_hw
            .checked_mul(channels)
            .ok_or_else(|| anyhow!("reduce_window stride overflow",))?;
        let in_stride_h = in_w
            .checked_mul(channels)
            .ok_or_else(|| anyhow!("reduce_window stride overflow",))?;

        for n_idx in 0..batch {
            let in_n_base = n_idx * in_stride_n;
            for oh in 0..out_h {
                let ih0 = (oh * s_h) as isize - p_h as isize;
                for ow in 0..out_w {
                    let iw0 = (ow * s_w) as isize - p_w as isize;
                    let dst_offset = ((n_idx * out_h + oh) * out_w + ow) * channels;
                    let dst = &mut out[dst_offset..dst_offset + channels];
                    self.window_max(values, in_n_base, in_stride_h, (ih0, iw0), dst);
                }
            }
        }

        output_tensor(output_spec, TensorData::F32(Buffer::from(out)))
    }

    /// Folds into `dst`, the results of one output pixel, the maximum over its window of the
    /// image at `in_n_base` of `values`, whose rows are `in_stride_h` apart. The window starts
    /// at input row and column `origin`.
    fn window_max(
        &self,
        values: &[f32],
        in_n_base: usize,
        in_stride_h: usize,
        origin: (isize, isize),
        dst: &mut [f32],
    ) {
        let (k_h, k_w) = self.window;
        let (ih0, iw0) = origin;
        let channels = self.channels;
        for kh in 0..k_h as isize {
            let ih = ih0 + kh;
            if ih < 0 || ih >= self.in_h as isize {
                continue;
            }
            let in_h_base = in_n_base + ih as usize * in_stride_h;
            for kw in 0..k_w as isize {
                let iw = iw0 + kw;
                if iw < 0 || iw >= self.in_w as isize {
                    continue;
                }
                let src_offset = in_h_base + iw as usize * channels;
                let src = &values[src_offset..src_offset + channels];
                for channel in 0..channels {
                    dst[channel] = maximum(dst[channel], src[channel]);
                }
            }
        }
    }
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
                .fold(4u64, |acc, &extent| acc.saturating_mul(extent as u64))
        };
        let out_dims = dims_of(output).unwrap_or_default();
        let out_bytes = bytes(&out_dims);
        // Each result element sums one window over the input channels of its group.
        let group_channels = if conv.depthwise {
            1
        } else {
            input.dims.last().copied().unwrap_or(0)
        };
        let window_len = patches.window.iter().product::<usize>() * group_channels;
        let flops = out_dims
            .iter()
            .chain([&window_len, &2])
            .fold(1u64, |acc, &factor| acc.saturating_mul(factor as u64));
        let bias_bytes = bias.map_or(0, |tensor| bytes(&tensor.dims));
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
    let &[channels, 1, 1, k_h, k_w] = weight.dims.as_slice() else {
        return Err(anyhow!("a depthwise weight of shape {:?}", weight.dims));
    };
    let flat_weight = HostTensor::new(vec![channels, k_h * k_w], weight.data.clone())?;
    depthwise_conv2d_nhwc_direct_f32(input, &flat_weight, bias, patches, output)
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

    let lhs_dims = lhs.dims;
    let rhs_dims = rhs.dims;
    let out_dims = match dims_of(output_spec) {
        Ok(dims) => dims,
        Err(err) => return Some(Err(err)),
    };
    let ranks = (lhs_dims.len(), rhs_dims.len());

    if is_matmul(spec, ranks, 2, &[], (1, 0)) {
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
    if is_matmul(spec, ranks, 2, &[], (1, 1)) && out_dims.len() == 2 {
        return Some(dot_general_batched_rhs_transposed(
            lhs_values,
            rhs_values,
            &[1, lhs_dims[0], lhs_dims[1]],
            &[1, rhs_dims[0], rhs_dims[1]],
            &[1, out_dims[0], out_dims[1]],
            output_spec,
        ));
    }

    if is_matmul(spec, ranks, 3, &[0], (2, 1)) {
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
    if is_matmul(spec, ranks, 3, &[0], (2, 2)) {
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

/// Whether `spec` multiplies an lhs and an rhs of ranks `ranks`, both `rank`, over the batch
/// axes `batch` of both, contracting axis `contract.0` of the lhs with axis `contract.1` of the
/// rhs.
fn is_matmul(
    spec: &DotGeneral,
    ranks: (usize, usize),
    rank: usize,
    batch: &[i64],
    contract: (i64, i64),
) -> bool {
    ranks == (rank, rank)
        && spec.batch_lhs.as_slice() == batch
        && spec.batch_rhs.as_slice() == batch
        && spec.contract_lhs.as_slice() == [contract.0]
        && spec.contract_rhs.as_slice() == [contract.1]
}

fn dot_general_2d(
    lhs: &[f32],
    rhs: &[f32],
    lhs_dims: &[usize],
    rhs_dims: &[usize],
    out_dims: &[usize],
    output_spec: &TensorType,
) -> Result<Vec<HostTensor>> {
    let (rows, k_lhs) = (lhs_dims[0], lhs_dims[1]);
    let (k_rhs, columns) = (rhs_dims[0], rhs_dims[1]);

    if k_lhs != k_rhs {
        return Err(anyhow!("dot_general lhs/rhs contract mismatch",));
    }

    if out_dims.len() != 2 || out_dims[0] != rows || out_dims[1] != columns {
        return Err(anyhow!("dot_general output shape mismatch"));
    }

    if lhs.len() != rows * k_lhs || rhs.len() != k_rhs * columns {
        return Err(anyhow!("dot_general operand size mismatch"));
    }

    if rows == 0 || columns == 0 || k_lhs == 0 {
        return output_tensor(
            output_spec,
            TensorData::F32(Buffer::from(vec![0.0f32; rows * columns])),
        )
        .map(|tensor| vec![tensor]);
    }

    let lhs_view = MatRef::from_row_major_slice(lhs, rows, k_lhs);
    let rhs_view = MatRef::from_row_major_slice(rhs, k_rhs, columns);

    // faer prefers column-major output. To avoid an explicit output transpose/copy, compute
    // C^T = B^T * A^T into a column-major (columns x rows) matrix. The underlying buffer layout
    // matches row-major (rows x columns) for C, so we can return it directly.
    let a_t = lhs_view.transpose();
    let b_t = rhs_view.transpose();
    let mut row_major = vec![0.0f32; rows * columns];
    let mut out_view = MatMut::from_column_major_slice_mut(row_major.as_mut_slice(), columns, rows);
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
    let (batch, rows, k_lhs) = (lhs_dims[0], lhs_dims[1], lhs_dims[2]);
    let (batch_rhs, k_rhs, columns) = (rhs_dims[0], rhs_dims[1], rhs_dims[2]);

    if batch != batch_rhs {
        return Err(anyhow!("dot_general batch mismatch"));
    }

    if k_lhs != k_rhs {
        return Err(anyhow!("dot_general lhs/rhs contract mismatch",));
    }

    if out_dims.len() != 3 || out_dims[0] != batch || out_dims[1] != rows || out_dims[2] != columns
    {
        return Err(anyhow!("dot_general output shape mismatch"));
    }

    if lhs.len() != batch * rows * k_lhs || rhs.len() != batch * k_rhs * columns {
        return Err(anyhow!("dot_general operand size mismatch"));
    }

    if batch == 0 || rows == 0 || columns == 0 || k_lhs == 0 {
        return output_tensor(
            output_spec,
            TensorData::F32(Buffer::from(vec![0.0f32; batch * rows * columns])),
        )
        .map(|tensor| vec![tensor]);
    }

    let lhs_batch_stride = rows * k_lhs;
    let rhs_batch_stride = k_rhs * columns;
    let out_batch_stride = rows * columns;

    let mut row_output = vec![0.0f32; batch * out_batch_stride];
    let par = faer_parallelism();

    for batch_index in 0..batch {
        let lhs_slice = &lhs[batch_index * lhs_batch_stride..(batch_index + 1) * lhs_batch_stride];
        let rhs_slice = &rhs[batch_index * rhs_batch_stride..(batch_index + 1) * rhs_batch_stride];

        let lhs_view = MatRef::from_row_major_slice(lhs_slice, rows, k_lhs);
        let rhs_view = MatRef::from_row_major_slice(rhs_slice, k_rhs, columns);

        let a_t = lhs_view.transpose();
        let b_t = rhs_view.transpose();
        let dst =
            &mut row_output[batch_index * out_batch_stride..(batch_index + 1) * out_batch_stride];
        let mut out_view = MatMut::from_column_major_slice_mut(dst, columns, rows);
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
    let (batch, rows, k_lhs) = (lhs_dims[0], lhs_dims[1], lhs_dims[2]);
    let (batch_rhs, columns, k_rhs) = (rhs_dims[0], rhs_dims[1], rhs_dims[2]);

    if batch != batch_rhs {
        return Err(anyhow!("dot_general batch mismatch"));
    }

    if k_lhs != k_rhs {
        return Err(anyhow!("dot_general lhs/rhs contract mismatch",));
    }

    if out_dims.len() != 3 || out_dims[0] != batch || out_dims[1] != rows || out_dims[2] != columns
    {
        return Err(anyhow!("dot_general output shape mismatch"));
    }

    if lhs.len() != batch * rows * k_lhs || rhs.len() != batch * columns * k_rhs {
        return Err(anyhow!("dot_general operand size mismatch"));
    }

    if batch == 0 || rows == 0 || columns == 0 || k_lhs == 0 {
        return output_tensor(
            output_spec,
            TensorData::F32(Buffer::from(vec![0.0f32; batch * rows * columns])),
        )
        .map(|tensor| vec![tensor]);
    }

    let lhs_batch_stride = rows * k_lhs;
    let rhs_batch_stride = columns * k_rhs;
    let out_batch_stride = rows * columns;

    let mut row_output = vec![0.0f32; batch * out_batch_stride];
    let par = faer_parallelism();

    for batch_index in 0..batch {
        let lhs_slice = &lhs[batch_index * lhs_batch_stride..(batch_index + 1) * lhs_batch_stride];
        let rhs_slice = &rhs[batch_index * rhs_batch_stride..(batch_index + 1) * rhs_batch_stride];

        let lhs_view = MatRef::from_row_major_slice(lhs_slice, rows, k_lhs);
        let rhs_view = MatRef::from_row_major_slice(rhs_slice, columns, k_rhs);

        // Compute C^T = B * A^T into a column-major (columns x rows) matrix.
        // The underlying buffer layout matches row-major (rows x columns) for C.
        let a_t = lhs_view.transpose();
        let dst =
            &mut row_output[batch_index * out_batch_stride..(batch_index + 1) * out_batch_stride];
        let mut out_view = MatMut::from_column_major_slice_mut(dst, columns, rows);
        matmul(&mut out_view, Accum::Replace, rhs_view, a_t, 1.0f32, par);
    }

    output_tensor(output_spec, TensorData::F32(Buffer::from(row_output))).map(|tensor| vec![tensor])
}
