//! faer's choice of kernels: the steps that compute a function, each one op or a sequence of ops
//! that one faer kernel computes.

use std::collections::{HashMap, HashSet};

use gpt_rs::ops::functional::{linear_weight, WeightLayout};
use gpt_rs_ir::dialects::tensor::{
    Binary, BinaryKind, BroadcastTo, DotGeneral, ExtractPatches, Reshape, TensorOp, Transpose,
};
use gpt_rs_ir::module::{Inst, Module, Value};
use gpt_rs_ir::types::DType;
use gpt_rs_ir::Op;

pub(crate) struct Step {
    pub kind: StepKind,
    /// The values no later step reads, which the run drops after this step.
    pub last_uses: Vec<Value>,
}

pub(crate) enum StepKind {
    /// An op, on faer's kernel for it when there is one, else on the interpreter.
    Op {
        inst: Inst,
        label: &'static str,
    },
    /// `minimum(maximum(input, 0), 6)` in one loop.
    Relu6 {
        input: Value,
        output: Value,
    },
    Conv2d(Conv2d),
    Linear(Linear),
}

/// An NHWC f32 convolution, with the bias add after it when there is one.
pub(crate) struct Conv2d {
    /// The NHWC input.
    pub input: Value,
    /// `[KH * KW * C_in, C_out]`, or `[C, 1, 1, KH, KW]` for a depthwise convolution.
    pub weight: Value,
    /// The bias, one value per output channel.
    pub bias: Option<Value>,
    /// The window, strides, dilation and padding of the convolution.
    pub patches: ExtractPatches,
    /// Whether each channel convolves with its own filter alone.
    pub depthwise: bool,
    /// The result, after the bias add when there is one.
    pub output: Value,
}

/// A projection that reads its weight as stored (see [`linear_weight`]).
pub(crate) struct Linear {
    pub x: Value,
    pub weight: Value,
    pub scale: Option<Value>,
    pub layout: WeightLayout,
    pub output: Value,
}

impl StepKind {
    fn reads(&self, module: &Module) -> Vec<Value> {
        match self {
            StepKind::Op { inst, .. } => module.operands(*inst).to_vec(),
            StepKind::Relu6 { input, .. } => vec![*input],
            StepKind::Conv2d(conv) => [conv.input, conv.weight]
                .into_iter()
                .chain(conv.bias)
                .collect(),
            StepKind::Linear(linear) => [linear.x, linear.weight]
                .into_iter()
                .chain(linear.scale)
                .collect(),
        }
    }
}

/// A kernel step, and the ops it covers besides the op whose result it gives.
struct Fused {
    kind: StepKind,
    covers: Vec<Inst>,
}

/// The steps that compute `results` from the ops `insts` of a function, in order.
pub(crate) fn steps(module: &Module, insts: &[Inst], results: &[Value]) -> Vec<Step> {
    // Each kernel by the op whose result it gives.
    let mut fused: HashMap<Inst, Fused> = insts
        .iter()
        .filter_map(|&inst| Some((inst, kernel(module, inst)?)))
        .collect();
    for &add in insts {
        let Some((conv, bias, broadcast)) = bias_add(module, add, &fused) else {
            continue;
        };
        let mut joined = fused.remove(&conv).expect("a convolution kernel");
        let StepKind::Conv2d(kernel) = &mut joined.kind else {
            unreachable!("bias_add matches only convolutions");
        };
        kernel.bias = Some(bias);
        kernel.output = module.results(add)[0];
        joined.covers.extend([conv, broadcast]);
        fused.insert(add, joined);
    }
    // From the last op back, each op whose result a step that runs or a result needs. An op that a
    // kernel covers runs only when another step reads it.
    let covers: HashSet<Inst> = fused
        .values()
        .flat_map(|kernel| kernel.covers.iter().copied())
        .collect();
    let mut needed: HashSet<Value> = results.iter().copied().collect();
    let mut kinds = Vec::new();
    for &inst in insts.iter().rev() {
        let kind = match fused.remove(&inst) {
            Some(kernel) => kernel.kind,
            None if covers.contains(&inst)
                && !module.results(inst).iter().any(|v| needed.contains(v)) =>
            {
                continue;
            }
            None => StepKind::Op {
                inst,
                label: gpt_rs::profiling::op_label(&module.op(inst).name()),
            },
        };
        needed.extend(kind.reads(module));
        kinds.push(kind);
    }
    kinds.reverse();
    with_last_uses(module, kinds, results)
}

/// The steps of `kinds`, each with the values that no later step and no result reads.
fn with_last_uses(module: &Module, kinds: Vec<StepKind>, results: &[Value]) -> Vec<Step> {
    let mut last: HashMap<Value, usize> = HashMap::new();
    for (index, kind) in kinds.iter().enumerate() {
        for value in kind.reads(module) {
            last.insert(value, index);
        }
    }
    for value in results {
        last.remove(value);
    }
    let mut last_uses: Vec<Vec<Value>> = vec![Vec::new(); kinds.len()];
    for (value, index) in last {
        last_uses[index].push(value);
    }
    kinds
        .into_iter()
        .zip(last_uses)
        .map(|(kind, last_uses)| Step { kind, last_uses })
        .collect()
}

/// The faer kernel that gives the result of `inst`, with the ops it covers.
fn kernel(module: &Module, inst: Inst) -> Option<Fused> {
    conv2d(module, inst)
        .or_else(|| depthwise_conv2d(module, inst))
        .or_else(|| linear(module, inst))
        .or_else(|| relu6(module, inst))
}

/// The extents of `value` when it is an f32 tensor with static extents.
fn f32_dims(module: &Module, value: Value) -> Option<Vec<usize>> {
    let ty = module.ty(value).as_tensor()?;
    let dims = ty.static_dims().filter(|_| ty.dtype == DType::F32)?;
    Some(dims.into_vec())
}

/// Whether the faer convolution kernels read patches with these attributes: two spatial axes,
/// padded with zeros.
fn conv_patches(patches: &ExtractPatches) -> bool {
    patches.window.len() == 2 && patches.pad_value.0 == 0.0
}

/// `dot_general(extract_patches(x), w)` with the patches `[N, OH, OW, K]` contracted whole with
/// `w [K, C_out]`, the form `merge-contracted-axes` gives a convolution without groups.
fn conv2d(module: &Module, inst: Inst) -> Option<Fused> {
    let dot = module.op_as::<DotGeneral>(inst)?;
    let patches = module.def_as::<ExtractPatches>(dot.operands.lhs)?;
    let attrs = &dot.attrs;
    let plain = attrs.batch_lhs.is_empty()
        && attrs.batch_rhs.is_empty()
        && attrs.contract_lhs[..] == [3]
        && attrs.contract_rhs[..] == [0];
    let input = patches.operands.input;
    let weight = dot.operands.rhs;
    let &[_, _, _, k] = f32_dims(module, patches.results.output)?.as_slice() else {
        return None;
    };
    let &[k_w, c_out] = f32_dims(module, weight)?.as_slice() else {
        return None;
    };
    let &[_, _, _, out_c] = f32_dims(module, dot.results.output)?.as_slice() else {
        return None;
    };
    let shapes = k == k_w && c_out == out_c && f32_dims(module, input)?.len() == 4;
    (plain && shapes && conv_patches(&patches.attrs)).then(|| Fused {
        kind: StepKind::Conv2d(Conv2d {
            input,
            weight,
            bias: None,
            patches: patches.attrs.clone(),
            depthwise: false,
            output: dot.results.output,
        }),
        covers: vec![patches.inst],
    })
}

/// A convolution as `conv2d` records it with one input and one output channel per group, whose
/// result `inst` reshapes back to `[N, OH, OW, C]`. It views the patches as
/// `[N, OH, OW, KH, KW, G = C, C_in / G = 1]` and the weight `[C, 1, KH, KW]` as
/// `[G = C, C_out / G = 1, C_in / G = 1, KH, KW]`. The dot gives `[G, N, OH, OW, 1]`, which a
/// transpose and the reshape put back.
fn depthwise_conv2d(module: &Module, inst: Inst) -> Option<Fused> {
    let reshape = module.op_as::<Reshape>(inst)?;
    let transpose = module.def_as::<Transpose>(reshape.operands.input)?;
    let dot = module.def_as::<DotGeneral>(transpose.operands.input)?;
    let patches_reshape = module.def_as::<Reshape>(dot.operands.lhs)?;
    let patches = module.def_as::<ExtractPatches>(patches_reshape.operands.input)?;
    let input = patches.operands.input;
    let weight = dot.operands.rhs;
    let &[n, _, _, c] = f32_dims(module, input)?.as_slice() else {
        return None;
    };
    let &[k_h, k_w] = patches.attrs.window.as_slice() else {
        return None;
    };
    let &[_, out_h, out_w, _] = f32_dims(module, reshape.results.output)?.as_slice() else {
        return None;
    };
    let depthwise = conv_patches(&patches.attrs)
        && f32_dims(module, reshape.results.output)? == [n, out_h, out_w, c]
        && f32_dims(module, patches_reshape.results.output)? == [n, out_h, out_w, k_h, k_w, c, 1]
        && f32_dims(module, weight)? == [c, 1, 1, k_h, k_w]
        && dot.attrs.batch_lhs[..] == [5]
        && dot.attrs.batch_rhs[..] == [0]
        && dot.attrs.contract_lhs[..] == [3, 4, 6]
        && dot.attrs.contract_rhs[..] == [3, 4, 2]
        && transpose.attrs.perm[..] == [1, 2, 3, 0, 4];
    depthwise.then(|| Fused {
        kind: StepKind::Conv2d(Conv2d {
            input,
            weight,
            bias: None,
            patches: patches.attrs.clone(),
            depthwise: true,
            output: reshape.results.output,
        }),
        covers: vec![patches.inst, patches_reshape.inst, dot.inst, transpose.inst],
    })
}

/// A projection whose weight the faer linear kernel reads as stored.
fn linear(module: &Module, inst: Inst) -> Option<Fused> {
    let linear = linear_weight(module, inst)?;
    let (x, weight, scale) = linear.dtypes(module);
    if !crate::linear::reads(x, weight, scale) {
        return None;
    }
    Some(Fused {
        kind: StepKind::Linear(Linear {
            x: linear.x,
            weight: linear.weight,
            scale: linear.scale,
            layout: linear.layout,
            output: module.results(inst)[0],
        }),
        covers: linear.chain,
    })
}

/// `minimum(maximum(input, broadcast_to(0)), broadcast_to(6))` over f32 scalar constants, rooted
/// at the `minimum`. The fused loop computes with the constants, so a -0.0 scalar, which keeps a
/// -0.0 input, does not match.
fn relu6(module: &Module, inst: Inst) -> Option<Fused> {
    let binary = |inst: Inst, kind: BinaryKind| -> Option<[Value; 2]> {
        let Op::Tensor(TensorOp::Binary(op)) = module.op(inst) else {
            return None;
        };
        let &[lhs, rhs] = module.operands(inst) else {
            return None;
        };
        (op.kind == kind).then_some([lhs, rhs])
    };
    // The broadcast of an f32 scalar constant that is `scalar` by bits.
    let broadcast_of = |value: Value, scalar: f32| -> Option<Inst> {
        let broadcast = module.def_as::<BroadcastTo>(value)?;
        let constant = module.constant(broadcast.operands.input)?;
        (constant.dims.is_empty() && constant.is_splat(scalar)).then_some(broadcast.inst)
    };
    let [low, high] = binary(inst, BinaryKind::Minimum)?;
    let (clamped, six) = [(low, high), (high, low)]
        .into_iter()
        .find_map(|(clamped, six)| Some((clamped, broadcast_of(six, 6.0)?)))?;
    let maximum = module.def_inst(clamped)?;
    let [a, b] = binary(maximum, BinaryKind::Maximum)?;
    let (input, zero) = [(a, b), (b, a)]
        .into_iter()
        .find_map(|(input, zero)| Some((input, broadcast_of(zero, 0.0)?)))?;
    let output = module.results(inst)[0];
    let f32_input = f32_dims(module, input).is_some();
    (f32_input && module.ty(input) == module.ty(output)).then_some(Fused {
        kind: StepKind::Relu6 { input, output },
        covers: vec![maximum, zero, six],
    })
}

/// Matches `add` as a bias add on the result of a convolution kernel: one f32 value per output
/// channel. Returns the kernel, the bias, and the broadcast that reads the bias. The kernel's
/// result has no other reader.
fn bias_add(
    module: &Module,
    add: Inst,
    fused: &HashMap<Inst, Fused>,
) -> Option<(Inst, Value, Inst)> {
    let add = module.op_as::<Binary>(add)?;
    if add.attrs.kind != BinaryKind::Add {
        return None;
    }
    let (lhs, rhs) = (add.operands.lhs, add.operands.rhs);
    let (conv, other) = [(lhs, rhs), (rhs, lhs)]
        .into_iter()
        .find_map(|(conv, other)| {
            let root = module.def_inst(conv)?;
            let is_conv = matches!(
                fused.get(&root),
                Some(Fused {
                    kind: StepKind::Conv2d(Conv2d { bias: None, .. }),
                    ..
                })
            );
            (is_conv && module.uses(conv).len() == 1).then_some((root, other))
        })?;
    let broadcast = module.def_as::<BroadcastTo>(other)?;
    let bias = broadcast.operands.input;
    let &channels = f32_dims(module, add.results.output)?.last()?;
    let dims = f32_dims(module, bias)?;
    let per_channel = dims.last() == Some(&channels) && dims.iter().product::<usize>() == channels;
    per_channel.then_some((conv, bias, broadcast.inst))
}
