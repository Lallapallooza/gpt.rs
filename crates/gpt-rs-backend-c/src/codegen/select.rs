//! The C backend's choice of kernels: the instructions that codegen emits, each one op or a
//! sequence of ops that one kernel computes. An op that a kernel covers is emitted only when
//! another emitted instruction reads it.

use std::collections::{HashMap, HashSet};

use anyhow::{anyhow, Result};
use gpt_rs::ops::functional::{linear_weight, WeightLayout};
use gpt_rs_ir::dialects::tensor::{
    Add, BroadcastTo, Cast, DotGeneral, ExtractPatches, Reshape, Slice, TensorOp, Transpose,
};
use gpt_rs_ir::ir::{DType, Inst, Module, Value};
use gpt_rs_ir::rewrite::matched;
use gpt_rs_ir::Op;

use super::types::{Elementwise, ElementwiseInput, Instruction, Kind, Node, NodeRef};
use crate::kernels::PANEL_N;
use crate::targets::{binary_expr, fused_input_fits, linear_kernel, unary_expr};

/// A kernel, and the ops it covers besides the op whose result it gives.
struct Fused<'m> {
    instruction: Instruction<'m>,
    covers: Vec<Inst>,
}

/// The instructions that compute `results` from the ops `insts` of a function, in order.
pub(super) fn select<'m>(
    module: &'m Module,
    insts: &[Inst],
    results: &[Value],
) -> Result<Vec<Instruction<'m>>> {
    // Each kernel by the op whose result it gives.
    let mut fused: HashMap<Inst, Fused<'m>> = insts
        .iter()
        .filter_map(|&inst| {
            let kernel = conv2d(module, inst)
                .or_else(|| grouped_conv2d(module, inst))
                .or_else(|| linear(module, inst))?;
            Some((inst, kernel))
        })
        .collect();
    for &add in insts {
        let Some((conv, bias, covers)) = bias_add(module, add, &fused) else {
            continue;
        };
        let mut joined = fused.remove(&conv).expect("a convolution kernel");
        joined.instruction.operands.push(bias);
        joined.instruction.results = vec![module.results(add)[0]];
        joined.covers.push(conv);
        joined.covers.extend(covers);
        fused.insert(add, joined);
    }
    // Chains of elementwise ops, from the last op of each chain back to its inputs.
    let mut claimed: HashSet<Inst> = fused
        .iter()
        .flat_map(|(&root, kernel)| std::iter::once(root).chain(kernel.covers.iter().copied()))
        .collect();
    for &inst in insts.iter().rev() {
        if claimed.contains(&inst) {
            continue;
        }
        let Some(kernel) = elementwise(module, inst, &claimed) else {
            continue;
        };
        claimed.insert(inst);
        claimed.extend(kernel.covers.iter().copied());
        fused.insert(inst, kernel);
    }

    // From the last op back, each op whose result an emitted instruction or a result needs.
    let covers: HashSet<Inst> = fused
        .values()
        .flat_map(|kernel| kernel.covers.iter().copied())
        .collect();
    let mut needed: HashSet<Value> = results.iter().copied().collect();
    let mut emitted = Vec::new();
    for &inst in insts.iter().rev() {
        let instruction = match fused.remove(&inst) {
            Some(kernel) => kernel.instruction,
            None if covers.contains(&inst)
                && !module.results(inst).iter().any(|v| needed.contains(v)) =>
            {
                continue;
            }
            None => op(module, inst)?,
        };
        needed.extend(instruction.operands.iter().copied());
        emitted.push(instruction);
    }
    emitted.reverse();
    Ok(emitted)
}

/// The instruction that emits the op of `inst` alone.
fn op(module: &Module, inst: Inst) -> Result<Instruction<'_>> {
    let op = module.op(inst);
    let Op::Tensor(tensor_op) = op else {
        return Err(anyhow!("the C backend has no {} op", op.name()));
    };
    Ok(Instruction {
        kind: Kind::Op(tensor_op),
        operands: module.operands(inst).to_vec(),
        results: module.results(inst).to_vec(),
        label: gpt_rs::profiling::op_label(&op.name()),
    })
}

/// The extents of `value` when it is an f32 tensor with static extents.
pub(super) fn f32_dims(module: &Module, value: Value) -> Option<Vec<usize>> {
    let ty = module.ty(value).as_tensor()?;
    let dims = ty.static_dims().filter(|_| ty.dtype == DType::F32)?;
    Some(dims.to_vec())
}

/// `dot_general(extract_patches(x), w)`: the patches `[N, OH, OW, K]` of an f32 NHWC `x`, padded
/// with zeros, contracted whole with `w [K, C_out]`, or with `w` in the GEMM's panels
/// `[C_out / PANEL_N, K, PANEL_N]`. The convolution kernel reads `x` and `w`.
fn conv2d(module: &Module, inst: Inst) -> Option<Fused<'_>> {
    let dot = module.op_as::<DotGeneral>(inst)?;
    let patches = dot.lhs().defining_op::<ExtractPatches>()?;
    let attrs = dot.attrs();
    let weight = dot.rhs().id();
    let w_dims = f32_dims(module, weight)?;
    let layout = matches!(
        (w_dims.as_slice(), attrs.contract_rhs.as_slice()),
        ([_, _], [0]) | ([_, _, PANEL_N], [1])
    );
    let input = patches.input().id();
    let plain =
        attrs.batch_lhs.is_empty() && attrs.batch_rhs.is_empty() && attrs.contract_lhs[..] == [3];
    let zero_padded = patches.window.len() == 2 && patches.pad_value.0 == 0.0;
    let f32_operands = f32_dims(module, input)?.len() == 4
        && f32_dims(module, patches.output().id())?.len() == 4
        && f32_dims(module, dot.output().id()).is_some();
    (layout && plain && zero_padded && f32_operands).then(|| Fused {
        instruction: Instruction {
            kind: Kind::Conv2d(patches.attrs().clone()),
            operands: vec![input, weight],
            results: vec![dot.output().id()],
            label: gpt_rs::profiling::intern_label("backend.c.conv2d"),
        },
        covers: vec![patches.inst()],
    })
}

/// A grouped convolution as the `conv2d` functional records it, once `c-pack-weights` has laid out
/// its weight: `reshape(transpose(dot_general(reshape(extract_patches(x)), w)))`. The patches are
/// `[N, OH, OW, KH, KW, G, C_in / G]` of an f32 NHWC `x` padded with zeros, `w` is
/// `[KH, KW, C_in / G, G, C_out / G]`, the dot gives `[G, N, OH, OW, C_out / G]`, and the
/// transpose and the reshape put that back as `[N, OH, OW, C_out]`. The grouped kernel reads `x`
/// and `w` and writes the last.
fn grouped_conv2d(module: &Module, inst: Inst) -> Option<Fused<'_>> {
    let reshape = module.op_as::<Reshape>(inst)?;
    matched!(let transpose @ Transpose(dot @ DotGeneral(patches_reshape @ Reshape(patches @ ExtractPatches(input)), weight)) = reshape.input() else {
        return None;
    });
    let (input, weight) = (input.id(), weight.id());
    let &[n, _, _, c_in] = f32_dims(module, input)?.as_slice() else {
        return None;
    };
    let &[k_h, k_w] = patches.window.as_slice() else {
        return None;
    };
    let &[w_h, w_w, group_in, groups, group_out] = f32_dims(module, weight)?.as_slice() else {
        return None;
    };
    let &[out_n, out_h, out_w, c_out] = f32_dims(module, reshape.output().id())?.as_slice() else {
        return None;
    };
    let attrs = dot.attrs();
    let shapes = [w_h, w_w] == [k_h, k_w]
        && groups * group_in == c_in
        && groups * group_out == c_out
        && out_n == n
        && f32_dims(module, patches_reshape.output().id())?
            == [n, out_h, out_w, k_h, k_w, groups, group_in];
    let axes = attrs.batch_lhs[..] == [5]
        && attrs.batch_rhs[..] == [3]
        && attrs.contract_lhs[..] == [3, 4, 6]
        && attrs.contract_rhs[..] == [0, 1, 2]
        && attrs.accum.is_none()
        && attrs.out.is_none()
        && transpose.perm[..] == [1, 2, 3, 0, 4];
    let zero_padded = patches.pad_value.0 == 0.0;
    (shapes && axes && zero_padded).then(|| Fused {
        instruction: Instruction {
            kind: Kind::GroupedConv2d {
                patches: patches.attrs().clone(),
                groups,
            },
            operands: vec![input, weight],
            results: vec![reshape.output().id()],
            label: gpt_rs::profiling::intern_label("backend.c.grouped_conv2d"),
        },
        covers: vec![
            patches.inst(),
            patches_reshape.inst(),
            dot.inst(),
            transpose.inst(),
        ],
    })
}

/// A projection in the PyTorch layout `x [M, K] . w [N, K]` whose weight a C linear kernel reads
/// as stored: through its casts, and with the scale of an FP8 weight.
fn linear(module: &Module, inst: Inst) -> Option<Fused<'_>> {
    let projection = linear_weight(module, inst)?;
    let (x, w, scale) = projection.dtypes(module);
    if projection.layout != WeightLayout::OutIn || linear_kernel(x, w, scale).is_none() {
        return None;
    }
    let operands = [projection.x, projection.weight]
        .into_iter()
        .chain(projection.scale)
        .collect();
    Some(Fused {
        instruction: Instruction {
            kind: Kind::Linear,
            operands,
            results: vec![module.results(inst)[0]],
            label: gpt_rs::profiling::intern_label("backend.c.linear"),
        },
        covers: projection.chain,
    })
}

/// Matches `add` as a bias add on the result of a convolution kernel, dense or grouped: one f32
/// value per output channel. Returns the kernel, the bias, and the ops between them: the broadcast
/// of the bias, and the reshape of a result in panels. The kernel's result has no other reader.
fn bias_add(
    module: &Module,
    add: Inst,
    fused: &HashMap<Inst, Fused<'_>>,
) -> Option<(Inst, Value, Vec<Inst>)> {
    let add = module.op_as::<Add>(add)?;
    let is_conv = |inst: Inst| {
        let Some(Fused { instruction, .. }) = fused.get(&inst) else {
            return false;
        };
        let conv = matches!(
            instruction.kind,
            Kind::Conv2d(_) | Kind::GroupedConv2d { .. }
        );
        conv && instruction.operands.len() == 2
    };
    let single_use = |value: Value| module.uses(value).len() == 1;
    // The convolution behind `value`, and the reshape between them.
    let conv_of = |value: Value| -> Option<(Inst, Option<Inst>)> {
        let def = module.def_inst(value)?;
        if is_conv(def) {
            return single_use(value).then_some((def, None));
        }
        let reshape = module.op_as::<Reshape>(def)?;
        let inner = reshape.input().id();
        let conv = module.def_inst(inner)?;
        (is_conv(conv) && single_use(inner) && single_use(value)).then_some((conv, Some(def)))
    };
    let (lhs, rhs) = (add.lhs().id(), add.rhs().id());
    let ((conv, reshape), other) = [(lhs, rhs), (rhs, lhs)]
        .into_iter()
        .find_map(|(conv, other)| Some((conv_of(conv)?, other)))?;
    let broadcast = module.value_ref(other).defining_op::<BroadcastTo>()?;
    let bias = broadcast.input().id();
    let &channels = f32_dims(module, add.output().id())?.last()?;
    let dims = f32_dims(module, bias)?;
    let per_channel = dims.last() == Some(&channels) && dims.iter().product::<usize>() == channels;
    let covers = std::iter::once(broadcast.inst()).chain(reshape).collect();
    per_channel.then_some((conv, bias, covers))
}

/// The elementwise kernel that gives the result of `inst`, when `inst` is the last f32
/// elementwise op of a chain, or an f32 -> bf16 cast of one that it stores directly. A chain of
/// one op is worth a kernel only when it reads through a broadcast or a slice that no other op
/// reads, or stores through a cast.
fn elementwise<'m>(module: &'m Module, inst: Inst, claimed: &HashSet<Inst>) -> Option<Fused<'m>> {
    let (root, store_bf16) = match module.op_as::<Cast>(inst) {
        Some(cast) => {
            let value = cast.input().id();
            let root = module.def_inst(value)?;
            let single = module.uses(value).len() == 1;
            let narrowing = cast.to == DType::Bf16;
            (single && narrowing && !claimed.contains(&root)).then_some((root, true))?
        }
        None => (inst, false),
    };
    node_op(module, root)?;
    let output = module.results(root)[0];
    let root_dims = f32_dims(module, output)?;
    if module.uses(output).is_empty() {
        return None;
    }
    let mut chain = Chain::new(module, claimed, root_dims);
    chain.collect(output, true)?;
    if chain.nodes.len() < 2 && !chain.removes_view() && !store_bf16 {
        return None;
    }
    let mut covers: Vec<Inst> = chain.nodes_insts.iter().copied().collect();
    covers.extend(chain.views.iter().filter_map(|&view| module.def_inst(view)));
    covers.retain(|&cover| cover != inst);
    let (operands, inputs): (Vec<Value>, Vec<ElementwiseInput>) = chain
        .inputs
        .into_iter()
        .map(|(value, starts)| (value, ElementwiseInput { starts }))
        .unzip();
    Some(Fused {
        instruction: Instruction {
            kind: Kind::Elementwise(Elementwise {
                nodes: chain.nodes,
                inputs,
                store_bf16,
            }),
            operands,
            results: vec![module.results(inst)[0]],
            label: gpt_rs::profiling::intern_label("backend.c.fused_elementwise"),
        },
        covers,
    })
}

/// The op of an elementwise node and its operands, when `inst` is an elementwise op of one or
/// two operands that has a C expression.
fn node_op(module: &Module, inst: Inst) -> Option<(TensorOp, Value, Option<Value>)> {
    let Op::Tensor(op) = module.op(inst) else {
        return None;
    };
    match *module.operands(inst) {
        [input] if unary_expr(op, "").is_some() => Some((op.clone(), input, None)),
        [lhs, rhs] if binary_expr(op, "", "").is_some() => Some((op.clone(), lhs, Some(rhs))),
        _ => None,
    }
}

/// An elementwise chain, collected from its last op back to its inputs.
struct Chain<'a> {
    module: &'a Module,
    claimed: &'a HashSet<Inst>,
    root_dims: Vec<usize>,
    /// Each input with the offsets the kernel reads it at, when it is the source of a slice.
    inputs: Vec<(Value, Option<Vec<usize>>)>,
    nodes: Vec<Node>,
    nodes_insts: HashSet<Inst>,
    memo: HashMap<Value, NodeRef>,
    /// The broadcasts and slices the kernel reads through.
    views: Vec<Value>,
}

impl<'a> Chain<'a> {
    fn new(module: &'a Module, claimed: &'a HashSet<Inst>, root_dims: Vec<usize>) -> Self {
        Self {
            module,
            claimed,
            root_dims,
            inputs: Vec::new(),
            nodes: Vec::new(),
            nodes_insts: HashSet::new(),
            memo: HashMap::new(),
            views: Vec::new(),
        }
    }

    fn input(&mut self, value: Value, starts: Option<Vec<usize>>) -> NodeRef {
        let key = (value, starts);
        let index = match self.inputs.iter().position(|input| *input == key) {
            Some(index) => index,
            None => {
                self.inputs.push(key);
                self.inputs.len() - 1
            }
        };
        NodeRef::Input(index)
    }

    fn add_input(&mut self, value: Value) -> Option<NodeRef> {
        let dims = f32_dims(self.module, value)?;
        fused_input_fits(&self.root_dims, &dims, None).then(|| self.input(value, None))
    }

    /// Reads the source of a `slice` at an offset instead of materializing the slice.
    fn add_sliced_input(&mut self, value: Value) -> Option<NodeRef> {
        let slice = self.module.value_ref(value).defining_op::<Slice>()?;
        if slice.sizes[..] != self.root_dims[..] {
            return None;
        }
        let source = slice.input().id();
        let source_dims = f32_dims(self.module, source)?;
        if !fused_input_fits(&self.root_dims, &source_dims, Some(&slice.starts[..])) {
            return None;
        }
        self.views.push(value);
        Some(self.input(source, Some(slice.starts.to_vec())))
    }

    fn collect(&mut self, value: Value, is_root: bool) -> Option<NodeRef> {
        if let Some(&node) = self.memo.get(&value) {
            return Some(node);
        }
        let node = self.collect_def(value, is_root)?;
        self.memo.insert(value, node);
        Some(node)
    }

    fn collect_def(&mut self, value: Value, is_root: bool) -> Option<NodeRef> {
        let module = self.module;
        let Some(inst) = module.def_inst(value) else {
            return self.add_input(value);
        };
        // Another kernel computes this value.
        if self.claimed.contains(&inst) {
            return self.add_input(value);
        }
        if module.op(inst).get::<Slice>().is_some() {
            return self
                .add_sliced_input(value)
                .or_else(|| self.add_input(value));
        }
        if let Some(broadcast) = module.op_as::<BroadcastTo>(inst) {
            if f32_dims(module, value)? != self.root_dims {
                return None;
            }
            // Reading the broadcast source in place never costs more than reading a materialized
            // copy. Other readers keep the broadcast if they need it.
            return match self.collect(broadcast.input().id(), false) {
                Some(node) => {
                    self.views.push(value);
                    Some(node)
                }
                None => self.add_input(value),
            };
        }
        let Some((op, lhs, rhs)) = node_op(module, inst) else {
            return self.add_input(value);
        };
        // The kernel reads smaller producers, such as per-row statistics, as broadcast inputs.
        // Values with other readers are materialized once and read as inputs.
        if f32_dims(module, value)? != self.root_dims || (!is_root && module.uses(value).len() != 1)
        {
            return self.add_input(value);
        }
        let lhs = self.collect(lhs, false)?;
        let rhs = match rhs {
            Some(rhs) => Some(self.collect(rhs, false)?),
            None => None,
        };
        self.nodes_insts.insert(inst);
        self.nodes.push(Node { op, lhs, rhs });
        Some(NodeRef::Node(self.nodes.len() - 1))
    }

    /// Whether the kernel reads through a broadcast or a slice that no op outside it reads.
    fn removes_view(&self) -> bool {
        self.views.iter().any(|&view| {
            self.module
                .uses(view)
                .iter()
                .all(|user| self.nodes_insts.contains(&user.inst))
        })
    }
}
