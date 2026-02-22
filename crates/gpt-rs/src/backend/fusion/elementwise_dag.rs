use std::collections::{HashMap, HashSet};

use crate::backend::index::InstId;
use crate::backend::rewriter::ProgramRewriter;
use crate::backend::spec::{
    DType, ElementwiseBinaryOp, ElementwiseUnaryOp, Operand, Operation, TensorSpec, ValueId,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FusionNodeKind {
    Unary,
    Binary,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FusionRef {
    Input(usize),
    Node(usize),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FusionNode {
    pub kind: FusionNodeKind,
    pub code: i64,
    pub lhs: FusionRef,
    pub rhs: Option<FusionRef>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ElementwiseDag {
    pub inputs: Vec<Operand>,
    pub nodes: Vec<FusionNode>,
    pub node_values: Vec<ValueId>,
}

pub fn unary_opcode(op: ElementwiseUnaryOp) -> i64 {
    match op {
        ElementwiseUnaryOp::Neg => 0,
        ElementwiseUnaryOp::Abs => 1,
        ElementwiseUnaryOp::Exp => 2,
        ElementwiseUnaryOp::Log => 3,
        ElementwiseUnaryOp::Tanh => 4,
        ElementwiseUnaryOp::Erf => 5,
        ElementwiseUnaryOp::Rsqrt => 6,
        ElementwiseUnaryOp::Reciprocal => 7,
    }
}

pub fn binary_opcode(op: ElementwiseBinaryOp) -> i64 {
    match op {
        ElementwiseBinaryOp::Add => 0,
        ElementwiseBinaryOp::Sub => 1,
        ElementwiseBinaryOp::Mul => 2,
        ElementwiseBinaryOp::Div => 3,
        ElementwiseBinaryOp::Maximum => 4,
        ElementwiseBinaryOp::Minimum => 5,
    }
}

pub fn build_elementwise_dag(
    rewriter: &ProgramRewriter<'_>,
    root_inst: InstId,
) -> Option<ElementwiseDag> {
    let root_value = rewriter.value_of(root_inst);
    let root_spec = tensor_spec_of(rewriter, root_value)?;
    if root_spec.dtype != DType::F32 {
        return None;
    }
    let root_dims = root_spec.shape.static_dims()?;

    let mut inputs: Vec<Operand> = Vec::new();
    let mut input_map: HashMap<ValueId, usize> = HashMap::new();
    let mut nodes: Vec<FusionNode> = Vec::new();
    let mut node_values: Vec<ValueId> = Vec::new();
    let mut memo: HashMap<ValueId, FusionRef> = HashMap::new();
    let mut visiting: HashSet<ValueId> = HashSet::new();

    struct CollectCtx<'a, 'r> {
        rewriter: &'a ProgramRewriter<'r>,
        root_dims: &'a [usize],
        inputs: &'a mut Vec<Operand>,
        input_map: &'a mut HashMap<ValueId, usize>,
        nodes: &'a mut Vec<FusionNode>,
        node_values: &'a mut Vec<ValueId>,
        memo: &'a mut HashMap<ValueId, FusionRef>,
        visiting: &'a mut HashSet<ValueId>,
    }

    fn add_input_value(ctx: &mut CollectCtx<'_, '_>, value: ValueId) -> Option<FusionRef> {
        if let Some(idx) = ctx.input_map.get(&value).copied() {
            return Some(FusionRef::Input(idx));
        }
        let spec = tensor_spec_of(ctx.rewriter, value)?;
        if spec.dtype != DType::F32 {
            return None;
        }
        let dims = spec.shape.static_dims()?;
        if !is_broadcastable(ctx.root_dims, &dims) {
            return None;
        }
        let idx = ctx.inputs.len();
        ctx.inputs.push(Operand::Value(value));
        ctx.input_map.insert(value, idx);
        Some(FusionRef::Input(idx))
    }

    fn add_input_operand(ctx: &mut CollectCtx<'_, '_>, operand: Operand) -> Option<FusionRef> {
        match &operand {
            Operand::Value(value) => add_input_value(ctx, *value),
            Operand::Literal(literal) => {
                if literal.spec.dtype != DType::F32 {
                    return None;
                }
                let dims = literal.spec.shape.static_dims()?;
                if !is_broadcastable(ctx.root_dims, &dims) {
                    return None;
                }
                let idx = ctx.inputs.len();
                ctx.inputs.push(operand);
                Some(FusionRef::Input(idx))
            }
            Operand::TupleElement { .. } => None,
        }
    }

    fn collect_operand(operand: &Operand, ctx: &mut CollectCtx<'_, '_>) -> Option<FusionRef> {
        match operand {
            Operand::Value(id) => collect_value(*id, ctx, false),
            Operand::Literal(literal) => add_input_operand(ctx, Operand::Literal(literal.clone())),
            Operand::TupleElement { .. } => None,
        }
    }

    fn collect_value(
        value: ValueId,
        ctx: &mut CollectCtx<'_, '_>,
        force_fuse: bool,
    ) -> Option<FusionRef> {
        if let Some(idx) = ctx.memo.get(&value).copied() {
            return Some(idx);
        }
        if !ctx.visiting.insert(value) {
            return None;
        }

        let inst = ctx.rewriter.inst_of(value);
        let result = if let Some(inst) = inst {
            match ctx.rewriter.op(inst) {
                Operation::ElementwiseUnary(op) => {
                    let spec = tensor_spec_of(ctx.rewriter, value)?;
                    if spec.dtype != DType::F32
                        || spec.shape.static_dims().as_deref() != Some(ctx.root_dims)
                    {
                        None
                    } else if !force_fuse && ctx.rewriter.users_of(value).len() != 1 {
                        add_input_operand(ctx, Operand::Value(value))
                    } else {
                        let operand = ctx.rewriter.operands(inst).first()?;
                        let lhs = collect_operand(operand, ctx)?;
                        let idx = ctx.nodes.len();
                        ctx.nodes.push(FusionNode {
                            kind: FusionNodeKind::Unary,
                            code: unary_opcode(*op),
                            lhs,
                            rhs: None,
                        });
                        ctx.node_values.push(value);
                        Some(FusionRef::Node(idx))
                    }
                }
                Operation::ElementwiseBinary(op) => {
                    let spec = tensor_spec_of(ctx.rewriter, value)?;
                    if spec.dtype != DType::F32
                        || spec.shape.static_dims().as_deref() != Some(ctx.root_dims)
                    {
                        None
                    } else if !force_fuse && ctx.rewriter.users_of(value).len() != 1 {
                        add_input_operand(ctx, Operand::Value(value))
                    } else {
                        let operands = ctx.rewriter.operands(inst);
                        let lhs = collect_operand(operands.first()?, ctx)?;
                        let rhs = collect_operand(operands.get(1)?, ctx)?;
                        let idx = ctx.nodes.len();
                        ctx.nodes.push(FusionNode {
                            kind: FusionNodeKind::Binary,
                            code: binary_opcode(*op),
                            lhs,
                            rhs: Some(rhs),
                        });
                        ctx.node_values.push(value);
                        Some(FusionRef::Node(idx))
                    }
                }
                Operation::BroadcastTo(_) => {
                    let spec = tensor_spec_of(ctx.rewriter, value)?;
                    if spec.dtype != DType::F32
                        || spec.shape.static_dims().as_deref() != Some(ctx.root_dims)
                    {
                        None
                    } else if ctx.rewriter.users_of(value).len() == 1 {
                        let operand = ctx.rewriter.operands(inst).first()?;
                        collect_operand(operand, ctx)
                    } else {
                        add_input_operand(ctx, Operand::Value(value))
                    }
                }
                _ => add_input_operand(ctx, Operand::Value(value)),
            }
        } else {
            add_input_operand(ctx, Operand::Value(value))
        };

        ctx.visiting.remove(&value);
        if let Some(idx) = result {
            ctx.memo.insert(value, idx);
        }
        result
    }

    let mut ctx = CollectCtx {
        rewriter,
        root_dims: root_dims.as_slice(),
        inputs: &mut inputs,
        input_map: &mut input_map,
        nodes: &mut nodes,
        node_values: &mut node_values,
        memo: &mut memo,
        visiting: &mut visiting,
    };
    let root_idx = collect_value(root_value, &mut ctx, true)?;
    if matches!(root_idx, FusionRef::Input(_)) {
        return None;
    }
    if nodes.len() < 2 {
        return None;
    }
    Some(ElementwiseDag {
        inputs,
        nodes,
        node_values,
    })
}

fn is_broadcastable(root_dims: &[usize], in_dims: &[usize]) -> bool {
    if root_dims.len() < in_dims.len() {
        return false;
    }
    let offset = root_dims.len() - in_dims.len();
    for (idx, dim) in in_dims.iter().enumerate() {
        let out_dim = root_dims[idx + offset];
        if *dim != 1 && *dim != out_dim {
            return false;
        }
    }
    true
}

fn tensor_spec_of(rewriter: &ProgramRewriter<'_>, value: ValueId) -> Option<TensorSpec> {
    match rewriter.type_of(value) {
        Some(crate::backend::spec::ValueType::Tensor(spec)) => Some(spec.clone()),
        _ => None,
    }
}
