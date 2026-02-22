use std::collections::{BTreeMap, HashMap, HashSet};

use gpt_rs::backend::{
    index::InstId,
    optimizer::{FunctionPass, OptimizeContext, PassResult},
    rewriter::ProgramRewriter,
    spec::{
        CustomCallAttr, CustomCallSpec, DType, Function, Operand, Operation, ValueId, ValueType,
    },
};

use crate::targets::{
    binary_code, unary_code, FUSION_ATTR_KIND, FUSION_ATTR_VERSION, FUSION_KIND_ELEMENTWISE_DAG_V1,
    TARGET_ELEMENTWISE_FUSED_F32_V1,
};

use super::utils::{static_dims, tensor_spec_of};

enum FusedKind {
    Unary,
    Binary,
}

#[derive(Clone, Copy)]
enum FusedRef {
    Input(usize),
    Node(usize),
}

struct FusedNode {
    kind: FusedKind,
    code: i64,
    lhs: FusedRef,
    rhs: Option<FusedRef>,
}

struct FusionPlan {
    inputs: Vec<Operand>,
    nodes: Vec<FusedNode>,
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

fn build_fusion_plan<'a, 'r>(
    rewriter: &'a ProgramRewriter<'r>,
    root_inst: InstId,
    root_dims: &[usize],
) -> Option<FusionPlan> {
    let mut inputs: Vec<Operand> = Vec::new();
    let mut input_map: HashMap<ValueId, usize> = HashMap::new();
    let mut nodes: Vec<FusedNode> = Vec::new();
    let mut memo: HashMap<ValueId, FusedRef> = HashMap::new();
    let mut visiting: HashSet<ValueId> = HashSet::new();

    struct CollectCtx<'a, 'r> {
        rewriter: &'a ProgramRewriter<'r>,
        root_dims: &'a [usize],
        inputs: &'a mut Vec<Operand>,
        input_map: &'a mut HashMap<ValueId, usize>,
        nodes: &'a mut Vec<FusedNode>,
        memo: &'a mut HashMap<ValueId, FusedRef>,
        visiting: &'a mut HashSet<ValueId>,
    }

    fn add_input_value(ctx: &mut CollectCtx<'_, '_>, value: ValueId) -> Option<FusedRef> {
        if let Some(idx) = ctx.input_map.get(&value).copied() {
            return Some(FusedRef::Input(idx));
        }
        let spec = tensor_spec_of(ctx.rewriter, value)?;
        if spec.dtype != DType::F32 {
            return None;
        }
        let dims = static_dims(&spec)?;
        if !is_broadcastable(ctx.root_dims, &dims) {
            return None;
        }
        let idx = ctx.inputs.len();
        ctx.inputs.push(Operand::Value(value));
        ctx.input_map.insert(value, idx);
        Some(FusedRef::Input(idx))
    }

    fn add_input_operand(ctx: &mut CollectCtx<'_, '_>, operand: Operand) -> Option<FusedRef> {
        match &operand {
            Operand::Value(value) => add_input_value(ctx, *value),
            Operand::Literal(literal) => {
                if literal.spec.dtype != DType::F32 {
                    return None;
                }
                let dims = static_dims(&literal.spec)?;
                if !is_broadcastable(ctx.root_dims, &dims) {
                    return None;
                }
                let idx = ctx.inputs.len();
                ctx.inputs.push(operand);
                Some(FusedRef::Input(idx))
            }
            Operand::TupleElement { .. } => None,
        }
    }

    fn collect_operand(operand: &Operand, ctx: &mut CollectCtx<'_, '_>) -> Option<FusedRef> {
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
    ) -> Option<FusedRef> {
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
                        || static_dims(&spec).as_deref() != Some(ctx.root_dims)
                    {
                        None
                    } else if !force_fuse && ctx.rewriter.users_of(value).len() != 1 {
                        add_input_operand(ctx, Operand::Value(value))
                    } else {
                        let operand = ctx.rewriter.operands(inst).first()?;
                        let lhs = collect_operand(operand, ctx)?;
                        let idx = ctx.nodes.len();
                        ctx.nodes.push(FusedNode {
                            kind: FusedKind::Unary,
                            code: unary_code(*op),
                            lhs,
                            rhs: None,
                        });
                        Some(FusedRef::Node(idx))
                    }
                }
                Operation::ElementwiseBinary(op) => {
                    let spec = tensor_spec_of(ctx.rewriter, value)?;
                    if spec.dtype != DType::F32
                        || static_dims(&spec).as_deref() != Some(ctx.root_dims)
                    {
                        None
                    } else if !force_fuse && ctx.rewriter.users_of(value).len() != 1 {
                        add_input_operand(ctx, Operand::Value(value))
                    } else {
                        let operands = ctx.rewriter.operands(inst);
                        let lhs = collect_operand(operands.first()?, ctx)?;
                        let rhs = collect_operand(operands.get(1)?, ctx)?;
                        let idx = ctx.nodes.len();
                        ctx.nodes.push(FusedNode {
                            kind: FusedKind::Binary,
                            code: binary_code(*op),
                            lhs,
                            rhs: Some(rhs),
                        });
                        Some(FusedRef::Node(idx))
                    }
                }
                Operation::BroadcastTo(_) => {
                    let spec = tensor_spec_of(ctx.rewriter, value)?;
                    if spec.dtype != DType::F32
                        || static_dims(&spec).as_deref() != Some(ctx.root_dims)
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
        root_dims,
        inputs: &mut inputs,
        input_map: &mut input_map,
        nodes: &mut nodes,
        memo: &mut memo,
        visiting: &mut visiting,
    };
    let root_value = rewriter.value_of(root_inst);
    let root_idx = collect_value(root_value, &mut ctx, true)?;
    if matches!(root_idx, FusedRef::Input(_)) {
        return None;
    }
    if nodes.len() < 2 {
        return None;
    }
    Some(FusionPlan { inputs, nodes })
}

#[derive(Debug, Default)]
pub struct TritonElementwiseFusionPass;

impl TritonElementwiseFusionPass {
    const NAME: &'static str = "triton_elementwise_fusion";
}

impl FunctionPass<crate::TritonBackend> for TritonElementwiseFusionPass {
    fn name(&self) -> &'static str {
        Self::NAME
    }

    fn run(
        &self,
        function: &mut Function,
        _cx: &mut OptimizeContext<crate::TritonBackend>,
    ) -> PassResult {
        let mut rewriter = match ProgramRewriter::new(function) {
            Ok(r) => r,
            Err(_) => {
                return PassResult::default();
            }
        };

        let mut changed = false;
        let mut rewrites = 0usize;
        let insts = rewriter.insts_in_order();
        for inst in insts {
            if !rewriter.contains(inst) {
                continue;
            }
            let root_value = rewriter.value_of(inst);
            let root_spec = match tensor_spec_of(&rewriter, root_value) {
                Some(spec) => spec,
                None => continue,
            };
            if root_spec.dtype != DType::F32 {
                continue;
            }
            let root_dims = match static_dims(&root_spec) {
                Some(dims) => dims,
                None => continue,
            };
            match rewriter.op(inst) {
                Operation::ElementwiseUnary(_) | Operation::ElementwiseBinary(_) => {}
                _ => continue,
            }

            let Some(plan) = build_fusion_plan(&rewriter, inst, &root_dims) else {
                continue;
            };

            let input_count = plan.inputs.len();
            let encode_ref = |reference: FusedRef| -> i64 {
                match reference {
                    FusedRef::Input(idx) => idx as i64,
                    FusedRef::Node(idx) => (input_count + idx) as i64,
                }
            };

            let mut kinds = Vec::with_capacity(plan.nodes.len());
            let mut codes = Vec::with_capacity(plan.nodes.len());
            let mut lhs = Vec::with_capacity(plan.nodes.len());
            let mut rhs = Vec::with_capacity(plan.nodes.len());
            for node in &plan.nodes {
                match node.kind {
                    FusedKind::Unary => kinds.push(0),
                    FusedKind::Binary => kinds.push(1),
                }
                codes.push(node.code);
                lhs.push(encode_ref(node.lhs));
                rhs.push(node.rhs.map(encode_ref).unwrap_or(-1));
            }

            let mut attrs = BTreeMap::new();
            attrs.insert(FUSION_ATTR_VERSION.into(), CustomCallAttr::I64(1));
            attrs.insert(
                FUSION_ATTR_KIND.into(),
                CustomCallAttr::String(FUSION_KIND_ELEMENTWISE_DAG_V1.to_string()),
            );
            attrs.insert("ops_kind".into(), CustomCallAttr::I64Array(kinds));
            attrs.insert("ops_code".into(), CustomCallAttr::I64Array(codes));
            attrs.insert("lhs".into(), CustomCallAttr::I64Array(lhs));
            attrs.insert("rhs".into(), CustomCallAttr::I64Array(rhs));

            let op = Operation::CustomCall(CustomCallSpec {
                target: TARGET_ELEMENTWISE_FUSED_F32_V1.to_string(),
                attrs,
            });

            let output_ty = ValueType::Tensor(root_spec.clone());
            let Ok((_new_inst, new_value)) =
                rewriter.insert_before(inst, op, plan.inputs.clone(), output_ty)
            else {
                continue;
            };

            rewriter.replace_all_uses(root_value, new_value);
            for result_id in &mut rewriter.func.result_ids {
                if *result_id == root_value {
                    *result_id = new_value;
                }
            }
            if let Some(old_inst) = rewriter.inst_of(root_value) {
                if rewriter.users_of(root_value).is_empty() {
                    rewriter
                        .erase_inst(old_inst)
                        .expect("triton optimizer erase should succeed");
                }
            }

            changed = true;
            rewrites += 1;
        }

        PassResult {
            changed,
            iterations: 1,
            rewrites_applied: rewrites,
            erased_insts: 0,
        }
    }
}
