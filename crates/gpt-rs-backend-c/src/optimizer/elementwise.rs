use std::collections::{BTreeMap, HashMap, HashSet};

use gpt_rs::backend::{
    index::InstId,
    optimizer::{FunctionPass, OptimizeContext, PassResult},
    rewriter::ProgramRewriter,
    spec::{
        CastSpec, CustomCallAttr, CustomCallSpec, DType, Function, Operand, Operation, SliceSpec,
        ValueId, ValueType,
    },
};

use crate::targets::{binary_code, fused_input_fits, unary_code, TARGET_ELEMENTWISE_FUSED};

use super::utils::tensor_spec_of;

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

/// An input of a fused kernel. The kernel reads the operand broadcast into the output shape. When
/// `starts` is set, the kernel reads it at an offset inside a same-rank source, which is an
/// absorbed `slice`.
#[derive(Clone)]
struct FusedInput {
    operand: Operand,
    starts: Option<Vec<usize>>,
}

struct FusionPlan {
    inputs: Vec<FusedInput>,
    nodes: Vec<FusedNode>,
    removes_view: bool,
}

fn build_fusion_plan<'a, 'r>(
    rewriter: &'a ProgramRewriter<'r>,
    root_inst: InstId,
    root_dims: &[usize],
) -> Option<FusionPlan> {
    let mut inputs: Vec<FusedInput> = Vec::new();
    let mut input_map: HashMap<(ValueId, Option<Vec<usize>>), usize> = HashMap::new();
    let mut nodes: Vec<FusedNode> = Vec::new();
    let mut memo: HashMap<ValueId, FusedRef> = HashMap::new();
    let mut visiting: HashSet<ValueId> = HashSet::new();
    let mut fused_insts: HashSet<InstId> = HashSet::new();
    let mut absorbed_views: Vec<ValueId> = Vec::new();

    struct CollectCtx<'a, 'r> {
        rewriter: &'a ProgramRewriter<'r>,
        root_dims: &'a [usize],
        inputs: &'a mut Vec<FusedInput>,
        input_map: &'a mut HashMap<(ValueId, Option<Vec<usize>>), usize>,
        nodes: &'a mut Vec<FusedNode>,
        memo: &'a mut HashMap<ValueId, FusedRef>,
        visiting: &'a mut HashSet<ValueId>,
        fused_insts: &'a mut HashSet<InstId>,
        absorbed_views: &'a mut Vec<ValueId>,
    }

    fn add_input_value(ctx: &mut CollectCtx<'_, '_>, value: ValueId) -> Option<FusedRef> {
        if let Some(idx) = ctx.input_map.get(&(value, None)).copied() {
            return Some(FusedRef::Input(idx));
        }
        let spec = tensor_spec_of(ctx.rewriter, value)?;
        if spec.dtype != DType::F32 {
            return None;
        }
        let dims = spec.shape.static_dims()?;
        if !fused_input_fits(ctx.root_dims, &dims, None) {
            return None;
        }
        let idx = ctx.inputs.len();
        ctx.inputs.push(FusedInput {
            operand: Operand::Value(value),
            starts: None,
        });
        ctx.input_map.insert((value, None), idx);
        Some(FusedRef::Input(idx))
    }

    /// Reads the source of a `slice` at an offset instead of materialising the slice.
    fn add_sliced_input(
        ctx: &mut CollectCtx<'_, '_>,
        slice_inst: InstId,
        spec: &SliceSpec,
    ) -> Option<FusedRef> {
        if spec.sizes.as_slice() != ctx.root_dims {
            return None;
        }
        let Some(Operand::Value(source)) = ctx.rewriter.operands(slice_inst).first().cloned()
        else {
            return None;
        };
        let source_spec = tensor_spec_of(ctx.rewriter, source)?;
        if source_spec.dtype != DType::F32 {
            return None;
        }
        let source_dims = source_spec.shape.static_dims()?;
        if !fused_input_fits(ctx.root_dims, &source_dims, Some(&spec.starts)) {
            return None;
        }
        ctx.absorbed_views.push(ctx.rewriter.value_of(slice_inst));
        let key = (source, Some(spec.starts.clone()));
        if let Some(idx) = ctx.input_map.get(&key).copied() {
            return Some(FusedRef::Input(idx));
        }
        let idx = ctx.inputs.len();
        ctx.inputs.push(FusedInput {
            operand: Operand::Value(source),
            starts: Some(spec.starts.clone()),
        });
        ctx.input_map.insert(key, idx);
        Some(FusedRef::Input(idx))
    }

    fn add_input_operand(ctx: &mut CollectCtx<'_, '_>, operand: Operand) -> Option<FusedRef> {
        match &operand {
            Operand::Value(value) => add_input_value(ctx, *value),
            Operand::Literal(literal) => {
                if literal.spec.dtype != DType::F32 {
                    return None;
                }
                let dims = literal.spec.shape.static_dims()?;
                if !fused_input_fits(ctx.root_dims, &dims, None) {
                    return None;
                }
                let idx = ctx.inputs.len();
                ctx.inputs.push(FusedInput {
                    operand,
                    starts: None,
                });
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
                    if spec.dtype != DType::F32 {
                        None
                    } else if spec.shape.static_dims().as_deref() != Some(ctx.root_dims)
                        || (!force_fuse && ctx.rewriter.users_of(value).len() != 1)
                    {
                        // The kernel reads smaller producers, such as per-row statistics, as
                        // broadcast inputs. Values with other users are materialised once and
                        // read as inputs.
                        add_input_operand(ctx, Operand::Value(value))
                    } else {
                        let operand = ctx.rewriter.operands(inst).first()?;
                        let lhs = collect_operand(operand, ctx)?;
                        ctx.fused_insts.insert(inst);
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
                    if spec.dtype != DType::F32 {
                        None
                    } else if spec.shape.static_dims().as_deref() != Some(ctx.root_dims)
                        || (!force_fuse && ctx.rewriter.users_of(value).len() != 1)
                    {
                        add_input_operand(ctx, Operand::Value(value))
                    } else {
                        let operands = ctx.rewriter.operands(inst);
                        let lhs = collect_operand(operands.first()?, ctx)?;
                        let rhs = collect_operand(operands.get(1)?, ctx)?;
                        ctx.fused_insts.insert(inst);
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
                        || spec.shape.static_dims().as_deref() != Some(ctx.root_dims)
                    {
                        None
                    } else {
                        // Reading the broadcast source in place never costs more than reading a
                        // materialised copy. Other users keep the op alive if they need it.
                        let operand = ctx.rewriter.operands(inst).first()?;
                        match collect_operand(operand, ctx) {
                            Some(fused) => {
                                ctx.absorbed_views.push(value);
                                Some(fused)
                            }
                            None => add_input_operand(ctx, Operand::Value(value)),
                        }
                    }
                }
                Operation::Slice(spec) => {
                    let spec = spec.clone();
                    add_sliced_input(ctx, inst, &spec)
                        .or_else(|| add_input_operand(ctx, Operand::Value(value)))
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
        fused_insts: &mut fused_insts,
        absorbed_views: &mut absorbed_views,
    };
    let root_value = rewriter.value_of(root_inst);
    let root_idx = collect_value(root_value, &mut ctx, true)?;
    if matches!(root_idx, FusedRef::Input(_)) {
        return None;
    }
    let removes_view = absorbed_views.iter().any(|view| {
        !rewriter.func.result_ids.contains(view)
            && rewriter
                .users_of(*view)
                .iter()
                .all(|user| fused_insts.contains(user))
    });
    Some(FusionPlan {
        inputs,
        nodes,
        removes_view,
    })
}

/// Erases `root` and, transitively, the elementwise producers that only fed it.
fn erase_dead_fused_producers(rewriter: &mut ProgramRewriter<'_>, root: InstId) -> usize {
    let mut erased = 0;
    let mut worklist = vec![root];
    while let Some(inst) = worklist.pop() {
        if !rewriter.contains(inst) {
            continue;
        }
        let value = rewriter.value_of(inst);
        if !rewriter.users_of(value).is_empty() || rewriter.func.result_ids.contains(&value) {
            continue;
        }
        if !matches!(
            rewriter.op(inst),
            Operation::ElementwiseUnary(_)
                | Operation::ElementwiseBinary(_)
                | Operation::BroadcastTo(_)
                | Operation::Slice(_)
        ) {
            continue;
        }
        let producers: Vec<InstId> = rewriter
            .operands(inst)
            .iter()
            .filter_map(|operand| match operand {
                Operand::Value(value) => rewriter.inst_of(*value),
                _ => None,
            })
            .collect();
        rewriter
            .erase_inst(inst)
            .expect("a fused producer without users can be erased");
        erased += 1;
        worklist.extend(producers);
    }
    erased
}

pub struct CElementwiseFusionPass;

impl CElementwiseFusionPass {
    const NAME: &'static str = "c-elementwise-fusion";
}

impl FunctionPass<crate::CBackend> for CElementwiseFusionPass {
    fn name(&self) -> &'static str {
        Self::NAME
    }

    fn run(
        &self,
        function: &mut Function,
        _cx: &mut OptimizeContext<crate::CBackend>,
    ) -> PassResult {
        let mut rewriter = match ProgramRewriter::new(function) {
            Ok(r) => r,
            Err(_) => {
                return PassResult::default();
            }
        };

        let mut changed = false;
        let mut rewrites = 0usize;
        let mut erased = 0usize;
        // Visit consumers before producers. Each fusion then starts from the last op of a chain
        // and absorbs every single-use producer behind it.
        let insts = rewriter.insts_in_order();
        for inst in insts.into_iter().rev() {
            if !rewriter.contains(inst) {
                continue;
            }
            let root_value = rewriter.value_of(inst);
            if rewriter.users_of(root_value).is_empty()
                && !rewriter.func.result_ids.contains(&root_value)
            {
                continue;
            }
            let root_spec = match tensor_spec_of(&rewriter, root_value) {
                Some(spec) => spec,
                None => continue,
            };
            if root_spec.dtype != DType::F32 {
                continue;
            }
            let root_dims = match root_spec.shape.static_dims() {
                Some(dims) => dims,
                None => continue,
            };
            match rewriter.op(inst) {
                Operation::ElementwiseUnary(_) | Operation::ElementwiseBinary(_) => {}
                _ => continue,
            }

            // A root whose only use is an f32 -> bf16 cast stores bf16 directly. Such a cast
            // typically produces a bf16 matmul input.
            let output_cast = match rewriter.users_of(root_value) {
                [user]
                    if !rewriter.func.result_ids.contains(&root_value)
                        && matches!(
                            rewriter.op(*user),
                            Operation::Cast(CastSpec { dtype: DType::Bf16 })
                        ) =>
                {
                    Some(*user)
                }
                _ => None,
            };

            let Some(plan) = build_fusion_plan(&rewriter, inst, &root_dims) else {
                continue;
            };
            // A single op is worth rewriting only when it avoids materialising an intermediate.
            // It does so when it reads through a broadcast or slice and makes that view dead, or
            // when it writes through a narrowing cast.
            if plan.nodes.len() < 2 && !plan.removes_view && output_cast.is_none() {
                continue;
            }

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
            attrs.insert("ops_kind".into(), CustomCallAttr::I64Array(kinds));
            attrs.insert("ops_code".into(), CustomCallAttr::I64Array(codes));
            attrs.insert("lhs".into(), CustomCallAttr::I64Array(lhs));
            attrs.insert("rhs".into(), CustomCallAttr::I64Array(rhs));
            let slice_starts: Vec<i64> = plan
                .inputs
                .iter()
                .map(|input| &input.starts)
                .enumerate()
                .filter_map(|(idx, starts)| starts.as_ref().map(|starts| (idx, starts)))
                .flat_map(|(idx, starts)| {
                    std::iter::once(idx as i64).chain(starts.iter().map(|start| *start as i64))
                })
                .collect();
            if !slice_starts.is_empty() {
                attrs.insert(
                    "input_slice_starts".into(),
                    CustomCallAttr::I64Array(slice_starts),
                );
            }

            let op = Operation::CustomCall(CustomCallSpec {
                target: TARGET_ELEMENTWISE_FUSED.to_string(),
                attrs,
            });

            let mut output_spec = root_spec.clone();
            if output_cast.is_some() {
                output_spec.dtype = DType::Bf16;
            }
            let Ok((_new_inst, new_value)) = rewriter.insert_before(
                inst,
                op,
                plan.inputs
                    .iter()
                    .map(|input| input.operand.clone())
                    .collect(),
                ValueType::Tensor(output_spec),
            ) else {
                continue;
            };

            let replaced = match output_cast {
                Some(cast) => rewriter.value_of(cast),
                None => root_value,
            };
            rewriter.replace_all_uses(replaced, new_value);
            for result_id in &mut rewriter.func.result_ids {
                if *result_id == replaced {
                    *result_id = new_value;
                }
            }
            if let Some(cast) = output_cast {
                rewriter
                    .erase_inst(cast)
                    .expect("absorbed output cast has no users left");
                erased += 1;
            }
            erased += erase_dead_fused_producers(&mut rewriter, inst);

            changed = true;
            rewrites += 1;
        }

        PassResult {
            changed,
            iterations: 1,
            rewrites_applied: rewrites,
            erased_insts: erased,
        }
    }
}
