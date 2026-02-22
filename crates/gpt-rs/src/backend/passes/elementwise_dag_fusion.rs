use std::collections::BTreeMap;

use crate::backend::fusion::{
    build_elementwise_dag, FusionNodeKind, FusionRef, FUSION_ATTR_KIND, FUSION_ATTR_VERSION,
    FUSION_KIND_ELEMENTWISE_DAG_V1,
};
use crate::backend::optimizer::{FunctionPass, OptimizeContext, PassResult};
use crate::backend::rewriter::ProgramRewriter;
use crate::backend::spec::{CustomCallAttr, CustomCallSpec, DType, Function, Operation, ValueType};

#[derive(Debug, Clone)]
pub struct ElementwiseDagFusionPass {
    target: String,
}

impl ElementwiseDagFusionPass {
    pub fn new(target: impl Into<String>) -> Self {
        Self {
            target: target.into(),
        }
    }
}

impl<B: crate::backend::spec::PortableBackend + 'static> FunctionPass<B>
    for ElementwiseDagFusionPass
{
    fn name(&self) -> &'static str {
        "elementwise_dag_fusion"
    }

    fn run(&self, function: &mut Function, _cx: &mut OptimizeContext<B>) -> PassResult {
        let mut rewriter = match ProgramRewriter::new(function) {
            Ok(r) => r,
            Err(_) => return PassResult::default(),
        };

        let mut changed = false;
        let mut rewrites = 0usize;
        let insts = rewriter.insts_in_order();
        for inst in insts {
            if !rewriter.contains(inst) {
                continue;
            }
            let root_value = rewriter.value_of(inst);
            let Some(root_spec) = tensor_spec_of(&rewriter, root_value) else {
                continue;
            };
            if root_spec.dtype != DType::F32 {
                continue;
            }
            match rewriter.op(inst) {
                Operation::ElementwiseUnary(_) | Operation::ElementwiseBinary(_) => {}
                _ => continue,
            }

            let Some(plan) = build_elementwise_dag(&rewriter, inst) else {
                continue;
            };

            let input_count = plan.inputs.len();
            let encode_ref = |reference: FusionRef| -> i64 {
                match reference {
                    FusionRef::Input(idx) => idx as i64,
                    FusionRef::Node(idx) => (input_count + idx) as i64,
                }
            };

            let mut kinds = Vec::with_capacity(plan.nodes.len());
            let mut codes = Vec::with_capacity(plan.nodes.len());
            let mut lhs = Vec::with_capacity(plan.nodes.len());
            let mut rhs = Vec::with_capacity(plan.nodes.len());
            for node in &plan.nodes {
                match node.kind {
                    FusionNodeKind::Unary => kinds.push(0),
                    FusionNodeKind::Binary => kinds.push(1),
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
                target: self.target.clone(),
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
                        .expect("fusion erase should succeed");
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

fn tensor_spec_of(
    rewriter: &ProgramRewriter<'_>,
    value: crate::backend::spec::ValueId,
) -> Option<crate::backend::spec::TensorSpec> {
    match rewriter.type_of(value) {
        Some(ValueType::Tensor(spec)) => Some(spec.clone()),
        _ => None,
    }
}
