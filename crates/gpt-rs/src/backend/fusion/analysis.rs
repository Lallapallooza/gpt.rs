use crate::backend::index::FunctionIndexError;
use crate::backend::rewriter::ProgramRewriter;
use crate::backend::spec::{
    DType, DotGeneralSpec, ElementwiseBinaryOp, Function, HintKind, Operand, Operation, TensorSpec,
    ValueId, ValueType,
};

use super::{
    build_elementwise_dag, DotEpiloguePayload, ElementwiseDagPayload, FusionCandidate,
    FusionDetail, FusionNodeKind,
};

pub fn discover_candidates(
    function: &mut Function,
) -> Result<Vec<FusionCandidate>, FunctionIndexError> {
    let rewriter = ProgramRewriter::new(function)?;
    let mut next_id = 0u32;
    let mut out = Vec::new();

    for inst in rewriter.insts_in_order() {
        let op = rewriter.op(inst);
        match op {
            Operation::ElementwiseUnary(_) | Operation::ElementwiseBinary(_) => {
                if let Some(candidate) = elementwise_candidate(&rewriter, inst, next_id) {
                    out.push(candidate);
                    next_id = next_id.saturating_add(1);
                }
            }
            Operation::DotGeneral(spec) => {
                if let Some(candidate) = dot_epilogue_candidate(&rewriter, inst, spec, next_id) {
                    out.push(candidate);
                    next_id = next_id.saturating_add(1);
                }
            }
            _ => {}
        }
    }

    Ok(out)
}

fn elementwise_candidate(
    rewriter: &ProgramRewriter<'_>,
    root_inst: crate::backend::index::InstId,
    id: u32,
) -> Option<FusionCandidate> {
    let root_value = rewriter.value_of(root_inst);
    let plan = build_elementwise_dag(rewriter, root_inst)?;
    let mut inputs = Vec::with_capacity(plan.inputs.len());
    for operand in &plan.inputs {
        match operand {
            Operand::Value(value) => push_unique_value(&mut inputs, *value),
            Operand::TupleElement { .. } | Operand::Literal(_) => return None,
        }
    }

    let mut inst_values = Vec::with_capacity(plan.node_values.len());
    for value in &plan.node_values {
        push_unique_value(&mut inst_values, *value);
    }
    if inst_values.is_empty() {
        return None;
    }

    let input_count = inputs.len();
    let encode_ref = |reference: super::FusionRef| -> i64 {
        match reference {
            super::FusionRef::Input(idx) => idx as i64,
            super::FusionRef::Node(idx) => (input_count + idx) as i64,
        }
    };

    let mut ops_kind = Vec::with_capacity(plan.nodes.len());
    let mut ops_code = Vec::with_capacity(plan.nodes.len());
    let mut lhs = Vec::with_capacity(plan.nodes.len());
    let mut rhs = Vec::with_capacity(plan.nodes.len());
    for node in &plan.nodes {
        ops_kind.push(match node.kind {
            FusionNodeKind::Unary => 0,
            FusionNodeKind::Binary => 1,
        });
        ops_code.push(node.code);
        lhs.push(encode_ref(node.lhs));
        rhs.push(node.rhs.map(encode_ref).unwrap_or(-1));
    }

    Some(FusionCandidate {
        id,
        kind: HintKind::ElementwiseDag,
        anchor: root_value,
        inst_values,
        inputs,
        exports: vec![root_value],
        detail: FusionDetail::ElementwiseDag(ElementwiseDagPayload {
            ops_kind,
            ops_code,
            lhs,
            rhs,
        }),
    })
}

fn dot_epilogue_candidate(
    rewriter: &ProgramRewriter<'_>,
    dot_inst: crate::backend::index::InstId,
    dot_spec: &DotGeneralSpec,
    id: u32,
) -> Option<FusionCandidate> {
    let dot_value = rewriter.value_of(dot_inst);
    let dot_output_spec = tensor_spec_of_value(rewriter, dot_value)?;
    if dot_output_spec.dtype != DType::F32 {
        return None;
    }

    let users = rewriter.users_of(dot_value);
    if users.len() != 1 {
        return None;
    }
    let add_inst = users[0];
    if !matches!(
        rewriter.op(add_inst),
        Operation::ElementwiseBinary(ElementwiseBinaryOp::Add)
    ) {
        return None;
    }
    let add_value = rewriter.value_of(add_inst);
    let add_output_spec = tensor_spec_of_value(rewriter, add_value)?;
    if add_output_spec.dtype != DType::F32 {
        return None;
    }

    let add_operands = rewriter.operands(add_inst);
    if add_operands.len() != 2 {
        return None;
    }
    let lhs = operand_value(add_operands.first()?)?;
    let rhs = operand_value(add_operands.get(1)?)?;
    let bias = if lhs == dot_value {
        rhs
    } else if rhs == dot_value {
        lhs
    } else {
        return None;
    };
    let bias_spec = tensor_spec_of_value(rewriter, bias)?;
    if bias_spec.dtype != DType::F32 {
        return None;
    }
    if !is_static_broadcastable(&add_output_spec, &bias_spec)
        || !is_static_broadcastable(&add_output_spec, &dot_output_spec)
    {
        return None;
    }

    let mut inst_values = vec![dot_value];
    let mut inputs = Vec::with_capacity(3);
    let dot_operands = rewriter.operands(dot_inst);
    let lhs_input = operand_value(dot_operands.first()?)?;
    let rhs_input = operand_value(dot_operands.get(1)?)?;
    inputs.push(lhs_input);
    inputs.push(rhs_input);

    let mut add_input = 2usize;
    if let Some(bias_inst) = rewriter.inst_of(bias) {
        if matches!(rewriter.op(bias_inst), Operation::BroadcastTo(_))
            && rewriter.users_of(bias).len() == 1
        {
            let broadcast_src = operand_value(rewriter.operands(bias_inst).first()?)?;
            let src_spec = tensor_spec_of_value(rewriter, broadcast_src)?;
            if src_spec.dtype == DType::F32 && is_static_broadcastable(&add_output_spec, &src_spec)
            {
                inputs.push(broadcast_src);
                push_unique_value(&mut inst_values, bias);
            } else {
                inputs.push(bias);
            }
        } else {
            inputs.push(bias);
        }
    } else {
        inputs.push(bias);
    }
    add_input = add_input.min(inputs.len().saturating_sub(1));
    push_unique_value(&mut inst_values, add_value);

    if inst_values.len() < 2 {
        return None;
    }

    Some(FusionCandidate {
        id,
        kind: HintKind::DotEpilogue,
        anchor: add_value,
        inst_values,
        inputs,
        exports: vec![add_value],
        detail: FusionDetail::DotEpilogue(DotEpiloguePayload {
            dot: dot_spec.clone(),
            add_input,
        }),
    })
}

fn operand_value(operand: &Operand) -> Option<ValueId> {
    match operand {
        Operand::Value(value) => Some(*value),
        Operand::TupleElement { .. } | Operand::Literal(_) => None,
    }
}

fn tensor_spec_of_value(rewriter: &ProgramRewriter<'_>, value: ValueId) -> Option<TensorSpec> {
    match rewriter.type_of(value) {
        Some(ValueType::Tensor(spec)) => Some(spec.clone()),
        _ => None,
    }
}

fn is_static_broadcastable(out_spec: &TensorSpec, in_spec: &TensorSpec) -> bool {
    let out_dims = out_spec
        .shape
        .dims()
        .iter()
        .map(|dim| dim.as_static())
        .collect::<Option<Vec<_>>>();
    let in_dims = in_spec
        .shape
        .dims()
        .iter()
        .map(|dim| dim.as_static())
        .collect::<Option<Vec<_>>>();
    let (Some(out_dims), Some(in_dims)) = (out_dims, in_dims) else {
        return false;
    };
    if in_dims.len() > out_dims.len() {
        return false;
    }
    let offset = out_dims.len() - in_dims.len();
    for (idx, in_dim) in in_dims.iter().enumerate() {
        let out_dim = out_dims[offset + idx];
        if *in_dim != 1 && *in_dim != out_dim {
            return false;
        }
    }
    true
}

fn push_unique_value(values: &mut Vec<ValueId>, value: ValueId) {
    if values.iter().all(|existing| *existing != value) {
        values.push(value);
    }
}

trait StaticDimExt {
    fn as_static(&self) -> Option<usize>;
}

impl StaticDimExt for crate::backend::spec::Dimension {
    fn as_static(&self) -> Option<usize> {
        match self {
            crate::backend::spec::Dimension::Static(v) => Some(*v),
            crate::backend::spec::Dimension::Dynamic(_) => None,
        }
    }
}
