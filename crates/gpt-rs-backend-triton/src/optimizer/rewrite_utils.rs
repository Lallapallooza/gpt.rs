use std::convert::TryInto;

use gpt_rs::backend::index::InstId;
use gpt_rs::backend::rewriter::ProgramRewriter;
use gpt_rs::backend::spec::{
    DType, Dimension, Operand, Operation, TensorLiteral, TensorSpec, ValueId, ValueType,
};

pub(super) fn operand_value(operand: Option<&Operand>) -> Option<ValueId> {
    match operand {
        Some(Operand::Value(value)) => Some(*value),
        Some(Operand::TupleElement { .. }) | Some(Operand::Literal(_)) | None => None,
    }
}

pub(super) fn tensor_spec_of(rewriter: &ProgramRewriter, value: ValueId) -> Option<TensorSpec> {
    match rewriter.type_of(value) {
        Some(ValueType::Tensor(spec)) => Some(spec.clone()),
        _ => None,
    }
}

pub(super) fn replace_value_and_results(
    rewriter: &mut ProgramRewriter,
    from: ValueId,
    to: ValueId,
) {
    rewriter.replace_all_uses(from, to);
    for result in &mut rewriter.func.result_ids {
        if *result == from {
            *result = to;
        }
    }
}

pub(super) fn erase_insts_if_dead(rewriter: &mut ProgramRewriter, insts: &[InstId]) -> usize {
    let mut pending = insts.to_vec();
    pending.sort_by_key(|inst| inst.0);
    pending.reverse();

    let mut erased = 0usize;
    loop {
        let mut made_progress = false;
        let mut next = Vec::new();
        for inst in pending {
            if !rewriter.contains(inst) {
                continue;
            }
            let value = rewriter.value_of(inst);
            if !rewriter.users_of(value).is_empty() {
                next.push(inst);
                continue;
            }
            if rewriter.erase_inst(inst).is_ok() {
                erased += 1;
                made_progress = true;
            } else {
                next.push(inst);
            }
        }
        if !made_progress {
            break;
        }
        pending = next;
    }
    erased
}

pub(super) fn scalar_f32_from_operand(
    rewriter: &ProgramRewriter,
    operand: Option<&Operand>,
) -> Option<f32> {
    match operand {
        Some(Operand::Literal(literal)) => scalar_f32_from_literal(literal),
        Some(Operand::Value(value)) => scalar_f32_from_value(rewriter, *value),
        Some(Operand::TupleElement { .. }) | None => None,
    }
}

fn scalar_f32_from_value(rewriter: &ProgramRewriter, value: ValueId) -> Option<f32> {
    let inst = rewriter.inst_of(value)?;
    match rewriter.op(inst) {
        Operation::Constant(literal) => scalar_f32_from_literal(literal),
        Operation::BroadcastTo(_) | Operation::Reshape(_) | Operation::StopGradient => {
            scalar_f32_from_operand(rewriter, rewriter.operands(inst).first())
        }
        _ => None,
    }
}

fn scalar_f32_from_literal(literal: &TensorLiteral) -> Option<f32> {
    if literal.spec.dtype != DType::F32 {
        return None;
    }
    let rank_zero = literal.spec.shape.dims().is_empty();
    let unit_shape = literal
        .spec
        .shape
        .dims()
        .iter()
        .all(|dim| matches!(dim, Dimension::Static(value) if *value == 1 || *value == 0));
    if !rank_zero && !unit_shape {
        return None;
    }
    let bytes: &[u8] = literal.bytes.as_ref();
    if bytes.len() != 4 {
        return None;
    }
    let bytes: [u8; 4] = bytes.try_into().ok()?;
    Some(f32::from_le_bytes(bytes))
}
