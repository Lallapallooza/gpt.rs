use std::collections::BTreeMap;

use gpt_rs::backend::index::InstId;
use gpt_rs::backend::optimizer::{FunctionPass, OptimizeContext, PassResult};
use gpt_rs::backend::pattern::{OperationView, SoftmaxDecompositionView};
use gpt_rs::backend::rewriter::ProgramRewriter;
use gpt_rs::backend::spec::{
    BroadcastToSpec, CustomCallAttr, CustomCallSpec, DType, ElementwiseBinaryOp, Function, Operand,
    Operation, ReduceKind, Shape, TensorSpec, ValueId, ValueType,
};
use gpt_rs::ops::functional::activation::SoftmaxLastDimPattern;

use super::rewrite_utils::{
    erase_insts_if_dead, operand_value, replace_value_and_results, tensor_spec_of,
};
use crate::targets::TARGET_SOFTMAX_LAST_AXIS_FUSED_F32_V1;

pub struct TritonSoftmaxFusionPass;

impl TritonSoftmaxFusionPass {
    const NAME: &'static str = "triton-softmax-fusion";
}

impl FunctionPass<crate::TritonBackend> for TritonSoftmaxFusionPass {
    fn name(&self) -> &'static str {
        Self::NAME
    }

    fn run(
        &self,
        function: &mut Function,
        _cx: &mut OptimizeContext<crate::TritonBackend>,
    ) -> PassResult {
        let mut rewriter = match ProgramRewriter::new(function) {
            Ok(rewriter) => rewriter,
            Err(_) => return PassResult::default(),
        };

        let mut changed = false;
        let mut rewrites = 0usize;
        let mut erased = 0usize;
        let insts = rewriter.insts_in_order();
        for inst in insts {
            if !rewriter.contains(inst) {
                continue;
            }
            if SoftmaxLastDimPattern::extract(inst, &rewriter).is_some() {
                gpt_rs::profiling::cache_event("triton_softmax_pattern_match");
            }
            let rewrite = if let Some(view) = SoftmaxDecompositionView::extract(inst, &rewriter) {
                gpt_rs::profiling::cache_event("triton_softmax_decomposition_match");
                match_softmax_last_axis_from_decomposition_view(&rewriter, &view)
            } else {
                None
            };
            let Some(rewrite) = rewrite else {
                continue;
            };

            let mut attrs = BTreeMap::new();
            attrs.insert(
                "axis".to_string(),
                CustomCallAttr::I64(i64::try_from(rewrite.axis).unwrap_or(-1)),
            );

            let op = Operation::CustomCall(CustomCallSpec {
                target: TARGET_SOFTMAX_LAST_AXIS_FUSED_F32_V1.to_string(),
                attrs,
            });
            let operands = vec![Operand::Value(rewrite.input)];
            let output_type = ValueType::Tensor(rewrite.out_spec.clone());
            let Ok((_new_inst, new_value)) =
                rewriter.insert_before(inst, op, operands, output_type)
            else {
                continue;
            };

            replace_value_and_results(&mut rewriter, rewrite.output, new_value);
            erased += erase_insts_if_dead(&mut rewriter, rewrite.roots.as_slice());
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

struct SoftmaxRewrite {
    input: ValueId,
    output: ValueId,
    out_spec: TensorSpec,
    axis: usize,
    roots: Vec<InstId>,
}

struct ReduceChain {
    reduce_inst: InstId,
    bcast_inst: Option<InstId>,
}

fn match_softmax_last_axis_from_decomposition_view(
    rewriter: &ProgramRewriter,
    view: &SoftmaxDecompositionView,
) -> Option<SoftmaxRewrite> {
    match_softmax_last_axis(
        rewriter,
        view.max.result,
        &view.max.spec,
        &view.max.operands,
        view.shifted.root,
        view.shifted.result,
        &view.shifted.operands,
        view.exp_values.root,
        view.exp_values.result,
        &view.exp_values.operands,
        view.sum.result,
        &view.sum.spec,
        &view.sum.operands,
        view.output.root,
        view.output.result,
        &view.output.operands,
    )
}

#[allow(clippy::too_many_arguments)]
fn match_softmax_last_axis(
    rewriter: &ProgramRewriter,
    max_result: ValueId,
    max_spec: &gpt_rs::backend::spec::ReduceSpec,
    max_operands: &[Operand],
    shifted_root: InstId,
    shifted_result: ValueId,
    shifted_operands: &[Operand],
    exp_root: InstId,
    exp_result: ValueId,
    exp_operands: &[Operand],
    sum_result: ValueId,
    sum_spec: &gpt_rs::backend::spec::ReduceSpec,
    sum_operands: &[Operand],
    output_root: InstId,
    output: ValueId,
    output_operands: &[Operand],
) -> Option<SoftmaxRewrite> {
    let out_spec = tensor_spec_of(rewriter, output)
        .or_else(|| reject_softmax("triton_softmax_reject_missing_out_spec"))?;
    if out_spec.dtype != DType::F32 {
        return reject_softmax("triton_softmax_reject_out_dtype");
    }
    let out_dims = out_spec.shape.static_dims()?;
    if out_dims.is_empty() {
        return reject_softmax("triton_softmax_reject_rank0");
    }
    let axis = out_dims.len() - 1;

    if max_spec.kind != ReduceKind::Max || !max_spec.keepdims || max_spec.axes.as_slice() != [axis]
    {
        return reject_softmax("triton_softmax_reject_reduce_max_shape");
    }
    if sum_spec.kind != ReduceKind::Sum || !sum_spec.keepdims || sum_spec.axes.as_slice() != [axis]
    {
        return reject_softmax("triton_softmax_reject_reduce_sum_shape");
    }

    let input = operand_value(max_operands.first())
        .or_else(|| reject_softmax("triton_softmax_reject_missing_reduce_max_operand"))?;
    let input_spec = tensor_spec_of(rewriter, input)
        .or_else(|| reject_softmax("triton_softmax_reject_missing_input_spec"))?;
    if input_spec != out_spec {
        return reject_softmax("triton_softmax_reject_input_out_mismatch");
    }

    let shifted_lhs = operand_value(shifted_operands.first())
        .or_else(|| reject_softmax("triton_softmax_reject_shifted_lhs"))?;
    let shifted_rhs = operand_value(shifted_operands.get(1))
        .or_else(|| reject_softmax("triton_softmax_reject_shifted_rhs"))?;
    if shifted_lhs != input {
        return reject_softmax("triton_softmax_reject_shifted_input_mismatch");
    }
    let max_chain = match_reduce_chain(
        rewriter,
        shifted_rhs,
        max_result,
        &out_spec.shape,
        "triton_softmax_reject_shifted_max_chain",
    )?;

    let exp_input = operand_value(exp_operands.first())
        .or_else(|| reject_softmax("triton_softmax_reject_exp_operand"))?;
    if exp_input != shifted_result {
        return reject_softmax("triton_softmax_reject_exp_shifted_mismatch");
    }

    let sum_input = operand_value(sum_operands.first())
        .or_else(|| reject_softmax("triton_softmax_reject_reduce_sum_operand"))?;
    if sum_input != exp_result {
        return reject_softmax("triton_softmax_reject_reduce_sum_input_mismatch");
    }

    let output_inst = rewriter
        .inst_of(output)
        .or_else(|| reject_softmax("triton_softmax_reject_missing_output_inst"))?;
    if output_inst != output_root {
        return reject_softmax("triton_softmax_reject_output_root_mismatch");
    }
    if !matches!(
        rewriter.op(output_inst),
        Operation::ElementwiseBinary(ElementwiseBinaryOp::Div)
    ) {
        return reject_softmax("triton_softmax_reject_output_not_div");
    }
    let div_lhs = operand_value(output_operands.first())
        .or_else(|| reject_softmax("triton_softmax_reject_div_lhs"))?;
    let div_rhs = operand_value(output_operands.get(1))
        .or_else(|| reject_softmax("triton_softmax_reject_div_rhs"))?;
    if div_lhs != exp_result {
        return reject_softmax("triton_softmax_reject_div_lhs_mismatch");
    }
    let sum_chain = match_reduce_chain(
        rewriter,
        div_rhs,
        sum_result,
        &out_spec.shape,
        "triton_softmax_reject_div_rhs_sum_chain",
    )?;

    let shifted_spec = tensor_spec_of(rewriter, shifted_result)
        .or_else(|| reject_softmax("triton_softmax_reject_missing_shifted_spec"))?;
    if shifted_spec != out_spec {
        return reject_softmax("triton_softmax_reject_shifted_out_mismatch");
    }
    let exp_spec = tensor_spec_of(rewriter, exp_result)
        .or_else(|| reject_softmax("triton_softmax_reject_missing_exp_spec"))?;
    if exp_spec != out_spec {
        return reject_softmax("triton_softmax_reject_exp_out_mismatch");
    }

    let mut roots = vec![
        output_inst,
        exp_root,
        shifted_root,
        sum_chain.reduce_inst,
        max_chain.reduce_inst,
    ];
    if let Some(inst) = sum_chain.bcast_inst {
        roots.push(inst);
    }
    if let Some(inst) = max_chain.bcast_inst {
        roots.push(inst);
    }

    Some(SoftmaxRewrite {
        input,
        output,
        out_spec,
        axis,
        roots,
    })
}

fn match_reduce_chain(
    rewriter: &ProgramRewriter,
    value: ValueId,
    expected_reduce: ValueId,
    expected_shape: &Shape,
    reason_prefix: &'static str,
) -> Option<ReduceChain> {
    let mut reduce_value = value;
    let mut bcast_inst = None;

    if let Some(inst) = rewriter.inst_of(value) {
        if let Operation::BroadcastTo(spec) = rewriter.op(inst) {
            if !broadcast_matches_shape(spec, expected_shape) {
                return reject_softmax(reason_prefix);
            }
            reduce_value = operand_value(rewriter.operands(inst).first())
                .or_else(|| reject_softmax(reason_prefix))?;
            bcast_inst = Some(inst);
        }
    }

    if reduce_value != expected_reduce {
        return reject_softmax(reason_prefix);
    }
    let reduce_inst = rewriter
        .inst_of(reduce_value)
        .or_else(|| reject_softmax(reason_prefix))?;
    let _input = operand_value(rewriter.operands(reduce_inst).first())
        .or_else(|| reject_softmax(reason_prefix))?;

    Some(ReduceChain {
        reduce_inst,
        bcast_inst,
    })
}

fn broadcast_matches_shape(spec: &BroadcastToSpec, expected_shape: &Shape) -> bool {
    &spec.result_shape == expected_shape
}

fn reject_softmax<T>(reason: &'static str) -> Option<T> {
    gpt_rs::profiling::cache_event(reason);
    None
}
