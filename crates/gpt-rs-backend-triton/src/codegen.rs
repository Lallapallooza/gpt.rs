mod hint_lowering;

use gpt_rs::backend::conversion::{BufferPlan, ConversionError, ConversionResult};
use gpt_rs::backend::fusion::{
    FUSION_ATTR_KIND, FUSION_ATTR_VERSION, FUSION_KIND_DOT_EPILOGUE_V1,
    FUSION_KIND_ELEMENTWISE_DAG_V1,
};
use gpt_rs::backend::spec::{
    CustomCallAttr, DType, ElementwiseUnaryOp, Function, Operand, Operation, Program, ReduceKind,
    TensorSpec, ValueType,
};

use crate::artifact::TritonArtifact;
use crate::kernels::{
    broadcast_kernel_spec, broadcast_si32_kernel_spec, compare_si32_i1_kernel_spec,
    concat_kernel_spec, dot_bias_rank2_kernel_spec, dynamic_update_slice_f32_kernel_spec,
    elementwise_binary_kernel_spec, elementwise_unary_kernel_spec,
    extract_patches_nhwc_kernel_spec, iota_si32_kernel_spec, prepacked_kernel_sources,
    reduce_max_last_axis_kernel_spec, reduce_sum_last_axis_kernel_spec,
    reduce_window_max_nhwc_kernel_spec, select_i1_f32_kernel_spec, slice_kernel_spec,
    take_f32_i32_kernel_spec, transpose_kernel_spec,
};
use crate::targets::{TARGET_DOT_BIAS_FUSED_F32_V1, TARGET_ELEMENTWISE_FUSED_F32_V1};

pub fn lower_program_to_artifact(
    program: &Program,
    entrypoint_symbol: &str,
    buffer_plan: BufferPlan,
) -> ConversionResult<TritonArtifact> {
    // Touch all prepacked assets so they are included and validated by the
    // compiler even before every kernel family is hooked into runtime dispatch.
    let _ = prepacked_kernel_sources();

    let lowered_program = hint_lowering::lower_hint_regions_to_custom_calls(program)?;
    let function = entry_function(&lowered_program)?;
    for instruction in &function.body {
        validate_instruction(function, instruction)?;
    }

    let kernels = vec![
        elementwise_unary_kernel_spec(),
        elementwise_binary_kernel_spec(),
        broadcast_kernel_spec(),
        broadcast_si32_kernel_spec(),
        slice_kernel_spec(),
        transpose_kernel_spec(),
        concat_kernel_spec(),
        reduce_sum_last_axis_kernel_spec(),
        reduce_max_last_axis_kernel_spec(),
        iota_si32_kernel_spec(),
        compare_si32_i1_kernel_spec(),
        select_i1_f32_kernel_spec(),
        take_f32_i32_kernel_spec(),
        dynamic_update_slice_f32_kernel_spec(),
        extract_patches_nhwc_kernel_spec(),
        reduce_window_max_nhwc_kernel_spec(),
        dot_bias_rank2_kernel_spec(),
    ];

    Ok(TritonArtifact::new(
        entrypoint_symbol.to_string(),
        lowered_program,
        buffer_plan,
        kernels,
    ))
}

fn validate_instruction(
    function: &Function,
    instruction: &gpt_rs::backend::spec::Instruction,
) -> ConversionResult<()> {
    match &instruction.op {
        Operation::Constant(literal) => {
            if !matches!(literal.spec.dtype, DType::F32 | DType::Si32 | DType::I1) {
                return Err(ConversionError::new(format!(
                    "triton constant lowering supports F32/Si32/I1 only, got {:?}",
                    literal.spec.dtype
                )));
            }
            ensure_tensor_output(&instruction.output)?;
            Ok(())
        }
        Operation::StopGradient | Operation::Reshape(_) => {
            let _ = operand_tensor_spec_for_operand(
                function,
                instruction.operands.first(),
                "alias source",
            )?;
            ensure_tensor_output(&instruction.output)?;
            Ok(())
        }
        Operation::ElementwiseUnary(op) => {
            let input_spec = operand_tensor_spec_for_operand(
                function,
                instruction.operands.first(),
                "elementwise unary input",
            )?;
            let out_spec = ensure_tensor_output(&instruction.output)?;
            if input_spec != out_spec || input_spec.dtype != DType::F32 {
                return Err(ConversionError::new(
                    "triton elementwise unary lowering requires matching F32 input/output specs",
                ));
            }
            match op {
                ElementwiseUnaryOp::Neg
                | ElementwiseUnaryOp::Abs
                | ElementwiseUnaryOp::Exp
                | ElementwiseUnaryOp::Log
                | ElementwiseUnaryOp::Tanh
                | ElementwiseUnaryOp::Erf
                | ElementwiseUnaryOp::Rsqrt
                | ElementwiseUnaryOp::Reciprocal => Ok(()),
            }
        }
        Operation::ElementwiseBinary(_) => {
            let lhs_spec = operand_tensor_spec_for_operand(
                function,
                instruction.operands.first(),
                "elementwise lhs",
            )?;
            let rhs_spec = operand_tensor_spec_for_operand(
                function,
                instruction.operands.get(1),
                "elementwise rhs",
            )?;
            let out_spec = ensure_tensor_output(&instruction.output)?;
            if lhs_spec != out_spec || rhs_spec != out_spec || out_spec.dtype != DType::F32 {
                return Err(ConversionError::new(
                    "triton elementwise lowering requires equal F32 lhs/rhs/out specs",
                ));
            }
            Ok(())
        }
        Operation::DotGeneral(_) => {
            let lhs_spec =
                operand_tensor_spec_for_operand(function, instruction.operands.first(), "dot lhs")?;
            let rhs_spec =
                operand_tensor_spec_for_operand(function, instruction.operands.get(1), "dot rhs")?;
            let out_spec = ensure_tensor_output(&instruction.output)?;
            if lhs_spec.dtype != DType::F32
                || rhs_spec.dtype != DType::F32
                || out_spec.dtype != DType::F32
            {
                return Err(ConversionError::new(
                    "triton dot lowering currently supports F32 only",
                ));
            }
            Ok(())
        }
        Operation::Reduce(spec) => {
            let input_spec = operand_tensor_spec_for_operand(
                function,
                instruction.operands.first(),
                "reduce input",
            )?;
            let out_spec = ensure_tensor_output(&instruction.output)?;
            if !matches!(spec.kind, ReduceKind::Sum | ReduceKind::Max) {
                return Err(ConversionError::new(
                    "triton reduce lowering supports sum/max only",
                ));
            }
            if input_spec.dtype != DType::F32 || out_spec.dtype != DType::F32 {
                return Err(ConversionError::new(
                    "triton reduce lowering currently supports F32 only",
                ));
            }
            Ok(())
        }
        Operation::BroadcastTo(_) => {
            let input_spec = operand_tensor_spec_for_operand(
                function,
                instruction.operands.first(),
                "broadcast input",
            )?;
            let out_spec = ensure_tensor_output(&instruction.output)?;
            if input_spec.dtype != out_spec.dtype {
                return Err(ConversionError::new(
                    "triton broadcast lowering requires matching input/output dtype",
                ));
            }
            if !matches!(out_spec.dtype, DType::F32 | DType::Si32) {
                return Err(ConversionError::new(
                    "triton broadcast lowering supports F32/Si32 only",
                ));
            }
            Ok(())
        }
        Operation::Slice(_) => {
            let input_spec = operand_tensor_spec_for_operand(
                function,
                instruction.operands.first(),
                "slice input",
            )?;
            let out_spec = ensure_tensor_output(&instruction.output)?;
            if input_spec.dtype != out_spec.dtype {
                return Err(ConversionError::new(
                    "triton slice lowering requires matching input/output dtype",
                ));
            }
            if !matches!(out_spec.dtype, DType::F32 | DType::Si32) {
                return Err(ConversionError::new(
                    "triton slice lowering supports F32/Si32 only",
                ));
            }
            Ok(())
        }
        Operation::Transpose(_) => {
            let input_spec = operand_tensor_spec_for_operand(
                function,
                instruction.operands.first(),
                "transpose input",
            )?;
            let out_spec = ensure_tensor_output(&instruction.output)?;
            if input_spec.dtype != DType::F32 || out_spec.dtype != DType::F32 {
                return Err(ConversionError::new(
                    "triton transpose lowering currently supports F32 only",
                ));
            }
            Ok(())
        }
        Operation::Concat(_) => {
            let out_spec = ensure_tensor_output(&instruction.output)?;
            if out_spec.dtype != DType::F32 {
                return Err(ConversionError::new(
                    "triton concat lowering currently supports F32 only",
                ));
            }
            for operand in &instruction.operands {
                let spec =
                    operand_tensor_spec_for_operand(function, Some(operand), "concat input")?;
                if spec.dtype != DType::F32 {
                    return Err(ConversionError::new(
                        "triton concat lowering currently supports F32 inputs only",
                    ));
                }
            }
            Ok(())
        }
        Operation::Iota(spec) => {
            let out_spec = ensure_tensor_output(&instruction.output)?;
            if spec.dtype != DType::Si32 || out_spec.dtype != DType::Si32 {
                return Err(ConversionError::new(
                    "triton iota lowering currently supports Si32 only",
                ));
            }
            Ok(())
        }
        Operation::Compare(_) => {
            let lhs_spec = operand_tensor_spec_for_operand(
                function,
                instruction.operands.first(),
                "compare lhs",
            )?;
            let rhs_spec = operand_tensor_spec_for_operand(
                function,
                instruction.operands.get(1),
                "compare rhs",
            )?;
            let out_spec = ensure_tensor_output(&instruction.output)?;
            if lhs_spec.dtype != DType::Si32
                || rhs_spec.dtype != DType::Si32
                || out_spec.dtype != DType::I1
            {
                return Err(ConversionError::new(
                    "triton compare lowering currently supports Si32 -> I1 only",
                ));
            }
            Ok(())
        }
        Operation::Select => {
            let pred_spec = operand_tensor_spec_for_operand(
                function,
                instruction.operands.first(),
                "select predicate",
            )?;
            let when_true_spec = operand_tensor_spec_for_operand(
                function,
                instruction.operands.get(1),
                "select true",
            )?;
            let when_false_spec = operand_tensor_spec_for_operand(
                function,
                instruction.operands.get(2),
                "select false",
            )?;
            let out_spec = ensure_tensor_output(&instruction.output)?;
            if pred_spec.dtype != DType::I1
                || when_true_spec.dtype != DType::F32
                || when_false_spec.dtype != DType::F32
                || out_spec.dtype != DType::F32
            {
                return Err(ConversionError::new(
                    "triton select lowering currently supports I1 predicate with F32 branches",
                ));
            }
            Ok(())
        }
        Operation::Take => {
            let params_spec = operand_tensor_spec_for_operand(
                function,
                instruction.operands.first(),
                "take params",
            )?;
            let indices_spec = operand_tensor_spec_for_operand(
                function,
                instruction.operands.get(1),
                "take indices",
            )?;
            let out_spec = ensure_tensor_output(&instruction.output)?;
            if params_spec.dtype != DType::F32
                || indices_spec.dtype != DType::Si32
                || out_spec.dtype != DType::F32
            {
                return Err(ConversionError::new(
                    "triton take lowering currently supports F32 params and Si32 indices",
                ));
            }
            Ok(())
        }
        Operation::DynamicSlice(_) => {
            let value_spec = operand_tensor_spec_for_operand(
                function,
                instruction.operands.first(),
                "dynamic_slice value",
            )?;
            let starts_spec = operand_tensor_spec_for_operand(
                function,
                instruction.operands.get(1),
                "dynamic_slice starts",
            )?;
            let out_spec = ensure_tensor_output(&instruction.output)?;
            if starts_spec.dtype != DType::Si32 || value_spec.dtype != out_spec.dtype {
                return Err(ConversionError::new(
                    "triton dynamic_slice lowering requires Si32 starts and matching value/output dtype",
                ));
            }
            if !matches!(out_spec.dtype, DType::F32 | DType::Si32) {
                return Err(ConversionError::new(
                    "triton dynamic_slice lowering supports F32/Si32 outputs only",
                ));
            }
            Ok(())
        }
        Operation::DynamicUpdateSlice(_) => {
            let base_spec = operand_tensor_spec_for_operand(
                function,
                instruction.operands.first(),
                "dynamic_update_slice base",
            )?;
            let update_spec = operand_tensor_spec_for_operand(
                function,
                instruction.operands.get(1),
                "dynamic_update_slice update",
            )?;
            let starts_spec = operand_tensor_spec_for_operand(
                function,
                instruction.operands.get(2),
                "dynamic_update_slice starts",
            )?;
            let out_spec = ensure_tensor_output(&instruction.output)?;
            if base_spec.dtype != DType::F32
                || update_spec.dtype != DType::F32
                || starts_spec.dtype != DType::Si32
                || out_spec.dtype != DType::F32
            {
                return Err(ConversionError::new(
                    "triton dynamic_update_slice lowering currently supports F32 base/update with Si32 starts",
                ));
            }
            Ok(())
        }
        Operation::ExtractPatches(_) => {
            let input_spec = operand_tensor_spec_for_operand(
                function,
                instruction.operands.first(),
                "extract_patches input",
            )?;
            let out_spec = ensure_tensor_output(&instruction.output)?;
            if input_spec.dtype != DType::F32 || out_spec.dtype != DType::F32 {
                return Err(ConversionError::new(
                    "triton extract_patches lowering currently supports F32 only",
                ));
            }
            Ok(())
        }
        Operation::ReduceWindow(spec) => {
            let input_spec = operand_tensor_spec_for_operand(
                function,
                instruction.operands.first(),
                "reduce_window input",
            )?;
            let out_spec = ensure_tensor_output(&instruction.output)?;
            if spec.reduce != ReduceKind::Max {
                return Err(ConversionError::new(
                    "triton reduce_window lowering currently supports max only",
                ));
            }
            if input_spec.dtype != DType::F32 || out_spec.dtype != DType::F32 {
                return Err(ConversionError::new(
                    "triton reduce_window lowering currently supports F32 only",
                ));
            }
            Ok(())
        }
        Operation::CustomCall(spec) => match spec.target.as_str() {
            TARGET_ELEMENTWISE_FUSED_F32_V1 => {
                let version =
                    custom_call_i64(spec.attrs.get(FUSION_ATTR_VERSION), FUSION_ATTR_VERSION)?;
                if version != 1 {
                    return Err(ConversionError::new(format!(
                        "unsupported fused elementwise payload version {version}"
                    )));
                }
                let kind = custom_call_string(spec.attrs.get(FUSION_ATTR_KIND), FUSION_ATTR_KIND)?;
                if kind != FUSION_KIND_ELEMENTWISE_DAG_V1 {
                    return Err(ConversionError::new(format!(
                        "unsupported fused elementwise payload kind '{kind}'"
                    )));
                }
                let out_spec = ensure_tensor_output(&instruction.output)?;
                if out_spec.dtype != DType::F32 {
                    return Err(ConversionError::new(
                        "triton fused elementwise custom_call requires F32 output",
                    ));
                }
                let kinds = custom_call_i64_array(spec.attrs.get("ops_kind"), "ops_kind")?;
                let codes = custom_call_i64_array(spec.attrs.get("ops_code"), "ops_code")?;
                let lhs = custom_call_i64_array(spec.attrs.get("lhs"), "lhs")?;
                let rhs = custom_call_i64_array(spec.attrs.get("rhs"), "rhs")?;
                let node_count = kinds.len();
                if node_count < 2 {
                    return Err(ConversionError::new(
                        "triton fused elementwise requires at least two fused nodes",
                    ));
                }
                if codes.len() != node_count || lhs.len() != node_count || rhs.len() != node_count {
                    return Err(ConversionError::new(
                        "triton fused elementwise attr arrays must have equal length",
                    ));
                }
                if instruction.operands.is_empty() {
                    return Err(ConversionError::new(
                        "triton fused elementwise requires at least one operand",
                    ));
                }
                for operand in &instruction.operands {
                    let input_spec = operand_tensor_spec_for_operand(
                        function,
                        Some(operand),
                        "fused elementwise input",
                    )?;
                    if input_spec.dtype != DType::F32 {
                        return Err(ConversionError::new(
                            "triton fused elementwise requires F32 inputs",
                        ));
                    }
                }
                Ok(())
            }
            TARGET_DOT_BIAS_FUSED_F32_V1 => {
                let version =
                    custom_call_i64(spec.attrs.get(FUSION_ATTR_VERSION), FUSION_ATTR_VERSION)?;
                if version != 1 {
                    return Err(ConversionError::new(format!(
                        "unsupported fused dot-epilogue payload version {version}"
                    )));
                }
                let kind = custom_call_string(spec.attrs.get(FUSION_ATTR_KIND), FUSION_ATTR_KIND)?;
                if kind != FUSION_KIND_DOT_EPILOGUE_V1 {
                    return Err(ConversionError::new(format!(
                        "unsupported fused dot-epilogue payload kind '{kind}'"
                    )));
                }
                if instruction.operands.len() < 3 {
                    return Err(ConversionError::new(
                        "triton fused dot+bias requires at least three operands",
                    ));
                }
                let out_spec = ensure_tensor_output(&instruction.output)?;
                if out_spec.dtype != DType::F32 {
                    return Err(ConversionError::new(
                        "triton fused dot+bias requires F32 output",
                    ));
                }
                for operand in &instruction.operands {
                    let input_spec =
                        operand_tensor_spec_for_operand(function, Some(operand), "dot+bias input")?;
                    if input_spec.dtype != DType::F32 {
                        return Err(ConversionError::new(
                            "triton fused dot+bias requires F32 inputs",
                        ));
                    }
                }
                let add_input = custom_call_i64(spec.attrs.get("dot_add_input"), "dot_add_input")?;
                if add_input < 0 || add_input as usize >= instruction.operands.len() {
                    return Err(ConversionError::new(
                        "triton fused dot+bias has out-of-range dot_add_input index",
                    ));
                }
                let _ = custom_call_i64_array(spec.attrs.get("dot_batch_lhs"), "dot_batch_lhs")?;
                let _ = custom_call_i64_array(spec.attrs.get("dot_batch_rhs"), "dot_batch_rhs")?;
                let _ =
                    custom_call_i64_array(spec.attrs.get("dot_contract_lhs"), "dot_contract_lhs")?;
                let _ =
                    custom_call_i64_array(spec.attrs.get("dot_contract_rhs"), "dot_contract_rhs")?;
                Ok(())
            }
            _ => Err(ConversionError::new(format!(
                "unsupported triton custom_call target '{}'",
                spec.target
            ))),
        },
        other => Err(ConversionError::new(format!(
            "triton lowering does not support operation: {:?}",
            other
        ))),
    }
}

fn custom_call_i64_array<'a>(
    attr: Option<&'a CustomCallAttr>,
    name: &str,
) -> ConversionResult<&'a [i64]> {
    match attr {
        Some(CustomCallAttr::I64Array(values)) => Ok(values.as_slice()),
        _ => Err(ConversionError::new(format!(
            "triton custom_call missing i64 array attr '{name}'"
        ))),
    }
}

fn custom_call_i64(attr: Option<&CustomCallAttr>, name: &str) -> ConversionResult<i64> {
    match attr {
        Some(CustomCallAttr::I64(value)) => Ok(*value),
        _ => Err(ConversionError::new(format!(
            "triton custom_call missing i64 attr '{name}'"
        ))),
    }
}

fn custom_call_string<'a>(
    attr: Option<&'a CustomCallAttr>,
    name: &str,
) -> ConversionResult<&'a str> {
    match attr {
        Some(CustomCallAttr::String(value)) => Ok(value.as_str()),
        _ => Err(ConversionError::new(format!(
            "triton custom_call missing string attr '{name}'"
        ))),
    }
}

fn entry_function(program: &Program) -> ConversionResult<&Function> {
    program
        .functions
        .iter()
        .find(|function| function.name == program.entry)
        .ok_or_else(|| ConversionError::new("entry function not found"))
}

fn ensure_tensor_output(output: &ValueType) -> ConversionResult<TensorSpec> {
    match output {
        ValueType::Tensor(spec) => Ok(spec.clone()),
        ValueType::Tuple(_) => Err(ConversionError::new(
            "tuple outputs are not supported by triton lowering",
        )),
    }
}

fn operand_tensor_spec_for_operand(
    function: &Function,
    operand: Option<&Operand>,
    what: &str,
) -> ConversionResult<TensorSpec> {
    match operand {
        Some(Operand::Value(id)) => operand_tensor_spec(function, id.0).ok_or_else(|| {
            ConversionError::new(format!(
                "failed to resolve operand tensor spec for {what} value {}",
                id.0
            ))
        }),
        Some(Operand::Literal(literal)) => Ok(literal.spec.clone()),
        Some(Operand::TupleElement { .. }) => Err(ConversionError::new(format!(
            "{what} uses tuple element operand, unsupported in triton lowering"
        ))),
        None => Err(ConversionError::new(format!("missing operand for {what}"))),
    }
}

fn operand_tensor_spec(function: &Function, value_id: u32) -> Option<TensorSpec> {
    if let Some((idx, _)) = function
        .parameter_ids
        .iter()
        .enumerate()
        .find(|(_, id)| id.0 == value_id)
    {
        return match function.parameters.get(idx) {
            Some(ValueType::Tensor(spec)) => Some(spec.clone()),
            _ => None,
        };
    }

    function.body.iter().find_map(|instruction| {
        if instruction.id.0 != value_id {
            return None;
        }
        match &instruction.output {
            ValueType::Tensor(spec) => Some(spec.clone()),
            ValueType::Tuple(_) => None,
        }
    })
}
