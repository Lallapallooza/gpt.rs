use gpt_rs::backend::conversion::{ConversionError, ConversionResult};
use gpt_rs::backend::fusion::{
    FUSION_ATTR_KIND, FUSION_ATTR_VERSION, FUSION_KIND_DOT_EPILOGUE_V1,
    FUSION_KIND_ELEMENTWISE_DAG_V1,
};
use gpt_rs::backend::spec::{DType, Operation};

use crate::targets::{
    TARGET_DOT_BIAS_FUSED_F32_V1, TARGET_ELEMENTWISE_FUSED_F32_V1, TARGET_LAYER_NORM_FUSED_F32_V1,
    TARGET_SOFTMAX_LAST_AXIS_FUSED_F32_V1,
};

use super::ValidateCtx;

pub(super) fn validate(cx: &ValidateCtx<'_>) -> ConversionResult<()> {
    let Operation::CustomCall(spec) = &cx.instruction.op else {
        return Err(ConversionError::new(
            "internal triton validator error: custom validator on non-custom op",
        ));
    };
    match spec.target.as_str() {
        TARGET_ELEMENTWISE_FUSED_F32_V1 => validate_fused_elementwise(cx, spec),
        TARGET_DOT_BIAS_FUSED_F32_V1 => validate_fused_dot_bias(cx, spec),
        TARGET_LAYER_NORM_FUSED_F32_V1 => validate_fused_layer_norm(cx, spec),
        TARGET_SOFTMAX_LAST_AXIS_FUSED_F32_V1 => validate_fused_softmax(cx, spec),
        _ => Err(ConversionError::new(format!(
            "unsupported triton custom_call target '{}'",
            spec.target
        ))),
    }
}

fn validate_fused_elementwise(
    cx: &ValidateCtx<'_>,
    spec: &gpt_rs::backend::spec::CustomCallSpec,
) -> ConversionResult<()> {
    let version = cx.custom_call_i64(spec.attrs.get(FUSION_ATTR_VERSION), FUSION_ATTR_VERSION)?;
    if version != 1 {
        return Err(ConversionError::new(format!(
            "unsupported fused elementwise payload version {version}"
        )));
    }
    let kind = cx.custom_call_string(spec.attrs.get(FUSION_ATTR_KIND), FUSION_ATTR_KIND)?;
    if kind != FUSION_KIND_ELEMENTWISE_DAG_V1 {
        return Err(ConversionError::new(format!(
            "unsupported fused elementwise payload kind '{kind}'"
        )));
    }
    let out_spec = cx.output_tensor_spec()?;
    if out_spec.dtype != DType::F32 {
        return Err(ConversionError::new(
            "triton fused elementwise custom_call requires F32 output",
        ));
    }
    let kinds = cx.custom_call_i64_array(spec.attrs.get("ops_kind"), "ops_kind")?;
    let codes = cx.custom_call_i64_array(spec.attrs.get("ops_code"), "ops_code")?;
    let lhs = cx.custom_call_i64_array(spec.attrs.get("lhs"), "lhs")?;
    let rhs = cx.custom_call_i64_array(spec.attrs.get("rhs"), "rhs")?;
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
    if cx.instruction.operands.is_empty() {
        return Err(ConversionError::new(
            "triton fused elementwise requires at least one operand",
        ));
    }
    for input_spec in cx.each_operand_tensor_spec("fused elementwise input")? {
        if input_spec.dtype != DType::F32 {
            return Err(ConversionError::new(
                "triton fused elementwise requires F32 inputs",
            ));
        }
    }
    Ok(())
}

fn validate_fused_dot_bias(
    cx: &ValidateCtx<'_>,
    spec: &gpt_rs::backend::spec::CustomCallSpec,
) -> ConversionResult<()> {
    let version = cx.custom_call_i64(spec.attrs.get(FUSION_ATTR_VERSION), FUSION_ATTR_VERSION)?;
    if version != 1 {
        return Err(ConversionError::new(format!(
            "unsupported fused dot-epilogue payload version {version}"
        )));
    }
    let kind = cx.custom_call_string(spec.attrs.get(FUSION_ATTR_KIND), FUSION_ATTR_KIND)?;
    if kind != FUSION_KIND_DOT_EPILOGUE_V1 {
        return Err(ConversionError::new(format!(
            "unsupported fused dot-epilogue payload kind '{kind}'"
        )));
    }
    if cx.instruction.operands.len() < 3 {
        return Err(ConversionError::new(
            "triton fused dot+bias requires at least three operands",
        ));
    }
    let out_spec = cx.output_tensor_spec()?;
    if out_spec.dtype != DType::F32 {
        return Err(ConversionError::new(
            "triton fused dot+bias requires F32 output",
        ));
    }
    for input_spec in cx.each_operand_tensor_spec("dot+bias input")? {
        if input_spec.dtype != DType::F32 {
            return Err(ConversionError::new(
                "triton fused dot+bias requires F32 inputs",
            ));
        }
    }
    let add_input = cx.custom_call_i64(spec.attrs.get("dot_add_input"), "dot_add_input")?;
    if add_input < 0 || add_input as usize >= cx.instruction.operands.len() {
        return Err(ConversionError::new(
            "triton fused dot+bias has out-of-range dot_add_input index",
        ));
    }
    let _ = cx.custom_call_i64_array(spec.attrs.get("dot_batch_lhs"), "dot_batch_lhs")?;
    let _ = cx.custom_call_i64_array(spec.attrs.get("dot_batch_rhs"), "dot_batch_rhs")?;
    let _ = cx.custom_call_i64_array(spec.attrs.get("dot_contract_lhs"), "dot_contract_lhs")?;
    let _ = cx.custom_call_i64_array(spec.attrs.get("dot_contract_rhs"), "dot_contract_rhs")?;
    Ok(())
}

fn validate_fused_layer_norm(
    cx: &ValidateCtx<'_>,
    spec: &gpt_rs::backend::spec::CustomCallSpec,
) -> ConversionResult<()> {
    if cx.instruction.operands.len() != 3 {
        return Err(ConversionError::new(
            "triton fused layer_norm requires exactly three inputs",
        ));
    }
    let out_spec = cx.output_tensor_spec()?;
    if out_spec.dtype != DType::F32 {
        return Err(ConversionError::new(
            "triton fused layer_norm requires F32 output",
        ));
    }
    let input_spec = cx.operand_tensor_spec(0, "input")?;
    let gamma_spec = cx.operand_tensor_spec(1, "gamma")?;
    let beta_spec = cx.operand_tensor_spec(2, "beta")?;
    if input_spec.dtype != DType::F32
        || gamma_spec.dtype != DType::F32
        || beta_spec.dtype != DType::F32
    {
        return Err(ConversionError::new(
            "triton fused layer_norm requires F32 input/gamma/beta",
        ));
    }
    if input_spec.shape != out_spec.shape {
        return Err(ConversionError::new(
            "triton fused layer_norm requires matching input/output shapes",
        ));
    }
    let input_dims = input_spec.shape.static_dims().ok_or_else(|| {
        ConversionError::new("triton fused layer_norm requires static input shape dimensions")
    })?;
    if input_dims.is_empty() {
        return Err(ConversionError::new(
            "triton fused layer_norm requires rank >= 1 input",
        ));
    }
    let axis = cx.custom_call_i64(spec.attrs.get("axis"), "axis")?;
    let expected = i64::try_from(input_dims.len() - 1)
        .map_err(|_| ConversionError::new("triton fused layer_norm rank exceeds i64 range"))?;
    if axis != expected {
        return Err(ConversionError::new(
            "triton fused layer_norm currently supports last-axis only",
        ));
    }
    let feature = input_dims[input_dims.len() - 1];
    let gamma_dims = gamma_spec.shape.static_dims().ok_or_else(|| {
        ConversionError::new("triton fused layer_norm requires static gamma shape dimensions")
    })?;
    let beta_dims = beta_spec.shape.static_dims().ok_or_else(|| {
        ConversionError::new("triton fused layer_norm requires static beta shape dimensions")
    })?;
    if gamma_dims.as_slice() != [feature] || beta_dims.as_slice() != [feature] {
        return Err(ConversionError::new(
            "triton fused layer_norm requires gamma/beta shape [hidden_size]",
        ));
    }
    let _ = cx.custom_call_f64(spec.attrs.get("eps"), "eps")?;
    Ok(())
}

fn validate_fused_softmax(
    cx: &ValidateCtx<'_>,
    spec: &gpt_rs::backend::spec::CustomCallSpec,
) -> ConversionResult<()> {
    if cx.instruction.operands.len() != 1 {
        return Err(ConversionError::new(
            "triton fused softmax requires exactly one input",
        ));
    }
    let out_spec = cx.output_tensor_spec()?;
    if out_spec.dtype != DType::F32 {
        return Err(ConversionError::new(
            "triton fused softmax requires F32 output",
        ));
    }
    let input_spec = cx.operand_tensor_spec(0, "input")?;
    if input_spec.dtype != DType::F32 || input_spec.shape != out_spec.shape {
        return Err(ConversionError::new(
            "triton fused softmax requires matching F32 input/output shape",
        ));
    }
    let axis = cx.custom_call_i64(spec.attrs.get("axis"), "axis")?;
    let rank = out_spec.shape.dims().len();
    let expected = i64::try_from(rank.saturating_sub(1))
        .map_err(|_| ConversionError::new("triton fused softmax rank exceeds i64 range"))?;
    if axis != expected {
        return Err(ConversionError::new(
            "triton fused softmax currently supports last-axis only",
        ));
    }
    Ok(())
}
