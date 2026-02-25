use gpt_rs::backend::conversion::{ConversionError, ConversionResult};
use gpt_rs::backend::spec::{DType, ElementwiseUnaryOp, Operation};

use super::{DTypeCapability, ValidateCtx};

const F32_ONLY: DTypeCapability = DTypeCapability::new(&[DType::F32]);

pub(super) fn validate(cx: &ValidateCtx<'_>) -> ConversionResult<()> {
    match &cx.instruction.op {
        Operation::ElementwiseUnary(op) => validate_unary(cx, *op),
        Operation::ElementwiseBinary(_) => validate_binary(cx),
        Operation::Compare(_) => validate_compare(cx),
        Operation::Select => validate_select(cx),
        Operation::Take => validate_take(cx),
        _ => Err(ConversionError::new(
            "internal triton validator error: elementwise validator on unsupported op",
        )),
    }
}

fn validate_unary(cx: &ValidateCtx<'_>, op: ElementwiseUnaryOp) -> ConversionResult<()> {
    let input_spec = cx.operand_tensor_spec(0, "elementwise unary input")?;
    let out_spec = cx.output_tensor_spec()?;
    if input_spec != out_spec {
        return Err(ConversionError::new(
            "triton elementwise unary lowering requires matching input/output specs",
        ));
    }
    F32_ONLY.require(&input_spec, "triton elementwise unary lowering")?;
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

fn validate_binary(cx: &ValidateCtx<'_>) -> ConversionResult<()> {
    let lhs_spec = cx.operand_tensor_spec(0, "elementwise lhs")?;
    let rhs_spec = cx.operand_tensor_spec(1, "elementwise rhs")?;
    let out_spec = cx.output_tensor_spec()?;
    if lhs_spec != out_spec || rhs_spec != out_spec {
        return Err(ConversionError::new(
            "triton elementwise lowering requires equal lhs/rhs/out specs",
        ));
    }
    F32_ONLY.require(&out_spec, "triton elementwise lowering")
}

fn validate_compare(cx: &ValidateCtx<'_>) -> ConversionResult<()> {
    let lhs_spec = cx.operand_tensor_spec(0, "compare lhs")?;
    let rhs_spec = cx.operand_tensor_spec(1, "compare rhs")?;
    let out_spec = cx.output_tensor_spec()?;
    if lhs_spec.dtype != DType::Si32 || rhs_spec.dtype != DType::Si32 || out_spec.dtype != DType::I1
    {
        return Err(ConversionError::new(
            "triton compare lowering currently supports Si32 -> I1 only",
        ));
    }
    Ok(())
}

fn validate_select(cx: &ValidateCtx<'_>) -> ConversionResult<()> {
    let pred_spec = cx.operand_tensor_spec(0, "select predicate")?;
    let when_true_spec = cx.operand_tensor_spec(1, "select true")?;
    let when_false_spec = cx.operand_tensor_spec(2, "select false")?;
    let out_spec = cx.output_tensor_spec()?;
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

fn validate_take(cx: &ValidateCtx<'_>) -> ConversionResult<()> {
    let params_spec = cx.operand_tensor_spec(0, "take params")?;
    let indices_spec = cx.operand_tensor_spec(1, "take indices")?;
    let out_spec = cx.output_tensor_spec()?;
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
