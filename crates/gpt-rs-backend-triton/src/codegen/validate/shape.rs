use gpt_rs::backend::conversion::{ConversionError, ConversionResult};
use gpt_rs::backend::spec::{DType, Operation, ReduceKind};

use super::{DTypeCapability, ValidateCtx};

const F32_ONLY: DTypeCapability = DTypeCapability::new(&[DType::F32]);
const F32_OR_SI32: DTypeCapability = DTypeCapability::new(&[DType::F32, DType::Si32]);

pub(super) fn validate(cx: &ValidateCtx<'_>) -> ConversionResult<()> {
    match &cx.instruction.op {
        Operation::Constant(literal) => {
            let out_spec = cx.output_tensor_spec()?;
            if !matches!(literal.spec.dtype, DType::F32 | DType::Si32 | DType::I1) {
                return Err(ConversionError::new(format!(
                    "triton constant lowering supports F32/Si32/I1 only, got {:?}",
                    literal.spec.dtype
                )));
            }
            if out_spec != literal.spec {
                return Err(ConversionError::new(
                    "triton constant lowering requires output spec to match literal spec",
                ));
            }
            Ok(())
        }
        Operation::StopGradient | Operation::Reshape(_) => {
            let _ = cx.operand_tensor_spec(0, "alias source")?;
            let _ = cx.output_tensor_spec()?;
            Ok(())
        }
        Operation::BroadcastTo(_) => validate_broadcast(cx),
        Operation::Slice(_) => validate_slice(cx),
        Operation::Transpose(_) => validate_transpose(cx),
        Operation::Concat(_) => validate_concat(cx),
        Operation::Iota(spec) => {
            let out_spec = cx.output_tensor_spec()?;
            if spec.dtype != DType::Si32 || out_spec.dtype != DType::Si32 {
                return Err(ConversionError::new(
                    "triton iota lowering currently supports Si32 only",
                ));
            }
            Ok(())
        }
        Operation::DynamicSlice(_) => validate_dynamic_slice(cx),
        Operation::DynamicUpdateSlice(_) => validate_dynamic_update_slice(cx),
        Operation::ExtractPatches(_) => {
            let input_spec = cx.operand_tensor_spec(0, "extract_patches input")?;
            let out_spec = cx.output_tensor_spec()?;
            F32_ONLY.require(&input_spec, "triton extract_patches lowering")?;
            F32_ONLY.require(&out_spec, "triton extract_patches lowering")
        }
        Operation::ReduceWindow(spec) => {
            if spec.reduce != ReduceKind::Max {
                return Err(ConversionError::new(
                    "triton reduce_window lowering currently supports max only",
                ));
            }
            let input_spec = cx.operand_tensor_spec(0, "reduce_window input")?;
            let out_spec = cx.output_tensor_spec()?;
            F32_ONLY.require(&input_spec, "triton reduce_window lowering")?;
            F32_ONLY.require(&out_spec, "triton reduce_window lowering")
        }
        _ => Err(ConversionError::new(
            "internal triton validator error: shape validator on unsupported op",
        )),
    }
}

fn validate_broadcast(cx: &ValidateCtx<'_>) -> ConversionResult<()> {
    let input_spec = cx.operand_tensor_spec(0, "broadcast input")?;
    let out_spec = cx.output_tensor_spec()?;
    if input_spec.dtype != out_spec.dtype {
        return Err(ConversionError::new(
            "triton broadcast lowering requires matching input/output dtype",
        ));
    }
    F32_OR_SI32.require(&out_spec, "triton broadcast lowering")
}

fn validate_slice(cx: &ValidateCtx<'_>) -> ConversionResult<()> {
    let input_spec = cx.operand_tensor_spec(0, "slice input")?;
    let out_spec = cx.output_tensor_spec()?;
    if input_spec.dtype != out_spec.dtype {
        return Err(ConversionError::new(
            "triton slice lowering requires matching input/output dtype",
        ));
    }
    F32_OR_SI32.require(&out_spec, "triton slice lowering")
}

fn validate_transpose(cx: &ValidateCtx<'_>) -> ConversionResult<()> {
    let input_spec = cx.operand_tensor_spec(0, "transpose input")?;
    let out_spec = cx.output_tensor_spec()?;
    F32_ONLY.require(&input_spec, "triton transpose lowering")?;
    F32_ONLY.require(&out_spec, "triton transpose lowering")
}

fn validate_concat(cx: &ValidateCtx<'_>) -> ConversionResult<()> {
    let out_spec = cx.output_tensor_spec()?;
    F32_ONLY.require(&out_spec, "triton concat lowering")?;
    for input_spec in cx.each_operand_tensor_spec("concat input")? {
        F32_ONLY.require(&input_spec, "triton concat lowering")?;
    }
    Ok(())
}

fn validate_dynamic_slice(cx: &ValidateCtx<'_>) -> ConversionResult<()> {
    let value_spec = cx.operand_tensor_spec(0, "dynamic_slice value")?;
    let starts_spec = cx.operand_tensor_spec(1, "dynamic_slice starts")?;
    let out_spec = cx.output_tensor_spec()?;
    if starts_spec.dtype != DType::Si32 || value_spec.dtype != out_spec.dtype {
        return Err(ConversionError::new(
            "triton dynamic_slice lowering requires Si32 starts and matching value/output dtype",
        ));
    }
    F32_OR_SI32.require(&out_spec, "triton dynamic_slice lowering")
}

fn validate_dynamic_update_slice(cx: &ValidateCtx<'_>) -> ConversionResult<()> {
    let base_spec = cx.operand_tensor_spec(0, "dynamic_update_slice base")?;
    let update_spec = cx.operand_tensor_spec(1, "dynamic_update_slice update")?;
    let starts_spec = cx.operand_tensor_spec(2, "dynamic_update_slice starts")?;
    let out_spec = cx.output_tensor_spec()?;
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
