use gpt_rs::backend::conversion::{ConversionError, ConversionResult};
use gpt_rs::backend::spec::{DType, Operation, ReduceKind};

use super::ValidateCtx;

pub(super) fn validate(cx: &ValidateCtx<'_>) -> ConversionResult<()> {
    let Operation::Reduce(spec) = &cx.instruction.op else {
        return Err(ConversionError::new(
            "internal triton validator error: reduction validator on non-reduce op",
        ));
    };
    let input_spec = cx.operand_tensor_spec(0, "reduce input")?;
    let out_spec = cx.output_tensor_spec()?;
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
