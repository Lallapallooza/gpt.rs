use gpt_rs::backend::conversion::{ConversionError, ConversionResult};
use gpt_rs::backend::spec::DType;

use super::ValidateCtx;

pub(super) fn validate(cx: &ValidateCtx<'_>) -> ConversionResult<()> {
    let lhs_spec = cx.operand_tensor_spec(0, "dot lhs")?;
    let rhs_spec = cx.operand_tensor_spec(1, "dot rhs")?;
    let out_spec = cx.output_tensor_spec()?;
    if lhs_spec.dtype != DType::F32 || rhs_spec.dtype != DType::F32 || out_spec.dtype != DType::F32
    {
        return Err(ConversionError::new(
            "triton dot lowering currently supports F32 only",
        ));
    }
    Ok(())
}
