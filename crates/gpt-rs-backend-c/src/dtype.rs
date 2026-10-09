use anyhow::{anyhow, Result};
use gpt_rs_ir::types::DType;

/// The tag of `dtype` in a `PtirTensor`.
pub(crate) fn dtype_tag(dtype: DType) -> Result<u32> {
    Ok(match dtype {
        DType::F32 => 0,
        DType::F16 => 1,
        DType::Bf16 => 2,
        DType::Si32 => 3,
        DType::I1 => 4,
        DType::F8E4M3 => 5,
        DType::Ui64 => 6,
        _ => return Err(anyhow!("the C backend has no {dtype} tensors")),
    })
}
