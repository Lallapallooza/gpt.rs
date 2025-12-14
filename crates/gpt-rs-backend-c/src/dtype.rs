use gpt_rs::backend::conversion::{ConversionError, ConversionResult};
use gpt_rs::backend::spec::DType;

pub(crate) fn dtype_tag(dtype: DType) -> Option<u32> {
    match dtype {
        DType::F32 => Some(0),
        DType::F16 => Some(1),
        DType::Bf16 => Some(2),
        DType::Si32 => Some(3),
        DType::I1 => Some(4),
        _ => None,
    }
}

pub(crate) fn dtype_tag_value(dtype: DType) -> ConversionResult<u32> {
    dtype_tag(dtype).ok_or_else(|| ConversionError::new("dtype not supported by C backend"))
}
