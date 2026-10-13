//! The state and the precision of the random ops.

use anyhow::{bail, Result};

use crate::ir::{DType, TensorType};

/// The significant bits of a float dtype that has uniform floats.
pub(crate) fn uniform_bits(dtype: DType) -> Result<u32> {
    match dtype {
        DType::F64 | DType::F32 | DType::F16 | DType::Bf16 => {
            Ok(dtype.significand_bits().expect("a float dtype"))
        }
        other => bail!("{other} has no uniform floats"),
    }
}

/// The type of a generator state.
pub fn state_type() -> TensorType {
    TensorType::new(DType::Ui64, [2])
}
