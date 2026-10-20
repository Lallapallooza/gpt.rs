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

/// The words of one block of the Philox stream: Philox4x32 computes four 32-bit words at a time.
pub(crate) const BLOCK_WORDS: usize = 4;

/// The rounds of Philox4x32-10.
pub(crate) const PHILOX_ROUNDS: u32 = 10;

/// The multipliers of Philox4x32: of words 0 and 2 of the block.
pub(crate) const PHILOX_MULTIPLIERS: [u64; 2] = [0xD251_1F53, 0xCD9E_8D57];

/// The steps that each round adds to the two words of the key.
pub(crate) const PHILOX_KEY_STEPS: [u32; 2] = [0x9E37_79B9, 0xBB67_AE85];

/// The type of a generator state.
pub fn state_type() -> TensorType {
    TensorType::new(DType::Ui64, [2])
}
