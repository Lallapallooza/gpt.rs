//! The type rules of the random ops: the types each op gives, and what it rejects.

use anyhow::{ensure, Result};

use crate::dialects::arith;
use crate::dialects::tensor::ir::*;
use crate::dialects::tensor::utils::class::ensure_dtype_class;
use crate::dialects::tensor::utils::random::state_type;
use crate::op::{NoRegions, OwnedTypes, TypeCx, Types};
use crate::types::TensorType;

/// The type rule of every [`Rng`] op.
pub(crate) fn rng<T: Rng>(
    op: &T,
    ins: RngOperands<Types<'_>>,
    _cx: &TypeCx<'_, NoRegions>,
) -> Result<RngResults<OwnedTypes>> {
    let state = state_type();
    ensure!(
        *ins.state == state,
        "the state must be {state}, got {}",
        ins.state
    );
    ensure_dtype_class("dtype", op.dtype, <T::Scalar as arith::Rng>::CLASSES)?;
    Ok(RngResults {
        new_state: state,
        output: TensorType {
            dtype: op.dtype,
            dims: op.shape.clone(),
        },
    })
}
