//! The type rules of `tensor.constant`: the types each op gives, and what it rejects.

use anyhow::{ensure, Result};

use crate::dialects::tensor::ir::*;
use crate::op::{NoRegions, OpDef, OwnedTypes, TypeCx, Types};

impl OpDef for Constant {
    fn result_types(
        &self,
        _ins: ConstantOperands<Types<'_>>,
        _cx: &TypeCx<'_, NoRegions>,
    ) -> Result<ConstantResults<OwnedTypes>> {
        let value = &self.value;
        let (count, len) = (value.element_count(), value.bytes.len());
        ensure!(
            value
                .dtype
                .size_in_bytes()
                .is_none_or(|size| size.checked_mul(count) == Some(len)),
            "{len} bytes do not hold {count} {} elements",
            value.dtype
        );
        Ok(ConstantResults {
            output: value.tensor_type(),
        })
    }
}
