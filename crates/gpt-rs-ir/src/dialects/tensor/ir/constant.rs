//! `tensor.constant`.

use anyhow::{ensure, Result};

use crate::dense::Dense;
use crate::op::{NoRegions, OpDef, OwnedTypes, TypeCx, Types};
use crate::Op;

/// `tensor.constant`: a tensor whose elements are an attribute.
///
/// - The result has the dtype and the static shape of its `value` attribute.
/// - The literal payload is interpreted as row-major, little-endian scalar storage, one byte per
///   `i1` element. Backends that do not support a dtype MAY reject the program as unimplemented.
/// - In the text form, elements narrower than a byte are written in hex, packed as tensors store
///   them.
#[derive(crate::Op, Debug, Clone, PartialEq, Eq, Hash)]
#[op("tensor.constant", results(output: Tensor))]
pub struct Constant {
    pub value: Dense,
}

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

impl Op {
    /// The dense value when this is a `tensor.constant`.
    pub fn as_constant(&self) -> Option<&Dense> {
        <Constant as crate::op::InOp>::cast(self).map(|c| &c.value)
    }
}
