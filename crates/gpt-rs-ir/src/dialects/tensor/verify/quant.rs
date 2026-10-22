//! The type rules of the quantization ops: the types each op gives, and what it rejects.

use anyhow::{ensure, Result};

use crate::dialects::tensor::ir::*;
use crate::dialects::tensor::utils::class::{ensure_class, ensure_dtype_class, Class};
use crate::dialects::tensor::utils::quant::scale_dims;
use crate::ir::{DType, Dim, NoRegions, OpDef, OwnedTypes, TensorType, TypeCx, Types};

impl OpDef for Dequantize {
    fn result_types(
        &self,
        ins: DequantizeOperands<Types>,
        _cx: &TypeCx<NoRegions>,
    ) -> Result<DequantizeResults<OwnedTypes>> {
        ensure_class("quantized", ins.quantized, CODES)?;
        ensure!(is_real(self.to), "cannot dequantize to {}", self.to);
        ensure_scales(&ins.quantized.dims, &self.block, ins.scale, ins.offset)?;
        Ok(DequantizeResults {
            output: ins.quantized.with_dtype(self.to),
        })
    }
}

impl OpDef for Quantize {
    fn result_types(
        &self,
        ins: QuantizeOperands<Types>,
        _cx: &TypeCx<NoRegions>,
    ) -> Result<QuantizeResults<OwnedTypes>> {
        ensure_quantized(ins.x, self.to)?;
        ensure_scales(&ins.x.dims, &self.block, ins.scale, ins.offset)?;
        Ok(QuantizeResults {
            output: ins.x.with_dtype(self.to),
        })
    }
}

impl OpDef for QuantizeStochastic {
    fn result_types(
        &self,
        ins: QuantizeStochasticOperands<Types>,
        _cx: &TypeCx<NoRegions>,
    ) -> Result<QuantizeStochasticResults<OwnedTypes>> {
        ensure_quantized(ins.x, self.to)?;
        ensure_scales(&ins.x.dims, &self.block, ins.scale, ins.offset)?;
        let draws = ins.x.with_dtype(DType::Ui32);
        ensure!(
            *ins.random == draws,
            "the random bits must be {draws}, got {}",
            ins.random
        );
        Ok(QuantizeStochasticResults {
            output: ins.x.with_dtype(self.to),
        })
    }
}

impl OpDef for QuantizeTable {
    fn result_types(
        &self,
        ins: QuantizeTableOperands<Types>,
        _cx: &TypeCx<NoRegions>,
    ) -> Result<QuantizeTableResults<OwnedTypes>> {
        ensure!(is_real(ins.x.dtype), "cannot quantize {}", ins.x);
        ensure_dtype_class("to", self.to, &[Class::Int])?;
        ensure_class("the table", ins.table, &[Class::Float, Class::Narrow])?;
        ensure!(
            ins.table.rank() == 1,
            "the table must have rank 1, got {}",
            ins.table
        );
        ensure_scales(&ins.x.dims, &self.block, ins.scale, ins.offset)?;
        Ok(QuantizeTableResults {
            output: ins.x.with_dtype(self.to),
        })
    }
}

impl OpDef for BlockScaleRange {
    fn result_types(
        &self,
        ins: BlockScaleRangeOperands<Types>,
        _cx: &TypeCx<NoRegions>,
    ) -> Result<BlockScaleRangeResults<OwnedTypes>> {
        ensure_quantized(ins.x, self.to)?;
        ensure_dtype_class("scale_dtype", self.scale_dtype, &[Class::Float])?;
        let scale = TensorType {
            dtype: self.scale_dtype,
            dims: scale_dims(&ins.x.dims, &self.block)?,
        };
        Ok(BlockScaleRangeResults {
            offset: scale.clone(),
            scale,
        })
    }
}

/// The type rule of every [`BlockScale`] op.
pub(crate) fn block_scale<T: BlockScale>(
    op: &T,
    ins: BlockScaleOperands<Types>,
    _cx: &TypeCx<NoRegions>,
) -> Result<BlockScaleResults<OwnedTypes>> {
    ensure_quantized(ins.x, op.to)?;
    ensure_dtype_class("scale_dtype", op.scale_dtype, SCALES)?;
    ensure!(
        op.scale_dtype != DType::E8M0 || T::POWER_OF_TWO,
        "e8m0 scales are powers of two, which this op does not give"
    );
    Ok(BlockScaleResults {
        scale: TensorType {
            dtype: op.scale_dtype,
            dims: scale_dims(&ins.x.dims, &op.block)?,
        },
    })
}

/// The classes of quantized elements: integers and narrow floats.
const ELEMENTS: &[Class] = &[Class::Int, Class::Narrow];

/// The classes of what `tensor.dequantize` reads: quantized elements, or the values of a code
/// table.
const CODES: &[Class] = &[Class::Int, Class::Narrow, Class::Float];

/// The classes of scales.
const SCALES: &[Class] = &[Class::Float, Class::Narrow, Class::Scale];

/// Checks that `x` is real-valued and `to` a quantized dtype.
fn ensure_quantized(x: &TensorType, to: DType) -> Result<()> {
    ensure!(is_real(x.dtype), "cannot quantize {x}");
    ensure_dtype_class("to", to, ELEMENTS)
}

/// Whether `dtype` is a real-valued input or output of quantization.
fn is_real(dtype: DType) -> bool {
    matches!(dtype, DType::F32 | DType::Bf16 | DType::F16)
}

/// Checks that `scale`, and `offset` when there is one, fit a tensor of `dims` with `block`.
fn ensure_scales(
    dims: &[Dim],
    block: &[usize],
    scale: &TensorType,
    offset: Option<&TensorType>,
) -> Result<()> {
    ensure_class("the scale", scale, SCALES)?;
    let expected = scale_dims(dims, block)?;
    let extents: Vec<String> = expected.iter().map(ToString::to_string).collect();
    ensure!(
        scale.dims == expected,
        "the scale must have the extents [{}] for block {block:?}, got {scale}",
        extents.join(", ")
    );
    if let Some(offset_type) = offset {
        ensure_class("the offset", offset_type, &[Class::Float])?;
        ensure!(
            offset_type.dims == scale.dims,
            "the offset must have the scale's extents, got {offset_type} and {scale}"
        );
    }
    Ok(())
}
