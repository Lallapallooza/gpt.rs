//! The type rules of the quantization ops: the types each op gives, and what it rejects.

use anyhow::{bail, ensure, Result};

use crate::dialects::tensor::ir::*;
use crate::dialects::tensor::utils::class::{ensure_class, ensure_dtype_class, Class};
use crate::ir::{DType, Dim, NoRegions, OpDef, OwnedTypes, Ranked, TensorType, TypeCx, Types};

impl OpDef for Dequantize {
    fn result_types(
        &self,
        ins: DequantizeOperands<Types<'_>>,
        _cx: &TypeCx<'_, NoRegions>,
    ) -> Result<DequantizeResults<OwnedTypes>> {
        ensure_class("q", ins.q, CODES)?;
        ensure!(is_real(self.to), "cannot dequantize to {}", self.to);
        ensure_scales(&ins.q.dims, &self.block, ins.scale, ins.offset)?;
        Ok(DequantizeResults {
            output: ins.q.with_dtype(self.to),
        })
    }
}

impl OpDef for Quantize {
    fn result_types(
        &self,
        ins: QuantizeOperands<Types<'_>>,
        _cx: &TypeCx<'_, NoRegions>,
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
        ins: QuantizeStochasticOperands<Types<'_>>,
        _cx: &TypeCx<'_, NoRegions>,
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
        ins: QuantizeTableOperands<Types<'_>>,
        _cx: &TypeCx<'_, NoRegions>,
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
        ins: BlockScaleRangeOperands<Types<'_>>,
        _cx: &TypeCx<'_, NoRegions>,
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
    ins: BlockScaleOperands<Types<'_>>,
    _cx: &TypeCx<'_, NoRegions>,
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

/// The scale extents of a tensor of `dims` with `block`, `ceil(extent / block)` on each axis.
fn scale_dims(dims: &[Dim], block: &[usize]) -> Result<Ranked<Dim>> {
    ensure!(
        block.len() == dims.len(),
        "block {block:?} must have one extent per axis of rank {}",
        dims.len()
    );
    dims.iter()
        .zip(block)
        .map(|(dim, &extent)| {
            ensure!(extent > 0, "block {block:?} must be positive");
            Ok(match dim {
                Dim::Static(size) => Dim::Static(size.div_ceil(extent)),
                Dim::Dynamic(_) if extent == 1 => dim.clone(),
                Dim::Dynamic(_) => bail!("a block of {extent} needs a static extent, got {dim}"),
            })
        })
        .collect()
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
    if let Some(offset) = offset {
        ensure_class("the offset", offset, &[Class::Float])?;
        ensure!(
            offset.dims == scale.dims,
            "the offset must have the scale's extents, got {offset} and {scale}"
        );
    }
    Ok(())
}
