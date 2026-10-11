//! The type rules of the elementwise ops: the types each op gives, and what it rejects.

use anyhow::{ensure, Result};

use crate::dialects::arith;
use crate::dialects::tensor::ir::*;
use crate::dialects::tensor::utils::class::{
    ensure_class, ensure_dtype_class, ensure_equal_types, CASTABLE, ORDERED,
};
use crate::op::{NoRegions, OpDef, OwnedTypes, TypeCx, Types};
use crate::types::DType;

impl OpDef for Select {
    fn result_types(
        &self,
        ins: SelectOperands<Types<'_>>,
        _cx: &TypeCx<'_, NoRegions>,
    ) -> Result<SelectResults<OwnedTypes>> {
        ensure!(
            ins.pred.dtype == DType::I1,
            "pred must be an i1 tensor, got {}",
            ins.pred
        );
        ensure_equal_types("on_true and on_false", &ins.on_true, &ins.on_false)?;
        ensure!(
            ins.pred.dims == ins.on_true.dims,
            "pred and the branches must have equal shapes, got {} and {}",
            ins.pred,
            ins.on_true
        );
        Ok(SelectResults {
            output: ins.on_true.clone(),
        })
    }
}

impl OpDef for Cast {
    fn result_types(
        &self,
        ins: CastOperands<Types<'_>>,
        _cx: &TypeCx<'_, NoRegions>,
    ) -> Result<CastResults<OwnedTypes>> {
        // e8m0 scales come from the `block_scale_*` ops, which state how they round them.
        ensure_dtype_class("to", self.to, CASTABLE)?;
        Ok(CastResults {
            output: ins.input.with_dtype(self.to),
        })
    }
}

impl OpDef for StopGradient {
    fn result_types(
        &self,
        ins: StopGradientOperands<Types<'_>>,
        _cx: &TypeCx<'_, NoRegions>,
    ) -> Result<StopGradientResults<OwnedTypes>> {
        Ok(StopGradientResults {
            output: ins.input.clone(),
        })
    }
}

/// The type rule of every [`Unary`] op.
pub(crate) fn unary<T: Unary>(
    _op: &T,
    ins: UnaryOperands<Types<'_>>,
    _cx: &TypeCx<'_, NoRegions>,
) -> Result<UnaryResults<OwnedTypes>> {
    ensure_class("the input", ins.input, <T::Scalar as arith::Unary>::CLASSES)?;
    Ok(UnaryResults {
        output: ins.input.clone(),
    })
}

/// The type rule of every [`Binary`] op.
pub(crate) fn binary<T: Binary>(
    _op: &T,
    ins: BinaryOperands<Types<'_>>,
    _cx: &TypeCx<'_, NoRegions>,
) -> Result<BinaryResults<OwnedTypes>> {
    ensure_equal_types("operands", &ins.lhs, &ins.rhs)?;
    ensure_class(
        "each operand",
        ins.lhs,
        <T::Scalar as arith::Binary>::CLASSES,
    )?;
    Ok(BinaryResults {
        output: ins.lhs.clone(),
    })
}

/// The type rule of every [`Compare`] op.
pub(crate) fn compare<T: Compare>(
    _op: &T,
    ins: CompareOperands<Types<'_>>,
    _cx: &TypeCx<'_, NoRegions>,
) -> Result<CompareResults<OwnedTypes>> {
    ensure_equal_types("operands", &ins.lhs, &ins.rhs)?;
    ensure_class("each operand", ins.lhs, ORDERED)?;
    Ok(CompareResults {
        output: ins.lhs.with_dtype(DType::I1),
    })
}
