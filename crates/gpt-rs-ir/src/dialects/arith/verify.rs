//! The type rules of the `arith` ops: the types each op gives, and what it rejects.

use anyhow::{ensure, Result};

use crate::dialects::arith::ir::*;
use crate::dialects::arith::{ensure_bitcast, ensure_fma};
use crate::dialects::tensor::utils::class::{
    ensure_dtype_class, ensure_equal_types, ARITHMETIC, CASTABLE, INTEGERS, ORDERED,
};
use crate::ir::{DType, NoRegions, OpDef, OwnedTypes, TypeCx, Types};

impl OpDef for Constant {
    fn result_types(
        &self,
        _ins: ConstantOperands<Types>,
        _cx: &TypeCx<NoRegions>,
    ) -> Result<ConstantResults<OwnedTypes>> {
        Ok(ConstantResults {
            output: self.value.dtype(),
        })
    }
}

impl OpDef for Select {
    fn result_types(
        &self,
        ins: SelectOperands<Types>,
        _cx: &TypeCx<NoRegions>,
    ) -> Result<SelectResults<OwnedTypes>> {
        ensure!(ins.pred == DType::I1, "pred must be i1, got {}", ins.pred);
        ensure_equal_types("on_true and on_false", &ins.on_true, &ins.on_false)?;
        Ok(SelectResults {
            output: ins.on_true,
        })
    }
}

impl OpDef for Fma {
    fn result_types(
        &self,
        ins: FmaOperands<Types>,
        _cx: &TypeCx<NoRegions>,
    ) -> Result<FmaResults<OwnedTypes>> {
        ensure_equal_types("lhs and rhs", &ins.lhs, &ins.rhs)?;
        ensure_equal_types("lhs and acc", &ins.lhs, &ins.acc)?;
        ensure_fma(ins.lhs)?;
        Ok(FmaResults { output: ins.lhs })
    }
}

impl OpDef for Cast {
    fn result_types(
        &self,
        _ins: CastOperands<Types>,
        _cx: &TypeCx<NoRegions>,
    ) -> Result<CastResults<OwnedTypes>> {
        ensure_dtype_class("to", self.to, CASTABLE)?;
        Ok(CastResults { output: self.to })
    }
}

impl OpDef for FromIndex {
    fn result_types(
        &self,
        _ins: FromIndexOperands<Types>,
        _cx: &TypeCx<NoRegions>,
    ) -> Result<FromIndexResults<OwnedTypes>> {
        ensure_dtype_class("dtype", self.dtype, ARITHMETIC)?;
        Ok(FromIndexResults { output: self.dtype })
    }
}

impl OpDef for ToIndex {
    fn result_types(
        &self,
        ins: ToIndexOperands<Types>,
        _cx: &TypeCx<NoRegions>,
    ) -> Result<ToIndexResults<OwnedTypes>> {
        ensure_dtype_class("the input", ins.input, INTEGERS)?;
        Ok(ToIndexResults { output: () })
    }
}

impl OpDef for Bitcast {
    fn result_types(
        &self,
        ins: BitcastOperands<Types>,
        _cx: &TypeCx<NoRegions>,
    ) -> Result<BitcastResults<OwnedTypes>> {
        ensure_bitcast(ins.input, self.to)?;
        Ok(BitcastResults { output: self.to })
    }
}

/// The type rule of every [`Unary`] op.
pub(crate) fn unary<T: Unary>(
    _op: &T,
    ins: UnaryOperands<Types>,
    _cx: &TypeCx<NoRegions>,
) -> Result<UnaryResults<OwnedTypes>> {
    ensure_dtype_class("the input", ins.input, T::CLASSES)?;
    Ok(UnaryResults { output: ins.input })
}

/// The type rule of every [`Binary`] op.
pub(crate) fn binary<T: Binary>(
    _op: &T,
    ins: BinaryOperands<Types>,
    _cx: &TypeCx<NoRegions>,
) -> Result<BinaryResults<OwnedTypes>> {
    ensure_equal_types("operands", &ins.lhs, &ins.rhs)?;
    ensure_dtype_class("each operand", ins.lhs, T::CLASSES)?;
    Ok(BinaryResults { output: ins.lhs })
}

/// The type rule of every [`Rng`] op.
pub(crate) fn rng<T: Rng>(
    op: &T,
    ins: RngOperands<Types>,
    _cx: &TypeCx<NoRegions>,
) -> Result<RngResults<OwnedTypes>> {
    ensure!(
        ins.key == DType::Ui64 && ins.counter == DType::Ui64,
        "the key and the counter are ui64, got {} and {}",
        ins.key,
        ins.counter
    );
    ensure_dtype_class("the index", ins.index, INTEGERS)?;
    ensure_dtype_class("dtype", op.dtype, T::CLASSES)?;
    Ok(RngResults { output: op.dtype })
}

/// The type rule of every [`Compare`] op.
pub(crate) fn compare<T: Compare>(
    _op: &T,
    ins: CompareOperands<Types>,
    _cx: &TypeCx<NoRegions>,
) -> Result<CompareResults<OwnedTypes>> {
    ensure_equal_types("operands", &ins.lhs, &ins.rhs)?;
    ensure_dtype_class("each operand", ins.lhs, ORDERED)?;
    Ok(CompareResults { output: DType::I1 })
}
