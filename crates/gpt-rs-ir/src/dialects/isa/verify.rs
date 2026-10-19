//! The type rules of the `isa` ops: the types each op gives, and what it rejects.

use anyhow::{ensure, Result};

use crate::dialects::arith;
use crate::dialects::isa::ir::*;
use crate::dialects::tensor::utils::class::{
    ensure_dtype_class, ensure_equal_types, CASTABLE, ORDERED,
};
use crate::ir::{
    BufferType, DType, NoRegions, OpDef, OwnedTypes, Ranked, TypeCx, Types, VectorType, F64,
};

/// The type rule of every [`Unary`] op.
pub(crate) fn unary<T: Unary>(
    _op: &T,
    ins: UnaryOperands<Types>,
    _cx: &TypeCx<NoRegions>,
) -> Result<UnaryResults<OwnedTypes>> {
    let classes = <T::Scalar as arith::Unary>::CLASSES;
    ensure_dtype_class("the input", ins.input.dtype, classes)?;
    Ok(UnaryResults {
        output: ins.input.clone(),
    })
}

/// The type rule of every [`Binary`] op.
pub(crate) fn binary<T: Binary>(
    _op: &T,
    ins: BinaryOperands<Types>,
    _cx: &TypeCx<NoRegions>,
) -> Result<BinaryResults<OwnedTypes>> {
    ensure_equal_types("operands", ins.lhs, ins.rhs)?;
    let classes = <T::Scalar as arith::Binary>::CLASSES;
    ensure_dtype_class("each operand", ins.lhs.dtype, classes)?;
    Ok(BinaryResults {
        output: ins.lhs.clone(),
    })
}

/// The type rule of every [`Compare`] op.
pub(crate) fn compare<T: Compare>(
    _op: &T,
    ins: CompareOperands<Types>,
    _cx: &TypeCx<NoRegions>,
) -> Result<CompareResults<OwnedTypes>> {
    ensure_equal_types("operands", ins.lhs, ins.rhs)?;
    ensure_dtype_class("each operand", ins.lhs.dtype, ORDERED)?;
    Ok(CompareResults {
        output: ins.lhs.with_dtype(DType::I1),
    })
}

impl OpDef for Select {
    fn result_types(
        &self,
        ins: SelectOperands<Types>,
        _cx: &TypeCx<NoRegions>,
    ) -> Result<SelectResults<OwnedTypes>> {
        ensure_equal_types("on_true and on_false", ins.on_true, ins.on_false)?;
        ensure_mask("pred", ins.pred, &ins.on_true.dims)?;
        Ok(SelectResults {
            output: ins.on_true.clone(),
        })
    }
}

impl OpDef for Cast {
    fn result_types(
        &self,
        ins: CastOperands<Types>,
        _cx: &TypeCx<NoRegions>,
    ) -> Result<CastResults<OwnedTypes>> {
        ensure_dtype_class("to", self.to, CASTABLE)?;
        Ok(CastResults {
            output: ins.input.with_dtype(self.to),
        })
    }
}

impl OpDef for Fma {
    fn result_types(
        &self,
        ins: FmaOperands<Types>,
        _cx: &TypeCx<NoRegions>,
    ) -> Result<FmaResults<OwnedTypes>> {
        ensure_equal_types("lhs and rhs", ins.lhs, ins.rhs)?;
        ensure_equal_types("lhs and acc", ins.lhs, ins.acc)?;
        arith::ensure_fma(ins.lhs.dtype)?;
        Ok(FmaResults {
            output: ins.lhs.clone(),
        })
    }
}

impl OpDef for Bitcast {
    fn result_types(
        &self,
        ins: BitcastOperands<Types>,
        _cx: &TypeCx<NoRegions>,
    ) -> Result<BitcastResults<OwnedTypes>> {
        arith::ensure_bitcast(ins.input.dtype, self.to)?;
        Ok(BitcastResults {
            output: ins.input.with_dtype(self.to),
        })
    }
}

impl OpDef for Splat {
    fn result_types(
        &self,
        ins: SplatOperands<Types>,
        _cx: &TypeCx<NoRegions>,
    ) -> Result<SplatResults<OwnedTypes>> {
        Ok(SplatResults {
            output: VectorType::try_new(ins.input, &self.dims)?,
        })
    }
}

impl OpDef for Expand {
    fn result_types(
        &self,
        ins: ExpandOperands<Types>,
        _cx: &TypeCx<NoRegions>,
    ) -> Result<ExpandResults<OwnedTypes>> {
        let output = VectorType::try_new(ins.input.dtype, &self.dims)?;
        ensure!(
            ins.input.dims.len() == output.dims.len() && ins.input.lanes() == output.lanes(),
            "{} has other lanes or another number of axes than {output}",
            ins.input
        );
        for (&from, &to) in ins.input.group().iter().zip(output.group()) {
            ensure!(
                from == to || from == 1,
                "{} repeats only axes of extent 1 to {output}",
                ins.input
            );
        }
        Ok(ExpandResults { output })
    }
}

impl OpDef for TileMask {
    fn result_types(
        &self,
        ins: TileMaskOperands<Types>,
        _cx: &TypeCx<NoRegions>,
    ) -> Result<TileMaskResults<OwnedTypes>> {
        let output = VectorType::try_new(DType::I1, &self.dims)?;
        let axes = output.group().len().max(1);
        ensure!(
            ins.extents.len() == axes,
            "the extent count is {}, but the tile has rank {axes}",
            ins.extents.len()
        );
        Ok(TileMaskResults { output })
    }
}

/// The type rule of every [`Read`] op.
pub(crate) fn read<T: Read>(
    op: &T,
    ins: ReadOperands<Types>,
    _cx: &TypeCx<NoRegions>,
) -> Result<ReadResults<OwnedTypes>> {
    Ok(ReadResults {
        output: read_registers(op, ins.buffer, ins.base.len())?,
    })
}

/// The type rule of every [`MaskedRead`] op.
pub(crate) fn masked_read<T: MaskedRead>(
    op: &T,
    ins: MaskedReadOperands<Types>,
    _cx: &TypeCx<NoRegions>,
) -> Result<MaskedReadResults<OwnedTypes>> {
    let output = read_registers(op, ins.buffer, ins.base.len())?;
    ensure_mask("the mask", ins.mask, &output.dims)?;
    Ok(MaskedReadResults { output })
}

/// The registers that a read of `buffer` at `base` indices gives with the attributes `read`.
fn read_registers(read: &ReadAttrs, buffer: &BufferType, base: usize) -> Result<VectorType> {
    let output = VectorType::try_new(buffer.dtype, &read.dims)?;
    ensure_access(buffer, base, &read.steps, &output)?;
    ensure_pad(buffer, read.pad, &read.checked, &output)?;
    Ok(output)
}

impl OpDef for Prefetch {
    fn result_types(
        &self,
        ins: PrefetchOperands<Types>,
        _cx: &TypeCx<NoRegions>,
    ) -> Result<PrefetchResults<OwnedTypes>> {
        let registers = VectorType::try_new(ins.buffer.dtype, &self.dims)?;
        ensure_access(ins.buffer, ins.base.len(), &self.steps, &registers)?;
        Ok(PrefetchResults::default())
    }
}

impl OpDef for Store {
    fn result_types(
        &self,
        ins: StoreOperands<Types>,
        _cx: &TypeCx<NoRegions>,
    ) -> Result<StoreResults<OwnedTypes>> {
        ensure_stored(ins.value, ins.buffer)?;
        ensure_access(ins.buffer, ins.base.len(), &self.steps, ins.value)?;
        Ok(StoreResults::default())
    }
}

impl OpDef for MaskedStore {
    fn result_types(
        &self,
        ins: MaskedStoreOperands<Types>,
        _cx: &TypeCx<NoRegions>,
    ) -> Result<MaskedStoreResults<OwnedTypes>> {
        ensure_stored(ins.value, ins.buffer)?;
        ensure_mask("the mask", ins.mask, &ins.value.dims)?;
        ensure_access(ins.buffer, ins.base.len(), &self.steps, ins.value)?;
        Ok(MaskedStoreResults::default())
    }
}

/// Checks that `buffer` holds the dtype of `value`.
fn ensure_stored(value: &VectorType, buffer: &BufferType) -> Result<()> {
    ensure!(
        value.dtype == buffer.dtype,
        "the value is {value}, but {buffer} holds {}",
        buffer.dtype
    );
    Ok(())
}

/// Checks that `mask` is a mask of the extents `dims`.
fn ensure_mask(what: &str, mask: &VectorType, dims: &[usize]) -> Result<()> {
    ensure!(
        mask.dtype == DType::I1 && mask.dims[..] == *dims,
        "{what} is {mask}, not a mask of the extents {dims:?}"
    );
    Ok(())
}

/// Checks that a read of `buffer` pads, if it does, with a value of its dtype.
fn ensure_pad(
    buffer: &BufferType,
    pad: Option<F64>,
    checked: &[Ranked<usize>],
    registers: &VectorType,
) -> Result<()> {
    if let Some(pad_value) = pad {
        ensure!(
            buffer.dtype.represents(pad_value.0),
            "it pads with {}, which {} cannot hold",
            pad_value.0,
            buffer.dtype
        );
    }
    if checked.is_empty() {
        return Ok(());
    }
    ensure!(pad.is_some(), "it names checked axes but has no pad");
    ensure!(
        checked.len() == registers.registers(),
        "`checked` has one entry per register, and holds {} for the {} registers of {registers}",
        checked.len(),
        registers.registers()
    );
    for axes in checked {
        for &axis in axes.iter() {
            ensure!(
                axis < buffer.dims.len(),
                "it checks axis {axis} of {buffer}, which has no axis {axis}"
            );
        }
    }
    Ok(())
}

/// Checks a memory op's access of `buffer` by registers of the type `registers`: one base index
/// per axis of the buffer, and one step per group axis with one entry per axis of the buffer.
fn ensure_access(
    buffer: &BufferType,
    base: usize,
    steps: &[Ranked<i64>],
    registers: &VectorType,
) -> Result<()> {
    ensure!(
        base == buffer.rank(),
        "the base index count is {base}, but {buffer} has rank {}",
        buffer.rank()
    );
    let group = registers.group();
    ensure!(
        steps.len() == group.len(),
        "the step count is {}, but the group rank of {registers} is {}",
        steps.len(),
        group.len()
    );
    for step in steps {
        ensure!(
            step.len() == buffer.rank(),
            "a step's length is {}, but {buffer} has rank {}",
            step.len(),
            buffer.rank()
        );
    }
    Ok(())
}
