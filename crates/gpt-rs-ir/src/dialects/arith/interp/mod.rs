//! Semantics of the `arith` ops. The ops of a family compute with their function, which this file
//! implements: [`UnaryMath`], [`BinaryMath`] and [`CompareMath`]. The `tensor` ops apply the same
//! functions to each element.

pub mod math;

use anyhow::Result;

use self::math::{draw_one, BinaryMath, CompareMath, Float, IntBinary, UnaryMath};
use super::ir::*;
use crate::ir::{Eval, EvalCx, Host, IntFormat, NoRegions, OwnedHost};

impl UnaryMath for Neg {
    fn float<F: Float>() -> Option<fn(F) -> F> {
        Some(|x| -x)
    }
    fn int() -> Option<fn(i128) -> i128> {
        Some(i128::wrapping_neg)
    }
}

impl UnaryMath for Abs {
    fn float<F: Float>() -> Option<fn(F) -> F> {
        Some(F::abs)
    }
    fn int() -> Option<fn(i128) -> i128> {
        Some(i128::wrapping_abs)
    }
}

impl UnaryMath for Exp {
    fn float<F: Float>() -> Option<fn(F) -> F> {
        Some(F::exp)
    }
}

impl UnaryMath for Log {
    fn float<F: Float>() -> Option<fn(F) -> F> {
        Some(F::ln)
    }
}

impl UnaryMath for Tanh {
    fn float<F: Float>() -> Option<fn(F) -> F> {
        Some(F::tanh)
    }
}

impl UnaryMath for Erf {
    fn float<F: Float>() -> Option<fn(F) -> F> {
        Some(F::erf)
    }
}

impl UnaryMath for Rsqrt {
    fn float<F: Float>() -> Option<fn(F) -> F> {
        Some(|x| F::ONE / x.sqrt())
    }
}

impl UnaryMath for Reciprocal {
    fn float<F: Float>() -> Option<fn(F) -> F> {
        Some(|x| F::ONE / x)
    }
}

impl UnaryMath for Sin {
    fn float<F: Float>() -> Option<fn(F) -> F> {
        Some(F::sin)
    }
}

impl UnaryMath for Cos {
    fn float<F: Float>() -> Option<fn(F) -> F> {
        Some(F::cos)
    }
}

impl UnaryMath for Not {
    fn int() -> Option<fn(i128) -> i128> {
        Some(|x| !x)
    }
}

impl BinaryMath for Add {
    fn float<F: Float>() -> Option<fn(F, F) -> F> {
        Some(|lhs, rhs| lhs + rhs)
    }
    fn int() -> Option<IntBinary> {
        Some(|lhs, rhs, _| Some(lhs.wrapping_add(rhs)))
    }
}

impl BinaryMath for Sub {
    fn float<F: Float>() -> Option<fn(F, F) -> F> {
        Some(|lhs, rhs| lhs - rhs)
    }
    fn int() -> Option<IntBinary> {
        Some(|lhs, rhs, _| Some(lhs.wrapping_sub(rhs)))
    }
}

impl BinaryMath for Mul {
    fn float<F: Float>() -> Option<fn(F, F) -> F> {
        Some(|lhs, rhs| lhs * rhs)
    }
    fn int() -> Option<IntBinary> {
        Some(|lhs, rhs, _| Some(lhs.wrapping_mul(rhs)))
    }
}

/// Integer division truncates toward zero; a divisor of 0 leaves it undefined.
impl BinaryMath for Div {
    fn float<F: Float>() -> Option<fn(F, F) -> F> {
        Some(|lhs, rhs| lhs / rhs)
    }
    fn int() -> Option<IntBinary> {
        Some(|lhs, rhs, _| (rhs != 0).then(|| lhs / rhs))
    }
}

/// A NaN operand gives NaN, and `-0.0` orders below `0.0`.
impl BinaryMath for Maximum {
    fn float<F: Float>() -> Option<fn(F, F) -> F> {
        // Equal operands differ at most in the sign of a zero.
        Some(
            |lhs, rhs| match (lhs.is_nan() || rhs.is_nan(), lhs == rhs) {
                (true, _) => F::NAN,
                (false, true) if lhs.is_sign_negative() => rhs,
                (false, _) if lhs >= rhs => lhs,
                _ => rhs,
            },
        )
    }
    fn int() -> Option<IntBinary> {
        Some(|lhs, rhs, _| Some(lhs.max(rhs)))
    }
}

/// A NaN operand gives NaN, and `-0.0` orders below `0.0`.
impl BinaryMath for Minimum {
    fn float<F: Float>() -> Option<fn(F, F) -> F> {
        Some(
            |lhs, rhs| match (lhs.is_nan() || rhs.is_nan(), lhs == rhs) {
                (true, _) => F::NAN,
                (false, true) if lhs.is_sign_negative() => lhs,
                (false, _) if lhs < rhs => lhs,
                _ => rhs,
            },
        )
    }
    fn int() -> Option<IntBinary> {
        Some(|lhs, rhs, _| Some(lhs.min(rhs)))
    }
}

impl BinaryMath for And {
    fn int() -> Option<IntBinary> {
        Some(|lhs, rhs, _| Some(lhs & rhs))
    }
}

impl BinaryMath for Or {
    fn int() -> Option<IntBinary> {
        Some(|lhs, rhs, _| Some(lhs | rhs))
    }
}

impl BinaryMath for Xor {
    fn int() -> Option<IntBinary> {
        Some(|lhs, rhs, _| Some(lhs ^ rhs))
    }
}

/// The amount a shift of an integer of `format` moves by: its unsigned value, or `None` when it
/// is at least the width.
fn shift_amount(format: IntFormat, amount: i128) -> Option<i128> {
    let unsigned = format.pattern(amount);
    (unsigned < i128::from(format.bits)).then_some(unsigned)
}

impl BinaryMath for ShiftLeft {
    fn int() -> Option<IntBinary> {
        Some(|x, amount, format| Some(shift_amount(format, amount).map_or(0, |by| x << by)))
    }
}

impl BinaryMath for ShiftRightLogical {
    fn int() -> Option<IntBinary> {
        Some(|x, amount, format| {
            Some(shift_amount(format, amount).map_or(0, |by| format.pattern(x) >> by))
        })
    }
}

impl BinaryMath for ShiftRightArithmetic {
    fn int() -> Option<IntBinary> {
        Some(|x, amount, format| match shift_amount(format, amount) {
            Some(by) => Some(x >> by),
            None if x < 0 => Some(-1),
            None => Some(0),
        })
    }
}

impl Eval for Constant {
    fn eval(
        &self,
        _ins: ConstantOperands<Host>,
        _cx: &mut EvalCx<NoRegions>,
    ) -> Result<ConstantResults<OwnedHost>> {
        Ok(ConstantResults {
            output: self.value.scalar(),
        })
    }
}

impl CompareMath for Less {
    fn holds<T: PartialOrd>(lhs: T, rhs: T) -> bool {
        lhs < rhs
    }
}

impl CompareMath for LessEqual {
    fn holds<T: PartialOrd>(lhs: T, rhs: T) -> bool {
        lhs <= rhs
    }
}

impl CompareMath for Equal {
    fn holds<T: PartialOrd>(lhs: T, rhs: T) -> bool {
        lhs == rhs
    }
}

impl CompareMath for GreaterEqual {
    fn holds<T: PartialOrd>(lhs: T, rhs: T) -> bool {
        lhs >= rhs
    }
}

impl CompareMath for Greater {
    fn holds<T: PartialOrd>(lhs: T, rhs: T) -> bool {
        lhs > rhs
    }
}

impl CompareMath for NotEqual {
    fn holds<T: PartialOrd>(lhs: T, rhs: T) -> bool {
        lhs != rhs
    }
}

impl Eval for Select {
    fn eval(
        &self,
        ins: SelectOperands<Host>,
        _cx: &mut EvalCx<NoRegions>,
    ) -> Result<SelectResults<OwnedHost>> {
        Ok(SelectResults {
            output: math::select(ins.pred, ins.on_true, ins.on_false)?,
        })
    }
}

impl Eval for Fma {
    fn eval(
        &self,
        ins: FmaOperands<Host>,
        _cx: &mut EvalCx<NoRegions>,
    ) -> Result<FmaResults<OwnedHost>> {
        Ok(FmaResults {
            output: math::fma(ins.lhs, ins.rhs, ins.acc)?,
        })
    }
}

impl Eval for Cast {
    fn eval(
        &self,
        ins: CastOperands<Host>,
        _cx: &mut EvalCx<NoRegions>,
    ) -> Result<CastResults<OwnedHost>> {
        Ok(CastResults {
            output: math::cast(ins.input, self.to)?,
        })
    }
}

impl Eval for FromIndex {
    fn eval(
        &self,
        ins: FromIndexOperands<Host>,
        _cx: &mut EvalCx<NoRegions>,
    ) -> Result<FromIndexResults<OwnedHost>> {
        Ok(FromIndexResults {
            output: math::from_index(ins.input, self.dtype)?,
        })
    }
}

impl Eval for ToIndex {
    fn eval(
        &self,
        ins: ToIndexOperands<Host>,
        _cx: &mut EvalCx<NoRegions>,
    ) -> Result<ToIndexResults<OwnedHost>> {
        Ok(ToIndexResults {
            output: ins.input.as_index()?,
        })
    }
}

impl Eval for Bitcast {
    fn eval(
        &self,
        ins: BitcastOperands<Host>,
        _cx: &mut EvalCx<NoRegions>,
    ) -> Result<BitcastResults<OwnedHost>> {
        Ok(BitcastResults {
            output: math::bitcast(ins.input, self.to)?,
        })
    }
}

/// The semantics of every [`Unary`] op.
pub(crate) fn unary<T: Unary>(
    _op: &T,
    ins: UnaryOperands<Host>,
    _cx: &mut EvalCx<NoRegions>,
) -> Result<UnaryResults<OwnedHost>> {
    Ok(UnaryResults {
        output: math::unary::<T>(ins.input)?,
    })
}

/// The semantics of every [`Binary`] op.
pub(crate) fn binary<T: Binary>(
    _op: &T,
    ins: BinaryOperands<Host>,
    _cx: &mut EvalCx<NoRegions>,
) -> Result<BinaryResults<OwnedHost>> {
    Ok(BinaryResults {
        output: math::binary::<T>(ins.lhs, ins.rhs)?,
    })
}

/// The semantics of every [`Rng`] op.
pub(crate) fn rng<T: Rng>(
    op: &T,
    ins: RngOperands<Host>,
    _cx: &mut EvalCx<NoRegions>,
) -> Result<RngResults<OwnedHost>> {
    Ok(RngResults {
        output: draw_one::<T>(op.dtype, ins.key, ins.counter, ins.index)?,
    })
}

/// The semantics of every [`Compare`] op.
pub(crate) fn compare<T: Compare>(
    _op: &T,
    ins: CompareOperands<Host>,
    _cx: &mut EvalCx<NoRegions>,
) -> Result<CompareResults<OwnedHost>> {
    Ok(CompareResults {
        output: math::compare::<T>(ins.lhs, ins.rhs)?,
    })
}
