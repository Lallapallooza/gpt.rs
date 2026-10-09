//! Semantics of the ops in `ir/elementwise.rs`.

use std::cmp::Ordering;

use anyhow::{anyhow, bail, ensure, Result};

use super::kernels::{cast, from_f32, from_f64, from_ints, values, Chunks, Values};
use crate::attr::Mnemonic;
use crate::dialects::tensor::ir::{
    Binary, BinaryKind, BinaryOperands, BinaryResults, Cast, CastOperands, CastResults, Compare,
    CompareKind, CompareOperands, CompareResults, ReduceKind, Select, SelectOperands,
    SelectResults, StopGradient, StopGradientOperands, StopGradientResults, Unary, UnaryKind,
    UnaryOperands, UnaryResults,
};
use crate::host::{Buffer, HostTensor, TensorData};
use crate::op::{Eval, EvalCx, Host, NoRegions, OwnedHost};
use crate::types::{DType, IntFormat};

/// The arithmetic of the types the interpreter computes in: f32, f64, and i128 for the integer
/// dtypes and `i1`, whose results the caller wraps to the dtype.
pub(super) trait Arith: Copy {
    /// `kind` on one element, or `None` when the type has no such op.
    fn unary(kind: UnaryKind) -> Option<fn(Self) -> Self>;
    /// `kind` on one pair of elements, or `None` when the type has no such op. The shifts depend
    /// on the dtype's width, which [`shift`] reads.
    fn binary(kind: BinaryKind) -> Option<fn(Self, Self) -> Self>;
    /// The value a `kind` reduction of no `dtype` elements gives, which every reduction starts
    /// from.
    fn reduce_init(kind: ReduceKind, dtype: DType) -> Option<Self>;
    /// `a` against `b`, with NaN above every number and equal to NaN.
    fn order(a: Self, b: Self) -> Ordering;
}

/// The float arithmetic: IEEE 754 operations. `maximum` and `minimum` give NaN when either
/// operand is NaN, and order `-0.0` below `0.0`.
macro_rules! float_arith {
    ($t:ty, $erf:path) => {
        impl Arith for $t {
            fn unary(kind: UnaryKind) -> Option<fn($t) -> $t> {
                Some(match kind {
                    UnaryKind::Neg => |x| -x,
                    UnaryKind::Abs => <$t>::abs,
                    UnaryKind::Exp => <$t>::exp,
                    UnaryKind::Log => <$t>::ln,
                    UnaryKind::Tanh => <$t>::tanh,
                    UnaryKind::Erf => $erf,
                    UnaryKind::Rsqrt => |x| 1.0 / x.sqrt(),
                    UnaryKind::Reciprocal => |x| 1.0 / x,
                    UnaryKind::Sin => <$t>::sin,
                    UnaryKind::Cos => <$t>::cos,
                    UnaryKind::Not => return None,
                })
            }

            fn binary(kind: BinaryKind) -> Option<fn($t, $t) -> $t> {
                Some(match kind {
                    BinaryKind::Add => |x, y| x + y,
                    BinaryKind::Sub => |x, y| x - y,
                    BinaryKind::Mul => |x, y| x * y,
                    BinaryKind::Div => |x, y| x / y,
                    // Equal operands differ at most in the sign of a zero.
                    BinaryKind::Maximum => |x, y| match (x.is_nan() || y.is_nan(), x == y) {
                        (true, _) => <$t>::NAN,
                        (false, true) if x.is_sign_negative() => y,
                        (false, _) if x >= y => x,
                        _ => y,
                    },
                    BinaryKind::Minimum => |x, y| match (x.is_nan() || y.is_nan(), x == y) {
                        (true, _) => <$t>::NAN,
                        (false, true) if x.is_sign_negative() => x,
                        (false, _) if x < y => x,
                        _ => y,
                    },
                    _ => return None,
                })
            }

            fn reduce_init(kind: ReduceKind, _dtype: DType) -> Option<$t> {
                Some(match kind {
                    ReduceKind::Sum => 0.0,
                    ReduceKind::Max => <$t>::NEG_INFINITY,
                    ReduceKind::Min => <$t>::INFINITY,
                    ReduceKind::And | ReduceKind::Or => return None,
                })
            }

            fn order(a: $t, b: $t) -> Ordering {
                match (a.is_nan(), b.is_nan()) {
                    (true, true) => Ordering::Equal,
                    (true, false) => Ordering::Greater,
                    (false, true) => Ordering::Less,
                    (false, false) => a.partial_cmp(&b).expect("neither is NaN"),
                }
            }
        }
    };
}

float_arith!(f32, libm::erff);
float_arith!(f64, libm::erf);

/// The integer arithmetic on exact values. `div` truncates toward zero and needs a divisor other
/// than 0.
impl Arith for i128 {
    fn unary(kind: UnaryKind) -> Option<fn(i128) -> i128> {
        Some(match kind {
            UnaryKind::Neg => i128::wrapping_neg,
            UnaryKind::Abs => i128::wrapping_abs,
            UnaryKind::Not => |x| !x,
            _ => return None,
        })
    }

    fn binary(kind: BinaryKind) -> Option<fn(i128, i128) -> i128> {
        Some(match kind {
            BinaryKind::Add => i128::wrapping_add,
            BinaryKind::Sub => i128::wrapping_sub,
            BinaryKind::Mul => i128::wrapping_mul,
            BinaryKind::Div => |x, y| x / y,
            BinaryKind::Maximum => i128::max,
            BinaryKind::Minimum => i128::min,
            BinaryKind::And => |x, y| x & y,
            BinaryKind::Or => |x, y| x | y,
            BinaryKind::Xor => |x, y| x ^ y,
            _ => return None,
        })
    }

    fn reduce_init(kind: ReduceKind, dtype: DType) -> Option<i128> {
        let int = dtype.int_format()?;
        Some(match kind {
            ReduceKind::Sum | ReduceKind::Or => 0,
            ReduceKind::Max => int.min(),
            ReduceKind::Min => int.max(),
            ReduceKind::And => -1,
        })
    }

    fn order(a: i128, b: i128) -> Ordering {
        a.cmp(&b)
    }
}

/// `x` shifted as `kind` says by `amount`, both integers of `int`. The amount reads as unsigned;
/// shifting by at least the width gives 0, or the sign in every bit for an arithmetic right shift.
fn shift(kind: BinaryKind, int: IntFormat, x: i128, amount: i128) -> i128 {
    let amount = int.pattern(amount);
    let outside = amount >= i128::from(int.bits);
    match (kind, outside) {
        (BinaryKind::ShiftLeft, false) => x << amount,
        (BinaryKind::ShiftRightLogical, false) => int.pattern(x) >> amount,
        (BinaryKind::ShiftRightArithmetic, false) => x >> amount,
        (BinaryKind::ShiftRightArithmetic, true) if x < 0 => -1,
        _ => 0,
    }
}

impl Eval for Unary {
    fn eval(
        &self,
        ins: UnaryOperands<Host<'_>>,
        _cx: &mut EvalCx<'_, NoRegions>,
    ) -> Result<UnaryResults<OwnedHost>> {
        let input = ins.input;
        let dtype = input.dtype();
        let missing = || anyhow!("the interpreter has no {dtype} {}", self.kind.mnemonic());
        let data = match values(&input.data) {
            Values::Int(x) => {
                let (f, int) = (
                    i128::unary(self.kind).ok_or_else(missing)?,
                    int_format(dtype)?,
                );
                from_ints(dtype, x.into_iter().map(|x| int.wrap(f(x))))?
            }
            Values::F32(x) => {
                let f = f32::unary(self.kind).ok_or_else(missing)?;
                from_f32(dtype, x.iter().map(|&x| f(x)))?
            }
            Values::F64(x) => {
                let f = f64::unary(self.kind).ok_or_else(missing)?;
                from_f64(dtype, x.iter().map(|&x| f(x)))?
            }
        };
        Ok(UnaryResults {
            output: HostTensor::new(input.dims.clone(), data)?,
        })
    }
}

/// The width and signedness of the integer dtype `dtype`.
pub(super) fn int_format(dtype: DType) -> Result<IntFormat> {
    dtype
        .int_format()
        .ok_or_else(|| anyhow!("{dtype} is not an integer dtype"))
}

impl Eval for Binary {
    fn eval(
        &self,
        ins: BinaryOperands<Host<'_>>,
        _cx: &mut EvalCx<'_, NoRegions>,
    ) -> Result<BinaryResults<OwnedHost>> {
        let dtype = ins.lhs.dtype();
        let kind = self.kind;
        let missing = || anyhow!("the interpreter has no {dtype} {}", kind.mnemonic());
        let data = match (values(&ins.lhs.data), values(&ins.rhs.data)) {
            (Values::Int(a), Values::Int(b)) => {
                let int = int_format(dtype)?;
                let pairs = a.into_iter().zip(b);
                let results: Vec<i128> = match kind {
                    BinaryKind::ShiftLeft
                    | BinaryKind::ShiftRightLogical
                    | BinaryKind::ShiftRightArithmetic => {
                        pairs.map(|(x, y)| shift(kind, int, x, y)).collect()
                    }
                    _ => {
                        let f = i128::binary(kind).ok_or_else(missing)?;
                        let pairs: Vec<(i128, i128)> = pairs.collect();
                        ensure!(
                            kind != BinaryKind::Div || pairs.iter().all(|&(_, y)| y != 0),
                            "integer division by zero"
                        );
                        pairs.into_iter().map(|(x, y)| f(x, y)).collect()
                    }
                };
                from_ints(dtype, results.into_iter().map(|x| int.wrap(x)))?
            }
            (Values::F32(a), Values::F32(b)) => {
                let f = f32::binary(kind).ok_or_else(missing)?;
                from_f32(dtype, a.iter().zip(b.iter()).map(|(&x, &y)| f(x, y)))?
            }
            (Values::F64(a), Values::F64(b)) => {
                let f = f64::binary(kind).ok_or_else(missing)?;
                from_f64(dtype, a.iter().zip(b.iter()).map(|(&x, &y)| f(x, y)))?
            }
            _ => bail!("{} operands differ in dtype", kind.mnemonic()),
        };
        Ok(BinaryResults {
            output: HostTensor::new(ins.lhs.dims.clone(), data)?,
        })
    }
}

impl Eval for Compare {
    fn eval(
        &self,
        ins: CompareOperands<Host<'_>>,
        _cx: &mut EvalCx<'_, NoRegions>,
    ) -> Result<CompareResults<OwnedHost>> {
        let flags = match (values(&ins.lhs.data), values(&ins.rhs.data)) {
            (Values::Int(a), Values::Int(b)) => self.kind.flags(&a, &b),
            (Values::F32(a), Values::F32(b)) => self.kind.flags(&a, &b),
            (Values::F64(a), Values::F64(b)) => self.kind.flags(&a, &b),
            _ => bail!("compared operands differ in dtype"),
        };
        Ok(CompareResults {
            output: HostTensor::new(ins.lhs.dims.clone(), TensorData::I1(flags))?,
        })
    }
}

impl CompareKind {
    /// 1 where the comparison holds between the elements of `a` and `b` at a position, else 0.
    /// Every comparison with NaN fails, except `not_equal`.
    fn flags<T: PartialOrd>(self, a: &[T], b: &[T]) -> Buffer<u8> {
        let holds = |x, y| match self {
            CompareKind::Less => x < y,
            CompareKind::LessEqual => x <= y,
            CompareKind::Equal => x == y,
            CompareKind::GreaterEqual => x >= y,
            CompareKind::Greater => x > y,
            CompareKind::NotEqual => x != y,
        };
        a.iter()
            .zip(b)
            .map(|(x, y)| u8::from(holds(x, y)))
            .collect()
    }
}

impl Eval for Select {
    fn eval(
        &self,
        ins: SelectOperands<Host<'_>>,
        _cx: &mut EvalCx<'_, NoRegions>,
    ) -> Result<SelectResults<OwnedHost>> {
        let (on_true, on_false) = (
            Chunks::of(&ins.on_true.data),
            Chunks::of(&ins.on_false.data),
        );
        let pred = ins.pred.as_i1()?;
        let mut out = Chunks::zeroed(ins.on_true.dtype(), pred.len());
        for (i, &pred) in pred.iter().enumerate() {
            let branch = if pred != 0 { &on_true } else { &on_false };
            out.set(i, branch.get(i));
        }
        Ok(SelectResults {
            output: HostTensor::new(ins.on_true.dims.clone(), out.into_data()?)?,
        })
    }
}

impl Eval for Cast {
    fn eval(
        &self,
        ins: CastOperands<Host<'_>>,
        _cx: &mut EvalCx<'_, NoRegions>,
    ) -> Result<CastResults<OwnedHost>> {
        Ok(CastResults {
            output: HostTensor::new(ins.input.dims.clone(), cast(&ins.input.data, self.to)?)?,
        })
    }
}

impl Eval for StopGradient {
    fn eval(
        &self,
        ins: StopGradientOperands<Host<'_>>,
        _cx: &mut EvalCx<'_, NoRegions>,
    ) -> Result<StopGradientResults<OwnedHost>> {
        Ok(StopGradientResults {
            output: ins.input.clone(),
        })
    }
}
