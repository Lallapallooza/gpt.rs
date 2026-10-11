//! Semantics of the elementwise ops.

use std::borrow::Cow;
use std::cmp::Ordering;

use anyhow::{anyhow, bail, Result};

use super::kernels::{cast, from_f32, from_f64, from_ints, rounder, values, Chunks, Values};
use crate::dialects::tensor::ir::*;
use crate::host::{Buffer, HostTensor, TensorData};
use crate::op::{Eval, EvalCx, Host, NoRegions, OwnedHost};
use crate::scalar::{BinaryMath, Combiner, CompareMath, UnaryMath};
use crate::types::{DType, IntFormat};

/// The arithmetic of the types the interpreter computes in: f32, f64, and i128 for the integer
/// dtypes and `i1`, whose results the caller wraps to the dtype.
pub(crate) trait Arith: Copy {
    /// The op `T` on one element of `dtype`.
    fn unary<T: UnaryMath>(dtype: DType) -> Result<impl Fn(Self) -> Self>;
    /// The op `T` on two elements of `dtype`, `None` for operands it leaves undefined.
    fn binary<T: BinaryMath>(dtype: DType) -> Result<impl Fn(Self, Self) -> Option<Self>>;
    /// The value a fold with `C` of no elements of `dtype` gives, which every fold starts from.
    fn fold_init<C: Combiner>(dtype: DType) -> Result<Self>;
    /// `a` against `b`, with NaN above every number and equal to NaN.
    fn order(a: Self, b: Self) -> Ordering;
    /// The elements of `values` when the type computes them.
    fn of<'a>(values: &'a Values<'_>) -> Option<&'a [Self]>;
    /// `values` as the elements of a tensor.
    fn into_values(values: Vec<Self>) -> Values<'static>;
    /// What a value of `dtype` keeps of a result: the result wrapped into an integer dtype's
    /// range, or rounded to a float dtype.
    fn rounder(dtype: DType) -> Result<impl Fn(Self) -> Self>;
}

/// The error of an op that has no form for `dtype`.
fn missing(op: &str, dtype: DType) -> anyhow::Error {
    anyhow!("the interpreter has no {dtype} {op}")
}

macro_rules! float_arith {
    ($t:ty, $variant:ident, $round:expr) => {
        impl Arith for $t {
            fn unary<T: UnaryMath>(dtype: DType) -> Result<impl Fn($t) -> $t> {
                T::float::<$t>().ok_or_else(|| missing(T::INFO.name, dtype))
            }

            fn binary<T: BinaryMath>(dtype: DType) -> Result<impl Fn($t, $t) -> Option<$t>> {
                let op = T::float::<$t>().ok_or_else(|| missing(T::INFO.name, dtype))?;
                Ok(move |x, y| Some(op(x, y)))
            }

            fn fold_init<C: Combiner>(dtype: DType) -> Result<$t> {
                C::float_init::<$t>().ok_or_else(|| missing(C::INFO.name, dtype))
            }

            fn order(a: $t, b: $t) -> Ordering {
                match (a.is_nan(), b.is_nan()) {
                    (true, true) => Ordering::Equal,
                    (true, false) => Ordering::Greater,
                    (false, true) => Ordering::Less,
                    (false, false) => a.partial_cmp(&b).expect("neither is NaN"),
                }
            }

            fn of<'a>(values: &'a Values<'_>) -> Option<&'a [$t]> {
                match values {
                    Values::$variant(v) => Some(v),
                    _ => None,
                }
            }

            fn into_values(values: Vec<$t>) -> Values<'static> {
                Values::$variant(Cow::Owned(values))
            }

            fn rounder(dtype: DType) -> Result<impl Fn($t) -> $t> {
                Ok($round(dtype))
            }
        }
    };
}

float_arith!(f32, F32, rounder);
float_arith!(f64, F64, |_| |x| x);

/// The integer arithmetic on exact values.
impl Arith for i128 {
    fn unary<T: UnaryMath>(dtype: DType) -> Result<impl Fn(i128) -> i128> {
        T::int().ok_or_else(|| missing(T::INFO.name, dtype))
    }

    fn binary<T: BinaryMath>(dtype: DType) -> Result<impl Fn(i128, i128) -> Option<i128>> {
        let op = T::int().ok_or_else(|| missing(T::INFO.name, dtype))?;
        let format = int_format(dtype)?;
        Ok(move |x, y| op(x, y, format))
    }

    fn fold_init<C: Combiner>(dtype: DType) -> Result<i128> {
        Ok(C::int_init(int_format(dtype)?))
    }

    fn order(a: i128, b: i128) -> Ordering {
        a.cmp(&b)
    }

    fn of<'a>(values: &'a Values<'_>) -> Option<&'a [i128]> {
        match values {
            Values::Int(v) => Some(v),
            _ => None,
        }
    }

    fn into_values(values: Vec<i128>) -> Values<'static> {
        Values::Int(values)
    }

    fn rounder(dtype: DType) -> Result<impl Fn(i128) -> i128> {
        let int = int_format(dtype)?;
        Ok(move |x| int.wrap(x))
    }
}

/// The op `T` on each element of `x`, rounded to its dtype.
pub(crate) fn unary_data<T: UnaryMath>(x: &TensorData) -> Result<TensorData> {
    let dtype = x.dtype();
    match values(x) {
        Values::Int(x) => {
            let (op, int) = (i128::unary::<T>(dtype)?, int_format(dtype)?);
            from_ints(dtype, x.into_iter().map(|x| int.wrap(op(x))))
        }
        Values::F32(x) => {
            let op = f32::unary::<T>(dtype)?;
            from_f32(dtype, x.iter().map(|&x| op(x)))
        }
        Values::F64(x) => {
            let op = f64::unary::<T>(dtype)?;
            from_f64(dtype, x.iter().map(|&x| op(x)))
        }
    }
}

/// The width and signedness of the integer dtype `dtype`.
pub(crate) fn int_format(dtype: DType) -> Result<IntFormat> {
    dtype
        .int_format()
        .ok_or_else(|| anyhow!("{dtype} is not an integer dtype"))
}

/// The op `T` on each pair of elements of `a` and `b`, of one dtype, rounded to it.
pub(crate) fn binary_data<T: BinaryMath>(a: &TensorData, b: &TensorData) -> Result<TensorData> {
    let dtype = a.dtype();
    let name = T::INFO.name;
    match (values(a), values(b)) {
        (Values::Int(a), Values::Int(b)) => {
            let (op, int) = (i128::binary::<T>(dtype)?, int_format(dtype)?);
            let results = a
                .into_iter()
                .zip(b)
                .map(|(x, y)| op(x, y).ok_or_else(|| anyhow!("{name}: integer division by zero")))
                .collect::<Result<Vec<i128>>>()?;
            from_ints(dtype, results.into_iter().map(|x| int.wrap(x)))
        }
        (Values::F32(a), Values::F32(b)) => {
            let op = f32::binary::<T>(dtype)?;
            let results = a.iter().zip(b.iter()).map(|(&x, &y)| op(x, y));
            from_f32(
                dtype,
                results.map(|x| x.expect("float ops are defined everywhere")),
            )
        }
        (Values::F64(a), Values::F64(b)) => {
            let op = f64::binary::<T>(dtype)?;
            let results = a.iter().zip(b.iter()).map(|(&x, &y)| op(x, y));
            from_f64(
                dtype,
                results.map(|x| x.expect("float ops are defined everywhere")),
            )
        }
        _ => bail!("{name} operands differ in dtype"),
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
            output: HostTensor::new(ins.on_true.dims, out.into_data()?)?,
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
            output: HostTensor::new(ins.input.dims, cast(&ins.input.data, self.to)?)?,
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

/// The semantics of every [`Unary`] op.
pub(crate) fn unary<T: Unary>(
    _op: &T,
    ins: UnaryOperands<Host<'_>>,
    _cx: &mut EvalCx<'_, NoRegions>,
) -> Result<UnaryResults<OwnedHost>> {
    let data = unary_data::<T::Scalar>(&ins.input.data)?;
    Ok(UnaryResults {
        output: HostTensor::new(ins.input.dims, data)?,
    })
}

/// The semantics of every [`Binary`] op.
pub(crate) fn binary<T: Binary>(
    _op: &T,
    ins: BinaryOperands<Host<'_>>,
    _cx: &mut EvalCx<'_, NoRegions>,
) -> Result<BinaryResults<OwnedHost>> {
    let data = binary_data::<T::Scalar>(&ins.lhs.data, &ins.rhs.data)?;
    Ok(BinaryResults {
        output: HostTensor::new(ins.lhs.dims, data)?,
    })
}

/// 1 where the relation of `T` holds between the elements of `a` and `b` at a position, else 0.
pub(crate) fn compare_data<T: CompareMath>(a: &TensorData, b: &TensorData) -> Result<TensorData> {
    fn flags<T: CompareMath, V: PartialOrd + Copy>(a: &[V], b: &[V]) -> Buffer<u8> {
        a.iter()
            .zip(b)
            .map(|(&lhs, &rhs)| u8::from(T::holds(lhs, rhs)))
            .collect()
    }
    let flags = match (values(a), values(b)) {
        (Values::Int(a), Values::Int(b)) => flags::<T, _>(&a, &b),
        (Values::F32(a), Values::F32(b)) => flags::<T, _>(&a, &b),
        (Values::F64(a), Values::F64(b)) => flags::<T, _>(&a, &b),
        _ => bail!("compared operands differ in dtype"),
    };
    Ok(TensorData::I1(flags))
}

/// The semantics of every [`Compare`] op.
pub(crate) fn compare<T: Compare>(
    _op: &T,
    ins: CompareOperands<Host<'_>>,
    _cx: &mut EvalCx<'_, NoRegions>,
) -> Result<CompareResults<OwnedHost>> {
    let flags = compare_data::<T::Scalar>(&ins.lhs.data, &ins.rhs.data)?;
    Ok(CompareResults {
        output: HostTensor::new(ins.lhs.dims, flags)?,
    })
}
