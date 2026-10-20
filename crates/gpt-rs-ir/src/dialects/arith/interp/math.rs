//! Scalar arithmetic: the semantics of the elementwise ops on one element. Each `arith` op that a
//! `tensor` op applies to every element implements [`UnaryMath`], [`BinaryMath`] or
//! [`CompareMath`]: its function on each type the interpreter computes in. The `tensor`
//! interpreter, the `arith` ops and the `tile` interpreter all call those functions, so each op's
//! semantics are written once.
//!
//! A [`HostScalar`] holds its value in the type the interpreter computes its dtype in. Each op
//! computes in that type and rounds once to the result dtype, as the `tensor` ops do element by
//! element.

use std::cmp::Ordering;
use std::ops::{Add, Div, Mul, Neg, Sub};

use anyhow::{anyhow, ensure, Result};

use crate::dialects::arith::{ensure_bitcast, ensure_fma};
use crate::dialects::tensor::interp::elementwise::{binary_data, compare_data, unary_data, Arith};
use crate::dialects::tensor::interp::kernels::cast as cast_data;
use crate::dialects::tensor::interp::random::element_words;
use crate::dialects::tensor::ir::Binary;
use crate::dialects::tile::CombineKind;
use crate::host::{element, repr, HostScalar, HostTensor, Repr, ScalarValue};
use crate::ir::{DType, IntFormat, OpKind, TensorType};

/// A float type the interpreter computes in, f32 or f64: the operations the per-element
/// functions use, each IEEE 754 in the type's own precision. `exp`, `ln`, `tanh`, `erf`, `sin` and
/// `cos` are the `libm` crate's, so the reference gives the same bits on every host.
pub trait Float:
    Copy
    + PartialOrd
    + Neg<Output = Self>
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
{
    const ZERO: Self;
    const ONE: Self;
    const NAN: Self;
    const INFINITY: Self;
    const NEG_INFINITY: Self;
    fn abs(self) -> Self;
    fn exp(self) -> Self;
    fn ln(self) -> Self;
    fn tanh(self) -> Self;
    fn erf(self) -> Self;
    fn sqrt(self) -> Self;
    fn sin(self) -> Self;
    fn cos(self) -> Self;
    fn is_nan(self) -> bool;
    fn is_sign_negative(self) -> bool;
}

macro_rules! float {
    ($t:ty, $exp:path, $ln:path, $tanh:path, $erf:path, $sin:path, $cos:path) => {
        impl Float for $t {
            const ZERO: $t = 0.0;
            const ONE: $t = 1.0;
            const NAN: $t = <$t>::NAN;
            const INFINITY: $t = <$t>::INFINITY;
            const NEG_INFINITY: $t = <$t>::NEG_INFINITY;
            fn abs(self) -> $t {
                <$t>::abs(self)
            }
            fn exp(self) -> $t {
                $exp(self)
            }
            fn ln(self) -> $t {
                $ln(self)
            }
            fn tanh(self) -> $t {
                $tanh(self)
            }
            fn erf(self) -> $t {
                $erf(self)
            }
            fn sqrt(self) -> $t {
                <$t>::sqrt(self)
            }
            fn sin(self) -> $t {
                $sin(self)
            }
            fn cos(self) -> $t {
                $cos(self)
            }
            fn is_nan(self) -> bool {
                <$t>::is_nan(self)
            }
            fn is_sign_negative(self) -> bool {
                <$t>::is_sign_negative(self)
            }
        }
    };
}

/// `$name`, the f32 function that rounds the `libm` crate's f64 function `$wide` to f32: the
/// reference of the f32 math functions. It is closer to the true value than the crate's f32
/// functions, whose `tanhf` is 2 ulp off on some inputs.
macro_rules! rounded_from_f64 {
    ($name:ident, $wide:path) => {
        fn $name(value: f32) -> f32 {
            $wide(f64::from(value)) as f32
        }
    };
}

rounded_from_f64!(exp_f32, libm::exp);
rounded_from_f64!(log_f32, libm::log);
rounded_from_f64!(tanh_f32, libm::tanh);
rounded_from_f64!(erf_f32, libm::erf);
rounded_from_f64!(sin_f32, libm::sin);
rounded_from_f64!(cos_f32, libm::cos);

float!(f32, exp_f32, log_f32, tanh_f32, erf_f32, sin_f32, cos_f32);

float!(
    f64,
    libm::exp,
    libm::log,
    libm::tanh,
    libm::erf,
    libm::sin,
    libm::cos
);

/// An op on one element: its function on floats and on integers. Integer results are exact; the
/// caller wraps them to the dtype.
pub trait UnaryMath: OpKind {
    /// The op on a float, or `None` when it has no float form.
    fn float<F: Float>() -> Option<fn(F) -> F> {
        None
    }
    /// The op on an integer, or `None` when it has no integer form.
    fn int() -> Option<fn(i128) -> i128> {
        None
    }
}

/// A binary op on two integers of a format: the exact result, or `None` for operands it leaves
/// undefined.
pub type IntBinary = fn(i128, i128, IntFormat) -> Option<i128>;

/// An op on two elements of one dtype: its function on floats and on integers. Integer results
/// are exact; the caller wraps them to the dtype.
pub trait BinaryMath: OpKind {
    /// The op on two floats, or `None` when it has no float form.
    fn float<F: Float>() -> Option<fn(F, F) -> F> {
        None
    }
    /// The op on two integers of `format`, or `None` when it has no integer form. The function
    /// gives `None` for operands the op leaves undefined: a divisor of 0.
    fn int() -> Option<IntBinary> {
        None
    }
}

/// A comparison of two elements of one dtype: whether its relation holds. NaN is unordered: every
/// relation with a NaN operand fails, except `not_equal`, which holds.
pub trait CompareMath: OpKind {
    fn holds<T: PartialOrd>(lhs: T, rhs: T) -> bool;
}

/// What an rng op draws for one element from its words of the stream: an integer of the dtype's
/// range, or a float before rounding to the dtype.
pub enum Drawn {
    Int(i128),
    Float(f64),
}

/// How an rng op draws its elements from the Philox stream of its state.
pub trait Draw: OpKind {
    /// The words of the stream that each element of `dtype` takes.
    fn words_per_element(dtype: DType) -> Result<usize>;
    /// The element of `dtype` that its `words` give.
    fn draw(dtype: DType, words: &[u32]) -> Result<Drawn>;
}

/// Element `index` of the stream of the state `[key, counter]`, drawn by `T` in `dtype`.
pub fn draw_one<T: Draw>(
    dtype: DType,
    key: HostScalar,
    counter: HostScalar,
    index: HostScalar,
) -> Result<HostScalar> {
    let word = |scalar: HostScalar| -> Result<u64> {
        let value = scalar
            .as_int()
            .ok_or_else(|| anyhow!("a {} state word", scalar.dtype))?;
        u64::try_from(value).map_err(|_| anyhow!("state word {value} is not a ui64"))
    };
    let signed_index = index.as_index()?;
    let element = u64::try_from(signed_index)
        .map_err(|_| anyhow!("negative element index {signed_index}"))?;
    let per = T::words_per_element(dtype)?;
    let words = element_words(word(key)?, word(counter)?, element, per);
    match T::draw(dtype, &words)? {
        Drawn::Int(value) => HostScalar::int(dtype, value),
        Drawn::Float(value) => HostScalar::from_f64(dtype, value),
    }
}

/// A `tensor` op that reductions fold their elements with, by its `arith` op: the fold of
/// `combine` in `tile.block`, from the value a fold of no elements gives.
pub trait Combiner: Binary + Default {
    const COMBINE: CombineKind;
    /// The value a fold of no floats gives.
    fn float_init<F: Float>() -> Option<F>;
    /// The value a fold of no integers of `format` gives.
    fn int_init(format: IntFormat) -> i128;
}

/// The op `T` on `x`.
pub fn unary<T: UnaryMath>(x: HostScalar) -> Result<HostScalar> {
    Ok(element(&unary_data::<T>(&x.to_data()?)?))
}

/// The op `T` on `lhs` and `rhs`, of one dtype.
pub fn binary<T: BinaryMath>(lhs: HostScalar, rhs: HostScalar) -> Result<HostScalar> {
    ensure!(
        rhs.dtype == lhs.dtype,
        "{} operands differ in dtype: {} and {}",
        T::INFO.name,
        lhs.dtype,
        rhs.dtype
    );
    Ok(element(&binary_data::<T>(
        &lhs.to_data()?,
        &rhs.to_data()?,
    )?))
}

/// Whether the relation of `T` holds between `lhs` and `rhs`, as an `i1` scalar.
pub fn compare<T: CompareMath>(lhs: HostScalar, rhs: HostScalar) -> Result<HostScalar> {
    ensure!(
        lhs.dtype == rhs.dtype,
        "compared operands differ in dtype: {} and {}",
        lhs.dtype,
        rhs.dtype
    );
    Ok(element(&compare_data::<T>(
        &lhs.to_data()?,
        &rhs.to_data()?,
    )?))
}

/// How `lhs` orders against `rhs`, of one dtype, with NaN above every number and equal to NaN:
/// the order of `tensor.argmax` and `tensor.top_k`.
pub fn order(lhs: HostScalar, rhs: HostScalar) -> Ordering {
    match (lhs.value, rhs.value) {
        (ScalarValue::Int(left), ScalarValue::Int(right)) => i128::order(left, right),
        (ScalarValue::F32(left), ScalarValue::F32(right)) => f32::order(left, right),
        (ScalarValue::F64(left), ScalarValue::F64(right)) => f64::order(left, right),
        _ => unreachable!("ordered scalars share a dtype"),
    }
}

/// The index `value` as an element of `dtype`, converted as `tensor.iota` converts its indices: as
/// an si64 cast to `dtype`.
pub fn from_index(value: i64, dtype: DType) -> Result<HostScalar> {
    cast(HostScalar::int(DType::Si64, i128::from(value))?, dtype)
}

/// `x` converted to `to` as `tensor.cast` converts an element.
pub fn cast(x: HostScalar, to: DType) -> Result<HostScalar> {
    if x.dtype == to {
        return Ok(x);
    }
    Ok(element(&cast_data(&x.to_data()?, to)?))
}

/// The bits of `x` read as an element of `to`, of the same width.
pub fn bitcast(x: HostScalar, to: DType) -> Result<HostScalar> {
    ensure_bitcast(x.dtype, to)?;
    let data = x.to_data()?;
    let tensor = HostTensor::from_le_bytes(&TensorType::new(to, [1usize]), data.bytes())?;
    Ok(element(&tensor.data))
}

/// `lhs * rhs + acc`, rounded once to the dtype, f32 or f64.
pub fn fma(lhs: HostScalar, rhs: HostScalar, acc: HostScalar) -> Result<HostScalar> {
    ensure!(
        lhs.dtype == rhs.dtype && lhs.dtype == acc.dtype,
        "fma operands differ in dtype: {}, {} and {}",
        lhs.dtype,
        rhs.dtype,
        acc.dtype
    );
    ensure_fma(lhs.dtype)?;
    let value = match (lhs.value, rhs.value, acc.value) {
        (ScalarValue::F32(factor), ScalarValue::F32(multiplier), ScalarValue::F32(addend)) => {
            ScalarValue::F32(factor.mul_add(multiplier, addend))
        }
        (ScalarValue::F64(factor), ScalarValue::F64(multiplier), ScalarValue::F64(addend)) => {
            ScalarValue::F64(factor.mul_add(multiplier, addend))
        }
        _ => unreachable!("an f32 or f64 scalar holds its value in its own type"),
    };
    Ok(HostScalar {
        dtype: lhs.dtype,
        value,
    })
}

/// `on_true` where `pred` is set and `on_false` elsewhere.
pub fn select(pred: HostScalar, on_true: HostScalar, on_false: HostScalar) -> Result<HostScalar> {
    ensure!(
        pred.dtype == DType::I1,
        "pred must be i1, got {}",
        pred.dtype
    );
    ensure!(
        on_true.dtype == on_false.dtype,
        "select branches differ in dtype: {} and {}",
        on_true.dtype,
        on_false.dtype
    );
    Ok(if pred.as_int() != Some(0) {
        on_true
    } else {
        on_false
    })
}

/// The scalar of `dtype` that a fold with `C` starts from: the value a fold of no elements gives.
pub fn fold_init<C: Combiner>(dtype: DType) -> Result<HostScalar> {
    Ok(match repr(dtype) {
        Repr::Int => HostScalar::int(dtype, i128::fold_init::<C>(dtype)?)?,
        Repr::F32 => HostScalar {
            dtype,
            value: ScalarValue::F32(f32::fold_init::<C>(dtype)?),
        },
        Repr::F64 => HostScalar {
            dtype,
            value: ScalarValue::F64(f64::fold_init::<C>(dtype)?),
        },
    })
}
