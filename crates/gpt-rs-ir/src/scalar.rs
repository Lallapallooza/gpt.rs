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

use anyhow::{anyhow, bail, ensure, Result};

use crate::dense::Dense;
use crate::dialects::tensor::interp::elementwise::{
    binary_data, compare_data, int_format, unary_data, Arith,
};
use crate::dialects::tensor::interp::kernels::{
    cast as cast_data, encode_f32, from_f64, from_ints, values, Values,
};
use crate::dialects::tensor::interp::random::element_words;
use crate::dialects::tensor::ir::Binary;
use crate::dialects::tile::CombineKind;
use crate::host::{HostScalar, HostTensor, ScalarValue, TensorData};
use crate::minifloat::e8m0_encode;
use crate::op::OpKind;
use crate::types::{DType, IntFormat, TensorType};

/// A float type the interpreter computes in, f32 or f64: the operations the per-element
/// functions use, each IEEE 754 in the type's own precision.
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
    ($t:ty, $erf:path) => {
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
                <$t>::exp(self)
            }
            fn ln(self) -> $t {
                <$t>::ln(self)
            }
            fn tanh(self) -> $t {
                <$t>::tanh(self)
            }
            fn erf(self) -> $t {
                $erf(self)
            }
            fn sqrt(self) -> $t {
                <$t>::sqrt(self)
            }
            fn sin(self) -> $t {
                <$t>::sin(self)
            }
            fn cos(self) -> $t {
                <$t>::cos(self)
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

float!(f32, libm::erff);
float!(f64, libm::erf);

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

/// The type the interpreter computes `dtype` in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Repr {
    Int,
    F32,
    F64,
}

fn repr(dtype: DType) -> Repr {
    if dtype == DType::I1 || dtype.int_format().is_some() {
        Repr::Int
    } else if dtype == DType::F64 {
        Repr::F64
    } else {
        Repr::F32
    }
}

impl HostScalar {
    /// The value `x` of the integer dtype or `i1` `dtype`, wrapped into its range.
    pub fn int(dtype: DType, x: i128) -> Result<Self> {
        let value = match dtype {
            DType::I1 => x & 1,
            _ => int_format(dtype)?.wrap(x),
        };
        Ok(Self {
            dtype,
            value: ScalarValue::Int(value),
        })
    }

    /// The value 0 of `dtype`. Fails for a dtype without 0, such as `e8m0`.
    pub fn zero(dtype: DType) -> Result<Self> {
        match repr(dtype) {
            Repr::Int => Self::int(dtype, 0),
            Repr::F32 | Repr::F64 => Self::from_f64(dtype, 0.0),
        }
    }

    /// `x` converted to `dtype` as `tensor.cast` converts an f64: rounded to nearest-even for a
    /// float, truncated and saturated for an integer.
    pub fn from_f64(dtype: DType, x: f64) -> Result<Self> {
        let source = Self {
            dtype: DType::F64,
            value: ScalarValue::F64(x),
        };
        cast(source, dtype)
    }

    /// The integer value of an integer or `i1` scalar.
    pub fn as_int(&self) -> Option<i128> {
        match self.value {
            ScalarValue::Int(x) => Some(x),
            _ => None,
        }
    }

    /// The value of a float scalar, exactly.
    pub fn as_f64(&self) -> Option<f64> {
        match self.value {
            ScalarValue::F32(x) => Some(f64::from(x)),
            ScalarValue::F64(x) => Some(x),
            ScalarValue::Int(_) => None,
        }
    }

    /// The scalar as an index: its integer value, which must fit `i64`.
    pub fn as_index(&self) -> Result<i64> {
        let value = self
            .as_int()
            .ok_or_else(|| anyhow!("a {} element is not an index", self.dtype))?;
        i64::try_from(value).map_err(|_| anyhow!("index {value} does not fit i64"))
    }

    /// The scalar as a one-element tensor of its dtype.
    pub fn to_data(&self) -> Result<TensorData> {
        encode(self.dtype, std::slice::from_ref(self))
    }

    /// The scalar as a rank-0 tensor.
    pub fn to_tensor(&self) -> Result<HostTensor> {
        HostTensor::new(Vec::<usize>::new(), self.to_data()?)
    }

    /// The scalar as a rank-0 dense constant.
    pub fn to_dense(&self) -> Result<Dense> {
        Ok(Dense {
            dtype: self.dtype,
            dims: Vec::new(),
            bytes: self.to_data()?.bytes().into(),
        })
    }

    /// The element of the rank-0 dense constant `dense`.
    pub fn from_dense(dense: &Dense) -> Result<Self> {
        ensure!(
            dense.dims.is_empty(),
            "a scalar is a rank-0 constant, not {}",
            dense.tensor_type()
        );
        let tensor = HostTensor::from_le_bytes(&dense.tensor_type(), &dense.bytes)?;
        Ok(element(&tensor.data))
    }
}

/// The elements of a tensor, decoded once into the types the interpreter computes with.
pub struct Decoded {
    dtype: DType,
    values: Values<'static>,
}

impl Decoded {
    pub fn new(data: &TensorData) -> Self {
        Self {
            dtype: data.dtype(),
            values: values(data).into_owned(),
        }
    }

    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Element `index`.
    pub fn get(&self, index: usize) -> HostScalar {
        let value = match &self.values {
            Values::Int(v) => ScalarValue::Int(v[index]),
            Values::F32(v) => ScalarValue::F32(v[index]),
            Values::F64(v) => ScalarValue::F64(v[index]),
        };
        HostScalar {
            dtype: self.dtype,
            value,
        }
    }
}

/// The elements `scalars`, each a value of `dtype`, as the data of a tensor of `dtype`.
pub fn encode(dtype: DType, scalars: &[HostScalar]) -> Result<TensorData> {
    if let Some(other) = scalars.iter().find(|s| s.dtype != dtype) {
        bail!("a {} element in a {dtype} tensor", other.dtype);
    }
    if dtype == DType::E8M0 {
        // An e8m0 element is a power of two, or NaN: its code is its exponent plus the bias.
        let codes = scalars.iter().map(|s| match s.value {
            ScalarValue::F32(x) if x.is_nan() => 255,
            ScalarValue::F32(x) => {
                let exponent = ((f64::from(x).to_bits() >> 52) & 0x7ff) as i32 - 1023;
                e8m0_encode(exponent)
            }
            _ => unreachable!("f32 is the computation type of e8m0"),
        });
        return Ok(TensorData::E8M0(codes.collect()));
    }
    match repr(dtype) {
        Repr::Int => from_ints(
            dtype,
            scalars
                .iter()
                .map(|s| s.as_int().expect("an integer dtype")),
        ),
        Repr::F32 => encode_f32(
            dtype,
            scalars.iter().map(|s| match s.value {
                ScalarValue::F32(x) => x,
                _ => unreachable!("f32 is the computation type of {dtype}"),
            }),
        ),
        Repr::F64 => from_f64(
            dtype,
            scalars.iter().map(|s| match s.value {
                ScalarValue::F64(x) => x,
                _ => unreachable!("f64 is the computation type of f64"),
            }),
        ),
    }
}

/// A tensor of type `ty` with the elements `scalars`, in row-major order.
pub fn tensor(ty: &TensorType, scalars: &[HostScalar]) -> Result<HostTensor> {
    let dims = ty
        .static_dims()
        .ok_or_else(|| anyhow!("a host tensor has static extents, not {ty}"))?;
    HostTensor::new(dims, encode(ty.dtype, scalars)?)
}

/// The one element of `data`.
fn element(data: &TensorData) -> HostScalar {
    Decoded::new(data).get(0)
}

/// The op `T` on `x`.
pub fn unary<T: UnaryMath>(x: HostScalar) -> Result<HostScalar> {
    Ok(element(&unary_data::<T>(&x.to_data()?)?))
}

/// The op `T` on `x` and `y`, of one dtype.
pub fn binary<T: BinaryMath>(x: HostScalar, y: HostScalar) -> Result<HostScalar> {
    ensure!(
        y.dtype == x.dtype,
        "{} operands differ in dtype: {} and {}",
        T::INFO.name,
        x.dtype,
        y.dtype
    );
    Ok(element(&binary_data::<T>(&x.to_data()?, &y.to_data()?)?))
}

/// Whether the relation of `T` holds between `x` and `y`, as an `i1` scalar.
pub fn compare<T: CompareMath>(x: HostScalar, y: HostScalar) -> Result<HostScalar> {
    ensure!(
        x.dtype == y.dtype,
        "compared operands differ in dtype: {} and {}",
        x.dtype,
        y.dtype
    );
    Ok(element(&compare_data::<T>(&x.to_data()?, &y.to_data()?)?))
}

/// How `x` orders against `y`, of one dtype, with NaN above every number and equal to NaN: the
/// order of `tensor.argmax` and `tensor.top_k`.
pub fn order(x: HostScalar, y: HostScalar) -> Ordering {
    match (x.value, y.value) {
        (ScalarValue::Int(a), ScalarValue::Int(b)) => i128::order(a, b),
        (ScalarValue::F32(a), ScalarValue::F32(b)) => f32::order(a, b),
        (ScalarValue::F64(a), ScalarValue::F64(b)) => f64::order(a, b),
        _ => unreachable!("ordered scalars share a dtype"),
    }
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
    ensure!(
        x.dtype.bits() == to.bits() && x.dtype != DType::I1 && to != DType::I1,
        "a bitcast of one element keeps its width: {} to {to}",
        x.dtype
    );
    let data = x.to_data()?;
    let tensor = HostTensor::from_le_bytes(&TensorType::new(to, [1usize]), data.bytes())?;
    Ok(element(&tensor.data))
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
