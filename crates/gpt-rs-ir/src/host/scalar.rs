//! Scalars in host memory: a value of a dtype held in the type the interpreter computes it in, and
//! its conversions to and from the data of tensors.

use anyhow::{anyhow, bail, ensure, Result};

use super::{HostTensor, TensorData};
use crate::dialects::arith::math::cast;
use crate::dialects::tensor::interp::elementwise::int_format;
use crate::dialects::tensor::interp::kernels::{encode_f32, from_f64, from_ints, values, Values};
use crate::ir::{e8m0_encode, DType, Dense, TensorType};

/// A scalar run-time value: a value of `dtype`, held in the type the interpreter computes that
/// dtype in. The `scalar` module gives its arithmetic.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HostScalar {
    pub dtype: DType,
    pub value: ScalarValue,
}

/// A scalar in the type the interpreter computes with: `i128` for the integers and `i1`, f64 for
/// f64, and f32 for every other float and `e8m0`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ScalarValue {
    Int(i128),
    F32(f32),
    F64(f64),
}

impl HostScalar {
    pub fn dtype(&self) -> DType {
        self.dtype
    }
}

/// The type the interpreter computes `dtype` in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Repr {
    Int,
    F32,
    F64,
}

pub(crate) fn repr(dtype: DType) -> Repr {
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
pub(crate) fn element(data: &TensorData) -> HostScalar {
    Decoded::new(data).get(0)
}
