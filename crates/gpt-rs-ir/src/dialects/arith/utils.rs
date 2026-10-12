//! Literals of the `arith` dialect, their values and their text form, and the value of
//! `arith.constant`.

use anyhow::{anyhow, ensure, Result};

use super::ir::*;
use crate::attr::{print_float, Number};
use crate::dense::Dense;
use crate::host::{HostScalar, ScalarValue};
use crate::op::ConstantOp;
use crate::types::DType;

impl Literal {
    pub fn new(value: HostScalar) -> Result<Self> {
        Ok(Literal(value.to_dense()?))
    }

    /// The literal as a rank-0 dense constant.
    pub fn dense(&self) -> &Dense {
        &self.0
    }

    pub fn dtype(&self) -> DType {
        self.0.dtype
    }

    /// The value of the literal.
    pub fn scalar(&self) -> HostScalar {
        HostScalar::from_dense(&self.0).expect("a literal holds one element of its dtype")
    }
}

/// The text form of a literal: its value, then its dtype, `0.5 : f32`.
#[derive(crate::Attr, Debug, Clone, PartialEq, Eq, Hash)]
#[attr(format = "$value `:` $dtype")]
pub(super) struct LiteralText {
    value: Number,
    dtype: DType,
}

impl From<&Literal> for LiteralText {
    fn from(literal: &Literal) -> Self {
        let scalar = literal.scalar();
        let mut value = String::new();
        match scalar.value {
            ScalarValue::Int(x) => value.push_str(&x.to_string()),
            ScalarValue::F32(x) => print_float(x, &mut value),
            ScalarValue::F64(x) => print_float(x, &mut value),
        }
        LiteralText {
            value: Number(value),
            dtype: scalar.dtype,
        }
    }
}

impl TryFrom<LiteralText> for Literal {
    type Error = anyhow::Error;

    fn try_from(text: LiteralText) -> Result<Self> {
        let (dtype, number) = (text.dtype, &text.value.0);
        let scalar = if dtype == DType::I1 || dtype.int_format().is_some() {
            ensure!(!text.value.is_float(), "`{number}` is not a {dtype} value");
            let value: i128 = number
                .parse()
                .map_err(|err| anyhow!("bad integer `{number}`: {err}"))?;
            let scalar = HostScalar::int(dtype, value)?;
            ensure!(
                scalar.as_int() == Some(value),
                "{value} is not a {dtype} value"
            );
            scalar
        } else if dtype == DType::F64 {
            let value: f64 = number
                .parse()
                .map_err(|_| anyhow!("bad float `{number}`"))?;
            HostScalar::from_f64(dtype, value)?
        } else {
            // The text of an f32-computed value is its shortest f32 decimal, so it reads as f32.
            let value: f32 = number
                .parse()
                .map_err(|_| anyhow!("bad float `{number}`"))?;
            HostScalar::from_f64(dtype, f64::from(value))?
        };
        Literal::new(scalar)
    }
}

impl ConstantOp for Constant {
    fn value(&self) -> &Dense {
        self.value.dense()
    }
}
