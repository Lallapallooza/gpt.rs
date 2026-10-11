//! Literals of the `arith` dialect, their values and their text form, and the value of
//! `arith.constant`.

use anyhow::{bail, ensure, Result};

use super::ir::*;
use crate::attr::{print_float, Attr};
use crate::dense::Dense;
use crate::host::{HostScalar, ScalarValue};
use crate::op::ConstantOp;
use crate::text::Cursor;
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

impl Attr for Literal {
    fn print(&self, out: &mut String) {
        let scalar = self.scalar();
        match scalar.value {
            ScalarValue::Int(x) => out.push_str(&x.to_string()),
            ScalarValue::F32(x) => print_float(x, out),
            ScalarValue::F64(x) => print_float(x, out),
        }
        out.push_str(" : ");
        out.push_str(&scalar.dtype.to_string());
    }

    fn parse(c: &mut Cursor<'_>) -> Result<Self> {
        let text = c.number()?;
        c.expect(":")?;
        let dtype = DType::parse(c)?;
        let is_float = text.contains(['.', 'e', 'E', 'i', 'N']);
        let scalar = if dtype == DType::I1 || dtype.int_format().is_some() {
            if is_float {
                bail!(c.error(format!("`{text}` is not a {dtype} value")));
            }
            let value: i128 = text
                .parse()
                .map_err(|err| c.error(format!("bad integer `{text}`: {err}")))?;
            let scalar = HostScalar::int(dtype, value)?;
            ensure!(
                scalar.as_int() == Some(value),
                c.error(format!("{value} is not a {dtype} value"))
            );
            scalar
        } else if dtype == DType::F64 {
            let value: f64 = text
                .parse()
                .map_err(|_| c.error(format!("bad float `{text}`")))?;
            HostScalar::from_f64(dtype, value)?
        } else {
            // The text of an f32-computed value is its shortest f32 decimal, so it reads as f32.
            let value: f32 = text
                .parse()
                .map_err(|_| c.error(format!("bad float `{text}`")))?;
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
