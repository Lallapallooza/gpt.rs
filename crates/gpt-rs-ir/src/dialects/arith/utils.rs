//! Literals of the `arith` dialect, their values and their text form, the value of
//! `arith.constant`, and the builder of scalar code.

use anyhow::{anyhow, ensure, Result};

use super::ir::*;
use crate::host::{HostScalar, ScalarValue};
use crate::ir::{
    print_float, ConstantValue, Created, DType, Dense, InOp, Module, Number, Operands, Value,
    Values,
};
use crate::rewrite::Build;

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

impl ConstantValue for Constant {
    fn value(&self) -> &Dense {
        self.value.dense()
    }
}

/// The dtypes that `arith.fma` and its lane-wise form compute on: those for which the interpreter
/// computes `lhs * rhs + acc` with one rounding (`mul_add`).
pub(crate) const FMA_DTYPES: &[DType] = &[DType::F32, DType::F64];

/// The `arith.mul` that gives `value`, when nothing else uses the product and `arith.fma` computes
/// on its dtype: a product that a numerics contract lets round once with the sum it enters.
pub(crate) fn lone_product(module: &Module, value: Value) -> Option<MulOp<'_>> {
    let product = module.value_ref(value);
    let dtype = product.value_type().as_scalar()?;
    if !product.has_one_use() || !FMA_DTYPES.contains(&dtype) {
        return None;
    }
    product.defining_op::<Mul>()
}

/// Whether `dtype` is one that `arith.fma` and its lane-wise form compute on.
pub(crate) fn ensure_fma(dtype: DType) -> Result<()> {
    ensure!(
        FMA_DTYPES.contains(&dtype),
        "fma computes on f32 and f64, not {dtype}"
    );
    Ok(())
}

/// Whether a bitcast from `from` to `to` keeps the width, as `arith.bitcast` and its lane-wise
/// form require.
pub(crate) fn ensure_bitcast(from: DType, to: DType) -> Result<()> {
    ensure!(
        from != DType::I1 && to != DType::I1,
        "i1 has no bits to reinterpret"
    );
    ensure!(
        from.bits() == to.bits(),
        "a bitcast keeps the width: {} has {} bits and {} has {}",
        from,
        from.bits(),
        to,
        to.bits()
    );
    Ok(())
}

/// Scalar code while it is built, through a `Builder`, such as the body of a `tile.block`, or
/// through a pattern's `Rewriter`. `create` creates one op, and most other methods build a
/// constant or a value from arith ops. An op its type rule rejects is a bug in the caller, and
/// panics.
pub struct ScalarBuilder<'b, B> {
    pub builder: &'b mut B,
}

impl<B: Build> ScalarBuilder<'_, B> {
    /// The dtype of the scalar `value`.
    pub fn dtype(&self, value: Value) -> DType {
        self.builder
            .value_type(value)
            .as_scalar()
            .expect("a block body computes on scalars")
    }

    /// The constant `value`.
    #[track_caller]
    pub fn lit(&mut self, value: HostScalar) -> Value {
        let literal = Literal::new(value).expect("a scalar is a literal");
        self.builder.create(Constant { value: literal }, ())
    }

    /// The float constant `value` of `dtype`, rounded to it.
    #[track_caller]
    pub fn float(&mut self, dtype: DType, value: f64) -> Value {
        let scalar = HostScalar::from_f64(dtype, value).expect("a float dtype");
        self.lit(scalar)
    }

    /// The integer constant `value` of `dtype`.
    #[track_caller]
    pub fn int(&mut self, dtype: DType, value: i128) -> Value {
        let scalar = HostScalar::int(dtype, value).expect("an integer dtype");
        self.lit(scalar)
    }

    /// Creates `op` on `operands`, and gives its results.
    #[track_caller]
    pub fn create<T: InOp>(
        &mut self,
        op: T,
        operands: impl Into<Operands<T, Values>>,
    ) -> Created<T> {
        self.builder.create(op, operands)
    }

    /// `value` converted to `to`, or `value` itself when it is a `to` already.
    #[track_caller]
    pub fn cast(&mut self, value: Value, to: DType) -> Value {
        if self.dtype(value) == to {
            return value;
        }
        self.builder.create(Cast { to }, value)
    }

    /// The bits of `value` as a `to`, of equal width, or `value` itself when it is a `to`
    /// already.
    #[track_caller]
    pub fn bitcast(&mut self, value: Value, to: DType) -> Value {
        if self.dtype(value) == to {
            return value;
        }
        self.builder.create(Bitcast { to }, value)
    }

    /// Whether `value` is NaN: whether it differs from itself.
    pub fn is_nan(&mut self, value: Value) -> Value {
        self.create(NotEqual, (value, value))
    }

    /// `value` rounded to an integer, ties to even, in its own float dtype: adding and subtracting
    /// `2^p`, `p` the dtype's fraction bits, rounds a magnitude below it, and one at or above it is
    /// whole already.
    pub fn round_even(&mut self, value: Value) -> Value {
        let dtype = self.dtype(value);
        let significand_bits = dtype
            .significand_bits()
            .unwrap_or_else(|| panic!("round_even rounds a float, not {dtype}"));
        let magnitude = self.create(Abs, value);
        let first_whole = self.float(dtype, 2f64.powi(significand_bits as i32 - 1));
        let shifted = self.create(Add, (magnitude, first_whole));
        let below_whole = self.create(Sub, (shifted, first_whole));
        let whole = self.create(GreaterEqual, (magnitude, first_whole));
        let rounded = self.create(Select, (whole, magnitude, below_whole));
        let negative = self.sign_bit(value);
        let negated = self.create(Neg, rounded);
        self.create(Select, (negative, negated, rounded))
    }

    /// The largest integer at or below `value`, for `0 <= value < 2^p`, `p` the dtype's fraction
    /// bits, where `round_even` is exact.
    pub fn floor_nonnegative(&mut self, value: Value) -> Value {
        let dtype = self.dtype(value);
        let rounded = self.round_even(value);
        let above = self.create(Greater, (rounded, value));
        let one = self.float(dtype, 1.0);
        let lower = self.create(Sub, (rounded, one));
        self.create(Select, (above, lower, rounded))
    }

    /// Whether the sign bit of the float `value` is set, -0.0 and negative NaNs included.
    pub fn sign_bit(&mut self, value: Value) -> Value {
        let width = self.dtype(value).bits();
        let unsigned = DType::unsigned(width).expect("an integer has the float's width");
        let bits = self.bitcast(value, unsigned);
        let shift = self.int(unsigned, width as i128 - 1);
        let top = self.create(ShiftRightLogical, (bits, shift));
        let one = self.int(unsigned, 1);
        self.create(Equal, (top, one))
    }

    /// The exponent field of the float `value` minus its bias, in the signed integer of its width:
    /// `floor(log2(value))` for a normal `value > 0`.
    pub fn exponent(&mut self, value: Value) -> Value {
        let dtype = self.dtype(value);
        let format = dtype
            .float_format()
            .unwrap_or_else(|| panic!("exponent reads a float, not {dtype}"));
        let unsigned = DType::unsigned(dtype.bits()).expect("an integer has the float's width");
        let signed = DType::signed(dtype.bits()).expect("an integer has the float's width");
        let bits = self.bitcast(value, unsigned);
        let shift = self.int(unsigned, i128::from(format.man_bits));
        let shifted = self.create(ShiftRightLogical, (bits, shift));
        let mask = self.int(unsigned, (1 << format.exp_bits) - 1);
        let field = self.create(And, (shifted, mask));
        let exponent = self.cast(field, signed);
        let bias = self.int(signed, i128::from(format.bias));
        self.create(Sub, (exponent, bias))
    }

    /// `2^exponent` as a `dtype`, f32 or f64, for an `exponent` of a normal `dtype` in the signed
    /// integer of its width.
    pub fn pow2(&mut self, dtype: DType, exponent: Value) -> Value {
        let format = dtype
            .float_format()
            .unwrap_or_else(|| panic!("pow2 builds a float, not {dtype}"));
        let signed = self.dtype(exponent);
        let bias = self.int(signed, i128::from(format.bias));
        let biased = self.create(Add, (exponent, bias));
        let shift = self.int(signed, i128::from(format.man_bits));
        let bits = self.create(ShiftLeft, (biased, shift));
        self.bitcast(bits, dtype)
    }

    /// The value of `values` at `position`, an integer below their count.
    pub fn pick(&mut self, position: Value, values: &[Value]) -> Value {
        let index_dtype = self.dtype(position);
        let (&last, earlier) = values.split_last().expect("a pick has values");
        let mut picked = last;
        for (at, &candidate) in earlier.iter().enumerate().rev() {
            let at_value = self.int(index_dtype, at as i128);
            let here = self.create(Equal, (position, at_value));
            picked = self.create(Select, (here, candidate, picked));
        }
        picked
    }
}
