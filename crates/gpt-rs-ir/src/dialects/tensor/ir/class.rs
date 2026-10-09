//! The classes of dtypes that the type rules accept, as the spec's dtype table names them.

use anyhow::{ensure, Result};

use crate::types::{DType, TensorType};

/// A class of dtypes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Class {
    /// `i1`.
    Bool,
    /// The integers, `si4` to `ui64`.
    Int,
    /// bf16, f16, f32 and f64: the floats with arithmetic.
    Float,
    /// The floats narrower than 16 bits, which quantized and FP8 tensors hold.
    Narrow,
    /// `e8m0`, the scales of microscaling formats.
    Scale,
}

impl Class {
    pub fn of(dtype: DType) -> Class {
        match dtype {
            DType::I1 => Class::Bool,
            DType::E8M0 => Class::Scale,
            DType::Bf16 | DType::F16 | DType::F32 | DType::F64 => Class::Float,
            other if other.is_float() => Class::Narrow,
            _ => Class::Int,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Class::Bool => "i1",
            Class::Int => "integer",
            Class::Float => "float",
            Class::Narrow => "narrow float",
            Class::Scale => "e8m0",
        }
    }
}

/// Integers and floats: the dtypes with arithmetic.
pub const ARITHMETIC: &[Class] = &[Class::Int, Class::Float];
/// `i1` and integers: the dtypes with bitwise logic.
pub const LOGIC: &[Class] = &[Class::Bool, Class::Int];
pub const INTEGERS: &[Class] = &[Class::Int];
pub const FLOATS: &[Class] = &[Class::Float];
/// The dtypes that order their values.
pub const ORDERED: &[Class] = &[Class::Bool, Class::Int, Class::Float];

/// Checks that `ty` holds elements of one of `classes`. `what` names it in the message.
pub fn ensure_class(what: &str, ty: &TensorType, classes: &[Class]) -> Result<()> {
    ensure!(
        classes.contains(&Class::of(ty.dtype)),
        "{what} must be {} tensor, got {ty}",
        describe(classes)
    );
    Ok(())
}

/// Checks that `dtype` is of one of `classes`. `what` names it in the message.
pub fn ensure_dtype_class(what: &str, dtype: DType, classes: &[Class]) -> Result<()> {
    ensure!(
        classes.contains(&Class::of(dtype)),
        "{what} must be {} dtype, got {dtype}",
        describe(classes)
    );
    Ok(())
}

/// `classes` as an indefinite noun phrase: "an integer or float".
fn describe(classes: &[Class]) -> String {
    let names: Vec<&str> = classes.iter().map(|class| class.name()).collect();
    let article = match names.first().and_then(|name| name.chars().next()) {
        Some('a' | 'e' | 'i' | 'o' | 'u') => "an",
        _ => "a",
    };
    let list = match names.as_slice() {
        [] => String::new(),
        [one] => one.to_string(),
        [rest @ .., last] => format!("{} or {last}", rest.join(", ")),
    };
    format!("{article} {list}")
}
