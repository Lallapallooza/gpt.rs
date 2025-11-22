//! Defines the scalar element trait implemented by host-side tensors.

use std::ops::{Add, Div, Mul, Sub};

/// Trait describing numeric behaviour required by tensor storages.
///
/// Implementations must provide zero/one constructors alongside exponential and logarithmic
/// helpers so higher-level math kernels can run generically over the element type.
pub trait StorageElement:
    Copy
    + Default
    + Send
    + Sync
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
{
    /// Returns the additive identity for the element type.
    fn zero() -> Self;
    /// Returns the multiplicative identity for the element type.
    fn one() -> Self;
    /// Applies the natural exponential function element-wise.
    fn exp(self) -> Self;
    /// Applies the natural logarithm element-wise.
    fn ln(self) -> Self;
    /// Converts from a 32-bit float into this element type.
    fn from_f32(v: f32) -> Self;
    /// Converts the element into a 32-bit float for interoperability.
    fn to_f32(self) -> f32;
}

impl StorageElement for f32 {
    fn zero() -> Self {
        0.0
    }

    fn one() -> Self {
        1.0
    }

    fn exp(self) -> Self {
        f32::exp(self)
    }

    fn ln(self) -> Self {
        f32::ln(self)
    }

    fn from_f32(v: f32) -> Self {
        v
    }

    fn to_f32(self) -> f32 {
        self
    }
}
