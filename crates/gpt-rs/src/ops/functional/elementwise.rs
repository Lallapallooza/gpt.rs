//! Elementwise arithmetic on tensors of equal shape. The same functionals are also
//! [`DeviceTensorOps`] methods (`a.add(&b)?`).

use anyhow::Result;

use crate::backend::spec::PortableBackend;
use crate::tensor::DeviceTensor;
use crate::{capture, ensure_same_backend, functional};

/// `lhs + rhs`.
#[functional]
pub fn add(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor> {
    ensure_same_backend!(lhs, rhs);
    capture!(|lhs, rhs| lhs.try_add(&rhs)?)
}

/// `lhs - rhs`.
#[functional]
pub fn sub(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor> {
    ensure_same_backend!(lhs, rhs);
    capture!(|lhs, rhs| lhs.try_sub(&rhs)?)
}

/// `lhs * rhs`.
#[functional]
pub fn mul(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor> {
    ensure_same_backend!(lhs, rhs);
    capture!(|lhs, rhs| lhs.try_mul(&rhs)?)
}

/// `lhs / rhs`.
#[functional]
pub fn div(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor> {
    ensure_same_backend!(lhs, rhs);
    capture!(|lhs, rhs| lhs.try_div(&rhs)?)
}

/// `max(lhs, rhs)`.
#[functional]
pub fn maximum(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor> {
    ensure_same_backend!(lhs, rhs);
    capture!(|lhs, rhs| lhs.try_maximum(&rhs)?)
}

/// `min(lhs, rhs)`.
#[functional]
pub fn minimum(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor> {
    ensure_same_backend!(lhs, rhs);
    capture!(|lhs, rhs| lhs.try_minimum(&rhs)?)
}

/// `-x`.
#[functional]
pub fn neg(x: &Tensor) -> Result<Tensor> {
    capture!(|x| x.try_neg()?)
}

/// `|x|`.
#[functional]
pub fn abs(x: &Tensor) -> Result<Tensor> {
    capture!(|x| x.try_abs()?)
}

/// `min(max(x, min), max)`. Either bound is optional. Returns `x` itself when both bounds are
/// absent.
#[functional]
pub fn clamp(x: &Tensor, min: Option<&Tensor>, max: Option<&Tensor>) -> Result<Tensor> {
    if let Some(min) = min {
        ensure_same_backend!(x, min);
    }
    if let Some(max) = max {
        ensure_same_backend!(x, max);
    }
    match (min, max) {
        (Some(min), Some(max)) => capture!(|x, min, max| {
            let lower = x.try_maximum(&min)?;
            lower.try_minimum(&max)?
        }),
        (Some(min), None) => capture!(|x, min| x.try_maximum(&min)?),
        (None, Some(max)) => capture!(|x, max| x.try_minimum(&max)?),
        (None, None) => Ok(x.clone()),
    }
}

/// Extension trait exposing the elementwise functionals and [`matmul`](super::matmul) as methods.
pub trait DeviceTensorOps<B: PortableBackend + 'static>: Sized {
    fn add(&self, rhs: &Self) -> Result<Self>;
    fn sub(&self, rhs: &Self) -> Result<Self>;
    fn mul(&self, rhs: &Self) -> Result<Self>;
    fn div(&self, rhs: &Self) -> Result<Self>;
    fn maximum(&self, rhs: &Self) -> Result<Self>;
    fn minimum(&self, rhs: &Self) -> Result<Self>;
    fn neg(&self) -> Result<Self>;
    fn abs(&self) -> Result<Self>;
    fn clamp(&self, min: Option<&Self>, max: Option<&Self>) -> Result<Self>;
    fn matmul(&self, rhs: &Self) -> Result<Self>;
}

impl<B: PortableBackend + 'static> DeviceTensorOps<B> for DeviceTensor<B> {
    fn add(&self, rhs: &Self) -> Result<Self> {
        add(self, rhs)
    }

    fn sub(&self, rhs: &Self) -> Result<Self> {
        sub(self, rhs)
    }

    fn mul(&self, rhs: &Self) -> Result<Self> {
        mul(self, rhs)
    }

    fn div(&self, rhs: &Self) -> Result<Self> {
        div(self, rhs)
    }

    fn maximum(&self, rhs: &Self) -> Result<Self> {
        maximum(self, rhs)
    }

    fn minimum(&self, rhs: &Self) -> Result<Self> {
        minimum(self, rhs)
    }

    fn neg(&self) -> Result<Self> {
        neg(self)
    }

    fn abs(&self) -> Result<Self> {
        abs(self)
    }

    fn clamp(&self, min: Option<&Self>, max: Option<&Self>) -> Result<Self> {
        clamp(self, min, max)
    }

    fn matmul(&self, rhs: &Self) -> Result<Self> {
        super::matmul(self, rhs)
    }
}
