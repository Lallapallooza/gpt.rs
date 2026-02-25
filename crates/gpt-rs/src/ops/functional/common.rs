//! Shared helpers backing the functional tensor API.
//!
//! These utilities provide trait adapters so `DeviceTensor` instances can call math helpers like
//! `tensor.add(&other)` as well as validation and graph-resolution routines reused across
//! functional kernels.

use std::sync::Arc;

use anyhow::{anyhow, bail, ensure, Result};
use gpt_rs_macros::capture_ptir;

use crate::backend::spec::{PortableBackend, ValueId};
use crate::ops::graph::GraphArena;
use crate::ops::ptir;
pub(crate) use crate::ops::ptir::scalar_broadcast;
use crate::tensor::spec_utils::{frontend_dtype, shape_from_spec};
use crate::tensor::{DType as TensorDType, DeviceTensor};

/// Builds numerically-stable softmax over a selected axis using PTIR tensor ops.
///
/// This helper centralizes softmax decomposition so all callsites emit the same IR pattern.
pub(crate) fn softmax_last_axis_ptir<'ctx, 'gb, B: PortableBackend + 'static>(
    input: &ptir::Tensor<'ctx, 'gb, B>,
    axis: usize,
) -> ptir::Tensor<'ctx, 'gb, B> {
    let max = input.reduce_max([axis], true);
    let shifted = *input - max.broadcast_like(input);
    let exp_values = shifted.exp();
    let sum = exp_values.reduce_sum([axis], true);
    exp_values / sum.broadcast_like(input)
}

/// Extension trait exposing ergonomic math helpers on device tensors.
pub trait DeviceTensorOps<B: PortableBackend + 'static>: Sized {
    /// Elementwise addition of two tensors, returning a lazily captured result.
    fn add(&self, rhs: &Self) -> Result<Self>;
    /// Elementwise subtraction that defers execution to the backend program builder.
    fn sub(&self, rhs: &Self) -> Result<Self>;
    /// Elementwise multiplication of two tensors.
    fn mul(&self, rhs: &Self) -> Result<Self>;
    /// Elementwise division with validation of dtype and shape.
    fn div(&self, rhs: &Self) -> Result<Self>;
    /// Elementwise maximum that reuses the binary op helper.
    fn maximum(&self, rhs: &Self) -> Result<Self>;
    /// Elementwise minimum blending two tensors element by element.
    fn minimum(&self, rhs: &Self) -> Result<Self>;
    /// Unary negation expressed as a portable unary op.
    fn neg(&self) -> Result<Self>;
    /// Elementwise absolute value.
    fn abs(&self) -> Result<Self>;
    /// Elementwise clamp with optional lower and upper bounds.
    fn clamp(&self, min: Option<&Self>, max: Option<&Self>) -> Result<Self>;
    /// Batched matrix multiplication with broadcasting-aware validation.
    fn matmul(&self, rhs: &Self) -> Result<Self>;
}

impl<B: PortableBackend + 'static> DeviceTensorOps<B> for DeviceTensor<B> {
    fn add(&self, rhs: &Self) -> Result<Self> {
        ensure_same_backend("add", self, rhs)?;
        capture_ptir!({ lhs = self, rhs }, |_session| {
            let result = lhs.try_add(&rhs)?;
            Ok(result.id())
        })?
        .into_device_tensor()
    }

    fn sub(&self, rhs: &Self) -> Result<Self> {
        ensure_same_backend("sub", self, rhs)?;
        capture_ptir!({ lhs = self, rhs }, |_session| {
            let result = lhs.try_sub(&rhs)?;
            Ok(result.id())
        })?
        .into_device_tensor()
    }

    fn mul(&self, rhs: &Self) -> Result<Self> {
        ensure_same_backend("mul", self, rhs)?;
        capture_ptir!({ lhs = self, rhs }, |_session| {
            let result = lhs.try_mul(&rhs)?;
            Ok(result.id())
        })?
        .into_device_tensor()
    }

    fn div(&self, rhs: &Self) -> Result<Self> {
        ensure_same_backend("div", self, rhs)?;
        capture_ptir!({ lhs = self, rhs }, |_session| {
            let result = lhs.try_div(&rhs)?;
            Ok(result.id())
        })?
        .into_device_tensor()
    }

    fn maximum(&self, rhs: &Self) -> Result<Self> {
        ensure_same_backend("maximum", self, rhs)?;
        capture_ptir!({ lhs = self, rhs }, |_session| {
            let result = lhs.try_maximum(&rhs)?;
            Ok(result.id())
        })?
        .into_device_tensor()
    }

    fn minimum(&self, rhs: &Self) -> Result<Self> {
        ensure_same_backend("minimum", self, rhs)?;
        capture_ptir!({ lhs = self, rhs }, |_session| {
            let result = lhs.try_minimum(&rhs)?;
            Ok(result.id())
        })?
        .into_device_tensor()
    }

    fn neg(&self) -> Result<Self> {
        capture_ptir!({ input = self }, |_session| {
            let result = input.try_neg()?;
            Ok(result.id())
        })?
        .into_device_tensor()
    }

    fn abs(&self) -> Result<Self> {
        capture_ptir!({ input = self }, |_session| {
            let result = input.try_abs()?;
            Ok(result.id())
        })?
        .into_device_tensor()
    }

    fn clamp(&self, min: Option<&Self>, max: Option<&Self>) -> Result<Self> {
        capture_clamp(self, min, max)
    }

    fn matmul(&self, rhs: &Self) -> Result<Self> {
        let backend = ensure_same_backend("matmul", self, rhs)?;
        crate::ops::functional::linalg::matmul(backend.as_ref(), self, rhs)
    }
}

/// Verifies that all operands live on the same backend instance and returns it.
pub(crate) fn ensure_same_backend<B: PortableBackend + 'static>(
    op_name: &str,
    lhs: &DeviceTensor<B>,
    rhs: &DeviceTensor<B>,
) -> Result<Arc<B>> {
    let lhs_backend = lhs.backend();
    let rhs_backend = rhs.backend();

    if !Arc::ptr_eq(&lhs_backend, &rhs_backend) {
        bail!(
            "{} operands must be placed on the same backend: {} vs {}",
            op_name,
            lhs_backend.backend_name(),
            rhs_backend.backend_name()
        );
    }

    Ok(lhs_backend)
}

/// Ensures two tensors share the same dtype.
pub(crate) fn ensure_same_dtype<B: PortableBackend + 'static>(
    lhs_name: &str,
    lhs: &DeviceTensor<B>,
    rhs_name: &str,
    rhs: &DeviceTensor<B>,
) -> Result<()> {
    ensure!(
        lhs.dtype() == rhs.dtype(),
        "{lhs_name} dtype {:?} must match {rhs_name} dtype {:?}",
        lhs.dtype(),
        rhs.dtype()
    );
    Ok(())
}

/// Ensures a tensor has the expected rank, returning a helpful error otherwise.
pub(crate) fn ensure_rank<B: PortableBackend + 'static>(
    tensor_name: &str,
    tensor: &DeviceTensor<B>,
    expected_rank: usize,
) -> Result<()> {
    ensure!(
        tensor.shape().rank() == expected_rank,
        "{tensor_name} must have rank {expected_rank}, got {:?}",
        tensor.shape().dims()
    );
    Ok(())
}

/// Validates that the final dimension matches an expected size.
pub(crate) fn ensure_last_dim<B: PortableBackend + 'static>(
    tensor_name: &str,
    tensor: &DeviceTensor<B>,
    expected: usize,
) -> Result<()> {
    ensure!(
        tensor
            .shape()
            .dims()
            .last()
            .copied()
            .ok_or_else(|| anyhow!("{tensor_name} missing trailing dimension data"))?
            == expected,
        "{tensor_name} last dimension must be {expected}, got {:?}",
        tensor.shape().dims()
    );
    Ok(())
}

/// Verifies a tensor rank is at least the requested minimum.
pub(crate) fn ensure_rank_at_least<B: PortableBackend + 'static>(
    tensor_name: &str,
    tensor: &DeviceTensor<B>,
    min_rank: usize,
) -> Result<()> {
    ensure!(
        tensor.shape().rank() >= min_rank,
        "{tensor_name} must have rank >= {min_rank}, got {:?}",
        tensor.shape().dims()
    );
    Ok(())
}

/// Confirms two tensors share identical shapes for portable kernels.
pub(crate) fn ensure_shape_matches<B: PortableBackend + 'static>(
    lhs_name: &str,
    lhs: &DeviceTensor<B>,
    rhs_name: &str,
    rhs: &DeviceTensor<B>,
) -> Result<()> {
    ensure!(
        lhs.shape().dims() == rhs.shape().dims(),
        "{lhs_name} shape {:?} must match {rhs_name} shape {:?}",
        lhs.shape().dims(),
        rhs.shape().dims()
    );
    Ok(())
}

/// Guards against dtype mismatches so PTIR programs remain well-defined.
pub(crate) fn ensure_dtype_equals<B: PortableBackend + 'static>(
    tensor_name: &str,
    tensor: &DeviceTensor<B>,
    expected: TensorDType,
) -> Result<()> {
    ensure!(
        tensor.dtype() == expected,
        "{tensor_name} must have dtype {:?}, got {:?}",
        expected,
        tensor.dtype()
    );
    Ok(())
}

/// Ensures two tensors match on every axis except an explicit concatenation axis.
pub(crate) fn ensure_dims_match_except_axis<B: PortableBackend + 'static>(
    lhs_name: &str,
    lhs: &DeviceTensor<B>,
    rhs_name: &str,
    rhs: &DeviceTensor<B>,
    axis: usize,
) -> Result<()> {
    let lhs_dims = lhs.shape().dims();
    let rhs_dims = rhs.shape().dims();
    ensure!(
        lhs_dims.len() == rhs_dims.len(),
        "{lhs_name} rank {} must match {rhs_name} rank {}",
        lhs_dims.len(),
        rhs_dims.len()
    );
    for (idx, (lhs_dim, rhs_dim)) in lhs_dims.iter().zip(rhs_dims.iter()).enumerate() {
        if idx != axis && lhs_dim != rhs_dim {
            bail!(
                "{lhs_name} dim {} ({}) must match {rhs_name} dim {} ({})",
                idx,
                lhs_dim,
                idx,
                rhs_dim
            );
        }
    }
    Ok(())
}

/// Ensures an axis index is valid for the provided tensor.
pub(crate) fn ensure_axis_in_bounds<B: PortableBackend + 'static>(
    tensor_name: &str,
    tensor: &DeviceTensor<B>,
    axis: usize,
) -> Result<()> {
    ensure!(
        axis < tensor.shape().rank(),
        "{} axis {} out of range for rank {}",
        tensor_name,
        axis,
        tensor.shape().rank()
    );
    Ok(())
}

/// Ensures a slice `[start, start + len)` fits within the selected axis length.
pub(crate) fn ensure_slice_within_bounds<B: PortableBackend + 'static>(
    tensor_name: &str,
    tensor: &DeviceTensor<B>,
    axis: usize,
    start: usize,
    len: usize,
) -> Result<()> {
    ensure!(
        start + len <= tensor.shape().dims()[axis],
        "{} slice [{}..{}) exceeds dimension {} (len {})",
        tensor_name,
        start,
        start + len,
        axis,
        tensor.shape().dims()[axis]
    );
    Ok(())
}

fn capture_clamp<B: PortableBackend + 'static>(
    x: &DeviceTensor<B>,
    min: Option<&DeviceTensor<B>>,
    max: Option<&DeviceTensor<B>>,
) -> Result<DeviceTensor<B>> {
    if let Some(min_tensor) = min {
        ensure_same_backend("clamp", x, min_tensor)?;
    }
    if let Some(max_tensor) = max {
        ensure_same_backend("clamp", x, max_tensor)?;
    }

    let capture_result = match (min, max) {
        (Some(min_tensor), Some(max_tensor)) => {
            capture_ptir!({ x, min_tensor, max_tensor }, |_session| {
                let mut current = x.try_maximum(&min_tensor)?;
                current = current.try_minimum(&max_tensor)?;
                Ok(current.id())
            })
        }
        (Some(min_tensor), None) => capture_ptir!({ x, min_tensor }, |_session| {
            let current = x.try_maximum(&min_tensor)?;
            Ok(current.id())
        }),
        (None, Some(max_tensor)) => capture_ptir!({ x, max_tensor }, |_session| {
            let current = x.try_minimum(&max_tensor)?;
            Ok(current.id())
        }),
        (None, None) => capture_ptir!({ x }, |_session| Ok(x.id())),
    };

    capture_result.into_device_tensor()
}

pub trait CaptureIntoDeviceTensor<B: PortableBackend + 'static> {
    fn into_device_tensor(self) -> Result<DeviceTensor<B>>;
}

impl<B: PortableBackend + 'static> CaptureIntoDeviceTensor<B> for (Arc<GraphArena<B>>, ValueId) {
    fn into_device_tensor(self) -> Result<DeviceTensor<B>> {
        let spec = self
            .0
            .tensor_spec_for(self.1)
            .ok_or_else(|| anyhow!("value {:?} missing tensor spec", self.1))?;
        let dtype = frontend_dtype(spec.dtype)?;
        let shape = shape_from_spec(&spec)?;
        DeviceTensor::from_lazy(self.0, shape, dtype, self.1)
    }
}

impl<B: PortableBackend + 'static> CaptureIntoDeviceTensor<B>
    for Result<(Arc<GraphArena<B>>, ValueId)>
{
    fn into_device_tensor(self) -> Result<DeviceTensor<B>> {
        self.and_then(|capture| capture.into_device_tensor())
    }
}
