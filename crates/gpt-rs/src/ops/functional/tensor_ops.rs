//! Miscellaneous tensor helpers built on top of the lazy graph interface.
//!
//! These routines combine portable broadcasting and elementwise primitives to implement
//! higher-level conveniences like bias addition.

use anyhow::Result;

use crate::tensor::DType;
use crate::{
    capture, ensure_dtype, ensure_last_dim, ensure_rank, ensure_rank_at_least, ensure_same_backend,
    ensure_same_dtype, functional,
};

/// Adds a bias vector to the last dimension of `x`, broadcasting it over the other axes.
#[functional]
pub fn add_bias(x: &Tensor, bias: &Tensor) -> Result<Tensor> {
    ensure_same_dtype!(x, bias);
    ensure_rank_at_least!(x, 1);
    ensure_rank!(bias, 1);
    let shape = x.shape().dims().to_vec();
    ensure_last_dim!(bias, shape[shape.len() - 1]);
    ensure_same_backend!(x, bias);
    capture!(|x, bias| {
        let bias_broadcast = bias.broadcast_to(shape);
        x + bias_broadcast
    })
}

/// Converts `x` to `dtype` with a PTIR `cast`. Float-to-float casts round to nearest even. Returns
/// `x` unchanged when it already has the requested dtype.
#[functional]
pub fn cast(x: &Tensor, dtype: DType) -> Result<Tensor> {
    if x.dtype() == dtype {
        return Ok(x.clone());
    }
    let target = crate::tensor::spec_utils::backend_dtype(dtype);
    capture!(|x| x.cast(target))
}

/// Multiplies `x` by a vector broadcast along its last dimension (`x * scale[None, ..., :]`).
#[functional]
pub fn mul_last_dim(x: &Tensor, scale: &Tensor) -> Result<Tensor> {
    ensure_same_dtype!(x, scale);
    ensure_rank_at_least!(x, 1);
    ensure_rank!(scale, 1);
    let shape = x.shape().dims().to_vec();
    ensure_last_dim!(scale, shape[shape.len() - 1]);
    ensure_same_backend!(x, scale);
    capture!(|x, scale| {
        let scale_broadcast = scale.broadcast_to(shape);
        x * scale_broadcast
    })
}

/// Elementwise `x + scalar` for f32 `x`.
#[functional]
pub fn add_scalar(x: &Tensor, scalar: f32) -> Result<Tensor> {
    ensure_dtype!(x, F32);
    capture!(|x| x.add_scalar(scalar))
}

/// Elementwise `exp(x)` for f32 `x`.
#[functional]
pub fn exp(x: &Tensor) -> Result<Tensor> {
    ensure_dtype!(x, F32);
    capture!(|x| x.exp())
}
