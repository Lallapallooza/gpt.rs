//! Normalization primitives including layer norm expressed through portable graphs.
//!
//! Results expose intermediate statistics for reuse and debugging without recomputing reductions.

use anyhow::Result;

use crate::backend::spec::PortableBackend;
use crate::ops::ptir::scalar_broadcast;
use crate::tensor::DeviceTensor;
use crate::{
    capture, ensure_dtype, ensure_last_dim, ensure_rank, ensure_rank_at_least, ensure_same_backend,
    ensure_same_dtype, functional,
};

/// Outputs produced by [`layer_norm`], including cached intermediates.
pub struct LayerNormResult<B: PortableBackend + 'static> {
    pub output: DeviceTensor<B>,
    pub normalized: DeviceTensor<B>,
    pub mean: DeviceTensor<B>,
    pub inv_std: DeviceTensor<B>,
}

/// Outputs produced by [`rms_norm`], including cached intermediates.
pub struct RmsNormResult<B: PortableBackend + 'static> {
    pub output: DeviceTensor<B>,
    pub normalized: DeviceTensor<B>,
    pub inv_rms: DeviceTensor<B>,
}

/// Applies layer normalization across the last tensor dimension.
///
/// The captured graph mirrors the textbook algorithm:
/// - compute the per-sample mean by summing across the normalized axis and scaling by `1/N`;
/// - subtract the mean to centre the activations and accumulate squared deviations;
/// - compute variance, add `eps`, and take the reciprocal square root to obtain `1/std`;
/// - multiply the centred activations by `1/std` to produce the normalized tensor;
/// - broadcast the affine parameters (`gamma`, `beta`) and apply the final scaling and shift.
///
/// Intermediate tensors (mean, inv_std, normalized) are returned alongside the final output so
/// downstream consumers can reuse them without recomputing reductions.
#[functional]
pub fn layer_norm(
    x: &Tensor,
    gamma: &Tensor,
    beta: &Tensor,
    eps: f32,
) -> Result<LayerNormResult<B>> {
    ensure_same_dtype!(x, gamma, beta);
    ensure_dtype!(x, F32);
    ensure_rank_at_least!(x, 1);
    ensure_same_backend!(x, gamma, beta);
    let input_shape = x.shape().dims().to_vec();
    let axis = input_shape.len() - 1;
    let features = input_shape[axis];
    ensure_rank!(gamma, 1);
    ensure_rank!(beta, 1);
    ensure_last_dim!(gamma, features);
    ensure_last_dim!(beta, features);
    let mut reduce_shape = input_shape.clone();
    reduce_shape[axis] = 1;

    let (output, normalized, mean, inv_std) = capture!(session, |x, gamma, beta| {
        // Mean reduction across the last dimension.
        let sum = x.reduce_sum(vec![axis], true);
        let inv_count = scalar_broadcast(&session, 1.0f32 / features as f32, &reduce_shape);
        let mean = sum * inv_count;
        let mean_broadcast = mean.broadcast_to(input_shape.clone());

        // Variance computation (centered squared values) plus epsilon stabilisation.
        let centered = x - mean_broadcast;
        let centered_sq = centered * centered;
        let var_sum = centered_sq.reduce_sum(vec![axis], true);
        let var_mean = var_sum * inv_count;
        let var_eps = var_mean + scalar_broadcast(&session, eps, &reduce_shape);
        let inv_std = var_eps.rsqrt();
        let inv_std_broadcast = inv_std.broadcast_to(input_shape.clone());
        let normalized = centered * inv_std_broadcast;

        // Apply affine transform: broadcast gamma/beta to the full shape and scale/shift.
        let gamma_broadcast = gamma.broadcast_to(input_shape.clone());
        let beta_broadcast = beta.broadcast_to(input_shape);
        let output = normalized * gamma_broadcast + beta_broadcast;
        (output, normalized, mean, inv_std)
    })?;
    Ok(LayerNormResult {
        output,
        normalized,
        mean,
        inv_std,
    })
}

/// Applies RMS normalization across the last tensor dimension.
///
/// The captured graph computes:
/// - squared activations and their mean along the last axis;
/// - reciprocal root mean square via `rsqrt(mean(x^2) + eps)`;
/// - normalized output `x * inv_rms`;
/// - affine scale by `gamma`.
#[functional]
pub fn rms_norm(x: &Tensor, gamma: &Tensor, eps: f32) -> Result<RmsNormResult<B>> {
    ensure_same_dtype!(x, gamma);
    ensure_dtype!(x, F32);
    ensure_rank_at_least!(x, 1);
    ensure_same_backend!(x, gamma);
    let input_shape = x.shape().dims().to_vec();
    let axis = input_shape.len() - 1;
    let features = input_shape[axis];
    ensure_rank!(gamma, 1);
    ensure_last_dim!(gamma, features);
    let mut reduce_shape = input_shape.clone();
    reduce_shape[axis] = 1;

    let (output, normalized, inv_rms) = capture!(session, |x, gamma| {
        let squared = x * x;
        let sum = squared.reduce_sum(vec![axis], true);
        let inv_count = scalar_broadcast(&session, 1.0f32 / features as f32, &reduce_shape);
        let mean_square = sum * inv_count;
        let inv_rms = (mean_square + scalar_broadcast(&session, eps, &reduce_shape)).rsqrt();
        let normalized = x * inv_rms.broadcast_to(input_shape.clone());
        let gamma_broadcast = gamma.broadcast_to(input_shape);
        let output = normalized * gamma_broadcast;
        (output, normalized, inv_rms)
    })?;
    Ok(RmsNormResult {
        output,
        normalized,
        inv_rms,
    })
}
