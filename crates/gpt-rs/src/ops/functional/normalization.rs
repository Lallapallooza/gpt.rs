//! Layer, RMS and batch normalization.

use crate::{
    ensure_dtype, ensure_last_dim, ensure_rank, ensure_rank_at_least, ensure_same_dtype, functional,
};

/// Applies layer normalization across the last tensor dimension:
/// `(x - mean) * rsqrt(var + eps) * gamma + beta`, with the mean and the biased variance over the
/// last axis.
#[functional]
pub fn layer_norm(x: &Tensor, gamma: &Tensor, beta: &Tensor, eps: f32) -> Tensor {
    ensure_same_dtype!(x, gamma, beta);
    ensure_dtype!(x, F32);
    ensure_rank_at_least!(x, 1);
    let features = x.size(-1);
    ensure_rank!(gamma, 1);
    ensure_rank!(beta, 1);
    ensure_last_dim!(gamma, features);
    ensure_last_dim!(beta, features);
    let inv_count = 1.0 / features as f32;

    let mean = x.reduce_sum([-1], true) * inv_count;
    let centered = x - &mean;
    let var = (&centered * &centered).reduce_sum([-1], true) * inv_count;
    let inv_std = (var + eps).rsqrt();
    centered * inv_std * gamma + beta
}

/// Applies RMS normalization across the last tensor dimension:
/// `x * rsqrt(mean(x^2) + eps) * gamma`, with the mean over the last axis.
#[functional]
pub fn rms_norm(x: &Tensor, gamma: &Tensor, eps: f32) -> Tensor {
    ensure_same_dtype!(x, gamma);
    ensure_dtype!(x, F32);
    ensure_rank_at_least!(x, 1);
    let features = x.size(-1);
    ensure_rank!(gamma, 1);
    ensure_last_dim!(gamma, features);

    let mean_square = (x * x).reduce_sum([-1], true) * (1.0 / features as f32);
    x * (mean_square + eps).rsqrt() * gamma
}

/// Applies batch normalization with fixed statistics (PyTorch `BatchNorm` in eval mode) per channel
/// of the last tensor dimension, the channel axis of channels-last activations.
///
/// It computes `x * scale + shift` with the per-channel `scale = weight * rsqrt(running_var + eps)`
/// and `shift = bias - running_mean * scale`, which depend on parameters only, so `outline-static`
/// moves them to load time.
#[functional]
pub fn batch_norm(
    x: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
    running_mean: &Tensor,
    running_var: &Tensor,
    eps: f32,
) -> Tensor {
    ensure_same_dtype!(x, weight, bias, running_mean, running_var);
    ensure_dtype!(x, F32);
    ensure_rank_at_least!(x, 1);
    let channels = x.size(-1);
    ensure_rank!(weight, 1);
    ensure_rank!(bias, 1);
    ensure_rank!(running_mean, 1);
    ensure_rank!(running_var, 1);
    ensure_last_dim!(weight, channels);
    ensure_last_dim!(bias, channels);
    ensure_last_dim!(running_mean, channels);
    ensure_last_dim!(running_var, channels);

    let scale = weight * (running_var + eps).rsqrt();
    let shift = bias - running_mean * &scale;
    x * scale + shift
}
