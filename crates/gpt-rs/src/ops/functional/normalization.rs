//! Normalization primitives including layer norm expressed through portable graphs.
//!
//! Results expose intermediate statistics for reuse and debugging without recomputing reductions.

use std::sync::Arc;

use anyhow::Result;
use gpt_rs_macros::{capture_ptir, ptir_pattern, support_runtime_overload};

use crate::backend::spec::PortableBackend;
use crate::ops::functional::common::{
    ensure_dtype_equals, ensure_last_dim, ensure_rank, ensure_rank_at_least, ensure_same_backend,
    ensure_same_dtype, scalar_broadcast, CaptureIntoDeviceTensor,
};
use crate::tensor::{DType as TensorDType, DeviceTensor};

/// Outputs produced by [`layer_norm`], including cached intermediates.
pub struct LayerNormResult<B: PortableBackend + 'static> {
    pub output: DeviceTensor<B>,
    pub normalized: DeviceTensor<B>,
    pub mean: DeviceTensor<B>,
    pub inv_std: DeviceTensor<B>,
}

struct LayerNormPlan {
    last_dim_axis: usize,
    feature_dim: usize,
    reduce_shape: Vec<usize>,
    input_shape: Vec<usize>,
}

/// Validates layer-norm tensors and prepares broadcast metadata reused during capture.
///
/// Enforces dtype alignment (currently f32), rank expectations, and gamma/beta shapes, then records
/// the reduction axes and broadcast shapes. Structural coverage lives in
/// `tests/functional_softmax.rs::layer_norm_captures_reduction_chain`, and numerical parity runs in
/// `crates/gpt-rs/tests/torch_parity.rs` plus backend suites.
fn validate_layer_norm<B: PortableBackend + 'static>(
    x: &DeviceTensor<B>,
    gamma: &DeviceTensor<B>,
    beta: &DeviceTensor<B>,
) -> Result<LayerNormPlan> {
    ensure_same_dtype("layer_norm x", x, "gamma", gamma)?;
    ensure_same_dtype("layer_norm x", x, "beta", beta)?;
    ensure_dtype_equals("layer_norm x", x, TensorDType::F32)?;

    ensure_rank_at_least("layer_norm input", x, 1)?;
    let rank = x.shape().rank();
    let last_dim = rank - 1;
    let feature_dim = x.shape().dims()[last_dim];

    ensure_same_backend("layer_norm", x, gamma)?;
    ensure_same_backend("layer_norm", x, beta)?;

    ensure_rank("layer_norm gamma", gamma, 1)?;
    ensure_rank("layer_norm beta", beta, 1)?;
    ensure_last_dim("layer_norm gamma", gamma, feature_dim)?;
    ensure_last_dim("layer_norm beta", beta, feature_dim)?;

    let mut reduce_shape = x.shape().dims().to_vec();
    reduce_shape[last_dim] = 1;

    Ok(LayerNormPlan {
        last_dim_axis: last_dim,
        feature_dim,
        reduce_shape,
        input_shape: x.shape().dims().to_vec(),
    })
}

/// Applies layer normalization across the last tensor dimension.
///
/// The captured graph mirrors the textbook algorithm:
/// - compute the per-sample mean by summing across the normalized axis and scaling by `1/N`;
/// - subtract the mean to centre the activations and accumulate squared deviations;
/// - compute variance, add `eps`, and take the reciprocal square root to obtain `1/std`;
/// - multiply the centred activations by `1/std` to produce the normalized tensor;
/// - broadcast the affine parameters (`gamma`, `beta`) and apply the final scaling and shift.
/// Intermediate tensors (mean, inv_std, normalized) are returned alongside the final output so
/// downstream consumers can reuse them without recomputing reductions.
#[support_runtime_overload]
#[ptir_pattern(target = "gpt_rs.layer_norm_f32")]
pub fn layer_norm<B: PortableBackend + 'static>(
    _backend: &B,
    x: &DeviceTensor<B>,
    gamma: &DeviceTensor<B>,
    beta: &DeviceTensor<B>,
    eps: f32,
) -> Result<LayerNormResult<B>> {
    let plan = validate_layer_norm(x, gamma, beta)?;
    let (graph, (output_id, normalized_id, mean_id, inv_std_id)) = capture_ptir!({ x, gamma, beta }, |session| {
        // Mean reduction across the last dimension.
        let sum = x.reduce_sum(vec![plan.last_dim_axis], true);
        let inv_count = scalar_broadcast(
            &session,
            1.0f32 / plan.feature_dim as f32,
            &plan.reduce_shape,
        );
        let mean = sum * inv_count;
        let mean_broadcast = mean.broadcast_to(plan.input_shape.clone());

        // Variance computation (centered squared values) plus epsilon stabilisation.
        let centered = x - mean_broadcast;
        let centered_sq = centered * centered;
        let var_sum = centered_sq.reduce_sum(vec![plan.last_dim_axis], true);
        let var_mean = var_sum * inv_count;
        let var_eps = var_mean + scalar_broadcast(&session, eps, &plan.reduce_shape);
        let inv_std = var_eps.rsqrt();
        let inv_std_broadcast = inv_std.broadcast_to(plan.input_shape.clone());
        let normalized = centered * inv_std_broadcast;

        // Apply affine transform: broadcast gamma/beta to the full shape and scale/shift.
        let gamma_broadcast = gamma.broadcast_to(plan.input_shape.clone());
        let beta_broadcast = beta.broadcast_to(plan.input_shape.clone());

        let output = normalized * gamma_broadcast + beta_broadcast;

        Ok((output.id(), normalized.id(), mean.id(), inv_std.id()))
    })?;

    let output = (Arc::clone(&graph), output_id).into_device_tensor()?;
    let normalized = (Arc::clone(&graph), normalized_id).into_device_tensor()?;
    let mean = (Arc::clone(&graph), mean_id).into_device_tensor()?;
    let inv_std = (graph, inv_std_id).into_device_tensor()?;

    Ok(LayerNormResult {
        output,
        normalized,
        mean,
        inv_std,
    })
}
