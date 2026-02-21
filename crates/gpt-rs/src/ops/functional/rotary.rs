//! Rotary position embedding (RoPE) helpers for attention projections.
//!
//! This module provides:
//! - host-side cache generation for cosine/sine tables, including linear and YARN-style scaling;
//! - backend-portable application of RoPE to projected head tensors.

use anyhow::{ensure, Result};
use gpt_rs_macros::{capture_ptir, ptir_pattern, support_runtime_overload};

use crate::backend::spec::PortableBackend;
use crate::ops::functional::common::{
    ensure_dtype_equals, ensure_rank, ensure_same_backend, ensure_same_dtype, ensure_shape_matches,
    CaptureIntoDeviceTensor,
};
use crate::ops::ptir;
use crate::tensor::{DType as TensorDType, DeviceTensor, Shape, Tensor};
use std::sync::Arc;

/// Position-scaling modes for rotary cache generation.
#[derive(Debug, Clone, Copy, Default, serde::Serialize, serde::Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum RopeScaling {
    #[default]
    None,
    Linear {
        factor: f32,
    },
    Yarn {
        factor: f32,
        mscale: f32,
    },
}

impl RopeScaling {
    fn position_factor(self) -> f32 {
        match self {
            RopeScaling::None => 1.0,
            RopeScaling::Linear { factor } | RopeScaling::Yarn { factor, .. } => factor,
        }
    }

    fn magnitude_scale(self) -> f32 {
        match self {
            RopeScaling::Yarn { mscale, .. } => mscale,
            RopeScaling::None | RopeScaling::Linear { .. } => 1.0,
        }
    }
}

/// Configuration for rotary cache generation.
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct RopeConfig {
    pub rotary_dim: usize,
    pub theta: f32,
    #[serde(default)]
    pub scaling: RopeScaling,
}

impl Default for RopeConfig {
    fn default() -> Self {
        Self {
            rotary_dim: 128,
            theta: 10_000.0,
            scaling: RopeScaling::None,
        }
    }
}

/// Host-side cosine/sine cache for RoPE.
pub struct RopeCache {
    pub cos: Tensor,
    pub sin: Tensor,
}

/// Builds RoPE cosine/sine lookup tables with optional scaling.
///
/// `cos` and `sin` have shape `[seq_len, rotary_dim / 2]`.
pub fn rotary_cos_sin_cache(seq_len: usize, config: RopeConfig) -> Result<RopeCache> {
    ensure!(
        config.rotary_dim > 0 && config.rotary_dim.is_multiple_of(2),
        "rotary_dim must be a positive even number (got {})",
        config.rotary_dim
    );
    ensure!(
        config.theta.is_finite() && config.theta > 0.0,
        "theta must be finite and positive (got {})",
        config.theta
    );

    let pos_factor = config.scaling.position_factor();
    ensure!(
        pos_factor.is_finite() && pos_factor > 0.0,
        "rope scaling factor must be finite and positive (got {pos_factor})"
    );
    let magnitude = config.scaling.magnitude_scale();
    ensure!(
        magnitude.is_finite() && magnitude > 0.0,
        "rope magnitude scale must be finite and positive (got {magnitude})"
    );

    let half = config.rotary_dim / 2;
    let mut cos = Vec::with_capacity(seq_len * half);
    let mut sin = Vec::with_capacity(seq_len * half);

    for pos in 0..seq_len {
        let scaled_pos = (pos as f32) / pos_factor;
        for i in 0..half {
            let exponent = (2.0 * i as f32) / (config.rotary_dim as f32);
            let inv_freq = config.theta.powf(-exponent);
            let angle = scaled_pos * inv_freq;
            cos.push(angle.cos() * magnitude);
            sin.push(angle.sin() * magnitude);
        }
    }

    Ok(RopeCache {
        cos: Tensor::from_vec(Shape::new([seq_len, half]), cos)?,
        sin: Tensor::from_vec(Shape::new([seq_len, half]), sin)?,
    })
}

struct RopePlan {
    num_heads: usize,
    seq_len: usize,
    head_dim: usize,
    rotary_dim: usize,
    half_rotary_dim: usize,
}

fn validate_apply_rope<B: PortableBackend + 'static>(
    x: &DeviceTensor<B>,
    cos: &DeviceTensor<B>,
    sin: &DeviceTensor<B>,
) -> Result<RopePlan> {
    ensure_rank("rope input", x, 3)?;
    ensure_rank("rope cos", cos, 2)?;
    ensure_rank("rope sin", sin, 2)?;
    ensure_same_backend("rope apply", x, cos)?;
    ensure_same_backend("rope apply", x, sin)?;
    ensure_same_dtype("rope input", x, "cos", cos)?;
    ensure_same_dtype("rope input", x, "sin", sin)?;
    ensure_shape_matches("rope cos", cos, "sin", sin)?;
    ensure_dtype_equals("rope input", x, TensorDType::F32)?;

    let x_dims = x.shape().dims();
    let cos_dims = cos.shape().dims();
    let seq_len = x_dims[1];
    let head_dim = x_dims[2];
    let half_rotary_dim = cos_dims[1];
    let rotary_dim = half_rotary_dim * 2;

    ensure!(
        cos_dims[0] == seq_len,
        "rope cos sequence length {} must match input sequence length {}",
        cos_dims[0],
        seq_len
    );
    ensure!(
        rotary_dim > 0 && rotary_dim <= head_dim,
        "rope rotary dimension {} must be in (0, head_dim={}]",
        rotary_dim,
        head_dim
    );

    Ok(RopePlan {
        num_heads: x_dims[0],
        seq_len,
        head_dim,
        rotary_dim,
        half_rotary_dim,
    })
}

/// Applies RoPE to the leading `rotary_dim` channels of `x`.
///
/// Input `x` is expected to have shape `[num_heads, seq_len, head_dim]`, and `cos`/`sin` must have
/// shape `[seq_len, rotary_dim / 2]`.
#[support_runtime_overload]
#[ptir_pattern(target = "gpt_rs.apply_rope_f32")]
pub fn apply_rope<B: PortableBackend + 'static>(
    _backend: &B,
    x: &DeviceTensor<B>,
    cos: &DeviceTensor<B>,
    sin: &DeviceTensor<B>,
) -> Result<DeviceTensor<B>> {
    let plan = validate_apply_rope(x, cos, sin)?;

    capture_ptir!({ x, cos, sin }, |_session| {
        let x_rot = x.slice(
            vec![0, 0, 0],
            vec![plan.num_heads, plan.seq_len, plan.rotary_dim],
        );
        let x_pairs = x_rot.reshape(vec![
            plan.num_heads,
            plan.seq_len,
            plan.half_rotary_dim,
            2,
        ]);

        let x_even = x_pairs.slice(
            vec![0, 0, 0, 0],
            vec![plan.num_heads, plan.seq_len, plan.half_rotary_dim, 1],
        );
        let x_even = x_even.reshape(vec![plan.num_heads, plan.seq_len, plan.half_rotary_dim]);
        let x_odd = x_pairs.slice(
            vec![0, 0, 0, 1],
            vec![plan.num_heads, plan.seq_len, plan.half_rotary_dim, 1],
        );
        let x_odd = x_odd.reshape(vec![plan.num_heads, plan.seq_len, plan.half_rotary_dim]);

        let cos_b = cos
            .reshape(vec![1, plan.seq_len, plan.half_rotary_dim])
            .broadcast_to(vec![plan.num_heads, plan.seq_len, plan.half_rotary_dim]);
        let sin_b = sin
            .reshape(vec![1, plan.seq_len, plan.half_rotary_dim])
            .broadcast_to(vec![plan.num_heads, plan.seq_len, plan.half_rotary_dim]);

        let rotated_even = x_even * cos_b - x_odd * sin_b;
        let rotated_odd = x_odd * cos_b + x_even * sin_b;

        let rotated_even = rotated_even.reshape(vec![
            plan.num_heads,
            plan.seq_len,
            plan.half_rotary_dim,
            1,
        ]);
        let rotated_odd = rotated_odd.reshape(vec![
            plan.num_heads,
            plan.seq_len,
            plan.half_rotary_dim,
            1,
        ]);
        let rotated_pairs = ptir::Tensor::concat(3, &[rotated_even, rotated_odd]);
        let rotated = rotated_pairs.reshape(vec![plan.num_heads, plan.seq_len, plan.rotary_dim]);

        let output = if plan.rotary_dim == plan.head_dim {
            rotated
        } else {
            let passthrough = x.slice(
                vec![0, 0, plan.rotary_dim],
                vec![plan.num_heads, plan.seq_len, plan.head_dim - plan.rotary_dim],
            );
            ptir::Tensor::concat(2, &[rotated, passthrough])
        };
        Ok(output.id())
    })?
    .into_device_tensor()
}

/// Applies RoPE to both query and key tensors.
pub struct RopeQkResult<B: PortableBackend + 'static> {
    pub query: DeviceTensor<B>,
    pub key: DeviceTensor<B>,
}

/// Convenience wrapper around [`apply_rope`] for `(q, k)` pairs.
pub fn apply_rope_qk<B: PortableBackend + 'static>(
    backend: &B,
    query: &DeviceTensor<B>,
    key: &DeviceTensor<B>,
    cos: &DeviceTensor<B>,
    sin: &DeviceTensor<B>,
) -> Result<RopeQkResult<B>> {
    let rotated_q = apply_rope(backend, query, cos, sin)?;
    let rotated_k = apply_rope(backend, key, cos, sin)?;
    Ok(RopeQkResult {
        query: rotated_q,
        key: rotated_k,
    })
}

impl RopeCache {
    /// Uploads host cos/sin tables to a backend.
    pub fn to_device<B: PortableBackend + 'static>(
        &self,
        backend: Arc<B>,
    ) -> Result<(DeviceTensor<B>, DeviceTensor<B>)> {
        let cos = DeviceTensor::from_host(Arc::clone(&backend), self.cos.clone())?;
        let sin = DeviceTensor::from_host(backend, self.sin.clone())?;
        Ok((cos, sin))
    }
}
