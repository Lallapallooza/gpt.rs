//! Rotary position embedding (RoPE) helpers for attention projections.
//!
//! This module provides:
//! - host-side cache generation for cosine/sine tables, including linear and YARN-style scaling;
//! - backend-portable application of RoPE to projected head tensors.

use anyhow::{ensure, Result};
use gpt_rs_macros::{capture_ptir, ptir_pattern, support_runtime_overload};

use crate::backend::spec::PortableBackend;
use crate::nn::layers::AttentionConfig;
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

fn build_rotary_cache_range(start: usize, len: usize, config: RopeConfig) -> Result<RopeCache> {
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
    let mut cos = Vec::with_capacity(len * half);
    let mut sin = Vec::with_capacity(len * half);

    for pos in start..start + len {
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
        cos: Tensor::from_vec(Shape::new([len, half]), cos)?,
        sin: Tensor::from_vec(Shape::new([len, half]), sin)?,
    })
}

/// Builds RoPE cosine/sine lookup tables with optional scaling.
///
/// `cos` and `sin` have shape `[seq_len, rotary_dim / 2]`.
pub fn rotary_cos_sin_cache(seq_len: usize, config: RopeConfig) -> Result<RopeCache> {
    build_rotary_cache_range(0, seq_len, config)
}

/// Builds a RoPE cache for a contiguous position range `[start, start + len)`.
pub fn rotary_cos_sin_cache_slice(
    start: usize,
    len: usize,
    config: RopeConfig,
) -> Result<RopeCache> {
    build_rotary_cache_range(start, len, config)
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

struct PackedQkvRopePlan {
    seq_len: usize,
    q_proj_dim: usize,
    kv_proj_dim: usize,
    num_query_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    kv_head_dim: usize,
}

fn validate_apply_rope_qkv_packed<B: PortableBackend + 'static>(
    config: &AttentionConfig,
    qkv: &DeviceTensor<B>,
    cos: &DeviceTensor<B>,
    sin: &DeviceTensor<B>,
) -> Result<PackedQkvRopePlan> {
    ensure_rank("rope packed qkv", qkv, 2)?;
    ensure_rank("rope cos", cos, 2)?;
    ensure_rank("rope sin", sin, 2)?;
    ensure_same_backend("rope packed qkv", qkv, cos)?;
    ensure_same_backend("rope packed qkv", qkv, sin)?;
    ensure_same_dtype("rope packed qkv", qkv, "cos", cos)?;
    ensure_same_dtype("rope packed qkv", qkv, "sin", sin)?;
    ensure_shape_matches("rope cos", cos, "sin", sin)?;
    ensure_dtype_equals("rope packed qkv", qkv, TensorDType::F32)?;

    let dims = qkv.shape().dims();
    let seq_len = dims[0];
    ensure!(
        dims[1] == config.total_projection_dim(),
        "rope packed qkv projection dimension mismatch: expected {}, got {}",
        config.total_projection_dim(),
        dims[1]
    );
    ensure!(
        cos.shape().dims()[0] == seq_len,
        "rope cos sequence length {} must match qkv sequence length {}",
        cos.shape().dims()[0],
        seq_len
    );

    let half_rotary = cos.shape().dims()[1];
    let rotary_dim = half_rotary * 2;
    ensure!(
        rotary_dim > 0 && rotary_dim <= config.head_dim,
        "rope rotary dimension {} must be in (0, head_dim={}]",
        rotary_dim,
        config.head_dim
    );
    ensure!(
        rotary_dim <= config.kv_head_dim,
        "rope rotary dimension {} must be <= kv_head_dim {}",
        rotary_dim,
        config.kv_head_dim
    );

    Ok(PackedQkvRopePlan {
        seq_len,
        q_proj_dim: config.query_projection_dim(),
        kv_proj_dim: config.key_value_projection_dim(),
        num_query_heads: config.num_query_heads,
        num_key_value_heads: config.num_key_value_heads,
        head_dim: config.head_dim,
        kv_head_dim: config.kv_head_dim,
    })
}

/// Applies RoPE to packed QKV projections in-place layout form `[seq_len, total_projection_dim]`.
///
/// The packed tensor uses `[Q | K | V]` layout where `Q` and `K` are rotated while `V` stays
/// untouched.
#[support_runtime_overload]
#[ptir_pattern(target = "gpt_rs.apply_rope_qkv_packed_f32")]
pub fn apply_rope_qkv_packed<B: PortableBackend + 'static>(
    _backend: &B,
    config: &AttentionConfig,
    qkv: &DeviceTensor<B>,
    cos: &DeviceTensor<B>,
    sin: &DeviceTensor<B>,
) -> Result<DeviceTensor<B>> {
    let plan = validate_apply_rope_qkv_packed(config, qkv, cos, sin)?;
    let half_rotary = cos.shape().dims()[1];
    let rotary_dim = half_rotary * 2;

    capture_ptir!({ qkv, cos, sin }, |_session| {
        let q_slice = qkv.slice(vec![0, 0], vec![plan.seq_len, plan.q_proj_dim]);
        let k_slice = qkv.slice(
            vec![0, plan.q_proj_dim],
            vec![plan.seq_len, plan.kv_proj_dim],
        );
        let v_slice = qkv.slice(
            vec![0, plan.q_proj_dim + plan.kv_proj_dim],
            vec![plan.seq_len, plan.kv_proj_dim],
        );

        let cos_q = cos
            .reshape(vec![1, plan.seq_len, half_rotary])
            .broadcast_to(vec![plan.num_query_heads, plan.seq_len, half_rotary]);
        let sin_q = sin
            .reshape(vec![1, plan.seq_len, half_rotary])
            .broadcast_to(vec![plan.num_query_heads, plan.seq_len, half_rotary]);
        let q_heads = q_slice
            .reshape(vec![plan.seq_len, plan.num_query_heads, plan.head_dim])
            .transpose(vec![1, 0, 2]);
        let q_rot = q_heads.slice(
            vec![0, 0, 0],
            vec![plan.num_query_heads, plan.seq_len, rotary_dim],
        );
        let q_rot_pairs = q_rot.reshape(vec![plan.num_query_heads, plan.seq_len, half_rotary, 2]);
        let q_even = q_rot_pairs
            .slice(
                vec![0, 0, 0, 0],
                vec![plan.num_query_heads, plan.seq_len, half_rotary, 1],
            )
            .reshape(vec![plan.num_query_heads, plan.seq_len, half_rotary]);
        let q_odd = q_rot_pairs
            .slice(
                vec![0, 0, 0, 1],
                vec![plan.num_query_heads, plan.seq_len, half_rotary, 1],
            )
            .reshape(vec![plan.num_query_heads, plan.seq_len, half_rotary]);
        let q_even_rot = q_even * cos_q - q_odd * sin_q;
        let q_odd_rot = q_odd * cos_q + q_even * sin_q;
        let q_even_col =
            q_even_rot.reshape(vec![plan.num_query_heads, plan.seq_len, half_rotary, 1]);
        let q_odd_col = q_odd_rot.reshape(vec![plan.num_query_heads, plan.seq_len, half_rotary, 1]);
        let q_rot_pairs = ptir::Tensor::concat(3, &[q_even_col, q_odd_col]);
        let q_rot = q_rot_pairs.reshape(vec![plan.num_query_heads, plan.seq_len, rotary_dim]);
        let q_heads = if rotary_dim == plan.head_dim {
            q_rot
        } else {
            let q_pass = q_heads.slice(
                vec![0, 0, rotary_dim],
                vec![plan.num_query_heads, plan.seq_len, plan.head_dim - rotary_dim],
            );
            ptir::Tensor::concat(2, &[q_rot, q_pass])
        };
        let q_flat = q_heads
            .transpose(vec![1, 0, 2])
            .reshape(vec![plan.seq_len, plan.q_proj_dim]);

        let cos_k = cos
            .reshape(vec![1, plan.seq_len, half_rotary])
            .broadcast_to(vec![plan.num_key_value_heads, plan.seq_len, half_rotary]);
        let sin_k = sin
            .reshape(vec![1, plan.seq_len, half_rotary])
            .broadcast_to(vec![plan.num_key_value_heads, plan.seq_len, half_rotary]);
        let k_heads = k_slice
            .reshape(vec![plan.seq_len, plan.num_key_value_heads, plan.kv_head_dim])
            .transpose(vec![1, 0, 2]);
        let k_rot = k_heads.slice(
            vec![0, 0, 0],
            vec![plan.num_key_value_heads, plan.seq_len, rotary_dim],
        );
        let k_rot_pairs =
            k_rot.reshape(vec![plan.num_key_value_heads, plan.seq_len, half_rotary, 2]);
        let k_even = k_rot_pairs
            .slice(
                vec![0, 0, 0, 0],
                vec![plan.num_key_value_heads, plan.seq_len, half_rotary, 1],
            )
            .reshape(vec![plan.num_key_value_heads, plan.seq_len, half_rotary]);
        let k_odd = k_rot_pairs
            .slice(
                vec![0, 0, 0, 1],
                vec![plan.num_key_value_heads, plan.seq_len, half_rotary, 1],
            )
            .reshape(vec![plan.num_key_value_heads, plan.seq_len, half_rotary]);
        let k_even_rot = k_even * cos_k - k_odd * sin_k;
        let k_odd_rot = k_odd * cos_k + k_even * sin_k;
        let k_even_col = k_even_rot.reshape(vec![
            plan.num_key_value_heads,
            plan.seq_len,
            half_rotary,
            1,
        ]);
        let k_odd_col = k_odd_rot.reshape(vec![
            plan.num_key_value_heads,
            plan.seq_len,
            half_rotary,
            1,
        ]);
        let k_rot_pairs = ptir::Tensor::concat(3, &[k_even_col, k_odd_col]);
        let k_rot = k_rot_pairs.reshape(vec![plan.num_key_value_heads, plan.seq_len, rotary_dim]);
        let k_heads = if rotary_dim == plan.kv_head_dim {
            k_rot
        } else {
            let k_pass = k_heads.slice(
                vec![0, 0, rotary_dim],
                vec![
                    plan.num_key_value_heads,
                    plan.seq_len,
                    plan.kv_head_dim - rotary_dim,
                ],
            );
            ptir::Tensor::concat(2, &[k_rot, k_pass])
        };
        let k_flat = k_heads
            .transpose(vec![1, 0, 2])
            .reshape(vec![plan.seq_len, plan.kv_proj_dim]);

        let packed = ptir::Tensor::concat(1, &[q_flat, k_flat, v_slice]);
        Ok(packed.id())
    })?
    .into_device_tensor()
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
