//! Rotary position embedding (RoPE) tables, with optional linear or YaRN frequency scaling.
//!
//! [`RotaryEmbedding`] holds no parameters, so it is not a module.

use anyhow::{ensure, Result};

use crate::backend::PortableBackend;
use crate::tensor::{DType, DeviceTensor, HostTensor};

/// Position scaling of the rotary frequencies: the `rope_type` of Hugging Face `rope_parameters`
/// and its keys.
#[derive(Debug, Clone, Copy, Default, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "rope_type", rename_all = "snake_case")]
pub enum RopeScaling {
    #[default]
    #[serde(rename = "default")]
    None,
    Linear {
        factor: f32,
    },
    Yarn {
        factor: f32,
        original_max_position_embeddings: usize,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        mscale: Option<f32>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        mscale_all_dim: Option<f32>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        beta_fast: Option<f32>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        beta_slow: Option<f32>,
        #[serde(default = "default_yarn_truncate")]
        truncate: bool,
    },
}

const fn default_yarn_truncate() -> bool {
    true
}

/// Hugging Face `rope_parameters` of a model config.
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct RopeParameters {
    pub rope_theta: f32,
    #[serde(default = "full_rotation")]
    pub partial_rotary_factor: f64,
    #[serde(flatten)]
    pub scaling: RopeScaling,
}

const fn full_rotation() -> f64 {
    1.0
}

impl RopeParameters {
    /// Rotary embedding for heads of `head_dim` channels. Like Hugging Face, it rotates the
    /// leading `int(head_dim * partial_rotary_factor)` channels.
    pub fn rope(&self, head_dim: usize) -> RopeConfig {
        RopeConfig {
            rotary_dim: (head_dim as f64 * self.partial_rotary_factor) as usize,
            theta: self.rope_theta,
            scaling: self.scaling,
        }
    }
}

/// Rotary embedding of the leading `rotary_dim` channels of each head.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RopeConfig {
    pub rotary_dim: usize,
    pub theta: f32,
    pub scaling: RopeScaling,
}

/// Cosine/sine tables of rotary position embeddings.
#[derive(Debug, Clone)]
pub struct RotaryEmbedding {
    /// Inverse wavelength of each rotated channel pair, `[rotary_dim / 2]`.
    inv_freq: Vec<f32>,
    /// Scale of both tables: the YaRN attention factor, or 1 without YaRN.
    attention_scaling: f32,
}

impl RotaryEmbedding {
    /// Computes the frequencies of `config`. YaRN follows Hugging Face.
    pub fn new(config: RopeConfig) -> Result<Self> {
        let RopeConfig {
            rotary_dim,
            theta,
            scaling,
        } = config;
        ensure!(
            rotary_dim > 0 && rotary_dim.is_multiple_of(2),
            "rotary_dim must be a positive even number (got {rotary_dim})"
        );
        ensure!(
            theta.is_finite() && theta > 0.0,
            "rope theta must be finite and positive (got {theta})"
        );
        let half = rotary_dim / 2;
        let base_inv_freq = |dim: usize| theta.powf(-(2.0 * dim as f32) / rotary_dim as f32);

        let (inv_freq, attention_scaling) = match scaling {
            RopeScaling::None => ((0..half).map(base_inv_freq).collect(), 1.0),
            RopeScaling::Linear { factor } => {
                ensure!(
                    factor.is_finite() && factor > 0.0,
                    "rope scaling factor must be finite and positive (got {factor})"
                );
                (
                    (0..half).map(|dim| base_inv_freq(dim) / factor).collect(),
                    1.0,
                )
            }
            RopeScaling::Yarn {
                factor,
                mscale,
                mscale_all_dim,
                beta_fast,
                beta_slow,
                original_max_position_embeddings: original_context,
                truncate,
            } => {
                ensure!(
                    factor.is_finite() && factor > 0.0,
                    "rope yarn factor must be finite and positive (got {factor})"
                );
                let fast_beta = beta_fast.unwrap_or(32.0);
                let slow_beta = beta_slow.unwrap_or(1.0);
                ensure!(
                    fast_beta.is_finite() && slow_beta.is_finite() && fast_beta >= slow_beta,
                    "rope yarn beta range is invalid: beta_fast={fast_beta} beta_slow={slow_beta}"
                );
                let attention_scaling = yarn_attention_scaling(factor, mscale, mscale_all_dim);
                let low = yarn_correction_dim(rotary_dim, theta, original_context, fast_beta);
                let high = yarn_correction_dim(rotary_dim, theta, original_context, slow_beta);
                let (ramp_start, ramp_end) = yarn_ramp(rotary_dim, low, high, truncate);

                // Extrapolated (unscaled) below the ramp, interpolated (`/ factor`) above it.
                let inv_freq = (0..half)
                    .map(|dim| {
                        let extrapolation = base_inv_freq(dim);
                        let interpolation = extrapolation / factor;
                        let ramp =
                            ((dim as f32 - ramp_start) / (ramp_end - ramp_start)).clamp(0.0, 1.0);
                        interpolation * ramp + extrapolation * (1.0 - ramp)
                    })
                    .collect();
                (inv_freq, attention_scaling)
            }
        };
        ensure!(
            attention_scaling.is_finite() && attention_scaling > 0.0,
            "rope attention scaling must be finite and positive (got {attention_scaling})"
        );
        Ok(Self {
            inv_freq,
            attention_scaling,
        })
    }

    /// The `(cos, sin)` tables of `positions` (`si32 [T]`), each `[T, rotary_dim / 2]`, computed
    /// in the graph like Hugging Face does: `cos(position * inv_freq) * attention_scaling`, and
    /// the same for `sin`.
    #[track_caller]
    pub fn tables<B: PortableBackend + 'static>(
        &self,
        positions: &DeviceTensor<B>,
    ) -> (DeviceTensor<B>, DeviceTensor<B>) {
        let half = self.inv_freq.len();
        let inv_freq_host = HostTensor::f32([1, half], self.inv_freq.clone()).expect("one row");
        let inv_freq = DeviceTensor::constant(positions, &inv_freq_host);
        let angles = positions.cast(DType::F32).reshape([positions.size(0), 1]) * inv_freq;
        let (cos, sin) = (angles.cos(), angles.sin());
        if self.attention_scaling == 1.0 {
            return (cos, sin);
        }
        (cos * self.attention_scaling, sin * self.attention_scaling)
    }
}

/// Hugging Face `get_mscale`: `0.1 * mscale * ln(scale) + 1` for a `scale` above 1, else 1.
fn yarn_mscale(scale: f32, mscale: f32) -> f32 {
    if scale <= 1.0 {
        1.0
    } else {
        0.1 * mscale * scale.ln() + 1.0
    }
}

/// The YaRN attention factor of `factor`: the ratio of its scales under `mscale` and
/// `mscale_all_dim` when both are set and nonzero, else its scale under 1.
fn yarn_attention_scaling(factor: f32, mscale: Option<f32>, mscale_all_dim: Option<f32>) -> f32 {
    match (mscale, mscale_all_dim) {
        (Some(scale), Some(scale_all)) if scale != 0.0 && scale_all != 0.0 => {
            yarn_mscale(factor, scale) / yarn_mscale(factor, scale_all)
        }
        _ => yarn_mscale(factor, 1.0),
    }
}

/// Hugging Face `find_correction_dim`: the channel pair that turns `num_rotations` times over the
/// `original_context` positions.
fn yarn_correction_dim(
    rotary_dim: usize,
    theta: f32,
    original_context: usize,
    num_rotations: f32,
) -> f32 {
    (rotary_dim as f32
        * (original_context as f32 / (num_rotations * 2.0 * std::f32::consts::PI)).ln())
        / (2.0 * theta.ln())
}

/// The YaRN ramp from the correction dimension `low` to `high`: rounded outward when `truncate`,
/// clamped to the channels of `rotary_dim`, and never empty.
fn yarn_ramp(rotary_dim: usize, low: f32, high: f32, truncate: bool) -> (f32, f32) {
    let (mut start, mut end) = if truncate {
        (low.floor(), high.ceil())
    } else {
        (low, high)
    };
    start = start.max(0.0);
    end = end.min((rotary_dim - 1) as f32);
    if (start - end).abs() < f32::EPSILON {
        end += 0.001;
    }
    (start, end)
}
