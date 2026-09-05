//! Rotary position embedding (RoPE) tables, with optional linear or YaRN frequency scaling.
//!
//! [`RotaryEmbedding`] holds no parameters, so it is not a module.

use std::sync::Arc;

use anyhow::{ensure, Result};

use crate::backend::spec::PortableBackend;
use crate::tensor::{DeviceTensor, Shape, Tensor};

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
        let base_inv_freq = |i: usize| theta.powf(-(2.0 * i as f32) / rotary_dim as f32);

        let (inv_freq, attention_scaling) = match scaling {
            RopeScaling::None => ((0..half).map(base_inv_freq).collect(), 1.0),
            RopeScaling::Linear { factor } => {
                ensure!(
                    factor.is_finite() && factor > 0.0,
                    "rope scaling factor must be finite and positive (got {factor})"
                );
                ((0..half).map(|i| base_inv_freq(i) / factor).collect(), 1.0)
            }
            RopeScaling::Yarn {
                factor,
                mscale,
                mscale_all_dim,
                beta_fast,
                beta_slow,
                original_max_position_embeddings,
                truncate,
            } => {
                ensure!(
                    factor.is_finite() && factor > 0.0,
                    "rope yarn factor must be finite and positive (got {factor})"
                );
                let beta_fast = beta_fast.unwrap_or(32.0);
                let beta_slow = beta_slow.unwrap_or(1.0);
                ensure!(
                    beta_fast.is_finite() && beta_slow.is_finite() && beta_fast >= beta_slow,
                    "rope yarn beta range is invalid: beta_fast={beta_fast} beta_slow={beta_slow}"
                );

                let get_mscale = |scale: f32, mscale: f32| {
                    if scale <= 1.0 {
                        1.0
                    } else {
                        0.1 * mscale * scale.ln() + 1.0
                    }
                };
                let attention_scaling = match (mscale, mscale_all_dim) {
                    (Some(ms), Some(ms_all)) if ms != 0.0 && ms_all != 0.0 => {
                        get_mscale(factor, ms) / get_mscale(factor, ms_all)
                    }
                    _ => get_mscale(factor, 1.0),
                };

                let find_correction_dim = |num_rotations: f32| {
                    (rotary_dim as f32
                        * (original_max_position_embeddings as f32
                            / (num_rotations * 2.0 * std::f32::consts::PI))
                            .ln())
                        / (2.0 * theta.ln())
                };
                let mut low = find_correction_dim(beta_fast);
                let mut high = find_correction_dim(beta_slow);
                if truncate {
                    low = low.floor();
                    high = high.ceil();
                }
                low = low.max(0.0);
                high = high.min((rotary_dim - 1) as f32);
                if (low - high).abs() < f32::EPSILON {
                    high += 0.001;
                }

                // Extrapolated (unscaled) below the ramp, interpolated (`/ factor`) above it.
                let inv_freq = (0..half)
                    .map(|i| {
                        let extrapolation = base_inv_freq(i);
                        let interpolation = extrapolation / factor;
                        let ramp = ((i as f32 - low) / (high - low)).clamp(0.0, 1.0);
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

    /// `(cos, sin)` on `backend`, each `[len, rotary_dim / 2]`, for positions `start..start + len`.
    pub fn tables<B: PortableBackend + 'static>(
        &self,
        backend: &Arc<B>,
        start: usize,
        len: usize,
    ) -> Result<(DeviceTensor<B>, DeviceTensor<B>)> {
        let half = self.inv_freq.len();
        let mut cos = Vec::with_capacity(len * half);
        let mut sin = Vec::with_capacity(len * half);
        for pos in start..start + len {
            for &inv_freq in &self.inv_freq {
                let angle = pos as f32 * inv_freq;
                cos.push(angle.cos() * self.attention_scaling);
                sin.push(angle.sin() * self.attention_scaling);
            }
        }
        let table = |data| {
            let host = Tensor::from_vec(Shape::new([len, half]), data)?;
            DeviceTensor::from_host(Arc::clone(backend), host)
        };
        Ok((table(cos)?, table(sin)?))
    }
}
