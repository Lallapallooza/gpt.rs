//! The norm of a decoder: layer or RMS normalization, under the name of the field that holds it.

use anyhow::Result;

use super::layer_norm::LayerNorm;
use super::rms_norm::{RmsNorm, RmsNormConfig};
use crate::nn::{self, Params};

/// Which norm a [`Norm`] is.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NormConfig {
    /// [`LayerNorm`] with this epsilon.
    Layer {
        eps: f32,
    },
    Rms(RmsNormConfig),
}

/// [`LayerNorm`] or [`RmsNorm`]. Its parameters are under the name of the field that holds it.
#[nn::module]
pub enum Norm {
    Layer(LayerNorm),
    Rms(RmsNorm),
}

#[nn::module]
impl Norm {
    pub fn new(p: &mut Params<B>, config: NormConfig) -> Result<Self> {
        Ok(match config {
            NormConfig::Layer { eps } => Self::Layer(LayerNorm::new(p, eps)?),
            NormConfig::Rms(rms) => Self::Rms(RmsNorm::new(p, rms)?),
        })
    }

    fn forward(&self, x: &Tensor) -> Tensor {
        match self {
            Norm::Layer(norm) => norm.call(x),
            Norm::Rms(norm) => norm.call(x),
        }
    }
}
