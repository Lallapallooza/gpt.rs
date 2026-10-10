//! RMS normalization built on [`functional::rms_norm`].

use anyhow::Result;

use crate::nn::{self, Params};
use crate::ops::functional;

/// Epsilon and scale convention of an [`RmsNorm`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RmsNormConfig {
    pub eps: f32,
    /// The scale is `1 + weight` (see [`RmsNorm::unit_offset`]).
    pub unit_offset: bool,
}

/// RMS normalization with a learnable scaling parameter.
#[nn::module]
pub struct RmsNorm {
    pub weight: Tensor,
    #[module(config)]
    pub eps: f32,
    /// Zero-centred weight: the scale is `1 + weight`.
    #[module(config)]
    pub unit_offset: bool,
}

#[nn::module]
impl RmsNorm {
    pub fn new(p: &mut Params<B>, config: RmsNormConfig) -> Result<Self> {
        Ok(Self {
            weight: p.param()?,
            eps: config.eps,
            unit_offset: config.unit_offset,
        })
    }

    fn forward(&self, x: &Tensor) -> Tensor {
        let scale = if self.unit_offset {
            &self.weight + 1.0
        } else {
            self.weight.clone()
        };
        functional::rms_norm(x, &scale, self.eps)
    }
}
