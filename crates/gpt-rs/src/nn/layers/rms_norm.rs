//! RMS normalization built on [`functional::rms_norm`].

use crate::nn;
use crate::ops::functional;
use anyhow::Result;

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
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let scale = if self.unit_offset {
            functional::add_scalar(&self.weight, 1.0)?
        } else {
            self.weight.clone()
        };
        Ok(functional::rms_norm(x, &scale, self.eps)?.output)
    }
}
