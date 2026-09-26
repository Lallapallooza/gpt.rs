//! Gated feed-forward block for SwiGLU-style transformer MLPs.
//!
//! Computes `down_proj(silu(gate_proj(x)) * up_proj(x))`.

use super::linear::Linear;
use crate::nn::{self, Params};
use crate::ops::functional;
use anyhow::Result;

/// Transformer gated MLP with `gate_proj`, `up_proj` and `down_proj` projections.
#[nn::module]
pub struct GatedFeedForward {
    pub gate_proj: Linear,
    pub up_proj: Linear,
    pub down_proj: Linear,
}

#[nn::module]
impl GatedFeedForward {
    pub fn new(p: &mut Params<B>) -> Result<Self> {
        Ok(Self {
            gate_proj: Linear::new(p)?,
            up_proj: Linear::new(p)?,
            down_proj: Linear::new(p)?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = self.gate_proj(x)?;
        let up = self.up_proj(x)?;
        let hidden = functional::swiglu(&gate, &up)?;
        self.down_proj(&hidden)
    }
}
