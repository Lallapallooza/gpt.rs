//! Gated feed-forward block for SwiGLU-style transformer MLPs.
//!
//! Computes `down_proj(silu(gate_proj(x)) * up_proj(x))`.

use super::linear::Linear;
use crate::nn::{self, LayerLoader};
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
    pub fn load(
        params: &mut LayerLoader<'_, B>,
        prefix: &str,
        dim: usize,
        hidden_dim: usize,
        bias: bool,
    ) -> Result<Self> {
        let mut linear = |name: &str, in_features: usize, out_features: usize| {
            params.linear(&format!("{prefix}.{name}"), in_features, out_features, bias)
        };
        Ok(Self {
            gate_proj: linear("gate_proj", dim, hidden_dim)?,
            up_proj: linear("up_proj", dim, hidden_dim)?,
            down_proj: linear("down_proj", hidden_dim, dim)?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = self.gate_proj(x)?;
        let up = self.up_proj(x)?;
        let hidden = functional::swiglu(&gate, &up)?;
        self.down_proj(&hidden)
    }
}
