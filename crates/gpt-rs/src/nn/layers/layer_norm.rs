//! Layer normalization (PyTorch `nn.LayerNorm`) built on [`functional::layer_norm`].

use crate::nn;
use crate::ops::functional;
use anyhow::Result;

/// Layer normalization with a learnable affine `weight` and `bias`.
#[nn::module]
pub struct LayerNorm {
    pub weight: Tensor,
    pub bias: Tensor,
    #[module(config)]
    pub eps: f32,
}

#[nn::module]
impl LayerNorm {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let outputs = functional::layer_norm(x, &self.weight, &self.bias, self.eps)?;
        Ok(outputs.output)
    }
}
