//! Layer normalization (PyTorch `nn.LayerNorm`) built on [`functional::layer_norm`].

use anyhow::Result;

use crate::nn::{self, Params};
use crate::ops::functional;

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
    pub fn new(p: &mut Params<B>, eps: f32) -> Result<Self> {
        Ok(Self {
            weight: p.param()?,
            bias: p.param()?,
            eps,
        })
    }

    fn forward(&self, x: &Tensor) -> Tensor {
        functional::layer_norm(x, &self.weight, &self.bias, self.eps)
    }
}
