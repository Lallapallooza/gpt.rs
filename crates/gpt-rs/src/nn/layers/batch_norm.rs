//! Batch normalization (PyTorch `nn.BatchNorm2d` in eval mode) built on
//! [`functional::batch_norm`].

use anyhow::Result;

use crate::nn::{self, Params};
use crate::ops::functional;

/// Batch normalization of NHWC activations by the running statistics `running_mean` and
/// `running_var`, then the affine `weight` and `bias`, all `[channels]`.
#[nn::module]
pub struct BatchNorm2d {
    pub weight: Tensor,
    pub bias: Tensor,
    pub running_mean: Tensor,
    pub running_var: Tensor,
    #[module(config)]
    pub eps: f32,
}

#[nn::module]
impl BatchNorm2d {
    pub fn new(p: &mut Params<B>, eps: f32) -> Result<Self> {
        Ok(Self {
            weight: p.param()?,
            bias: p.param()?,
            running_mean: p.param()?,
            running_var: p.param()?,
            eps,
        })
    }

    fn forward(&self, x: &Tensor) -> Tensor {
        functional::batch_norm(
            x,
            &self.weight,
            &self.bias,
            &self.running_mean,
            &self.running_var,
            self.eps,
        )
    }
}
