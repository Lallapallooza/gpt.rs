//! Convolution layers: 2D image convolutions (NHWC internal layout) and the depthwise causal 1D
//! convolution of linear attention.

use anyhow::{bail, Result};

use crate::nn;
use crate::ops::functional::{causal_conv1d, conv2d, reshape, CausalConv1dResult, Conv2dParams2d};

/// 2D convolution (PyTorch `nn.Conv2d`) over NHWC inputs, with an OIHW `weight`
/// (`[out_channels, in_channels / groups, kh, kw]`) and a `[out_channels]` `bias`.
#[nn::module]
pub struct Conv2d {
    pub weight: Tensor,
    pub bias: Tensor,
    #[module(config)]
    pub params: Conv2dParams2d,
}

#[nn::module]
impl Conv2d {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        conv2d(x, &self.weight, Some(&self.bias), self.params)
    }
}

/// Depthwise causal 1D convolution with a `[channels, 1, kernel]` `weight`. It matches PyTorch
/// `nn.Conv1d` with `groups = channels`, left padding `kernel - 1` and no bias. It carries its
/// input window from one call to the next.
#[nn::module]
pub struct CausalConv1d {
    pub weight: Tensor,
}

#[nn::module]
impl CausalConv1d {
    /// Convolves `x` (`[T, channels]`) after the earlier inputs in `state`, a
    /// `[kernel - 1, channels]` window (see [`causal_conv1d`]).
    fn forward(&self, x: &Tensor, state: &Tensor) -> Result<CausalConv1dResult<B>> {
        let [channels, _, kernel] = self.weight.shape().dims() else {
            bail!(
                "causal conv1d weight must be [channels, 1, kernel], got {:?}",
                self.weight.shape().dims()
            );
        };
        let weight = reshape(&self.weight, &[*channels, *kernel])?;
        causal_conv1d(x, state, &weight)
    }
}
