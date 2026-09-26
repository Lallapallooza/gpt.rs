//! Convolution layers: 2D image convolutions (NHWC internal layout) and the depthwise causal 1D
//! convolution of linear attention.

use anyhow::{bail, ensure, Result};

use crate::nn::{self, Params};
use crate::ops::functional::{
    causal_conv1d, conv2d, reshape, CausalConv1dResult, Conv2dParams2d, Padding2d,
};

/// How a [`Conv2d`] pads its input.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConvPadding {
    /// `(kernel - 1) / 2 * dilation` on every side, which keeps the size at stride 1 for odd
    /// kernels.
    Centered,
    /// TensorFlow `SAME` padding for the input size.
    TfSame,
}

/// 2D convolution (PyTorch `nn.Conv2d`) over NHWC inputs, with an OIHW `weight`
/// (`[out_channels, in_channels / groups, kh, kw]`) and an optional `[out_channels]` `bias`. The
/// kernel size comes from `weight`, and the groups from the input channels.
#[nn::module]
pub struct Conv2d {
    pub weight: Tensor,
    pub bias: Option<Tensor>,
    #[module(config)]
    pub stride: usize,
    #[module(config)]
    pub dilation: usize,
    #[module(config)]
    pub padding: ConvPadding,
}

#[nn::module]
impl Conv2d {
    pub fn new(
        p: &mut Params<B>,
        stride: usize,
        dilation: usize,
        padding: ConvPadding,
    ) -> Result<Self> {
        Ok(Self {
            weight: p.param()?,
            bias: p.optional_param()?,
            stride,
            dilation,
            padding,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (input, weight) = (x.shape().dims(), self.weight.shape().dims());
        let ([_, height, width, channels], [_, group_channels, kh, kw]) = (input, weight) else {
            bail!("conv2d expects NHWC input and an OIHW weight, got {input:?} and {weight:?}");
        };
        ensure!(
            *group_channels > 0 && channels.is_multiple_of(*group_channels),
            "conv2d input channels {channels} must be a multiple of the weight's {group_channels} input channels"
        );
        let mut params = Conv2dParams2d {
            kernel: [*kh, *kw],
            stride: [self.stride; 2],
            dilation: [self.dilation; 2],
            padding: Padding2d::zero(),
            groups: channels / group_channels,
        };
        params.padding = match self.padding {
            ConvPadding::Centered => {
                let [pad_h, pad_w] = params.kernel.map(|k| (k - 1) / 2 * self.dilation);
                Padding2d {
                    top: pad_h,
                    bottom: pad_h,
                    left: pad_w,
                    right: pad_w,
                }
            }
            ConvPadding::TfSame => tf_same_padding([*height, *width], &params),
        };
        conv2d(x, &self.weight, self.bias.as_ref(), params)
    }
}

/// TensorFlow `SAME` padding, as `transformers` applies it: the output has `ceil(input / stride)`
/// positions, and the odd padding row or column goes after the input.
fn tf_same_padding(input: [usize; 2], params: &Conv2dParams2d) -> Padding2d {
    let [(top, bottom), (left, right)] = [0, 1].map(|axis| {
        let (size, kernel, stride) = (input[axis], params.kernel[axis], params.stride[axis]);
        let remainder = match size % stride {
            0 => stride,
            remainder => remainder,
        };
        let total = kernel.saturating_sub(remainder) * params.dilation[axis];
        let before = kernel.saturating_sub(remainder) / 2 * params.dilation[axis];
        (before, total - before)
    });
    Padding2d {
        top,
        bottom,
        left,
        right,
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
    pub fn new(p: &mut Params<B>) -> Result<Self> {
        Ok(Self { weight: p.param()? })
    }

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
