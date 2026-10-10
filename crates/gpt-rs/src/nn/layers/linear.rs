//! The projection layer built on [`functional::linear`]: PyTorch `nn.Linear`, and `transformers`'
//! `Conv1D` with its transposed weight.

use anyhow::Result;

use crate::nn::{self, Params};
use crate::ops::functional::{self, WeightLayout};
use crate::tensor::DType;

/// Fully connected layer `y = x W^T + b`. `weight` is stored as `layout` says: `[out_features,
/// in_features]` like `nn.Linear`, or `[in_features, out_features]` like the `Conv1D` of GPT-2
/// checkpoints. An FP8 checkpoint stores `W` as `weight * weight_scale_inv`, in the dtype of the
/// scale.
#[nn::module]
pub struct Linear {
    pub weight: Tensor,
    pub weight_scale_inv: Option<Tensor>,
    pub bias: Option<Tensor>,
    #[module(config)]
    pub layout: WeightLayout,
    #[module(config)]
    pub input_dtype: Option<DType>,
}

#[nn::module]
impl Linear {
    /// A projection with an `[out_features, in_features]` weight, like `nn.Linear`.
    pub fn new(p: &mut Params<B>) -> Result<Self> {
        Self::with_layout(p, WeightLayout::OutIn)
    }

    /// A projection whose weight the checkpoint stores as `layout` says.
    pub fn with_layout(p: &mut Params<B>, layout: WeightLayout) -> Result<Self> {
        Ok(Self {
            weight: p.param()?,
            weight_scale_inv: p.optional_param()?,
            bias: p.optional_param()?,
            layout,
            input_dtype: p.linear_input_dtype(),
        })
    }

    /// A bias-free projection that shares `weight` with another layer, like a language-model head
    /// tied to the token embedding.
    pub fn tied(weight: &Tensor, input_dtype: Option<DType>) -> Self {
        Self {
            weight: weight.clone(),
            weight_scale_inv: None,
            bias: None,
            layout: WeightLayout::OutIn,
            input_dtype,
        }
    }

    /// Projects `input` (`[N, in_features]`) to `[N, out_features]` in f32.
    fn forward(&self, input: &Tensor) -> Tensor {
        let input = match self.input_dtype {
            Some(dtype) => input.cast(dtype),
            None => input.clone(),
        };
        // The dequantization stays in the graph, so a backend can read the FP8 weight in place.
        let weight = match &self.weight_scale_inv {
            Some(scale) => self.weight.cast(scale.dtype()) * scale,
            None => self.weight.clone(),
        };
        let output = functional::linear(&input, &weight, self.layout);
        match &self.bias {
            Some(bias) => output + bias,
            None => output,
        }
    }
}
