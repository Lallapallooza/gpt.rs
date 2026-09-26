//! Projection layers built on [`functional::linear`]: PyTorch `nn.Linear` and `transformers`'
//! `Conv1D`.

use crate::nn::{self, Params};
use crate::ops::functional::{self, WeightLayout};
use crate::tensor::DType;
use anyhow::Result;

/// Fully connected layer `y = x W^T + b` with `weight` stored as `[out_features, in_features]`.
#[nn::module]
pub struct Linear {
    pub weight: Tensor,
    pub bias: Option<Tensor>,
    #[module(config)]
    pub input_dtype: Option<DType>,
}

#[nn::module]
impl Linear {
    pub fn new(p: &mut Params<B>) -> Result<Self> {
        Ok(Self {
            weight: p.param()?,
            bias: p.optional_param()?,
            input_dtype: p.linear_input_dtype(),
        })
    }

    /// A bias-free projection that shares `weight` with another layer, like a language-model head
    /// tied to the token embedding.
    pub fn tied(weight: &Tensor, input_dtype: Option<DType>) -> Self {
        Self {
            weight: weight.clone(),
            bias: None,
            input_dtype,
        }
    }

    /// Projects `input` (`[N, in_features]`) to `[N, out_features]` in f32.
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let input = match self.input_dtype {
            Some(dtype) => functional::cast(input, dtype)?,
            None => input.clone(),
        };
        let output = functional::linear(&input, &self.weight, WeightLayout::OutIn)?;
        match &self.bias {
            Some(bias) => functional::add_bias(&output, bias),
            None => Ok(output),
        }
    }
}

/// Fully connected layer `y = x W + b` with `weight` stored as `[in_features, out_features]`, the
/// `Conv1D` of GPT-2 checkpoints.
#[nn::module]
pub struct Conv1D {
    pub weight: Tensor,
    pub bias: Tensor,
    #[module(config)]
    pub input_dtype: Option<DType>,
}

#[nn::module]
impl Conv1D {
    pub fn new(p: &mut Params<B>) -> Result<Self> {
        Ok(Self {
            weight: p.param()?,
            bias: p.param()?,
            input_dtype: p.linear_input_dtype(),
        })
    }

    /// Projects `input` (`[N, in_features]`) to `[N, out_features]` in f32.
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let input = match self.input_dtype {
            Some(dtype) => functional::cast(input, dtype)?,
            None => input.clone(),
        };
        let output = functional::linear(&input, &self.weight, WeightLayout::InOut)?;
        functional::add_bias(&output, &self.bias)
    }
}
