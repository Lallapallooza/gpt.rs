//! Linear projection layer (PyTorch `nn.Linear`) built on [`functional::linear`].

use crate::nn;
use crate::ops::functional;
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
    /// Projects `input` (`[N, in_features]`) to `[N, out_features]` in f32.
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let input = match self.input_dtype {
            Some(dtype) => functional::cast(input, dtype)?,
            None => input.clone(),
        };
        let output = functional::linear(&input, &self.weight)?;
        match &self.bias {
            Some(bias) => functional::add_bias(&output, bias),
            None => Ok(output),
        }
    }
}
