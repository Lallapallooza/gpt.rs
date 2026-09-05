//! Position-wise feed-forward block used inside transformer layers.

use super::linear::Linear;
use crate::backend::spec::PortableBackend;
use crate::nn::{self, LayerLoader};
use crate::ops::functional;
use anyhow::Result;

/// Activation function of a [`FeedForward`], serialized with its Hugging Face name.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ActivationFunction {
    /// Exact GELU.
    #[serde(rename = "gelu")]
    Gelu,
    /// Tanh-approximated GELU.
    #[serde(rename = "gelu_new", alias = "gelu_pytorch_tanh")]
    GeluTanh,
}

impl ActivationFunction {
    /// Returns the layer that applies this function.
    pub fn layer<B: PortableBackend + 'static>(self) -> Activation<B> {
        let function = match self {
            ActivationFunction::Gelu => functional::gelu,
            ActivationFunction::GeluTanh => functional::gelu_tanh,
        };
        Activation { function }
    }
}

/// Parameter-free elementwise activation layer, built by [`ActivationFunction::layer`].
#[nn::module]
pub struct Activation {
    #[module(config)]
    function: fn(&Tensor) -> Result<Tensor>,
}

#[nn::module]
impl Activation {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        (self.function)(x)
    }
}

/// Transformer feed-forward network `down_proj(activation(up_proj(x)))`.
#[nn::module]
pub struct FeedForward {
    pub up_proj: Linear,
    pub activation: Activation,
    pub down_proj: Linear,
}

#[nn::module]
impl FeedForward {
    pub fn load(
        params: &mut LayerLoader<'_, B>,
        prefix: &str,
        dim: usize,
        hidden_dim: usize,
        activation: ActivationFunction,
        bias: bool,
    ) -> Result<Self> {
        Ok(Self {
            up_proj: params.linear(&format!("{prefix}.up_proj"), dim, hidden_dim, bias)?,
            activation: activation.layer(),
            down_proj: params.linear(&format!("{prefix}.down_proj"), hidden_dim, dim, bias)?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let up = self.up_proj(x)?;
        let act = self.activation(&up)?;
        self.down_proj(&act)
    }
}
