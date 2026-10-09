//! Parameter-free elementwise activations, named as in Hugging Face configs.

use crate::backend::PortableBackend;
use crate::nn;
use crate::ops::functional;

/// Activation function, serialized with its Hugging Face name (`hidden_act`,
/// `activation_function`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ActivationFunction {
    /// Exact GELU.
    #[serde(rename = "gelu")]
    Gelu,
    /// Tanh-approximated GELU.
    #[serde(rename = "gelu_new", alias = "gelu_pytorch_tanh")]
    GeluTanh,
    /// `max(x, 0)`.
    #[serde(rename = "relu")]
    Relu,
    /// `min(max(x, 0), 6)`.
    #[serde(rename = "relu6")]
    Relu6,
    /// `x * sigmoid(x)`.
    #[serde(rename = "silu", alias = "swish")]
    Silu,
}

impl ActivationFunction {
    /// Returns the layer that applies this function.
    pub fn layer<B: PortableBackend + 'static>(self) -> Activation<B> {
        let function = match self {
            ActivationFunction::Gelu => functional::gelu,
            ActivationFunction::GeluTanh => functional::gelu_tanh,
            ActivationFunction::Relu => functional::relu,
            ActivationFunction::Relu6 => functional::relu6,
            ActivationFunction::Silu => functional::silu,
        };
        Activation { function }
    }
}

/// Parameter-free elementwise activation layer, built by [`ActivationFunction::layer`].
#[nn::module]
pub struct Activation {
    #[module(config)]
    function: fn(&Tensor) -> Tensor,
}

#[nn::module]
impl Activation {
    fn forward(&self, x: &Tensor) -> Tensor {
        (self.function)(x)
    }
}
