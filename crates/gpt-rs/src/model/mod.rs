pub mod config;
pub mod gpt;
pub mod ministral;
pub mod mobilenet_v2;
pub mod qwen3_5;
pub mod registry;
pub mod resnet;

pub use self::config::ModelConfig;
pub use self::gpt::{Gpt, GptConfig};
pub use self::ministral::{Ministral, MinistralConfig};
pub use self::mobilenet_v2::{MobileNetV2, MobileNetV2Config};
pub use self::qwen3_5::{Qwen35, Qwen35Config};
pub use self::resnet::{ResNet, ResNetConfig};

/// `silu`, the `hidden_act` that `transformers` configs of the Llama family default to.
fn silu() -> crate::nn::ActivationFunction {
    crate::nn::ActivationFunction::Silu
}
