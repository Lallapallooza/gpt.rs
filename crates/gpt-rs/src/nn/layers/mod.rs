//! Layers built from reusable functionals. Each layer is a [`crate::nn::module`].

pub mod activation;
pub mod attention;
pub mod batch_norm;
pub mod conv;
pub mod decoder_block;
pub mod embedding;
pub mod feed_forward;
pub mod gated_delta_net;
pub mod layer_norm;
pub mod linear;
pub mod names;
pub mod norm;
pub mod rms_norm;
pub mod rotary_embedding;

pub use self::activation::{Activation, ActivationFunction};
pub use self::attention::{AttentionConfig, CausalSelfAttention};
pub use self::batch_norm::BatchNorm2d;
pub use self::conv::{CausalConv1d, Conv2d, ConvNormActivation, ConvPadding};
pub use self::decoder_block::{DecoderBlock, Mixer, MixerConfig};
pub use self::embedding::Embedding;
pub use self::feed_forward::{FeedForward, MlpConfig};
pub use self::gated_delta_net::{GatedDeltaNet, GatedDeltaNetConfig};
pub use self::layer_norm::LayerNorm;
pub use self::linear::Linear;
pub use self::names::{DecoderNames, LlamaNames};
pub use self::norm::{Norm, NormConfig};
pub use self::rms_norm::{RmsNorm, RmsNormConfig};
pub use self::rotary_embedding::{RopeConfig, RopeParameters, RopeScaling, RotaryEmbedding};
