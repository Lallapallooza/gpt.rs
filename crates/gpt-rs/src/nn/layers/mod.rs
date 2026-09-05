//! Layers built from reusable functionals. Each layer is a [`crate::nn::module`].

pub mod attention;
pub mod conv;
pub mod decoder_block;
pub mod embedding;
pub mod feed_forward;
pub mod gated_delta_net;
pub mod gated_feed_forward;
pub mod layer_norm;
pub mod linear;
pub mod rms_norm;
pub mod rotary_embedding;

pub use attention::{AttentionConfig, AttentionPositions, CausalSelfAttention};
pub use conv::{CausalConv1d, Conv2d};
pub use decoder_block::{DecoderBlock, Mixer, MixerConfig, Mlp, MlpConfig, Norm, NormConfig};
pub use embedding::Embedding;
pub use feed_forward::{Activation, ActivationFunction, FeedForward};
pub use gated_delta_net::{GatedDeltaNet, GatedDeltaNetConfig};
pub use gated_feed_forward::GatedFeedForward;
pub use layer_norm::LayerNorm;
pub use linear::Linear;
pub use rms_norm::{RmsNorm, RmsNormConfig};
pub use rotary_embedding::{RopeConfig, RopeParameters, RopeScaling, RotaryEmbedding};
