//! GPT-style layers built from reusable functional primitives.
//!
//! Each layer wraps lower-level functional ops and exposes ergonomic `forward` helpers.
//! Layers intentionally focus on inference-only forward paths.

pub mod attention;
pub mod conv;
pub mod embedding;
pub mod feed_forward;
pub mod layer_norm;
pub mod linear;
pub mod rms_norm;

pub use crate::ops::functional::AttentionCache;
pub use attention::{AttentionConfig, CausalSelfAttention};
pub use conv::Conv2d;
pub use embedding::Embedding;
pub use feed_forward::FeedForward;
pub use layer_norm::LayerNorm;
pub use linear::Linear;
pub use rms_norm::RmsNorm;
