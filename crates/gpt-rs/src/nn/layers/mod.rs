//! GPT-style layers built from reusable functional primitives.
//!
//! Each layer wraps lower-level functional ops, exposes ergonomic `forward` / `forward_with_state`
//! entry points, and returns state structs that capture intermediate tensors for later reuse.
//! Layers intentionally leave backward paths unimplemented on the portable backend for now, but
//! the placeholder types keep the API stable until gradients land.

pub mod attention;
pub mod embedding;
pub mod feed_forward;
pub mod layer_norm;
pub mod linear;

pub use crate::ops::functional::AttentionCache;
pub use attention::{
    AttentionConfig, CausalSelfAttention, CausalSelfAttentionGradients, CausalSelfAttentionState,
};
pub use embedding::{Embedding, EmbeddingGradients, EmbeddingState};
pub use feed_forward::{FeedForward, FeedForwardGradients, FeedForwardState};
pub use layer_norm::{LayerNorm, LayerNormGradients, LayerNormState};
pub use linear::{Linear, LinearGradients, LinearState};
