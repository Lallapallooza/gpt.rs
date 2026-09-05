//! GPT-2.
//!
//! `embed_tokens + embed_positions` → `n_layer` × [`input_layernorm` → `self_attn` → +residual →
//! `post_attention_layernorm` → `mlp` → +residual] → `norm` → `lm_head`
//! ([`CausalDecoder`], [`DecoderBlock`](crate::nn::DecoderBlock)).
//!
//! Model-specific: learned absolute positions, [`LayerNorm`](crate::nn::LayerNorm)s, biased
//! multi-head [`CausalSelfAttention`](crate::nn::CausalSelfAttention) without rotary embedding and
//! a non-gated [`FeedForward`](crate::nn::FeedForward) MLP.

use crate::inference::decoder::{CausalDecoder, DecoderConfig, DecoderLayout};
use crate::nn::{ActivationFunction, AttentionConfig, MixerConfig, MlpConfig, NormConfig};
use anyhow::Result;

pub const KIND: &str = "gpt";

/// GPT-2 decoder.
pub type Gpt<B> = CausalDecoder<B, GptConfig>;

/// The keys of a Hugging Face `GPT2Config` that the model reads.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct GptConfig {
    pub vocab_size: usize,
    pub n_positions: usize,
    pub n_embd: usize,
    pub n_layer: usize,
    pub n_head: usize,
    pub n_inner: Option<usize>,
    pub layer_norm_epsilon: f32,
    pub activation_function: ActivationFunction,
}

impl DecoderConfig for GptConfig {
    const KIND: &'static str = KIND;

    fn layout(&self) -> Result<DecoderLayout> {
        let attention =
            AttentionConfig::with_equal_heads(self.n_embd, self.n_head)?.with_bias(true);
        Ok(DecoderLayout {
            vocab_size: self.vocab_size,
            context_length: self.n_positions,
            embed_dim: self.n_embd,
            learned_positions: true,
            norm: NormConfig::LayerNorm {
                eps: self.layer_norm_epsilon,
            },
            mixers: vec![MixerConfig::Attention(attention); self.n_layer],
            mlp: MlpConfig::FeedForward {
                hidden_dim: self.n_inner.unwrap_or(4 * self.n_embd),
                activation: self.activation_function,
                bias: true,
            },
        })
    }
}
