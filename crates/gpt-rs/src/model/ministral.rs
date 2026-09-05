//! Llama-style decoder (Mistral, Ministral).
//!
//! `embed_tokens` → `num_hidden_layers` × [`input_layernorm` → `self_attn` → +residual →
//! `post_attention_layernorm` → `mlp` → +residual] → `norm` → `lm_head`
//! ([`CausalDecoder`], [`DecoderBlock`](crate::nn::DecoderBlock)).
//!
//! Model-specific: [`RmsNorm`](crate::nn::RmsNorm)s, grouped-query
//! [`CausalSelfAttention`](crate::nn::CausalSelfAttention) with rotary embeddings
//! ([`RotaryEmbedding`](crate::nn::RotaryEmbedding), optionally linear or YaRN scaled) and a gated
//! MLP ([`GatedFeedForward`](crate::nn::GatedFeedForward)). The model has no biases.

use crate::inference::decoder::{CausalDecoder, DecoderConfig, DecoderLayout};
use crate::nn::{
    AttentionConfig, MixerConfig, MlpConfig, NormConfig, RmsNormConfig, RopeParameters,
};
use anyhow::Result;

pub const KIND: &str = "ministral";

/// Ministral decoder.
pub type Ministral<B> = CausalDecoder<B, MinistralConfig>;

/// The keys of a Hugging Face `MistralConfig` / `Ministral3Config` that the model reads.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MinistralConfig {
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub rms_norm_eps: f32,
    pub rope_parameters: RopeParameters,
}

impl DecoderConfig for MinistralConfig {
    const KIND: &'static str = KIND;

    fn layout(&self) -> Result<DecoderLayout> {
        let attention = AttentionConfig::with_projection_dims(
            self.hidden_size,
            self.num_attention_heads,
            self.num_key_value_heads,
            self.head_dim,
        )?
        .with_rope(self.rope_parameters.rope(self.head_dim))?;
        Ok(DecoderLayout {
            vocab_size: self.vocab_size,
            context_length: self.max_position_embeddings,
            embed_dim: self.hidden_size,
            learned_positions: false,
            norm: NormConfig::RmsNorm(RmsNormConfig {
                eps: self.rms_norm_eps,
                unit_offset: false,
            }),
            mixers: vec![MixerConfig::Attention(attention); self.num_hidden_layers],
            mlp: MlpConfig::Gated {
                hidden_dim: self.intermediate_size,
            },
        })
    }
}
