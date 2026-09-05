//! Qwen3.5-family hybrid decoder (text model).
//!
//! `embed_tokens` → one layer per `layer_types` entry: [`input_layernorm` → `linear_attn` or
//! `self_attn` → +residual → `post_attention_layernorm` → `mlp` → +residual] → `norm` → `lm_head`
//! ([`CausalDecoder`], [`DecoderBlock`](crate::nn::DecoderBlock)).
//!
//! Model-specific:
//! - `layer_types` interleaves Gated DeltaNet linear attention
//!   ([`GatedDeltaNet`](crate::nn::GatedDeltaNet)) with softmax attention
//!   ([`CausalSelfAttention`](crate::nn::CausalSelfAttention)).
//! - Attention has an output gate, q/k norms and a rotary embedding of the leading
//!   `partial_rotary_factor` of each head.
//! - Every RMSNorm except the DeltaNet output norm scales by `1 + weight`
//!   ([`RmsNorm::unit_offset`](crate::nn::RmsNorm::unit_offset)).
//! - The MLP is gated ([`GatedFeedForward`](crate::nn::GatedFeedForward)).

use anyhow::Result;

use crate::inference::decoder::{CausalDecoder, DecoderConfig, DecoderLayout};
use crate::nn::{
    AttentionConfig, GatedDeltaNetConfig, MixerConfig, MlpConfig, NormConfig, RmsNormConfig,
    RopeParameters,
};

pub const KIND: &str = "qwen3_5";

/// Qwen3.5 text decoder.
pub type Qwen35<B> = CausalDecoder<B, Qwen35Config>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Qwen35LayerType {
    LinearAttention,
    FullAttention,
}

/// The keys of a Hugging Face `Qwen3_5TextConfig` that the model reads.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Qwen35Config {
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub hidden_size: usize,
    pub layer_types: Vec<Qwen35LayerType>,
    pub intermediate_size: usize,
    pub rms_norm_eps: f32,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub attention_bias: bool,
    pub rope_parameters: RopeParameters,
    pub linear_num_key_heads: usize,
    pub linear_num_value_heads: usize,
    pub linear_key_head_dim: usize,
    pub linear_value_head_dim: usize,
    pub linear_conv_kernel_dim: usize,
}

impl DecoderConfig for Qwen35Config {
    const KIND: &'static str = KIND;

    fn layout(&self) -> Result<DecoderLayout> {
        let norm = RmsNormConfig {
            eps: self.rms_norm_eps,
            unit_offset: true,
        };
        let attention = AttentionConfig::with_projection_dims(
            self.hidden_size,
            self.num_attention_heads,
            self.num_key_value_heads,
            self.head_dim,
        )?
        .with_bias(self.attention_bias)
        .with_qk_norm(norm)
        .with_rope(self.rope_parameters.rope(self.head_dim))?
        .with_output_gate();
        let linear_attention = GatedDeltaNetConfig {
            embed_dim: self.hidden_size,
            num_key_heads: self.linear_num_key_heads,
            num_value_heads: self.linear_num_value_heads,
            key_head_dim: self.linear_key_head_dim,
            value_head_dim: self.linear_value_head_dim,
            conv_kernel_dim: self.linear_conv_kernel_dim,
            rms_norm_eps: self.rms_norm_eps,
        };
        let mixers = self
            .layer_types
            .iter()
            .map(|layer_type| match layer_type {
                Qwen35LayerType::FullAttention => MixerConfig::Attention(attention.clone()),
                Qwen35LayerType::LinearAttention => {
                    MixerConfig::LinearAttention(linear_attention.clone())
                }
            })
            .collect();
        Ok(DecoderLayout {
            vocab_size: self.vocab_size,
            context_length: self.max_position_embeddings,
            embed_dim: self.hidden_size,
            learned_positions: false,
            norm: NormConfig::RmsNorm(norm),
            mixers,
            mlp: MlpConfig::Gated {
                hidden_dim: self.intermediate_size,
            },
        })
    }
}
