//! Qwen3.5-family hybrid decoder (text model).
//!
//! `embed_tokens` -> `num_hidden_layers` x \[`input_layernorm` -> `linear_attn` or `self_attn` (by
//! `layer_types`) -> +residual -> `post_attention_layernorm` -> `mlp` -> +residual\] -> `norm` ->
//! `lm_head` ([`CausalDecoder`], [`DecoderBlock`](crate::nn::DecoderBlock)).
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

use anyhow::{anyhow, Result};

use crate::backend::spec::PortableBackend;
use crate::inference::decoder::{CausalDecoder, DecoderConfig, DecoderLayout, DecoderModel};
use crate::nn::{
    AttentionConfig, GatedDeltaNetConfig, MixerConfig, Params, RmsNormConfig, RopeParameters,
};

pub const KIND: &str = "qwen3_5_text";

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
    pub max_position_embeddings: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    /// Mixer of each layer. The model builds the first `num_hidden_layers` of them.
    pub layer_types: Vec<Qwen35LayerType>,
    pub rms_norm_eps: f32,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub rope_parameters: RopeParameters,
    pub linear_num_key_heads: usize,
    pub linear_num_value_heads: usize,
    pub linear_key_head_dim: usize,
    pub linear_value_head_dim: usize,
    pub linear_conv_kernel_dim: usize,
    #[serde(default)]
    pub tie_word_embeddings: bool,
}

impl DecoderConfig for Qwen35Config {
    const KIND: &'static str = KIND;
    const BASE_MODEL_PREFIX: &'static str = "model";

    type Model<B: PortableBackend + 'static> = DecoderModel<B>;

    fn context_length(&self) -> usize {
        self.max_position_embeddings
    }

    fn tie_word_embeddings(&self) -> bool {
        self.tie_word_embeddings
    }

    fn base_model<B: PortableBackend + 'static>(
        &self,
        p: &mut Params<B>,
    ) -> Result<DecoderModel<B>> {
        DecoderModel::new(p, self.layout()?)
    }
}

impl Qwen35Config {
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
        let layer_types = self
            .layer_types
            .get(..self.num_hidden_layers)
            .ok_or_else(|| {
                anyhow!(
                    "num_hidden_layers {} exceeds the {} layer_types",
                    self.num_hidden_layers,
                    self.layer_types.len()
                )
            })?;
        let mixers = layer_types
            .iter()
            .map(|layer_type| match layer_type {
                Qwen35LayerType::FullAttention => MixerConfig::Attention(attention.clone()),
                Qwen35LayerType::LinearAttention => {
                    MixerConfig::LinearAttention(linear_attention.clone())
                }
            })
            .collect();
        Ok(DecoderLayout { norm, mixers })
    }
}
