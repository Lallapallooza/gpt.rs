//! Llama-style decoder (Ministral).
//!
//! `embed_tokens` -> `num_hidden_layers` x \[`input_layernorm` -> `self_attn` -> +residual ->
//! `post_attention_layernorm` -> `mlp` -> +residual\] -> `norm` -> `lm_head`
//! ([`CausalDecoder`], [`DecoderBlock`](crate::nn::DecoderBlock)).
//!
//! Model-specific: [`RmsNorm`](crate::nn::RmsNorm)s, grouped-query
//! [`CausalSelfAttention`](crate::nn::CausalSelfAttention) with rotary embeddings
//! ([`RotaryEmbedding`](crate::nn::RotaryEmbedding), optionally linear or YaRN scaled) and a gated
//! MLP ([`FeedForward`](crate::nn::FeedForward)). The model has no biases.

use anyhow::Result;

use crate::inference::decoder::{CausalDecoder, DecoderConfig, DecoderLayout};
use crate::nn::{
    ActivationFunction, AttentionConfig, DecoderNames, MixerConfig, MlpConfig, NormConfig,
    RmsNormConfig, RopeParameters,
};
use crate::ops::functional::WeightLayout;

pub const KIND: &str = "ministral3";

/// Ministral decoder.
pub type Ministral<B> = CausalDecoder<B, MinistralConfig>;

/// The keys of a Hugging Face `Ministral3Config` that the model reads.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MinistralConfig {
    pub max_position_embeddings: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f32,
    pub rope_parameters: RopeParameters,
    #[serde(default = "super::silu")]
    pub hidden_act: ActivationFunction,
    #[serde(default)]
    pub tie_word_embeddings: bool,
}

impl DecoderNames for MinistralConfig {}

impl DecoderConfig for MinistralConfig {
    const KIND: &'static str = KIND;
    const BASE_MODEL_PREFIX: &'static str = "model";

    fn context_length(&self) -> usize {
        self.max_position_embeddings
    }

    fn tie_word_embeddings(&self) -> bool {
        self.tie_word_embeddings
    }

    fn layout(&self) -> Result<DecoderLayout> {
        let attention = AttentionConfig::with_projection_dims(
            self.hidden_size,
            self.num_attention_heads,
            self.num_key_value_heads,
            self.head_dim,
        )?;
        Ok(DecoderLayout {
            norm: NormConfig::Rms(RmsNormConfig {
                eps: self.rms_norm_eps,
                unit_offset: false,
            }),
            mixers: vec![MixerConfig::Attention(attention); self.num_hidden_layers],
            mlp: MlpConfig {
                gated: true,
                act: self.hidden_act,
                weight_layout: WeightLayout::OutIn,
            },
            rope: Some(self.rope_parameters.rope(self.head_dim)),
            absolute_positions: false,
        })
    }
}
