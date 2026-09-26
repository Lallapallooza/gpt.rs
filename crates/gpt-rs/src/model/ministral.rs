//! Llama-style decoder (Ministral).
//!
//! `embed_tokens` -> `num_hidden_layers` x \[`input_layernorm` -> `self_attn` -> +residual ->
//! `post_attention_layernorm` -> `mlp` -> +residual\] -> `norm` -> `lm_head`
//! ([`CausalDecoder`], [`DecoderBlock`](crate::nn::DecoderBlock)).
//!
//! Model-specific: [`RmsNorm`](crate::nn::RmsNorm)s, grouped-query
//! [`CausalSelfAttention`](crate::nn::CausalSelfAttention) with rotary embeddings
//! ([`RotaryEmbedding`](crate::nn::RotaryEmbedding), optionally linear or YaRN scaled) and a gated
//! MLP ([`GatedFeedForward`](crate::nn::GatedFeedForward)). The model has no biases.

use crate::backend::spec::PortableBackend;
use crate::inference::decoder::{CausalDecoder, DecoderConfig, DecoderLayout, DecoderModel};
use crate::nn::{AttentionConfig, MixerConfig, Params, RmsNormConfig, RopeParameters};
use anyhow::Result;

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
    #[serde(default)]
    pub tie_word_embeddings: bool,
}

impl DecoderConfig for MinistralConfig {
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

impl MinistralConfig {
    fn layout(&self) -> Result<DecoderLayout> {
        let attention = AttentionConfig::with_projection_dims(
            self.hidden_size,
            self.num_attention_heads,
            self.num_key_value_heads,
            self.head_dim,
        )?
        .with_rope(self.rope_parameters.rope(self.head_dim))?;
        Ok(DecoderLayout {
            norm: RmsNormConfig {
                eps: self.rms_norm_eps,
                unit_offset: false,
            },
            mixers: vec![MixerConfig::Attention(attention); self.num_hidden_layers],
        })
    }
}
