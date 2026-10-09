//! GPT-2, laid out like the Hugging Face `GPT2LMHeadModel`.
//!
//! `wte + wpe` -> `n_layer` x `h.{i}` \[`ln_1` -> `attn` -> +residual -> `ln_2` -> `mlp` ->
//! +residual\] -> `ln_f` -> `lm_head` ([`CausalDecoder`] with the base model as `transformer`).
//!
//! Model-specific: learned absolute positions (`wpe`), layer norms, multi-head attention whose
//! fused `c_attn` projection emits the queries, keys and values, and a non-gated MLP. Every
//! projection except `lm_head` stores its weight as `[in_features, out_features]`, the `Conv1D`
//! of `transformers`.

use anyhow::Result;

use crate::inference::decoder::{CausalDecoder, DecoderConfig, DecoderLayout};
use crate::nn::{
    ActivationFunction, AttentionConfig, DecoderNames, MixerConfig, MlpConfig, NormConfig,
};
use crate::ops::functional::WeightLayout;

pub const KIND: &str = "gpt2";

/// GPT-2 decoder.
pub type Gpt<B> = CausalDecoder<B, GptConfig>;

/// The keys of a Hugging Face `GPT2Config` that the model reads.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct GptConfig {
    pub n_positions: usize,
    pub n_embd: usize,
    pub n_layer: usize,
    pub n_head: usize,
    pub layer_norm_epsilon: f32,
    pub activation_function: ActivationFunction,
    #[serde(default = "tied_by_default")]
    pub tie_word_embeddings: bool,
}

/// `GPT2Config` ties `lm_head` to `wte` unless the config says otherwise.
fn tied_by_default() -> bool {
    true
}

impl DecoderNames for GptConfig {
    const EMBED_TOKENS: &'static str = "wte";
    const EMBED_POSITIONS: &'static str = "wpe";
    const LAYERS: &'static str = "h";
    const NORM: &'static str = "ln_f";
    const INPUT_LAYERNORM: &'static str = "ln_1";
    const POST_ATTENTION_LAYERNORM: &'static str = "ln_2";
    const SELF_ATTN: &'static str = "attn";
    const QKV_PROJ: &'static str = "c_attn";
    const O_PROJ: &'static str = "c_proj";
    const UP_PROJ: &'static str = "c_fc";
    const DOWN_PROJ: &'static str = "c_proj";
}

impl DecoderConfig for GptConfig {
    const KIND: &'static str = KIND;
    const BASE_MODEL_PREFIX: &'static str = "transformer";

    fn context_length(&self) -> usize {
        self.n_positions
    }

    fn tie_word_embeddings(&self) -> bool {
        self.tie_word_embeddings
    }

    fn layout(&self) -> Result<DecoderLayout> {
        let attention = AttentionConfig::with_equal_heads(self.n_embd, self.n_head)?
            .with_fused_qkv()
            .with_weight_layout(WeightLayout::InOut);
        Ok(DecoderLayout {
            norm: NormConfig::Layer {
                eps: self.layer_norm_epsilon,
            },
            mixers: vec![MixerConfig::Attention(attention); self.n_layer],
            mlp: MlpConfig {
                gated: false,
                act: self.activation_function,
                weight_layout: WeightLayout::InOut,
            },
            rope: None,
            absolute_positions: true,
        })
    }
}
