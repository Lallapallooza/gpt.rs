//! Pre-norm decoder layer of the Llama-style decoders: `h = x + mixer(norm1(x))`,
//! `out = h + mlp(norm2(h))`.

use std::sync::Arc;

use super::attention::{AttentionConfig, AttentionPositions, CausalSelfAttention};
use super::gated_delta_net::{GatedDeltaNet, GatedDeltaNetConfig};
use super::gated_feed_forward::GatedFeedForward;
use super::rms_norm::{RmsNorm, RmsNormConfig};
use crate::inference::LayerCache;
use crate::nn::{self, capture, Params};
use crate::ops::functional::DeviceTensorOps;
use anyhow::{bail, Result};

/// Hugging Face module names of the token mixers inside a decoder layer.
const SELF_ATTN: &str = "self_attn";
const LINEAR_ATTN: &str = "linear_attn";

/// Token mixer of one decoder layer.
#[derive(Debug, Clone)]
pub enum MixerConfig {
    /// Softmax attention (`self_attn`).
    Attention(AttentionConfig),
    /// Gated DeltaNet linear attention (`linear_attn`).
    LinearAttention(GatedDeltaNetConfig),
}

/// Token mixer of a decoder layer, named by its Hugging Face module.
#[nn::module]
pub enum Mixer {
    #[module(rename = SELF_ATTN)]
    Attention(CausalSelfAttention),
    #[module(rename = LINEAR_ATTN)]
    LinearAttention(GatedDeltaNet),
}

#[nn::module]
impl Mixer {
    pub fn new(p: &mut Params<B>, config: &MixerConfig) -> Result<Self> {
        Ok(match config {
            MixerConfig::Attention(config) => {
                Self::Attention(CausalSelfAttention::new(p, config.clone())?)
            }
            MixerConfig::LinearAttention(config) => {
                Self::LinearAttention(GatedDeltaNet::new(p, config.clone())?)
            }
        })
    }

    /// Hugging Face module name of the mixer inside its decoder layer.
    pub fn name(&self) -> &'static str {
        match self {
            Mixer::Attention(_) => SELF_ATTN,
            Mixer::LinearAttention(_) => LINEAR_ATTN,
        }
    }

    /// Mixes `x` (`[T, embed_dim]`) at `positions`, continuing from `cache`. Returns the output
    /// and the cache after the last token.
    fn forward(
        &self,
        x: &Tensor,
        cache: &LayerCache<B>,
        positions: &AttentionPositions<B>,
    ) -> Result<(Tensor, LayerCache<B>)> {
        match (self, cache) {
            (Mixer::Attention(attn), LayerCache::Attention(kv)) => {
                let (out, kv) = attn.call((x, kv, positions))?;
                Ok((out, LayerCache::Attention(kv)))
            }
            (Mixer::LinearAttention(lin), LayerCache::LinearAttention(state)) => {
                let (out, state) = lin.call((x, state))?;
                Ok((out, LayerCache::LinearAttention(state)))
            }
            _ => bail!("layer cache kind does not match the {} layer", self.name()),
        }
    }

    /// Empty cache on `backend` with room for `capacity` positions. Linear attention ignores
    /// `capacity` because its state has a fixed size.
    pub fn empty_cache(&self, backend: &Arc<B>, capacity: usize) -> Result<LayerCache<B>> {
        match self {
            Mixer::Attention(attn) => attn
                .config
                .empty_cache(backend, capacity)
                .map(LayerCache::Attention),
            Mixer::LinearAttention(lin) => {
                lin.empty_cache(backend).map(LayerCache::LinearAttention)
            }
        }
    }
}

/// Pre-norm residual decoder layer.
#[nn::module]
pub struct DecoderBlock {
    pub input_layernorm: RmsNorm,
    /// Its parameters are under the mixer's own name ([`Mixer::name`]).
    #[module(flatten)]
    pub mixer: Mixer,
    pub post_attention_layernorm: RmsNorm,
    pub mlp: GatedFeedForward,
}

#[nn::module]
impl DecoderBlock {
    pub fn new(p: &mut Params<B>, norm: RmsNormConfig, mixer: &MixerConfig) -> Result<Self> {
        Ok(Self {
            input_layernorm: RmsNorm::new(p, norm)?,
            mixer: Mixer::new(p, mixer)?,
            post_attention_layernorm: RmsNorm::new(p, norm)?,
            mlp: GatedFeedForward::new(p)?,
        })
    }

    /// Runs the layer on `x` (`[T, dim]`) at `positions`. Returns the output and the updated cache.
    fn forward(
        &self,
        x: &Tensor,
        cache: &LayerCache<B>,
        positions: &AttentionPositions<B>,
    ) -> Result<(Tensor, LayerCache<B>)> {
        let normed = self.input_layernorm(x)?;
        capture::record("input_layernorm", &normed)?;
        let (mixed, cache) = self.mixer((&normed, cache, positions))?;
        capture::record(self.mixer.name(), &mixed)?;
        let residual = mixed.add(x)?;
        let normed = self.post_attention_layernorm(&residual)?;
        capture::record("post_attention_layernorm", &normed)?;
        let mlp = self.mlp(&normed)?;
        capture::record("mlp", &mlp)?;
        Ok((mlp.add(&residual)?, cache))
    }
}
