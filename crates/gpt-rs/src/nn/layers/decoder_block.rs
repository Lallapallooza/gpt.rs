//! Pre-norm decoder layer: `h = x + mixer(norm1(x))`, `out = h + mlp(norm2(h))`.

use std::sync::Arc;

use anyhow::Result;

use super::attention::{AttentionConfig, CausalSelfAttention};
use super::feed_forward::{FeedForward, MlpConfig};
use super::gated_delta_net::{GatedDeltaNet, GatedDeltaNetConfig};
use super::names::DecoderNames;
use super::norm::{Norm, NormConfig};
use crate::inference::LayerCache;
use crate::nn::{self, capture, Params};
use crate::ops::functional::AttentionPositions;

/// Hugging Face module name of linear attention inside a decoder layer.
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
pub enum Mixer<N: DecoderNames> {
    #[module(rename = N::SELF_ATTN)]
    Attention(CausalSelfAttention<N>),
    #[module(rename = LINEAR_ATTN)]
    LinearAttention(GatedDeltaNet),
}

#[nn::module]
impl<N: DecoderNames> Mixer<N> {
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
            Mixer::Attention(_) => N::SELF_ATTN,
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
    ) -> (Tensor, LayerCache<B>) {
        let _path = capture::scope(format_args!("{}", self.name()));
        match (self, cache) {
            (Mixer::Attention(attn), LayerCache::Attention(kv)) => {
                let (out, kv) = attn.call((x, kv, positions));
                (out, LayerCache::Attention(kv))
            }
            (Mixer::LinearAttention(lin), LayerCache::LinearAttention(state)) => {
                let (out, state) = lin.call((x, state));
                (out, LayerCache::LinearAttention(state))
            }
            _ => panic!("layer cache kind does not match the {} layer", self.name()),
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

/// Pre-norm residual decoder layer, with the module names of the family `N`.
#[nn::module]
pub struct DecoderBlock<N: DecoderNames> {
    #[module(rename = N::INPUT_LAYERNORM)]
    pub input_layernorm: Norm,
    /// Its parameters are under the mixer's own name ([`Mixer::name`]).
    #[module(flatten)]
    pub mixer: Mixer<N>,
    #[module(rename = N::POST_ATTENTION_LAYERNORM)]
    pub post_attention_layernorm: Norm,
    pub mlp: FeedForward<N>,
}

#[nn::module]
impl<N: DecoderNames> DecoderBlock<N> {
    pub fn new(
        p: &mut Params<B>,
        norm: NormConfig,
        mixer: &MixerConfig,
        mlp: MlpConfig,
    ) -> Result<Self> {
        Ok(Self {
            input_layernorm: Norm::new(p, norm)?,
            mixer: Mixer::new(p, mixer)?,
            post_attention_layernorm: Norm::new(p, norm)?,
            mlp: FeedForward::new(p, mlp)?,
        })
    }

    /// Runs the layer on `x` (`[T, dim]`) at `positions`. Returns the output and the updated cache.
    fn forward(
        &self,
        x: &Tensor,
        cache: &LayerCache<B>,
        positions: &AttentionPositions<B>,
    ) -> (Tensor, LayerCache<B>) {
        let normed = self.input_layernorm(x);
        capture::record(N::INPUT_LAYERNORM, &normed);
        let (mixed, cache) = self.mixer((&normed, cache, positions));
        capture::record(self.mixer.name(), &mixed);
        let residual = mixed + x;
        let normed = self.post_attention_layernorm(&residual);
        capture::record(N::POST_ATTENTION_LAYERNORM, &normed);
        let mlp = self.mlp(&normed);
        capture::record("mlp", &mlp);
        (mlp + residual, cache)
    }
}
