//! Pre-norm decoder block shared by every causal decoder: `h = x + mixer(norm1(x))`,
//! `out = h + mlp(norm2(h))`.

use std::sync::Arc;

use super::attention::{AttentionConfig, AttentionPositions, CausalSelfAttention};
use super::feed_forward::{ActivationFunction, FeedForward};
use super::gated_delta_net::{GatedDeltaNet, GatedDeltaNetConfig};
use super::gated_feed_forward::GatedFeedForward;
use super::layer_norm::LayerNorm;
use super::rms_norm::{RmsNorm, RmsNormConfig};
use crate::backend::spec::PortableBackend;
use crate::inference::LayerCache;
use crate::nn::{self, capture, LayerLoader};
use crate::ops::functional::DeviceTensorOps;
use anyhow::{bail, Result};

/// Normalisation of the residual stream in a decoder.
#[derive(Debug, Clone, Copy)]
pub enum NormConfig {
    LayerNorm { eps: f32 },
    RmsNorm(RmsNormConfig),
}

impl NormConfig {
    pub fn load<B: PortableBackend + 'static>(
        self,
        params: &mut LayerLoader<'_, B>,
        prefix: &str,
        dim: usize,
    ) -> Result<Norm<B>> {
        Ok(match self {
            NormConfig::LayerNorm { eps } => Norm::LayerNorm(params.layer_norm(prefix, dim, eps)?),
            NormConfig::RmsNorm(config) => Norm::RmsNorm(params.rms_norm(prefix, dim, config)?),
        })
    }
}

#[nn::module]
pub enum Norm {
    LayerNorm(LayerNorm),
    RmsNorm(RmsNorm),
}

#[nn::module]
impl Norm {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            Norm::LayerNorm(norm) => norm.call(x),
            Norm::RmsNorm(norm) => norm.call(x),
        }
    }
}

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

impl MixerConfig {
    fn load<B: PortableBackend + 'static>(
        &self,
        params: &mut LayerLoader<'_, B>,
        layer: &str,
    ) -> Result<Mixer<B>> {
        Ok(match self {
            MixerConfig::Attention(config) => {
                let prefix = format!("{layer}.{SELF_ATTN}");
                Mixer::Attention(CausalSelfAttention::load(params, &prefix, config.clone())?)
            }
            MixerConfig::LinearAttention(config) => {
                let prefix = format!("{layer}.{LINEAR_ATTN}");
                Mixer::LinearAttention(GatedDeltaNet::load(params, &prefix, config.clone())?)
            }
        })
    }
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
                .empty_cache(backend, capacity)
                .map(LayerCache::Attention),
            Mixer::LinearAttention(lin) => {
                lin.empty_cache(backend).map(LayerCache::LinearAttention)
            }
        }
    }
}

/// Feed-forward network of a decoder layer (`mlp`).
#[derive(Debug, Clone, Copy)]
pub enum MlpConfig {
    /// `down_proj(activation(up_proj(x)))`, with biases when `bias` is set.
    FeedForward {
        hidden_dim: usize,
        activation: ActivationFunction,
        bias: bool,
    },
    /// `down_proj(silu(gate_proj(x)) * up_proj(x))`, without biases.
    Gated { hidden_dim: usize },
}

impl MlpConfig {
    fn load<B: PortableBackend + 'static>(
        self,
        params: &mut LayerLoader<'_, B>,
        prefix: &str,
        dim: usize,
    ) -> Result<Mlp<B>> {
        Ok(match self {
            MlpConfig::FeedForward {
                hidden_dim,
                activation,
                bias,
            } => Mlp::FeedForward(FeedForward::load(
                params, prefix, dim, hidden_dim, activation, bias,
            )?),
            MlpConfig::Gated { hidden_dim } => Mlp::Gated(GatedFeedForward::load(
                params, prefix, dim, hidden_dim, false,
            )?),
        })
    }
}

#[nn::module]
pub enum Mlp {
    FeedForward(FeedForward),
    Gated(GatedFeedForward),
}

#[nn::module]
impl Mlp {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            Mlp::FeedForward(mlp) => mlp.call(x),
            Mlp::Gated(mlp) => mlp.call(x),
        }
    }
}

/// Pre-norm residual decoder layer.
#[nn::module]
pub struct DecoderBlock {
    pub input_layernorm: Norm,
    /// Its parameters are under the mixer's own name ([`Mixer::name`]).
    #[module(flatten)]
    pub mixer: Mixer,
    pub post_attention_layernorm: Norm,
    pub mlp: Mlp,
}

#[nn::module]
impl DecoderBlock {
    pub fn load(
        params: &mut LayerLoader<'_, B>,
        prefix: &str,
        dim: usize,
        norm: NormConfig,
        mixer: &MixerConfig,
        mlp: MlpConfig,
    ) -> Result<Self> {
        Ok(Self {
            input_layernorm: norm.load(params, &format!("{prefix}.input_layernorm"), dim)?,
            mixer: mixer.load(params, prefix)?,
            post_attention_layernorm: norm.load(
                params,
                &format!("{prefix}.post_attention_layernorm"),
                dim,
            )?,
            mlp: mlp.load(params, &format!("{prefix}.mlp"), dim)?,
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
