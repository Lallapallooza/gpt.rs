//! Gated DeltaNet token mixer (Hugging Face `linear_attn` of Qwen3.5). It does not load Qwen3-Next
//! checkpoints, which fuse the input projections into `in_proj_qkvz` and `in_proj_ba`.
//!
//! The graph computes the DeltaNet decay `-exp(A_log)` and reshapes the depthwise
//! `conv1d.weight`, so the parameters keep their Hugging Face layouts and names.

use std::sync::Arc;

use anyhow::{ensure, Result};

use super::conv::CausalConv1d;
use super::linear::Linear;
use super::rms_norm::{RmsNorm, RmsNormConfig};
use crate::nn::{self, LayerLoader};
use crate::ops::functional::{self, DeviceTensorOps, LinearAttentionCache};

/// Head layout and convolution size of a [`GatedDeltaNet`].
#[derive(Debug, Clone)]
pub struct GatedDeltaNetConfig {
    pub embed_dim: usize,
    pub num_key_heads: usize,
    pub num_value_heads: usize,
    pub key_head_dim: usize,
    pub value_head_dim: usize,
    pub conv_kernel_dim: usize,
    /// Epsilon of the RMSNorm over each value head of the output.
    pub rms_norm_eps: f32,
}

impl GatedDeltaNetConfig {
    fn validate(&self) -> Result<()> {
        ensure!(
            self.num_key_heads > 0 && self.num_value_heads.is_multiple_of(self.num_key_heads),
            "linear value heads ({}) must be a multiple of linear key heads ({})",
            self.num_value_heads,
            self.num_key_heads
        );
        ensure!(self.conv_kernel_dim >= 2, "conv_kernel_dim must be >= 2");
        Ok(())
    }

    pub fn key_dim(&self) -> usize {
        self.num_key_heads * self.key_head_dim
    }

    pub fn value_dim(&self) -> usize {
        self.num_value_heads * self.value_head_dim
    }

    /// Channels of the causal convolution: queries, keys and values.
    pub fn conv_dim(&self) -> usize {
        2 * self.key_dim() + self.value_dim()
    }
}

/// Gated DeltaNet linear attention. A causal depthwise convolution over the projected queries,
/// keys and values feeds the gated delta rule.
#[nn::module]
pub struct GatedDeltaNet {
    #[module(config)]
    pub config: GatedDeltaNetConfig,
    pub in_proj_qkv: Linear,
    pub in_proj_z: Linear,
    pub in_proj_b: Linear,
    pub in_proj_a: Linear,
    pub conv1d: CausalConv1d,
    pub dt_bias: Tensor,
    #[module(rename = "A_log")]
    pub a_log: Tensor,
    pub norm: RmsNorm,
    pub out_proj: Linear,
}

#[nn::module]
impl GatedDeltaNet {
    pub fn load(
        params: &mut LayerLoader<'_, B>,
        prefix: &str,
        config: GatedDeltaNetConfig,
    ) -> Result<Self> {
        config.validate()?;
        let name = |leaf: &str| format!("{prefix}.{leaf}");
        let (hidden, hv) = (config.embed_dim, config.num_value_heads);
        let (conv_dim, value_dim) = (config.conv_dim(), config.value_dim());
        let conv_shape = [conv_dim, 1, config.conv_kernel_dim];
        let norm = RmsNormConfig {
            eps: config.rms_norm_eps,
            unit_offset: false,
        };
        Ok(Self {
            in_proj_qkv: params.linear(&name("in_proj_qkv"), hidden, conv_dim, false)?,
            in_proj_z: params.linear(&name("in_proj_z"), hidden, value_dim, false)?,
            in_proj_b: params.linear(&name("in_proj_b"), hidden, hv, false)?,
            in_proj_a: params.linear(&name("in_proj_a"), hidden, hv, false)?,
            conv1d: CausalConv1d {
                weight: params.tensor(&name("conv1d.weight"), &conv_shape)?,
            },
            dt_bias: params.tensor(&name("dt_bias"), &[hv])?,
            a_log: params.tensor(&name("A_log"), &[hv])?,
            norm: params.rms_norm(&name("norm"), config.value_head_dim, norm)?,
            out_proj: params.linear(&name("out_proj"), value_dim, hidden, false)?,
            config,
        })
    }

    /// Mixes `x` (`[T, embed_dim]`), continuing from the state in `cache`. Returns the output and
    /// the state after the last token.
    fn forward(
        &self,
        x: &Tensor,
        cache: &LinearAttentionCache<B>,
    ) -> Result<(Tensor, LinearAttentionCache<B>)> {
        let cfg = &self.config;
        let t = x.shape().dims()[0];
        let (hk, hv) = (cfg.num_key_heads, cfg.num_value_heads);
        let (dk, dv) = (cfg.key_head_dim, cfg.value_head_dim);
        let (key_dim, value_dim) = (cfg.key_dim(), cfg.value_dim());

        let mixed = self.in_proj_qkv(x)?;
        let conv = self.conv1d((&mixed, cache.conv_state()))?;
        let mixed = functional::silu(&conv.output)?;
        let q = functional::slice_along_axis(&mixed, 1, 0, key_dim)?;
        let k = functional::slice_along_axis(&mixed, 1, key_dim, key_dim)?;
        let v = functional::slice_along_axis(&mixed, 1, 2 * key_dim, value_dim)?;
        let q = functional::reshape(&q, &[t, hk, dk])?;
        let k = functional::reshape(&k, &[t, hk, dk])?;
        let v = functional::reshape(&v, &[t, hv, dv])?;

        let beta = functional::sigmoid(&self.in_proj_b(x)?)?;
        let a = functional::add_bias(&self.in_proj_a(x)?, &self.dt_bias)?;
        let neg_a = functional::exp(&self.a_log)?.neg()?;
        let g = functional::mul_last_dim(&functional::softplus(&a)?, &neg_a)?;

        let delta = functional::gated_delta_rule(&q, &k, &v, &g, &beta, cache.recurrent_state())?;
        let normed = self.norm(&delta.output)?;
        let z = functional::reshape(&self.in_proj_z(x)?, &[t, hv, dv])?;
        let gated = normed.mul(&functional::silu(&z)?)?;
        let gated = functional::reshape(&gated, &[t, value_dim])?;
        let out = self.out_proj(&gated)?;
        Ok((out, LinearAttentionCache::new(conv.state, delta.state)?))
    }

    /// Zero state of this layer on `backend`, as before the first token.
    pub fn empty_cache(&self, backend: &Arc<B>) -> Result<LinearAttentionCache<B>> {
        let cfg = &self.config;
        LinearAttentionCache::zeros(
            backend,
            cfg.conv_kernel_dim,
            cfg.conv_dim(),
            cfg.num_value_heads,
            cfg.key_head_dim,
            cfg.value_head_dim,
        )
    }
}
