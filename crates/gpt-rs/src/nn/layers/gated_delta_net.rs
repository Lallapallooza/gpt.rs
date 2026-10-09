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
use crate::nn::{self, Params};
use crate::ops::functional::{self, LinearAttentionCache};
use crate::s;

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
    pub fn new(p: &mut Params<B>, config: GatedDeltaNetConfig) -> Result<Self> {
        config.validate()?;
        let norm = RmsNormConfig {
            eps: config.rms_norm_eps,
            unit_offset: false,
        };
        Ok(Self {
            in_proj_qkv: Linear::new(p)?,
            in_proj_z: Linear::new(p)?,
            in_proj_b: Linear::new(p)?,
            in_proj_a: Linear::new(p)?,
            conv1d: CausalConv1d::new(p)?,
            dt_bias: p.param()?,
            a_log: p.param()?,
            norm: RmsNorm::new(p, norm)?,
            out_proj: Linear::new(p)?,
            config,
        })
    }

    /// Mixes `x` (`[T, embed_dim]`), continuing from the state in `cache`. Returns the output and
    /// the state after the last token.
    fn forward(
        &self,
        x: &Tensor,
        cache: &LinearAttentionCache<B>,
    ) -> (Tensor, LinearAttentionCache<B>) {
        let cfg = &self.config;
        let t = x.size(0);
        let (hk, hv) = (cfg.num_key_heads, cfg.num_value_heads);
        let (dk, dv) = (cfg.key_head_dim, cfg.value_head_dim);
        let (key_dim, value_dim) = (cfg.key_dim(), cfg.value_dim());

        let conv = self.conv1d((&self.in_proj_qkv(x), cache.conv_state()));
        let mixed = functional::silu(&conv.output);
        let q = mixed.slice(s![.., ..key_dim]).reshape([t, hk, dk]);
        let k = mixed
            .slice(s![.., key_dim..2 * key_dim])
            .reshape([t, hk, dk]);
        let v = mixed
            .slice(s![.., 2 * key_dim..2 * key_dim + value_dim])
            .reshape([t, hv, dv]);

        let beta = functional::sigmoid(&self.in_proj_b(x));
        let a = self.in_proj_a(x) + &self.dt_bias;
        let g = functional::softplus(&a) * -self.a_log.exp();

        let delta = functional::gated_delta_rule(&q, &k, &v, &g, &beta, cache.recurrent_state());
        let z = self.in_proj_z(x).reshape([t, hv, dv]);
        let gated = self.norm(&delta.output) * functional::silu(&z);
        let out = self.out_proj(&gated.reshape([t, value_dim]));
        (out, LinearAttentionCache::new(conv.state, delta.state))
    }

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
