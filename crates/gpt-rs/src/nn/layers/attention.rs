use std::marker::PhantomData;
use std::sync::Arc;

use anyhow::{ensure, Result};

use super::linear::Linear;
use super::names::DecoderNames;
use super::rms_norm::{RmsNorm, RmsNormConfig};
use crate::backend::PortableBackend;
use crate::nn::{self, Params};
use crate::ops::functional::{self, AttentionPositions, DecodeKvCache, WeightLayout};

/// Configuration of a [`CausalSelfAttention`] layer.
///
/// * `num_query_heads == num_key_value_heads` -> classic multi-head attention.
/// * `num_key_value_heads == 1` -> multi-query attention (shared KV).
/// * `1 < num_key_value_heads < num_query_heads` -> grouped-query attention.
///
/// Queries, keys and values share `head_dim`. The rotary embedding, when the model has one, comes
/// with the positions ([`AttentionPositions`]).
#[derive(Debug, Clone, PartialEq)]
pub struct AttentionConfig {
    pub embed_dim: usize,
    pub num_query_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    /// RMSNorm over each query and key head (`q_norm`, `k_norm`), before the rotary embedding.
    pub qk_norm: Option<RmsNormConfig>,
    /// `q_proj` emits `[query, gate]` per head. The layer multiplies the attention output by
    /// `sigmoid(gate)` before `o_proj`.
    pub output_gate: bool,
    /// One projection computes `[queries, keys, values]` in place of `q_proj`, `k_proj` and
    /// `v_proj`.
    pub fused_qkv: bool,
    /// How the checkpoint stores the projection weights.
    pub weight_layout: WeightLayout,
}

impl AttentionConfig {
    /// Creates a configuration where each query head owns its own key/value head (classic MHA).
    pub fn with_equal_heads(embed_dim: usize, num_heads: usize) -> Result<Self> {
        Self::with_kv(embed_dim, num_heads, num_heads)
    }

    /// Creates a configuration that shares key/value heads across groups of query heads (GQA),
    /// with `head_dim = embed_dim / num_query_heads`.
    pub fn with_kv(
        embed_dim: usize,
        num_query_heads: usize,
        num_key_value_heads: usize,
    ) -> Result<Self> {
        ensure!(
            num_query_heads > 0 && embed_dim.is_multiple_of(num_query_heads),
            "embed dim {embed_dim} must be a multiple of the number of query heads {num_query_heads}"
        );
        Self::with_projection_dims(
            embed_dim,
            num_query_heads,
            num_key_value_heads,
            embed_dim / num_query_heads,
        )
    }

    /// Creates a grouped-query configuration with an explicit head dimension.
    ///
    /// This constructor does not require `embed_dim / num_query_heads == head_dim`, which allows
    /// architectures that use wider query projections than residual width.
    pub fn with_projection_dims(
        embed_dim: usize,
        num_query_heads: usize,
        num_key_value_heads: usize,
        head_dim: usize,
    ) -> Result<Self> {
        ensure!(embed_dim > 0, "embed dim must be positive");
        ensure!(
            num_query_heads > 0 && num_key_value_heads > 0,
            "query and key/value head counts must be positive"
        );
        ensure!(
            num_query_heads.is_multiple_of(num_key_value_heads),
            "query heads ({num_query_heads}) must be a multiple of key/value heads \
             ({num_key_value_heads})"
        );
        ensure!(head_dim > 0, "head dimension must be positive");
        Ok(AttentionConfig {
            embed_dim,
            num_query_heads,
            num_key_value_heads,
            head_dim,
            qk_norm: None,
            output_gate: false,
            fused_qkv: false,
            weight_layout: WeightLayout::OutIn,
        })
    }

    /// Normalizes each query and key head (see [`Self::qk_norm`]).
    pub fn with_qk_norm(mut self, norm: RmsNormConfig) -> Self {
        self.qk_norm = Some(norm);
        self
    }

    /// Gates the attention output per channel (see [`Self::output_gate`]).
    pub fn with_output_gate(mut self) -> Self {
        self.output_gate = true;
        self
    }

    /// Computes the queries, keys and values with one projection (see [`Self::fused_qkv`]).
    pub fn with_fused_qkv(mut self) -> Self {
        self.fused_qkv = true;
        self
    }

    /// Reads the projection weights in `layout`.
    pub fn with_weight_layout(mut self, layout: WeightLayout) -> Self {
        self.weight_layout = layout;
        self
    }

    /// Width of the attention output (`num_query_heads * head_dim`), the input of `o_proj`.
    pub fn query_dim(&self) -> usize {
        self.num_query_heads * self.head_dim
    }

    /// Output dimension of `q_proj`. An output gate doubles it, with one gate channel per query
    /// channel.
    pub fn query_projection_dim(&self) -> usize {
        self.query_dim() * if self.output_gate { 2 } else { 1 }
    }

    /// Output dimension of the key and of the value projection.
    pub fn key_value_projection_dim(&self) -> usize {
        self.num_key_value_heads * self.head_dim
    }

    /// Empty KV cache on `backend` with room for `capacity` positions.
    pub fn empty_cache<B: PortableBackend + 'static>(
        &self,
        backend: &Arc<B>,
        capacity: usize,
    ) -> Result<DecodeKvCache<B>> {
        DecodeKvCache::zeros(backend, self.num_key_value_heads, capacity, self.head_dim)
    }
}

/// Decoder-style causal self-attention over a fixed-capacity KV cache, with the projection names
/// of the family `N`. A full sequence runs as a prefill into an empty cache.
#[nn::module]
pub struct CausalSelfAttention<N: DecoderNames> {
    #[module(config)]
    pub config: AttentionConfig,
    pub q_proj: Option<Linear>,
    pub k_proj: Option<Linear>,
    pub v_proj: Option<Linear>,
    #[module(rename = N::QKV_PROJ)]
    pub qkv_proj: Option<Linear>,
    #[module(rename = N::O_PROJ)]
    pub o_proj: Linear,
    pub q_norm: Option<RmsNorm>,
    pub k_norm: Option<RmsNorm>,
    #[module(config)]
    names: PhantomData<N>,
}

#[nn::module]
impl<N: DecoderNames> CausalSelfAttention<N> {
    pub fn new(p: &mut Params<B>, config: AttentionConfig) -> Result<Self> {
        ensure!(
            !(config.fused_qkv && config.output_gate),
            "an attention with a fused query, key and value projection has no output gate"
        );
        let (separate, fused) = (!config.fused_qkv, config.fused_qkv);
        let layout = config.weight_layout;
        let projection = |params: &mut Params<B>, present: bool| {
            present
                .then(|| Linear::with_layout(params, layout))
                .transpose()
        };
        Ok(Self {
            q_proj: projection(p, separate)?,
            k_proj: projection(p, separate)?,
            v_proj: projection(p, separate)?,
            qkv_proj: projection(p, fused)?,
            o_proj: Linear::with_layout(p, layout)?,
            q_norm: config
                .qk_norm
                .map(|norm| RmsNorm::new(p, norm))
                .transpose()?,
            k_norm: config
                .qk_norm
                .map(|norm| RmsNorm::new(p, norm))
                .transpose()?,
            config,
            names: PhantomData,
        })
    }

    /// Runs attention for `x` (`[T, embed_dim]`) at `positions`. It appends the keys and values
    /// of `x` to `cache`, which holds the earlier positions (see
    /// [`functional::attention_kv_cache`]).
    fn forward(
        &self,
        x: &Tensor,
        cache: &DecodeKvCache<B>,
        positions: &AttentionPositions<B>,
    ) -> (Tensor, DecodeKvCache<B>) {
        let cfg = &self.config;
        let (seq_len, head_dim) = (x.size(0), cfg.head_dim);
        let (query_proj, key_proj, value_proj) = match self.qkv_proj(x) {
            Some(qkv) => {
                let (q_dim, kv_dim) = (cfg.query_dim(), cfg.key_value_projection_dim());
                (
                    qkv.slice(crate::s![.., ..q_dim]),
                    qkv.slice(crate::s![.., q_dim..q_dim + kv_dim]),
                    qkv.slice(crate::s![.., q_dim + kv_dim..]),
                )
            }
            None => {
                let project = |proj: Option<Tensor>| proj.expect("separate q, k and v projections");
                (
                    project(self.q_proj(x)),
                    project(self.k_proj(x)),
                    project(self.v_proj(x)),
                )
            }
        };
        let (query_heads, gate) = if cfg.output_gate {
            let q_and_gate = query_proj.reshape([seq_len, cfg.num_query_heads, 2 * head_dim]);
            let gate = q_and_gate.slice(crate::s![.., .., head_dim..]);
            (
                q_and_gate.slice(crate::s![.., .., ..head_dim]),
                Some(gate.reshape([seq_len, cfg.query_dim()])),
            )
        } else {
            (
                query_proj.reshape([seq_len, cfg.num_query_heads, head_dim]),
                None,
            )
        };
        let key_heads = key_proj.reshape([seq_len, cfg.num_key_value_heads, head_dim]);
        let values = value_proj.reshape([seq_len, cfg.num_key_value_heads, head_dim]);
        let queries = self.q_norm(&query_heads).unwrap_or(query_heads);
        let keys = self.k_norm(&key_heads).unwrap_or(key_heads);

        let (attended, next_cache) =
            functional::attention_kv_cache(&queries, &keys, &values, cache, positions);
        let output = match gate {
            Some(gate_logits) => attended * functional::sigmoid(&gate_logits),
            None => attended,
        };
        (self.o_proj(&output), next_cache)
    }
}
