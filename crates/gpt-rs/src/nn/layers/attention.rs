use super::linear::Linear;
use super::rms_norm::{RmsNorm, RmsNormConfig};
use super::rotary_embedding::{RopeConfig, RotaryEmbedding};
use crate::backend::spec::PortableBackend;
use crate::nn::{self, LayerLoader};
use crate::ops::functional::{self, DecodeKvCache, DeviceTensorOps};
use crate::tensor::DeviceTensor;
use anyhow::{bail, ensure, Result};
use std::sync::Arc;

/// Configuration of a [`CausalSelfAttention`] layer.
///
/// * `num_query_heads == num_key_value_heads` -> classic multi-head attention.
/// * `num_key_value_heads == 1` -> multi-query attention (shared KV).
/// * `1 < num_key_value_heads < num_query_heads` -> grouped-query attention.
///
/// Queries, keys and values share `head_dim`.
#[derive(Debug, Clone, PartialEq)]
pub struct AttentionConfig {
    pub embed_dim: usize,
    pub num_query_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub bias: bool,
    /// RMSNorm over each query and key head (`q_norm`, `k_norm`), before the rotary embedding.
    pub qk_norm: Option<RmsNormConfig>,
    pub rope: Option<RopeConfig>,
    /// `q_proj` emits `[query, gate]` per head. The layer multiplies the attention output by
    /// `sigmoid(gate)` before `o_proj`.
    pub output_gate: bool,
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
            bias: false,
            qk_norm: None,
            rope: None,
            output_gate: false,
        })
    }

    /// Sets whether the projections have biases.
    pub fn with_bias(mut self, bias: bool) -> Self {
        self.bias = bias;
        self
    }

    /// Normalises each query and key head (see [`Self::qk_norm`]).
    pub fn with_qk_norm(mut self, norm: RmsNormConfig) -> Self {
        self.qk_norm = Some(norm);
        self
    }

    /// Rotates the leading `rope.rotary_dim` channels of each query and key head.
    pub fn with_rope(mut self, rope: RopeConfig) -> Result<Self> {
        let (rotary_dim, head_dim) = (rope.rotary_dim, self.head_dim);
        ensure!(
            rotary_dim > 0 && rotary_dim.is_multiple_of(2) && rotary_dim <= head_dim,
            "rotary_dim ({rotary_dim}) must be positive, even and <= head_dim ({head_dim})"
        );
        self.rope = Some(rope);
        Ok(self)
    }

    /// Gates the attention output per channel (see [`Self::output_gate`]).
    pub fn with_output_gate(mut self) -> Self {
        self.output_gate = true;
        self
    }

    /// Returns the number of query heads configured for the layer.
    pub fn num_heads(&self) -> usize {
        self.num_query_heads
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
}

/// Inputs shared by every attention layer for `T` new tokens at positions `position..position + T`.
pub struct AttentionPositions<B: PortableBackend + 'static> {
    /// `[0, position, 0]` (`i32`): where the new keys and values go in the KV caches.
    pub update_starts: DeviceTensor<B>,
    /// Rotary `(cos, sin)` tables of the positions, each `[T, rotary_dim / 2]`.
    pub rotary: Option<(DeviceTensor<B>, DeviceTensor<B>)>,
}

impl<B: PortableBackend + 'static> AttentionPositions<B> {
    /// Creates the inputs for positions `position..position + len`.
    pub fn new(
        backend: &Arc<B>,
        position: usize,
        len: usize,
        rotary: Option<&RotaryEmbedding>,
    ) -> Result<Self> {
        Ok(Self {
            update_starts: crate::inference::index_tensor(backend, [0, position, 0])?,
            rotary: rotary
                .map(|rope| rope.tables(backend, position, len))
                .transpose()?,
        })
    }
}

/// Decoder-style causal self-attention over a fixed-capacity KV cache. A full sequence runs as a
/// prefill into an empty cache.
#[nn::module]
pub struct CausalSelfAttention {
    #[module(config)]
    pub config: AttentionConfig,
    pub q_proj: Linear,
    pub k_proj: Linear,
    pub v_proj: Linear,
    pub o_proj: Linear,
    pub q_norm: Option<RmsNorm>,
    pub k_norm: Option<RmsNorm>,
}

#[nn::module]
impl CausalSelfAttention {
    pub fn load(
        params: &mut LayerLoader<'_, B>,
        prefix: &str,
        config: AttentionConfig,
    ) -> Result<Self> {
        let (embed, kv_dim) = (config.embed_dim, config.key_value_projection_dim());
        let mut linear = |name: &str, in_features: usize, out_features: usize| {
            params.linear(
                &format!("{prefix}.{name}"),
                in_features,
                out_features,
                config.bias,
            )
        };
        let q_proj = linear("q_proj", embed, config.query_projection_dim())?;
        let k_proj = linear("k_proj", embed, kv_dim)?;
        let v_proj = linear("v_proj", embed, kv_dim)?;
        let o_proj = linear("o_proj", config.query_dim(), embed)?;
        let mut norm = |name: &str| {
            config
                .qk_norm
                .map(|norm| params.rms_norm(&format!("{prefix}.{name}"), config.head_dim, norm))
                .transpose()
        };
        Ok(Self {
            q_norm: norm("q_norm")?,
            k_norm: norm("k_norm")?,
            config,
            q_proj,
            k_proj,
            v_proj,
            o_proj,
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
    ) -> Result<(Tensor, DecodeKvCache<B>)> {
        let cfg = &self.config;
        let (t, d) = (x.shape().dims()[0], cfg.head_dim);
        let split_heads =
            |y: &Tensor, heads: usize, width: usize| functional::reshape(y, &[t, heads, width]);

        let q_proj = self.q_proj(x)?;
        let (q, gate) = if cfg.output_gate {
            let q_and_gate = split_heads(&q_proj, cfg.num_query_heads, 2 * d)?;
            let gate = functional::slice_along_axis(&q_and_gate, 2, d, d)?;
            (
                functional::slice_along_axis(&q_and_gate, 2, 0, d)?,
                Some(functional::reshape(&gate, &[t, cfg.query_dim()])?),
            )
        } else {
            (split_heads(&q_proj, cfg.num_query_heads, d)?, None)
        };
        let k = split_heads(&self.k_proj(x)?, cfg.num_key_value_heads, d)?;
        let v = split_heads(&self.v_proj(x)?, cfg.num_key_value_heads, d)?;
        let q = self.q_norm(&q)?.unwrap_or(q);
        let k = self.k_norm(&k)?.unwrap_or(k);

        let heads_first = |y: &Tensor| functional::transpose(y, &[1, 0, 2]);
        let (mut q, mut k, v) = (heads_first(&q)?, heads_first(&k)?, heads_first(&v)?);
        match (&cfg.rope, &positions.rotary) {
            (Some(RopeConfig { rotary_dim, .. }), Some((cos, sin))) => {
                ensure!(
                    cos.shape().dims() == [t, rotary_dim / 2],
                    "rotary tables have shape {:?}, expected [{t}, {}]",
                    cos.shape().dims(),
                    rotary_dim / 2
                );
                q = functional::apply_rope(&q, cos, sin)?;
                k = functional::apply_rope(&k, cos, sin)?;
            }
            (None, None) => {}
            (Some(_), None) => bail!("rotary attention needs rotary tables in its positions"),
            (None, Some(_)) => bail!("attention without rotary embedding got rotary tables"),
        }

        let functional::DecodeAttentionComputation { output, cache } =
            functional::attention_kv_cache(&q, &k, &v, cache, &positions.update_starts)?;
        let output = match gate {
            Some(gate) => output.mul(&functional::sigmoid(&gate)?)?,
            None => output,
        };
        Ok((self.o_proj(&output)?, cache))
    }

    /// Empty KV cache of this layer on `backend` with room for `capacity` positions.
    pub fn empty_cache(&self, backend: &Arc<B>, capacity: usize) -> Result<DecodeKvCache<B>> {
        DecodeKvCache::zeros(
            backend,
            self.config.num_key_value_heads,
            capacity,
            self.config.head_dim,
        )
    }
}
