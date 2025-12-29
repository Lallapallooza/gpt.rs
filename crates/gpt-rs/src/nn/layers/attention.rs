use super::linear::Linear;
use crate::backend::spec::PortableBackend;
use crate::module::{Module, ParamVisitor, ParamVisitorMut, TensorRole};
use crate::ops::functional::{self, AttentionCache, DecodeKvCache};
use crate::tensor::DeviceTensor;
use anyhow::{ensure, Result};
use std::fmt;
use std::sync::Arc;

/// Configuration for causal self-attention head layout.
///
/// * `num_query_heads == num_key_value_heads` -> classic multi-head attention.
/// * `num_key_value_heads == 1` -> multi-query attention (shared KV).
/// * `1 < num_key_value_heads < num_query_heads` -> grouped-query attention.
///
/// The portable implementation currently requires the query and key/value head
/// dimensions to match (`kv_head_dim == head_dim`), which aligns with the LLaMA
/// style GQA factorisation.
#[derive(Debug, Clone)]
pub struct AttentionConfig {
    pub embed_dim: usize,
    pub num_query_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub kv_head_dim: usize,
}

impl AttentionConfig {
    /// Creates a configuration where each query head owns its own key/value head (classic MHA).
    pub fn with_equal_heads(embed_dim: usize, num_heads: usize) -> Self {
        Self::with_kv(embed_dim, num_heads, num_heads)
    }

    /// Creates a configuration that shares key/value heads across groups of query heads (GQA).
    ///
    /// Parameters must satisfy `embed_dim % num_query_heads == 0` and
    /// `num_query_heads % num_key_value_heads == 0`. The key/value head dimension matches the
    /// derived query head dimension.
    pub fn with_kv(embed_dim: usize, num_query_heads: usize, num_key_value_heads: usize) -> Self {
        assert!(
            embed_dim.is_multiple_of(num_query_heads),
            "embed dim must divide number of query heads"
        );
        assert!(
            num_query_heads.is_multiple_of(num_key_value_heads),
            "query heads must be divisible by key/value heads"
        );
        let head_dim = embed_dim / num_query_heads;
        AttentionConfig {
            embed_dim,
            num_query_heads,
            num_key_value_heads,
            head_dim,
            kv_head_dim: head_dim,
        }
    }

    /// Creates a grouped-query configuration where key/value heads use the provided dimension.
    ///
    /// The portable path currently requires `kv_head_dim == head_dim`, so an assertion guards
    /// against unsupported layouts.
    pub fn with_custom_kv_dim(
        embed_dim: usize,
        num_query_heads: usize,
        num_key_value_heads: usize,
        kv_head_dim: usize,
    ) -> Self {
        assert!(
            embed_dim.is_multiple_of(num_query_heads),
            "embed dim must divide number of query heads"
        );
        assert!(
            num_query_heads.is_multiple_of(num_key_value_heads),
            "query heads must be divisible by key/value heads"
        );
        let head_dim = embed_dim / num_query_heads;
        assert!(
            head_dim == kv_head_dim,
            "portable attention requires query and key head dimensions to match"
        );
        AttentionConfig {
            embed_dim,
            num_query_heads,
            num_key_value_heads,
            head_dim,
            kv_head_dim,
        }
    }

    /// Returns the number of query heads configured for the layer.
    pub fn num_heads(&self) -> usize {
        self.num_query_heads
    }

    /// Number of query heads served by each key/value head (GQA group size).
    pub fn kv_group_size(&self) -> usize {
        self.num_query_heads / self.num_key_value_heads
    }

    /// Dimension of the packed query projection (rows in `proj_qkv` assigned to queries).
    pub fn query_projection_dim(&self) -> usize {
        self.num_query_heads * self.head_dim
    }

    /// Dimension of either the packed key or value projection.
    pub fn key_value_projection_dim(&self) -> usize {
        self.num_key_value_heads * self.kv_head_dim
    }

    /// Total output dimension of the packed QKV projection.
    pub fn total_projection_dim(&self) -> usize {
        self.query_projection_dim() + 2 * self.key_value_projection_dim()
    }
}

/// Decoder-style causal self-attention with optional key/value cache sharing.
///
/// The layer projects an input sequence `x` into Q/K/V using a single linear map,
/// applies causal masking, and supports incremental decoding by concatenating the
/// provided key/value cache before running attention (old tokens receive zeroed
/// queries so only fresh positions produce new outputs). Head layout is determined by
/// [`AttentionConfig`]:
///
/// * `num_query_heads == num_key_value_heads` -> classic multi-head attention (MHA).
/// * `num_key_value_heads == 1` -> multi-query attention (MQA).
/// * `1 < num_key_value_heads < num_query_heads` -> grouped-query attention (GQA).
pub struct CausalSelfAttention<B: PortableBackend + 'static> {
    backend: Arc<B>,
    pub config: AttentionConfig,
    pub proj_qkv: Linear<B>,
    pub proj_out: Linear<B>,
}

impl<B: PortableBackend + 'static> CausalSelfAttention<B> {
    /// Constructs the attention layer by uploading packed QKV and output projection weights.
    ///
    /// The constructor validates that the packed QKV weight matches the configuration's embed and
    /// projection dimensions and wires the linear sublayers to the shared backend.
    pub fn new<WQ, WO, BQ, BO>(
        backend: Arc<B>,
        config: AttentionConfig,
        w_qkv: WQ,
        w_out: WO,
        b_qkv: BQ,
        b_out: BO,
    ) -> Result<Self>
    where
        WQ: crate::tensor::IntoDeviceTensor<B>,
        WO: crate::tensor::IntoDeviceTensor<B>,
        BQ: crate::tensor::IntoDeviceTensorOption<B>,
        BO: crate::tensor::IntoDeviceTensorOption<B>,
    {
        let proj_qkv = Linear::new(Arc::clone(&backend), w_qkv, b_qkv)?;
        let proj_out = Linear::new(Arc::clone(&backend), w_out, b_out)?;
        let qkv_dims = proj_qkv.weight.shape().dims();
        ensure!(
            qkv_dims[0] == config.embed_dim,
            "qkv weight input dimension {} must match embed dim {}",
            qkv_dims[0],
            config.embed_dim
        );
        ensure!(
            qkv_dims[1] == config.total_projection_dim(),
            "qkv weight output dimension {} must match attention projection dim {}",
            qkv_dims[1],
            config.total_projection_dim()
        );
        Ok(Self {
            backend,
            config,
            proj_qkv,
            proj_out,
        })
    }

    /// Runs attention without returning KV caches.
    #[deny(clippy::disallowed_methods, clippy::disallowed_types)]
    pub fn forward(&self, x: &DeviceTensor<B>) -> Result<DeviceTensor<B>> {
        let _prof_guard = crate::profiling::layer_scope("CausalSelfAttention::forward");
        let qkv = self.proj_qkv.forward(x)?;
        let functional::AttentionComputation { output, .. } =
            functional::attention(self.backend.as_ref(), &self.config, &qkv, None)?;
        self.proj_out.forward(&output)
    }

    /// Runs attention while accepting an optional existing cache.
    ///
    /// During incremental decoding the provided cache will be concatenated with the fresh
    /// key/value tensors before computing attention.
    #[deny(clippy::disallowed_methods, clippy::disallowed_types)]
    pub fn forward_with_cache(
        &self,
        x: &DeviceTensor<B>,
        cache: Option<&AttentionCache<B>>,
    ) -> Result<(DeviceTensor<B>, AttentionCache<B>)> {
        let _prof_guard = crate::profiling::layer_scope("CausalSelfAttention::forward_with_cache");
        let qkv = self.proj_qkv.forward(x)?;
        let functional::AttentionComputation { output, cache, .. } =
            functional::attention(self.backend.as_ref(), &self.config, &qkv, cache)?;
        let output = self.proj_out.forward(&output)?;
        Ok((output, cache))
    }

    /// Runs attention while updating a fixed-capacity decode KV cache.
    ///
    /// This is the decoding hot path used by the CLI when `--kv-cache` is enabled and lazy
    /// execution is active. The method expects `seq_len == 1` inputs and updates the cache at the
    /// position encoded by `update_starts`/`query_start` (see
    /// [`functional::attention_decode_cache`]).
    #[deny(clippy::disallowed_methods, clippy::disallowed_types)]
    pub fn forward_with_decode_cache(
        &self,
        x: &DeviceTensor<B>,
        cache: &DecodeKvCache<B>,
        update_starts: &DeviceTensor<B>,
        query_start: &DeviceTensor<B>,
    ) -> Result<(DeviceTensor<B>, DecodeKvCache<B>)> {
        let _prof_guard = crate::profiling::layer_scope("CausalSelfAttention::forward_decode");

        let qkv = self.proj_qkv.forward(x)?;
        let functional::DecodeAttentionComputation {
            output: attention_context,
            cache: updated_cache,
        } = functional::attention_decode_cache(
            self.backend.as_ref(),
            &self.config,
            &qkv,
            cache,
            update_starts,
            query_start,
        )?;

        let output = self.proj_out.forward(&attention_context)?;
        Ok((output, updated_cache))
    }

    /// Returns the backend that owns the layer parameters.
    pub fn backend(&self) -> Arc<B> {
        Arc::clone(&self.backend)
    }
}

impl<B: PortableBackend + 'static> Clone for CausalSelfAttention<B> {
    fn clone(&self) -> Self {
        CausalSelfAttention {
            backend: Arc::clone(&self.backend),
            config: self.config.clone(),
            proj_qkv: self.proj_qkv.clone(),
            proj_out: self.proj_out.clone(),
        }
    }
}

impl<B: PortableBackend> fmt::Debug for CausalSelfAttention<B> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CausalSelfAttention")
            .field("config", &self.config)
            .finish()
    }
}

impl<B: PortableBackend + 'static> Module<B> for CausalSelfAttention<B> {
    fn visit_params(&self, v: &mut ParamVisitor<'_, B>) -> Result<()> {
        v.param("w_qkv", TensorRole::Parameter, &self.proj_qkv.weight)?;
        if let Some(bias) = &self.proj_qkv.bias {
            v.param("b_qkv", TensorRole::Parameter, bias)?;
        }
        v.param("w_out", TensorRole::Parameter, &self.proj_out.weight)?;
        if let Some(bias) = &self.proj_out.bias {
            v.param("b_out", TensorRole::Parameter, bias)?;
        }
        Ok(())
    }

    fn visit_params_mut(&mut self, v: &mut ParamVisitorMut<'_, B>) -> Result<()> {
        v.param("w_qkv", TensorRole::Parameter, &mut self.proj_qkv.weight)?;
        if let Some(bias) = &mut self.proj_qkv.bias {
            v.param("b_qkv", TensorRole::Parameter, bias)?;
        }
        v.param("w_out", TensorRole::Parameter, &mut self.proj_out.weight)?;
        if let Some(bias) = &mut self.proj_out.bias {
            v.param("b_out", TensorRole::Parameter, bias)?;
        }
        Ok(())
    }
}
