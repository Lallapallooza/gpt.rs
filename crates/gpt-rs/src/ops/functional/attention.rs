//! Attention kernels and cache management routines built for GPT-style models.
//!
//! The implementations stay backend portable by capturing computation graphs for scaled dot
//! product attention, incremental cache updates, and supporting tensor reshapes used during
//! autoregressive decoding.

use anyhow::{ensure, Result};
use gpt_rs_macros::{capture_ptir, ptir_pattern, support_runtime_overload};

use crate::backend::spec::{DType, PortableBackend};
use crate::nn::layers::attention::AttentionConfig;
use crate::ops::functional::common::{
    ensure_axis_in_bounds, ensure_dims_match_except_axis, ensure_last_dim, ensure_rank,
    ensure_same_backend, ensure_same_dtype, ensure_shape_matches, ensure_slice_within_bounds,
    scalar_broadcast, CaptureIntoDeviceTensor,
};
use crate::ops::functional::resolve_graph_from_tensors;
use crate::ops::functional::runtime::CacheKeyArg;
use crate::ops::graph::GraphArena;
use crate::ops::ptir::{self, DotAttrs, DotDims};
use crate::tensor::DeviceTensor;
use std::sync::Arc;

/// Holds the key/value tensors used by incremental attention.
pub struct AttentionCache<B: PortableBackend + 'static> {
    keys: DeviceTensor<B>,
    values: DeviceTensor<B>,
}

/// Fixed-capacity key/value cache used by the decode hot path.
///
/// Unlike [`AttentionCache`], this cache keeps a stable backing shape so lazy execution can reuse
/// compiled plans across decoding steps. The active prefix length lives in `len`, while the
/// backing tensors use `capacity = keys.shape().dims()[1]`.
pub struct DecodeKvCache<B: PortableBackend + 'static> {
    keys: DeviceTensor<B>,
    values: DeviceTensor<B>,
    len: usize,
}

impl<B: PortableBackend + 'static> Clone for AttentionCache<B> {
    fn clone(&self) -> Self {
        AttentionCache {
            keys: self.keys.clone(),
            values: self.values.clone(),
        }
    }
}

impl<B: PortableBackend + 'static> AttentionCache<B> {
    /// Builds a cache from precomputed key/value tensors, validating shared shape and dtype.
    /// The constructor enforces the `[num_heads, seq_len, head_dim]` convention used by GPT models
    /// and guarantees that the cached tensors originate from the same backend instance.
    pub fn new(keys: DeviceTensor<B>, values: DeviceTensor<B>) -> Result<Self> {
        ensure_same_dtype("attention cache keys", &keys, "values", &values)?;
        ensure_shape_matches("attention cache keys", &keys, "values", &values)?;
        ensure_rank("attention cache keys", &keys, 3)?;
        Ok(Self { keys, values })
    }

    /// Returns the key tensor shaped as `[num_heads, seq_len, head_dim]`.
    /// The tensor is shared by reference so callers avoid redundant captures when composing ops.
    pub fn keys(&self) -> &DeviceTensor<B> {
        &self.keys
    }

    /// Returns the value tensor shaped as `[num_heads, seq_len, head_dim]`.
    /// Like the key tensor, it preserves backend ownership and gradient tracking flags.
    pub fn values(&self) -> &DeviceTensor<B> {
        &self.values
    }

    /// Sequence length currently stored in the cache.
    /// Useful when appending incremental key/value blocks or trimming history.
    pub fn len(&self) -> usize {
        self.keys.shape().dims()[1]
    }

    /// Whether the cache holds zero time steps.
    /// Empty caches typically indicate the first decoding step in autoregressive inference.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Concatenates another cache along the sequence dimension, producing a new cache.
    /// The resulting cache inherits the backend and gradient flags of the operands and validates
    /// matching head counts and head dimensions before stitching storage together.
    pub fn concat(&self, other: &AttentionCache<B>) -> Result<AttentionCache<B>> {
        let backend = ensure_same_backend("attention cache concat", self.keys(), other.keys())?;
        ensure_same_backend(
            "attention cache concat values",
            self.values(),
            other.values(),
        )?;
        let keys = concat_along_axis(backend.as_ref(), self.keys(), other.keys(), 1)?;
        let values = concat_along_axis(backend.as_ref(), self.values(), other.values(), 1)?;
        AttentionCache::new(keys, values)
    }
}

impl<B: PortableBackend + 'static> Clone for DecodeKvCache<B> {
    fn clone(&self) -> Self {
        DecodeKvCache {
            keys: self.keys.clone(),
            values: self.values.clone(),
            len: self.len,
        }
    }
}

impl<B: PortableBackend + 'static> DecodeKvCache<B> {
    /// Builds a decode cache from fixed-capacity key/value tensors plus the active prefix length.
    ///
    /// Layout: `[num_kv_heads, capacity, kv_head_dim]`.
    pub fn new(keys: DeviceTensor<B>, values: DeviceTensor<B>, len: usize) -> Result<Self> {
        ensure_same_dtype("decode cache keys", &keys, "values", &values)?;
        ensure_shape_matches("decode cache keys", &keys, "values", &values)?;
        ensure_rank("decode cache keys", &keys, 3)?;
        let capacity = keys.shape().dims()[1];
        ensure!(
            len <= capacity,
            "decode cache len {} exceeds capacity {}",
            len,
            capacity
        );
        Ok(Self { keys, values, len })
    }

    /// Returns the key buffer shaped as `[num_heads, capacity, head_dim]`.
    pub fn keys(&self) -> &DeviceTensor<B> {
        &self.keys
    }

    /// Returns the value buffer shaped as `[num_heads, capacity, head_dim]`.
    pub fn values(&self) -> &DeviceTensor<B> {
        &self.values
    }

    /// Active prefix length currently stored in the cache.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Total cache capacity (`keys.shape().dims()[1]`).
    pub fn capacity(&self) -> usize {
        self.keys.shape().dims()[1]
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn with_len(mut self, len: usize) -> Result<Self> {
        ensure!(
            len <= self.capacity(),
            "decode cache len {} exceeds capacity {}",
            len,
            self.capacity()
        );
        self.len = len;
        Ok(self)
    }
}

/// Bundle returned by the forward attention kernel containing context and cache tensors.
/// The `present` cache holds the current-step keys/values while `cache` merges them with prior
/// history when incremental decoding is enabled. Both caches reuse existing graph arenas to keep
/// incremental execution efficient.
pub struct AttentionComputation<B: PortableBackend + 'static> {
    pub output: DeviceTensor<B>,
    pub present: AttentionCache<B>,
    pub cache: AttentionCache<B>,
}

/// Bundle returned by the decode-cache attention kernel.
pub struct DecodeAttentionComputation<B: PortableBackend + 'static> {
    pub output: DeviceTensor<B>,
    pub cache: DecodeKvCache<B>,
}

struct AttentionPlan {
    seq_len: usize,
    cache_len: usize,
    requires_grad: bool,
}

struct DecodeAttentionPlan {
    seq_len: usize,
    requires_grad: bool,
}

fn validate_qkv_layout<B: PortableBackend + 'static>(
    config: &AttentionConfig,
    qkv: &DeviceTensor<B>,
) -> Result<usize> {
    ensure_rank("attention qkv", qkv, 2)?;
    let dims = qkv.shape().dims();
    let total_proj_dim = dims[1];
    ensure!(
        total_proj_dim == config.total_projection_dim(),
        "qkv projection dimension mismatch: expected {}, got {}",
        config.total_projection_dim(),
        total_proj_dim
    );
    Ok(dims[0])
}

fn validate_cache_layout<B: PortableBackend + 'static>(
    config: &AttentionConfig,
    cache: &AttentionCache<B>,
) -> Result<usize> {
    ensure_rank("attention cache keys", cache.keys(), 3)?;
    ensure_rank("attention cache values", cache.values(), 3)?;
    let dims = cache.keys().shape().dims();
    ensure!(
        dims[0] == config.num_key_value_heads,
        "cache heads mismatch: expected {}, got {}",
        config.num_key_value_heads,
        dims[0]
    );
    ensure_last_dim("attention cache keys", cache.keys(), config.kv_head_dim)?;
    ensure_last_dim("attention cache values", cache.values(), config.kv_head_dim)?;
    Ok(cache.len())
}

fn validate_decode_cache_layout<B: PortableBackend + 'static>(
    config: &AttentionConfig,
    cache: &DecodeKvCache<B>,
) -> Result<usize> {
    ensure_rank("decode cache keys", cache.keys(), 3)?;
    ensure_rank("decode cache values", cache.values(), 3)?;
    let dims = cache.keys().shape().dims();
    ensure!(
        dims[0] == config.num_key_value_heads,
        "decode cache heads mismatch: expected {}, got {}",
        config.num_key_value_heads,
        dims[0]
    );
    ensure_last_dim("decode cache keys", cache.keys(), config.kv_head_dim)?;
    ensure_last_dim("decode cache values", cache.values(), config.kv_head_dim)?;
    ensure!(
        cache.len() <= dims[1],
        "decode cache len {} exceeds capacity {}",
        cache.len(),
        dims[1]
    );
    Ok(dims[1])
}

fn validate_decode_starts<B: PortableBackend + 'static>(
    update_starts: &DeviceTensor<B>,
    query_start: &DeviceTensor<B>,
) -> Result<()> {
    ensure_rank("decode attention update starts", update_starts, 1)?;
    ensure_rank("decode attention query start", query_start, 1)?;
    ensure!(
        update_starts.dtype() == crate::tensor::DType::I32,
        "decode attention update starts must be i32"
    );
    ensure!(
        query_start.dtype() == crate::tensor::DType::I32,
        "decode attention query start must be i32"
    );
    ensure!(
        update_starts.shape().dims() == [3],
        "decode attention update starts must have shape [3]"
    );
    ensure!(
        query_start.shape().dims() == [1],
        "decode attention query start must have shape [1]"
    );
    Ok(())
}

/// Validates QKV tensors (shape/dtype) plus optional caches and derives metadata for capture.
fn validate_attention<B: PortableBackend + 'static>(
    config: &AttentionConfig,
    qkv: &DeviceTensor<B>,
    cache: Option<&AttentionCache<B>>,
) -> Result<AttentionPlan> {
    let seq_len = validate_qkv_layout(config, qkv)?;
    let cache_len = match cache {
        Some(existing) => validate_cache_layout(config, existing)?,
        None => 0,
    };

    Ok(AttentionPlan {
        seq_len,
        cache_len,
        requires_grad: qkv.requires_grad_flag(),
    })
}

fn validate_decode_attention<B: PortableBackend + 'static>(
    config: &AttentionConfig,
    qkv: &DeviceTensor<B>,
    cache: &DecodeKvCache<B>,
    update_starts: &DeviceTensor<B>,
    query_start: &DeviceTensor<B>,
) -> Result<DecodeAttentionPlan> {
    let seq_len = validate_qkv_layout(config, qkv)?;
    ensure!(
        seq_len == 1,
        "decode attention expects seq_len=1 (got {seq_len})"
    );
    let capacity = validate_decode_cache_layout(config, cache)?;
    ensure!(
        cache.len() + seq_len <= capacity,
        "decode cache update would exceed capacity (len {} + seq_len {} > {})",
        cache.len(),
        seq_len,
        capacity
    );
    validate_decode_starts(update_starts, query_start)?;

    Ok(DecodeAttentionPlan {
        seq_len,
        requires_grad: qkv.requires_grad_flag(),
    })
}

/// Helper that returns the most recent `seq_len` elements from a cache-aware tensor slice.
///
/// When `cache_len` is zero it simply clones the tensor; otherwise it slices along `axis`.
fn recent_window<B: PortableBackend + 'static>(
    backend: &B,
    tensor: &DeviceTensor<B>,
    axis: usize,
    cache_len: usize,
    seq_len: usize,
) -> Result<DeviceTensor<B>> {
    if cache_len == 0 {
        Ok(tensor.clone())
    } else {
        slice_along_axis(backend, tensor, axis, cache_len, seq_len)
    }
}

/// Executes scaled dot-product attention with optional incremental cache.
/// Returns the attention output alongside the cache fragments needed for subsequent tokens.
///
/// The forward graph proceeds in these stages:
/// - reshape the packed QKV tensor into query, key, and value projections;
/// - optionally splice in the existing cache so the attention window covers all past tokens;
/// - compute scaled dot products, apply the softmax with causal masking, and combine with values;
/// - emit three outputs in a single backend program: context, new keys, and new values.
#[support_runtime_overload]
#[ptir_pattern(target = "gpt_rs.attention_f32")]
pub fn attention<B: PortableBackend + 'static>(
    backend: &B,
    config: &AttentionConfig,
    qkv: &DeviceTensor<B>,
    cache: Option<&AttentionCache<B>>,
) -> Result<AttentionComputation<B>> {
    let plan = validate_attention(config, qkv, cache)?;
    let graph =
        resolve_graph_from_tensors(&[qkv]).unwrap_or_else(|| GraphArena::new(qkv.backend()));

    let total_len = plan.cache_len + plan.seq_len;
    let embed_dim = config.embed_dim;
    let num_query_heads = config.num_query_heads;
    let num_kv_heads = config.num_key_value_heads;
    let head_dim = config.head_dim;
    let kv_head_dim = config.kv_head_dim;
    let kv_group_size = config.kv_group_size();
    let q_proj_dim = config.query_projection_dim();
    let kv_proj_dim = config.key_value_projection_dim();

    let (graph, (context_id, keys_id, values_id)) = if let Some(cache) = cache {
        let captured = capture_ptir! {
            graph = Arc::clone(&graph);
            { qkv, cache_keys = cache.keys(), cache_values = cache.values() },
            |session| {
                let q_slice = qkv.slice(vec![0, 0], vec![plan.seq_len, q_proj_dim]);
                let k_slice = qkv.slice(vec![0, q_proj_dim], vec![plan.seq_len, kv_proj_dim]);
                let v_slice = qkv.slice(
                    vec![0, q_proj_dim + kv_proj_dim],
                    vec![plan.seq_len, kv_proj_dim],
                );

                let q_heads_new = q_slice
                    .reshape(vec![plan.seq_len, num_query_heads, head_dim])
                    .transpose(vec![1, 0, 2]);

                let new_k_cache = k_slice
                    .reshape(vec![plan.seq_len, num_kv_heads, kv_head_dim])
                    .transpose(vec![1, 0, 2]);

                let new_v_cache = v_slice
                    .reshape(vec![plan.seq_len, num_kv_heads, kv_head_dim])
                    .transpose(vec![1, 0, 2]);

                let k_cache = ptir::Tensor::concat(1, &[cache_keys, new_k_cache]);
                let v_cache = ptir::Tensor::concat(1, &[cache_values, new_v_cache]);

                let k_grouped = k_cache
                    .reshape(vec![num_kv_heads, 1, total_len, kv_head_dim])
                    .broadcast_to(vec![num_kv_heads, kv_group_size, total_len, kv_head_dim])
                    .reshape(vec![num_query_heads, total_len, kv_head_dim]);

                let v_grouped = v_cache
                    .reshape(vec![num_kv_heads, 1, total_len, kv_head_dim])
                    .broadcast_to(vec![num_kv_heads, kv_group_size, total_len, kv_head_dim])
                    .reshape(vec![num_query_heads, total_len, kv_head_dim]);

                let scores = q_heads_new.dot_general(
                    &k_grouped,
                    &DotDims::new(crate::axes!(0), crate::axes!(2), crate::axes!(2)),
                    &DotAttrs::default(),
                );

                let scale_shape = vec![num_query_heads, plan.seq_len, total_len];
                let scale =
                    scalar_broadcast(&session, 1.0f32 / (head_dim as f32).sqrt(), &scale_shape);
                let scaled_scores = scores * scale;

                let positions = session.iota(vec![total_len, total_len], 1, DType::Si32);
                let query_pos = session.iota(vec![total_len, total_len], 0, DType::Si32);
                let allowed = query_pos.greater_equal(&positions);
                let mask_plane = vec![total_len, total_len];
                let zero = scalar_broadcast(&session, 0.0, &mask_plane);
                let neg_large = scalar_broadcast(&session, -1e9, &mask_plane);
                let base_mask = ptir::Tensor::select(&allowed, &zero, &neg_large);
                let sliced_mask =
                    base_mask.slice(vec![plan.cache_len, 0], vec![plan.seq_len, total_len]);
                let mask = sliced_mask.broadcast_to(vec![num_query_heads, plan.seq_len, total_len]);

                let masked_scores = scaled_scores + mask;
                let max_scores = masked_scores.reduce_max(vec![2], true);
                let stabilized =
                    masked_scores - max_scores.broadcast_to(vec![num_query_heads, plan.seq_len, total_len]);

                let exp_scores = stabilized.exp();
                let sum_scores = exp_scores.reduce_sum(vec![2], true);
                let softmax =
                    exp_scores / sum_scores.broadcast_to(vec![num_query_heads, plan.seq_len, total_len]);

                let context = softmax.dot_general(
                    &v_grouped,
                    &DotDims::new(crate::axes!(0), crate::axes!(2), crate::axes!(1)),
                    &DotAttrs::default(),
                );

                let context_out = context
                    .transpose(vec![1, 0, 2])
                    .reshape(vec![plan.seq_len, embed_dim]);

                let context_id = context_out.id();
                let keys_id = k_cache.id();
                let values_id = v_cache.id();

                drop(session);
                ctx.export(keys_id);
                ctx.export(values_id);

                Ok((context_id, keys_id, values_id))
            }
        }?;
        captured
    } else {
        let captured = capture_ptir! {
            graph = Arc::clone(&graph);
            { qkv },
            |session| {
                let q_slice = qkv.slice(vec![0, 0], vec![plan.seq_len, q_proj_dim]);
                let k_slice = qkv.slice(vec![0, q_proj_dim], vec![plan.seq_len, kv_proj_dim]);
                let v_slice = qkv.slice(
                    vec![0, q_proj_dim + kv_proj_dim],
                    vec![plan.seq_len, kv_proj_dim],
                );

                let q_heads_new = q_slice
                    .reshape(vec![plan.seq_len, num_query_heads, head_dim])
                    .transpose(vec![1, 0, 2]);

                let new_k_cache = k_slice
                    .reshape(vec![plan.seq_len, num_kv_heads, kv_head_dim])
                    .transpose(vec![1, 0, 2]);

                let new_v_cache = v_slice
                    .reshape(vec![plan.seq_len, num_kv_heads, kv_head_dim])
                    .transpose(vec![1, 0, 2]);

                let k_grouped = new_k_cache
                    .reshape(vec![num_kv_heads, 1, total_len, kv_head_dim])
                    .broadcast_to(vec![num_kv_heads, kv_group_size, total_len, kv_head_dim])
                    .reshape(vec![num_query_heads, total_len, kv_head_dim]);

                let v_grouped = new_v_cache
                    .reshape(vec![num_kv_heads, 1, total_len, kv_head_dim])
                    .broadcast_to(vec![num_kv_heads, kv_group_size, total_len, kv_head_dim])
                    .reshape(vec![num_query_heads, total_len, kv_head_dim]);

                let scores = q_heads_new.dot_general(
                    &k_grouped,
                    &DotDims::new(crate::axes!(0), crate::axes!(2), crate::axes!(2)),
                    &DotAttrs::default(),
                );

                let scale_shape = vec![num_query_heads, plan.seq_len, total_len];
                let scale =
                    scalar_broadcast(&session, 1.0f32 / (head_dim as f32).sqrt(), &scale_shape);
                let scaled_scores = scores * scale;

                let positions = session.iota(vec![total_len, total_len], 1, DType::Si32);
                let query_pos = session.iota(vec![total_len, total_len], 0, DType::Si32);
                let allowed = query_pos.greater_equal(&positions);
                let mask_plane = vec![total_len, total_len];
                let zero = scalar_broadcast(&session, 0.0, &mask_plane);
                let neg_large = scalar_broadcast(&session, -1e9, &mask_plane);
                let base_mask = ptir::Tensor::select(&allowed, &zero, &neg_large);
                let sliced_mask =
                    base_mask.slice(vec![plan.cache_len, 0], vec![plan.seq_len, total_len]);
                let mask = sliced_mask.broadcast_to(vec![num_query_heads, plan.seq_len, total_len]);

                let masked_scores = scaled_scores + mask;
                let max_scores = masked_scores.reduce_max(vec![2], true);
                let stabilized =
                    masked_scores - max_scores.broadcast_to(vec![num_query_heads, plan.seq_len, total_len]);

                let exp_scores = stabilized.exp();
                let sum_scores = exp_scores.reduce_sum(vec![2], true);
                let softmax =
                    exp_scores / sum_scores.broadcast_to(vec![num_query_heads, plan.seq_len, total_len]);

                let context = softmax.dot_general(
                    &v_grouped,
                    &DotDims::new(crate::axes!(0), crate::axes!(2), crate::axes!(1)),
                    &DotAttrs::default(),
                );

                let context_out = context
                    .transpose(vec![1, 0, 2])
                    .reshape(vec![plan.seq_len, embed_dim]);

                let context_id = context_out.id();
                let keys_id = new_k_cache.id();
                let values_id = new_v_cache.id();

                drop(session);
                ctx.export(keys_id);
                ctx.export(values_id);

                Ok((context_id, keys_id, values_id))
            }
        }?;
        captured
    };

    let context_total = (Arc::clone(&graph), context_id)
        .into_device_tensor(false)?
        .requires_grad(plan.requires_grad);
    let combined_keys = (Arc::clone(&graph), keys_id)
        .into_device_tensor(false)?
        .requires_grad(plan.requires_grad);
    let combined_values = (graph, values_id)
        .into_device_tensor(false)?
        .requires_grad(plan.requires_grad);

    let existing_requires_grad = cache
        .map(|c| c.keys().requires_grad_flag() || c.values().requires_grad_flag())
        .unwrap_or(false);

    let combined_keys = combined_keys.requires_grad(plan.requires_grad || existing_requires_grad);
    let combined_values =
        combined_values.requires_grad(plan.requires_grad || existing_requires_grad);

    let present_keys = recent_window(backend, &combined_keys, 1, plan.cache_len, plan.seq_len)?;
    let present_values = recent_window(backend, &combined_values, 1, plan.cache_len, plan.seq_len)?;

    let present = AttentionCache::new(present_keys, present_values)?;
    let combined = AttentionCache::new(combined_keys, combined_values)?;
    let output = context_total;

    Ok(AttentionComputation {
        output,
        present,
        cache: combined,
    })
}

/// Concatenates tensors along the chosen axis, reusing shared validation helpers.
///
/// Used by attention cache operations; emits `ptir::Tensor::concat` after matching dims on all
/// other axes.
#[ptir_pattern(target = "gpt_rs.concat_along_axis")]
pub fn concat_along_axis<B: PortableBackend + 'static>(
    _backend: &B,
    lhs: &DeviceTensor<B>,
    rhs: &DeviceTensor<B>,
    axis: usize,
) -> Result<DeviceTensor<B>> {
    ensure_same_backend("concat", lhs, rhs)?;
    ensure_same_dtype("concat lhs", lhs, "rhs", rhs)?;
    ensure_axis_in_bounds("concat", lhs, axis)?;
    ensure_dims_match_except_axis("concat lhs", lhs, "rhs", rhs, axis)?;

    let requires_grad = lhs.requires_grad_flag() || rhs.requires_grad_flag();
    capture_ptir!({ lhs, rhs }, |_session| {
        let concatenated = ptir::Tensor::concat(axis, &[lhs, rhs]);
        Ok(concatenated.id())
    })?
    .into_device_tensor(requires_grad)
}

/// Extracts a slice along a single axis while preserving other dimensions (cache utility).
#[ptir_pattern(target = "gpt_rs.slice_along_axis")]
pub fn slice_along_axis<B: PortableBackend + 'static>(
    _backend: &B,
    tensor: &DeviceTensor<B>,
    axis: usize,
    start: usize,
    len: usize,
) -> Result<DeviceTensor<B>> {
    ensure_axis_in_bounds("slice_along_axis", tensor, axis)?;
    ensure_slice_within_bounds("slice_along_axis", tensor, axis, start, len)?;

    let mut starts = vec![0usize; tensor.shape().rank()];
    let mut sizes = tensor.shape().dims().to_vec();
    starts[axis] = start;
    sizes[axis] = len;

    let starts_clone = starts.clone();
    let sizes_clone = sizes.clone();

    capture_ptir!({ tensor }, |_session| {
        let sliced = tensor.slice(starts_clone.clone(), sizes_clone.clone());
        Ok(sliced.id())
    })?
    .into_device_tensor(tensor.requires_grad_flag())
}

/// Helper that updates `base` with `update` using a dynamic start index tensor.
///
/// This emits a `dynamic_update_slice` op and exports the result so callers can carry the updated
/// tensor across decode steps without forcing additional materializations.
#[ptir_pattern(target = "gpt_rs.dynamic_update_slice_into")]
pub fn dynamic_update_slice_into<B: PortableBackend + 'static>(
    _backend: &B,
    base: &DeviceTensor<B>,
    update: &DeviceTensor<B>,
    starts: &DeviceTensor<B>,
) -> Result<DeviceTensor<B>> {
    ensure_same_backend("dynamic_update_slice_into", base, update)?;
    ensure_same_backend("dynamic_update_slice_into starts", base, starts)?;
    ensure_same_dtype("dynamic_update_slice_into base", base, "update", update)?;
    ensure_rank("dynamic_update_slice_into starts", starts, 1)?;
    ensure!(
        starts.dtype() == crate::tensor::DType::I32,
        "dynamic_update_slice_into starts must be i32"
    );
    ensure!(
        starts.shape().dims() == [base.shape().rank()],
        "dynamic_update_slice_into starts must have shape [rank]"
    );

    let base_dims = base.shape().dims();
    let update_dims = update.shape().dims();
    ensure!(
        update_dims.len() == base_dims.len(),
        "dynamic_update_slice_into update rank mismatch (base {:?}, update {:?})",
        base_dims,
        update_dims
    );
    for (axis, (&u, &b)) in update_dims.iter().zip(base_dims.iter()).enumerate() {
        ensure!(
            u <= b,
            "dynamic_update_slice_into update dim {}={} exceeds base dim {}={}",
            axis,
            u,
            axis,
            b
        );
    }

    let requires_grad = base.requires_grad_flag() || update.requires_grad_flag();
    let sizes = update_dims.to_vec();

    capture_ptir!({ base, update, starts }, |session| {
        let updated = base.dynamic_update_slice(&update, &starts, sizes.clone());
        let updated_id = updated.id();
        drop(session);
        ctx.export(updated_id);
        Ok(updated_id)
    })?
    .into_device_tensor(requires_grad)
}

/// Decode-only attention path that updates a fixed-capacity KV cache with `dynamic_update_slice`.
///
/// This is the decoding hot path used to keep tensor shapes stable across steps (and therefore
/// reuse compiled plans in lazy mode). The caller is responsible for supplying:
/// - `cache`: fixed-capacity `[num_kv_heads, capacity, kv_head_dim]` buffers + active `len`
/// - `update_starts`: i32 tensor `[3]` holding `[0, pos, 0]` where `pos == cache.len()`
/// - `query_start`: i32 tensor `[1]` holding `[pos]` for selecting the causal mask position
///
/// The current implementation supports `seq_len == 1` only.
#[support_runtime_overload]
#[ptir_pattern(target = "gpt_rs.attention_decode_cache_f32")]
pub fn attention_decode_cache<B: PortableBackend + 'static>(
    _backend: &B,
    config: &AttentionConfig,
    qkv: &DeviceTensor<B>,
    cache: &DecodeKvCache<B>,
    update_starts: &DeviceTensor<B>,
    query_start: &DeviceTensor<B>,
) -> Result<DecodeAttentionComputation<B>> {
    let plan = validate_decode_attention(config, qkv, cache, update_starts, query_start)?;
    let graph =
        resolve_graph_from_tensors(&[qkv]).unwrap_or_else(|| GraphArena::new(qkv.backend()));

    let capacity = cache.capacity();
    let num_query_heads = config.num_query_heads;
    let num_kv_heads = config.num_key_value_heads;
    let head_dim = config.head_dim;
    let kv_head_dim = config.kv_head_dim;

    let (graph, (context_id, keys_id, values_id)) = capture_ptir! {
        graph = Arc::clone(&graph);
        {
            qkv,
            cache_keys = cache.keys(),
            cache_values = cache.values(),
            update_starts,
            query_start,
        },
        |session| {
            let embed_dim = config.embed_dim;
            let kv_group_size = config.kv_group_size();
            let q_proj_dim = config.query_projection_dim();
            let kv_proj_dim = config.key_value_projection_dim();

            let q_slice = qkv.slice(vec![0, 0], vec![plan.seq_len, q_proj_dim]);
            let k_slice = qkv.slice(vec![0, q_proj_dim], vec![plan.seq_len, kv_proj_dim]);
            let v_slice = qkv.slice(
                vec![0, q_proj_dim + kv_proj_dim],
                vec![plan.seq_len, kv_proj_dim],
            );

            let q_heads_new = q_slice.reshape(vec![num_query_heads, plan.seq_len, head_dim]);

            let new_k_cache = k_slice.reshape(vec![num_kv_heads, plan.seq_len, kv_head_dim]);

            let new_v_cache = v_slice.reshape(vec![num_kv_heads, plan.seq_len, kv_head_dim]);

            let sizes = vec![num_kv_heads, plan.seq_len, kv_head_dim];
            let k_cache =
                cache_keys.dynamic_update_slice(&new_k_cache, &update_starts, sizes.clone());
            let v_cache =
                cache_values.dynamic_update_slice(&new_v_cache, &update_starts, sizes);

            let k_grouped = k_cache
                .reshape(vec![num_kv_heads, 1, capacity, kv_head_dim])
                .broadcast_to(vec![num_kv_heads, kv_group_size, capacity, kv_head_dim])
                .reshape(vec![num_query_heads, capacity, kv_head_dim]);

            let v_grouped = v_cache
                .reshape(vec![num_kv_heads, 1, capacity, kv_head_dim])
                .broadcast_to(vec![num_kv_heads, kv_group_size, capacity, kv_head_dim])
                .reshape(vec![num_query_heads, capacity, kv_head_dim]);

            let scores = q_heads_new.dot_general(
                &k_grouped,
                &DotDims::new(crate::axes!(0), crate::axes!(2), crate::axes!(2)),
                &DotAttrs::default(),
            );

            let scale_shape = vec![num_query_heads, plan.seq_len, capacity];
            let scale =
                scalar_broadcast(&session, 1.0f32 / (head_dim as f32).sqrt(), &scale_shape);
            let scaled_scores = scores * scale;

            let positions = session.iota(vec![capacity], 0, DType::Si32);
            let query_pos = positions.dynamic_slice(&query_start, vec![1]);
            let query_pos = query_pos.broadcast_to(vec![capacity]);
            let allowed = query_pos.greater_equal(&positions);
            let zero = scalar_broadcast(&session, 0.0, &[capacity]);
            let neg_large = scalar_broadcast(&session, -1e9, &[capacity]);
            let base_mask = ptir::Tensor::select(&allowed, &zero, &neg_large);
            let mask_2d = base_mask.broadcast_to(vec![plan.seq_len, capacity]);
            let mask = mask_2d.broadcast_to(vec![num_query_heads, plan.seq_len, capacity]);

            let masked_scores = scaled_scores + mask;
            let max_scores = masked_scores.reduce_max(vec![2], true);
            let stabilized = masked_scores
                - max_scores.broadcast_to(vec![num_query_heads, plan.seq_len, capacity]);

            let exp_scores = stabilized.exp();
            let sum_scores = exp_scores.reduce_sum(vec![2], true);
            let softmax =
                exp_scores / sum_scores.broadcast_to(vec![num_query_heads, plan.seq_len, capacity]);

            let context = softmax.dot_general(
                &v_grouped,
                &DotDims::new(crate::axes!(0), crate::axes!(2), crate::axes!(1)),
                &DotAttrs::default(),
            );

            let context_out = context.reshape(vec![plan.seq_len, embed_dim]);

            let context_id = context_out.id();
            let keys_id = k_cache.id();
            let values_id = v_cache.id();

            drop(session);
            ctx.export(keys_id);
            ctx.export(values_id);

            Ok((context_id, keys_id, values_id))
        }
    }?;

    let output = (Arc::clone(&graph), context_id)
        .into_device_tensor(false)?
        .requires_grad(plan.requires_grad);
    let combined_keys = (Arc::clone(&graph), keys_id)
        .into_device_tensor(false)?
        .requires_grad(plan.requires_grad);
    let combined_values = (graph, values_id)
        .into_device_tensor(false)?
        .requires_grad(plan.requires_grad);

    let existing_requires_grad =
        cache.keys().requires_grad_flag() || cache.values().requires_grad_flag();
    let combined_keys = combined_keys.requires_grad(plan.requires_grad || existing_requires_grad);
    let combined_values =
        combined_values.requires_grad(plan.requires_grad || existing_requires_grad);

    let new_len = cache.len() + plan.seq_len;
    let updated_cache = DecodeKvCache::new(combined_keys, combined_values, new_len)?;

    Ok(DecodeAttentionComputation {
        output,
        cache: updated_cache,
    })
}

impl CacheKeyArg for AttentionConfig {
    fn add_to_cache_key(&self, builder: &mut crate::ops::functional::runtime::CacheKeyBuilder) {
        builder.combine_hash(&self.embed_dim);
        builder.combine_hash(&self.num_query_heads);
        builder.combine_hash(&self.num_key_value_heads);
        builder.combine_hash(&self.head_dim);
        builder.combine_hash(&self.kv_head_dim);
    }
}

impl<B: PortableBackend + 'static> CacheKeyArg for AttentionCache<B> {
    fn add_to_cache_key(&self, builder: &mut crate::ops::functional::runtime::CacheKeyBuilder) {
        builder.combine_hash(self.keys.shape().dims());
        builder.combine_hash(&self.keys.dtype());
        builder.combine_hash(&self.values.dtype());
    }
}

impl<B: PortableBackend + 'static> CacheKeyArg for DecodeKvCache<B> {
    fn add_to_cache_key(&self, builder: &mut crate::ops::functional::runtime::CacheKeyBuilder) {
        builder.combine_hash(self.keys.shape().dims());
        builder.combine_hash(&self.keys.dtype());
        builder.combine_hash(&self.values.dtype());
    }
}
