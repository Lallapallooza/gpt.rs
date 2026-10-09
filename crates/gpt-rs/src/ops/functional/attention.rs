//! Causal attention over a fixed-capacity KV cache ([`attention_kv_cache`]). The same functional
//! serves full-sequence prefill from an empty cache and autoregressive decode.

use std::sync::Arc;

use anyhow::Result;
use gpt_rs_ir::dialects::tensor::DotGeneral;

use crate::backend::PortableBackend;
use crate::ops::functional::{apply_rope, softmax};
use crate::tensor::{DType, DeviceTensor, HostTensor, Shape};
use crate::{ensure_dtype, ensure_rank, ensure_same_dtype, ensure_same_shape, functional, s};

/// Additive attention-score bias for masked positions. After the softmax subtracts the maximum,
/// `exp` maps this bias to exactly 0.
pub(crate) const MASKED_BIAS: f32 = -1e9;

/// Fixed-capacity KV cache that [`attention_kv_cache`] reads and writes.
///
/// The cache keeps a stable backing shape, so lazy execution can reuse compiled plans across
/// decode steps.
pub struct DecodeKvCache<B: PortableBackend + 'static> {
    keys: DeviceTensor<B>,
    values: DeviceTensor<B>,
    len: usize,
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
    /// Layout: `[num_kv_heads, capacity, head_dim]`.
    #[track_caller]
    pub fn new(keys: DeviceTensor<B>, values: DeviceTensor<B>, len: usize) -> Self {
        assert!(
            keys.dtype() == values.dtype() && keys.shape().dims() == values.shape().dims(),
            "decode cache keys ({:?} {:?}) and values ({:?} {:?}) must match",
            keys.dtype(),
            keys.shape().dims(),
            values.dtype(),
            values.shape().dims()
        );
        assert!(
            keys.shape().rank() == 3,
            "decode cache keys must have rank 3, got {:?}",
            keys.shape().dims()
        );
        let capacity = keys.size(1);
        assert!(
            len <= capacity,
            "decode cache len {len} exceeds capacity {capacity}"
        );
        Self { keys, values, len }
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

    /// The number of positions the cache has room for.
    pub fn capacity(&self) -> usize {
        self.keys.size(1)
    }

    /// Whether the cache holds no positions.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Creates an empty cache with room for `capacity` positions.
    pub fn zeros(
        backend: &Arc<B>,
        num_kv_heads: usize,
        capacity: usize,
        head_dim: usize,
    ) -> Result<Self> {
        let shape = Shape::new([num_kv_heads, capacity, head_dim]);
        let keys = DeviceTensor::zeros(Arc::clone(backend), shape.clone())?;
        let values = DeviceTensor::zeros(Arc::clone(backend), shape)?;
        Ok(Self::new(keys, values, 0))
    }
}

/// What every attention layer of a forward reads about its `T` new tokens:
///
/// - their positions (`si32 [T]`, consecutive);
/// - for rotary attention, the rotary `(cos, sin)` tables of those positions, each
///   `[T, rotary_dim / 2]`;
/// - where their keys and values go in a cache.
pub struct AttentionPositions<B: PortableBackend + 'static> {
    positions: DeviceTensor<B>,
    rotary: Option<(DeviceTensor<B>, DeviceTensor<B>)>,
    /// The offsets of the first token's row in a cache `[num_kv_heads, capacity, head_dim]`:
    /// `[0, position, 0]`.
    cache_starts: DeviceTensor<B>,
}

impl<B: PortableBackend + 'static> AttentionPositions<B> {
    /// The tokens at `positions` (`si32 [T]`), with the rotary tables of those positions for
    /// rotary attention.
    #[track_caller]
    pub fn new(
        positions: DeviceTensor<B>,
        rotary: Option<(DeviceTensor<B>, DeviceTensor<B>)>,
    ) -> Self {
        assert!(
            positions.dtype() == DType::Si32 && positions.shape().rank() == 1,
            "attention positions must be si32 [T], got {:?} {:?}",
            positions.dtype(),
            positions.shape().dims()
        );
        let zero = HostTensor::si32([1], vec![0]).expect("one element");
        let zero = DeviceTensor::constant(&positions, &zero);
        let cache_starts = DeviceTensor::concat(0, &[&zero, &positions.slice(s![..1]), &zero]);
        Self {
            positions,
            rotary,
            cache_starts,
        }
    }
}

/// Grouped-query causal attention for `T` new tokens appended to a fixed-capacity KV cache.
///
/// `q` is `[T, Hq, D]`, and `k` and `v` are `[T, Hkv, D]`. When `positions` holds rotary tables,
/// they rotate `q` and `k`. The first position must equal `cache.len()`.
///
/// The functional writes the new keys and values at the first position with
/// `dynamic_update_slice`. Query `t` then attends to the cache rows up to its own position, with
/// scale `D^-1/2`. The causal mask compares the positions with an integer iota. The functional
/// returns the context as `[T, Hq * D]` (query-head major) and the extended cache.
///
/// Shapes depend only on `(T, capacity)`, so all positions reuse the same compiled programs.
#[functional]
pub fn attention_kv_cache(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    cache: &DecodeKvCache<B>,
    positions: &AttentionPositions<B>,
) -> (Tensor, DecodeKvCache<B>) {
    let cache_keys = cache.keys();
    let token_positions = &positions.positions;
    ensure_rank!(q, 3);
    ensure_rank!(k, 3);
    ensure_dtype!(q, F32);
    ensure_same_dtype!(q, k, v, cache_keys);
    ensure_same_shape!(k, v);
    ensure_rank!(token_positions, 1);
    ensure_dtype!(token_positions, Si32);
    let [t, hq, d] = [0, 1, 2].map(|axis| q.size(axis));
    let hkv = k.size(1);
    assert!(
        k.shape().dims() == [t, hkv, d] && token_positions.size(0) == t,
        "{FUNCTIONAL}: k must be [{t}, {hkv}, {d}] and positions [{t}], got {:?} and {:?}",
        k.shape().dims(),
        token_positions.shape().dims()
    );
    assert!(
        hkv > 0 && hq.is_multiple_of(hkv),
        "{FUNCTIONAL}: query heads ({hq}) must be a multiple of kv heads ({hkv})"
    );
    let cache_dims = cache_keys.shape().dims();
    assert!(
        cache_dims[0] == hkv && cache_dims[2] == d,
        "{FUNCTIONAL}: cache must be [{hkv}, capacity, {d}], got {cache_dims:?}"
    );
    let c = cache_dims[1];
    assert!(
        cache.len() + t <= c,
        "{FUNCTIONAL}: update would exceed capacity (len {} + seq_len {t} > {c})",
        cache.len()
    );
    let group = hq / hkv;
    let scale = 1.0f32 / (d as f32).sqrt();

    let (mut q, mut k, v) = (
        q.transpose([1, 0, 2]),
        k.transpose([1, 0, 2]),
        v.transpose([1, 0, 2]),
    );
    if let Some((cos, sin)) = &positions.rotary {
        q = apply_rope(&q, cos, sin);
        k = apply_rope(&k, cos, sin);
    }

    let starts = &positions.cache_starts;
    let keys = cache_keys.dynamic_update_slice(&k, starts);
    let values = cache.values().dynamic_update_slice(&v, starts);
    keys.export();
    values.export();

    let scores = q
        .reshape([hkv, group * t, d])
        .dot_general(&keys, DotGeneral::contract([2], [2]).batch([0], [0]))
        * scale;
    let key_pos = DeviceTensor::iota(&q, [t, c], 1, DType::Si32);
    let bias = token_positions
        .reshape([t, 1])
        .greater_equal(&key_pos)
        .select(0.0, MASKED_BIAS)
        .reshape([1, 1, t, c])
        .broadcast_to([hkv, group, t, c])
        .reshape([hkv, group * t, c]);
    let probs = softmax(&(scores + bias), -1);
    let output = probs
        .dot_general(&values, DotGeneral::contract([2], [1]).batch([0], [0]))
        .reshape([hkv, group, t, d])
        .transpose([2, 0, 1, 3])
        .reshape([t, hq * d]);
    let cache = DecodeKvCache::new(keys, values, cache.len() + t);
    (output, cache)
}
