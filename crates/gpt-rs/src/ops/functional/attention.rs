//! Causal attention over a fixed-capacity KV cache ([`attention_kv_cache`]). The same functional
//! serves full-sequence prefill from an empty cache and autoregressive decode.

use std::sync::Arc;

use anyhow::{ensure, Result};

use crate::backend::spec::{DType, PortableBackend};
use crate::ops::functional::activation::softmax_ptir;
use crate::ops::ptir::{self, scalar_broadcast, DotAttrs, DotDims};
use crate::tensor::{DeviceTensor, Shape, Tensor as HostTensor};
use crate::{
    capture, ensure_dtype, ensure_rank, ensure_same_backend, ensure_same_dtype, ensure_same_shape,
    functional,
};

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
    pub fn new(keys: DeviceTensor<B>, values: DeviceTensor<B>, len: usize) -> Result<Self> {
        ensure!(
            keys.dtype() == values.dtype() && keys.shape().dims() == values.shape().dims(),
            "decode cache keys ({:?} {:?}) and values ({:?} {:?}) must match",
            keys.dtype(),
            keys.shape().dims(),
            values.dtype(),
            values.shape().dims()
        );
        ensure!(
            keys.shape().rank() == 3,
            "decode cache keys must have rank 3, got {:?}",
            keys.shape().dims()
        );
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
        Self::new(keys, values, 0)
    }

    /// Returns the cache with room for `capacity` positions. The result holds a copy of the stored
    /// prefix.
    pub fn grow(&self, capacity: usize) -> Result<Self> {
        ensure!(
            capacity >= self.capacity(),
            "cannot shrink a decode cache from {} to {capacity}",
            self.capacity()
        );
        if capacity == self.capacity() {
            return Ok(self.clone());
        }
        let backend = self.keys.backend();
        let dims = self.keys.shape().dims();
        let starts = DeviceTensor::from_host(
            Arc::clone(&backend),
            HostTensor::from_i32(Shape::new([3]), vec![0, 0, 0])?,
        )?;
        let grown = |old: &DeviceTensor<B>| {
            let base = DeviceTensor::zeros(
                Arc::clone(&backend),
                Shape::new([dims[0], capacity, dims[2]]),
            )?;
            dynamic_update_slice_into(&base, old, &starts)
        };
        Self::new(grown(&self.keys)?, grown(&self.values)?, self.len)
    }
}

/// Bundle returned by [`attention_kv_cache`].
pub struct DecodeAttentionComputation<B: PortableBackend + 'static> {
    pub output: DeviceTensor<B>,
    pub cache: DecodeKvCache<B>,
}

/// Extracts `len` entries starting at `start` along `axis`, keeping the other dimensions.
#[functional]
pub fn slice_along_axis(x: &Tensor, axis: usize, start: usize, len: usize) -> Result<Tensor> {
    let dims = x.shape().dims();
    ensure!(
        axis < dims.len(),
        "{FUNCTIONAL}: axis {axis} must be less than the rank of x {dims:?}"
    );
    ensure!(
        start + len <= dims[axis],
        "{FUNCTIONAL}: slice [{start}..{}) must fit in dimension {axis} of x {dims:?}",
        start + len
    );
    let mut starts = vec![0usize; dims.len()];
    let mut sizes = dims.to_vec();
    starts[axis] = start;
    sizes[axis] = len;
    capture!(|x| x.slice(starts, sizes))
}

/// Writes `update` into `base` at the dynamic offsets `starts` (`i32 [rank]`). The functional
/// exports the result, so later programs can read it.
#[functional]
pub fn dynamic_update_slice_into(
    base: &Tensor,
    update: &Tensor,
    starts: &Tensor,
) -> Result<Tensor> {
    ensure_same_backend!(base, update, starts);
    ensure_same_dtype!(base, update);
    ensure_rank!(starts, 1);
    ensure_dtype!(starts, I32);
    let base_dims = base.shape().dims();
    let update_dims = update.shape().dims();
    ensure!(
        starts.shape().dims() == [base_dims.len()],
        "{FUNCTIONAL}: starts must have shape [rank of base], got {:?} for base {base_dims:?}",
        starts.shape().dims()
    );
    ensure!(
        update_dims.len() == base_dims.len()
            && update_dims.iter().zip(base_dims).all(|(u, b)| u <= b),
        "{FUNCTIONAL}: update {update_dims:?} must fit in base {base_dims:?}"
    );
    let sizes = update_dims.to_vec();
    capture!(session, |base, update, starts| {
        let updated = base.dynamic_update_slice(&update, &starts, sizes);
        session.export(updated);
        updated
    })
}

/// Grouped-query causal attention for `T` new tokens appended to a fixed-capacity KV cache.
///
/// `q` is `[Hq, T, D]`, and `k` and `v` are `[Hkv, T, D]`, with RoPE already applied.
/// `update_starts` (`i32 [3]`) holds `[0, position, 0]`. `position` is the absolute position of
/// the first new token and must equal `cache.len()`.
///
/// The functional writes the new keys and values at `position` with `dynamic_update_slice`. Query
/// `t` then attends to cache rows `0..=position + t` with scale `D^-1/2`. The causal mask uses
/// only integer iotas and comparisons. The functional returns the context as `[T, Hq * D]`
/// (query-head major) and the updated cache.
///
/// Shapes depend only on `(T, capacity)`, so all positions reuse the same compiled programs.
#[functional]
pub fn attention_kv_cache(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    cache: &DecodeKvCache<B>,
    update_starts: &Tensor,
) -> Result<DecodeAttentionComputation<B>> {
    let cache_keys = cache.keys();
    let cache_values = cache.values();
    ensure_rank!(q, 3);
    ensure_rank!(k, 3);
    ensure_rank!(v, 3);
    ensure_same_backend!(q, k, v, cache_keys, update_starts);
    ensure_dtype!(q, F32);
    ensure_same_dtype!(q, k, v, cache_keys);
    ensure_same_shape!(k, v);
    ensure_dtype!(update_starts, I32);
    ensure!(
        update_starts.shape().dims() == [3],
        "{FUNCTIONAL}: update_starts must have shape [3], got {:?}",
        update_starts.shape().dims()
    );
    let (q_dims, k_dims) = (q.shape().dims(), k.shape().dims());
    let (hq, t, d) = (q_dims[0], q_dims[1], q_dims[2]);
    let hkv = k_dims[0];
    ensure!(
        k_dims[1] == t && k_dims[2] == d,
        "{FUNCTIONAL}: k must be [{hkv}, {t}, {d}], got {k_dims:?}"
    );
    ensure!(
        hkv > 0 && hq.is_multiple_of(hkv),
        "{FUNCTIONAL}: query heads ({hq}) must be a multiple of kv heads ({hkv})"
    );
    let cache_dims = cache_keys.shape().dims();
    ensure!(
        cache_dims[0] == hkv && cache_dims[2] == d,
        "{FUNCTIONAL}: cache must be [{hkv}, capacity, {d}], got {cache_dims:?}"
    );
    let c = cache_dims[1];
    ensure!(
        cache.len() + t <= c,
        "{FUNCTIONAL}: update would exceed capacity (len {} + seq_len {t} > {c})",
        cache.len()
    );
    let group = hq / hkv;
    let scale = 1.0f32 / (d as f32).sqrt();

    let (output, keys, values) =
        capture!(session, |q,
                           k,
                           v,
                           cache_keys,
                           cache_values,
                           update_starts| {
            let keys = cache_keys.dynamic_update_slice(&k, &update_starts, vec![hkv, t, d]);
            let values = cache_values.dynamic_update_slice(&v, &update_starts, vec![hkv, t, d]);

            let q_grouped = q.reshape(vec![hkv, group * t, d]);
            let scores = q_grouped.dot_general(
                &keys,
                &DotDims::new(crate::axes!(0), crate::axes!(2), crate::axes!(2)),
                &DotAttrs::default(),
            ) * scale;
            // Query positions `position..position + T`, read from an iota at the cache write
            // offset.
            let query_pos = session
                .iota(vec![1, c, 1], 1, DType::Si32)
                .dynamic_slice(&update_starts, vec![1, t, 1])
                .reshape(vec![t, 1])
                .broadcast_to(vec![t, c]);
            let key_pos = session.iota(vec![t, c], 1, DType::Si32);
            let allowed = query_pos.greater_equal(&key_pos);
            let bias = ptir::Tensor::select(
                &allowed,
                &scalar_broadcast(&session, 0.0, &[t, c]),
                &scalar_broadcast(&session, MASKED_BIAS, &[t, c]),
            )
            .reshape(vec![1, 1, t, c])
            .broadcast_to(vec![hkv, group, t, c])
            .reshape(vec![hkv, group * t, c]);
            let probs = softmax_ptir(&(scores + bias), 2);
            let context = probs
                .dot_general(
                    &values,
                    &DotDims::new(crate::axes!(0), crate::axes!(2), crate::axes!(1)),
                    &DotAttrs::default(),
                )
                .reshape(vec![hkv, group, t, d])
                .transpose(vec![2, 0, 1, 3])
                .reshape(vec![t, hq * d]);

            session.export(keys);
            session.export(values);
            (context, keys, values)
        })?;
    let cache = DecodeKvCache::new(keys, values, cache.len() + t)?;
    Ok(DecodeAttentionComputation { output, cache })
}
