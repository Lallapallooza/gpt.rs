pub mod decoder;
pub mod generate;
pub mod sampler;

use std::sync::Arc;

use anyhow::{anyhow, Result};

use crate::backend::spec::{DecodeSampleRequest, PortableBackend};
use crate::ops::functional::{DecodeKvCache, LinearAttentionCache};
use crate::tensor::{DeviceTensor, Shape, Tensor};

/// Incremental decode state of one decoder layer, carried between generation steps.
pub enum LayerCache<B: PortableBackend + 'static> {
    /// Fixed-capacity key/value cache of a softmax-attention layer.
    Attention(DecodeKvCache<B>),
    /// Convolution window and recurrent state of a linear-attention (Gated DeltaNet) layer.
    LinearAttention(LinearAttentionCache<B>),
}

impl<B: PortableBackend + 'static> Clone for LayerCache<B> {
    fn clone(&self) -> Self {
        match self {
            LayerCache::Attention(cache) => LayerCache::Attention(cache.clone()),
            LayerCache::LinearAttention(cache) => LayerCache::LinearAttention(cache.clone()),
        }
    }
}

impl<B: PortableBackend + 'static> LayerCache<B> {
    /// The state tensors: keys and values, or convolution window and recurrent state.
    pub fn tensors(&self) -> [&DeviceTensor<B>; 2] {
        match self {
            LayerCache::Attention(kv) => [kv.keys(), kv.values()],
            LayerCache::LinearAttention(state) => [state.conv_state(), state.recurrent_state()],
        }
    }

    /// Stops exporting the state of a cache that is being replaced.
    pub(crate) fn release_exports(&self) {
        for tensor in self.tensors() {
            tensor.release_export();
        }
    }

    /// Replaces this cache with `next` and releases the exports of the replaced state.
    pub(crate) fn replace(&mut self, next: LayerCache<B>) {
        self.release_exports();
        *self = next;
    }
}

/// Uploads token ids or positions as an `i32 [N]` tensor.
pub(crate) fn index_tensor<B: PortableBackend + 'static>(
    backend: &Arc<B>,
    indices: impl IntoIterator<Item = usize>,
) -> Result<DeviceTensor<B>> {
    let ids = indices
        .into_iter()
        .map(|i| i32::try_from(i).map_err(|_| anyhow!("index {i} exceeds i32::MAX")))
        .collect::<Result<Vec<i32>>>()?;
    let host = Tensor::from_i32(Shape::new([ids.len()]), ids)?;
    DeviceTensor::from_host(Arc::clone(backend), host)
}

pub trait CausalLanguageModel<B: PortableBackend + 'static> {
    fn context_length(&self) -> usize;
    fn num_layers(&self) -> usize;

    fn forward(&self, tokens: &[usize]) -> Result<Tensor>;

    /// Runs `tokens` at `position_offset` through the model, updating `caches` (one slot per layer).
    ///
    /// `capacity` pins the KV cache capacity. With `None`, the capacity is the end position rounded
    /// up to a power of two, capped at the context length. Returns logits `[1, vocab]` for the last
    /// position of `tokens`.
    fn forward_with_decode_cache(
        &self,
        tokens: &[usize],
        position_offset: usize,
        caches: &mut [Option<LayerCache<B>>],
        capacity: Option<usize>,
    ) -> Result<Tensor>;

    /// Runs `tokens` like [`CausalLanguageModel::forward_with_decode_cache`] and samples the next
    /// token on the backend according to `request`. Returns `None` and leaves `caches` untouched
    /// when the model or the backend cannot sample on the backend.
    fn forward_with_decode_cache_sample_next(
        &self,
        _tokens: &[usize],
        _position_offset: usize,
        _caches: &mut [Option<LayerCache<B>>],
        _capacity: Option<usize>,
        _request: DecodeSampleRequest,
    ) -> Result<Option<usize>> {
        Ok(None)
    }
}
