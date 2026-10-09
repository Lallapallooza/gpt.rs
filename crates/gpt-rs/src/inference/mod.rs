//! Running loaded models: the decoder language models ([`decoder`]) with token generation
//! ([`generate`]), and the image models ([`vision`]).

pub mod decoder;
pub mod generate;
pub mod vision;

use std::sync::Arc;

use anyhow::{anyhow, Result};

use crate::backend::PortableBackend;
use crate::ops::functional::{DecodeKvCache, LinearAttentionCache};
use crate::tensor::{DeviceTensor, HostTensor};
use generate::Sampling;

/// Incremental decode state of one decoder layer, carried between generation steps.
pub enum LayerCache<B: PortableBackend + 'static> {
    /// Fixed-capacity key/value cache of a softmax-attention layer.
    Attention(DecodeKvCache<B>),
    /// Convolution window and recurrent state of a linear-attention (Gated DeltaNet) layer.
    LinearAttention(LinearAttentionCache<B>),
}

impl<B: PortableBackend + 'static> LayerCache<B> {
    /// The state tensors: keys and values, or convolution window and recurrent state.
    pub fn tensors(&self) -> [&DeviceTensor<B>; 2] {
        match self {
            LayerCache::Attention(kv) => [kv.keys(), kv.values()],
            LayerCache::LinearAttention(state) => [state.conv_state(), state.recurrent_state()],
        }
    }

    /// Replaces this cache with `next` and stops exporting the replaced state.
    pub(crate) fn replace(&mut self, next: LayerCache<B>) {
        for tensor in self.tensors() {
            tensor.release_export();
        }
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
    let host = HostTensor::si32(vec![ids.len()], ids)?;
    DeviceTensor::from_host(Arc::clone(backend), host)
}

/// What a loaded causal language model does: logits of a token sequence, and decoding the tokens
/// after a prompt.
pub trait CausalLanguageModel {
    /// The most positions a sequence can hold.
    fn context_length(&self) -> usize;

    /// The logits `[T, vocab]` of every position of `tokens`, computed as one prefill.
    fn forward(&self, tokens: &[usize]) -> Result<HostTensor>;

    /// Prefills `prompt` and returns the state that decodes the tokens after it, each picked as
    /// `sampling` says. `max_len` bounds the tokens, the prompt included, that the state will
    /// hold. It sizes the caches.
    fn decode(
        &self,
        prompt: &[usize],
        max_len: usize,
        sampling: &Sampling,
    ) -> Result<Box<dyn Decode + '_>>;
}

/// The tokens that a causal language model decodes after a prompt, one at a time.
pub trait Decode {
    /// The next token. The first call returns the token after the prompt.
    fn next_token(&mut self) -> Result<usize>;
}
