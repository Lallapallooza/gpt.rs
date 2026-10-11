//! The decoder-only causal language model that every text model uses.
//!
//! A model contributes its configuration ([`DecoderConfig`]): its module names and the layout of
//! its base model ([`DecoderModel`]). [`CausalDecoder`] adds the `lm_head` and implements prefill,
//! decoding and cache management once for all models. The modules mirror the Hugging Face
//! `*ForCausalLM` tree, which names the parameters and the module outputs that
//! [`crate::nn::capture`] records.

use std::sync::Arc;

use anyhow::{anyhow, bail, ensure, Result};
use gpt_rs_ir::dialects::tensor::state_type;
use serde::de::DeserializeOwned;
use serde::Serialize;

use super::generate::Sampling;
use super::{index_tensor, CausalLanguageModel, Decode, LayerCache};
use crate::backend::PortableBackend;
use crate::model::ModelConfig;
use crate::nn::{
    self, capture, DecoderBlock, DecoderNames, Embedding, Linear, MixerConfig, MlpConfig, Norm,
    NormConfig, Params, RopeConfig, RotaryEmbedding,
};
use crate::ops::functional::{self, AttentionPositions};
use crate::ops::graph::context::{current_arena, push_default_arena, ArenaGuard};
use crate::ops::graph::GraphArena;
use crate::runtime::LoadedModel;
use crate::s;
use crate::tensor::{DeviceTensor, HostTensor};

/// Largest number of prompt tokens processed by one compiled forward program.
///
/// Prefill splits the prompt into chunks of this size plus one remainder chunk. Every chunk
/// streams all weights once, so larger chunks mean fewer weight passes. The chunk size also bounds
/// the attention scores and activations that one program holds.
pub const PREFILL_CHUNK: usize = 256;

/// Checkpoint configuration of a decoder model: its model kind, its module names, and the
/// structure of its base model.
pub trait DecoderConfig: DecoderNames + Serialize + DeserializeOwned + Send + 'static {
    const KIND: &'static str;
    /// Module name of the base model in the `*ForCausalLM` tree, the `base_model_prefix` of
    /// `transformers`.
    const BASE_MODEL_PREFIX: &'static str;

    fn context_length(&self) -> usize;

    fn tie_word_embeddings(&self) -> bool;

    /// Validates the configuration and returns the structure of the base model.
    fn layout(&self) -> Result<DecoderLayout>;
}

/// Decoder-only causal language model configured by `C`, laid out like a Hugging Face
/// `*ForCausalLM`: the base model, named [`DecoderConfig::BASE_MODEL_PREFIX`], and the `lm_head`
/// projection.
#[nn::module]
pub struct CausalDecoder<C: DecoderConfig> {
    #[module(config)]
    pub config: C,
    #[module(config)]
    vocab_size: usize,
    #[module(config)]
    context_length: usize,
    #[module(rename = C::BASE_MODEL_PREFIX)]
    pub base_model: DecoderModel<C>,
    pub lm_head: Linear,
}

/// Structure of a [`DecoderModel`].
#[derive(Debug, Clone)]
pub struct DecoderLayout {
    /// The two norms of each layer and the final norm.
    pub norm: NormConfig,
    /// The token mixer of each layer.
    pub mixers: Vec<MixerConfig>,
    /// The MLP of each layer.
    pub mlp: MlpConfig,
    /// The rotary embedding of the attention layers.
    pub rope: Option<RopeConfig>,
    /// Learned absolute positions: an embedding of the positions adds to the token embeddings.
    pub absolute_positions: bool,
}

/// Base model of a decoder, the Hugging Face `*Model` from the token embeddings to the final norm,
/// with the module names of the family `N`.
#[nn::module]
pub struct DecoderModel<N: DecoderNames> {
    #[module(rename = N::EMBED_TOKENS)]
    pub embed_tokens: Embedding,
    #[module(rename = N::EMBED_POSITIONS)]
    pub embed_positions: Option<Embedding>,
    #[module(rename = N::LAYERS)]
    pub layers: Vec<DecoderBlock<N>>,
    #[module(rename = N::NORM)]
    pub norm: Norm,
    #[module(config)]
    pub rotary_emb: Option<RotaryEmbedding>,
}

#[nn::module]
impl<N: DecoderNames> DecoderModel<N> {
    pub fn new(p: &mut Params<B>, layout: DecoderLayout) -> Result<Self> {
        let DecoderLayout {
            norm,
            mixers,
            mlp,
            rope,
            absolute_positions,
        } = layout;
        if let Some(RopeConfig { rotary_dim, .. }) = rope {
            for mixer in &mixers {
                let MixerConfig::Attention(attention) = mixer else {
                    continue;
                };
                let head_dim = attention.head_dim;
                ensure!(
                    rotary_dim > 0 && rotary_dim.is_multiple_of(2) && rotary_dim <= head_dim,
                    "rotary_dim ({rotary_dim}) must be positive, even and <= head_dim ({head_dim})"
                );
            }
        }
        Ok(Self {
            embed_tokens: Embedding::new(p)?,
            embed_positions: absolute_positions.then(|| Embedding::new(p)).transpose()?,
            layers: p.each(&mixers, |p, mixer| DecoderBlock::new(p, norm, mixer, mlp))?,
            norm: Norm::new(p, norm)?,
            rotary_emb: rope.map(RotaryEmbedding::new).transpose()?,
        })
    }

    /// The final hidden states `[T, hidden_size]` of the token ids `tokens` at `positions`, both
    /// `si32 [T]`. It replaces the cache of each layer in `caches` with the cache after the
    /// tokens.
    fn forward(&self, tokens: &Tensor, positions: &Tensor, caches: &mut [LayerCache<B>]) -> Tensor {
        let mut hidden = self.embed_tokens(tokens);
        capture::record(N::EMBED_TOKENS, &hidden);
        if let Some(position_embeds) = self.embed_positions(positions) {
            capture::record(N::EMBED_POSITIONS, &position_embeds);
            hidden = hidden + position_embeds;
        }
        let rotary = self.rotary_emb.as_ref().map(|rope| rope.tables(positions));
        let positions = AttentionPositions::new(positions.clone(), rotary);
        for (layer, (block, cache)) in self.layers.iter().zip(caches.iter_mut()).enumerate() {
            let scope = capture::scope(format_args!("{}.{layer}", N::LAYERS));
            let (output, updated) = block.call((&hidden, cache, &positions));
            scope.record(&output);
            cache.replace(updated);
            hidden = output;
        }
        let normed = self.norm(&hidden);
        capture::record(N::NORM, &normed);
        normed
    }

    /// Empty caches of every layer on `backend`, with room for `capacity` positions. Linear
    /// attention keeps a fixed-size state instead.
    pub fn empty_caches(&self, backend: &Arc<B>, capacity: usize) -> Result<Vec<LayerCache<B>>> {
        self.layers
            .iter()
            .map(|block| block.mixer.empty_cache(backend, capacity))
            .collect()
    }
}

/// Builds the decoder of `config.kind` from checkpoint tensors.
pub(crate) fn build_from_model_config<B: PortableBackend + 'static, C: DecoderConfig>(
    backend: Arc<B>,
    cfg: &ModelConfig,
    get: &mut dyn FnMut(&str) -> Result<Option<DeviceTensor<B>>>,
) -> Result<Box<dyn LoadedModel>> {
    let config: C = serde_json::from_value(cfg.config.clone())
        .map_err(|err| anyhow!("invalid {} config: {err}", C::KIND))?;
    let mut params = Params::new(backend, get)
        .with_linear_input_dtype(cfg.runtime.matmul_input_dtype)
        .with_base_model_prefix(C::BASE_MODEL_PREFIX);
    Ok(Box::new(CausalDecoder::new(&mut params, config)?))
}

#[nn::module]
impl<C: DecoderConfig> CausalDecoder<C> {
    pub fn new(p: &mut Params<B>, config: C) -> Result<Self> {
        // A tied `lm_head` shares the token embedding, so the base model comes first.
        let layout = config.layout()?;
        let base_model = p.scoped(C::BASE_MODEL_PREFIX, |p| DecoderModel::new(p, layout))?;
        let embeddings = &base_model.embed_tokens.weight;
        let &[vocab_size, _] = embeddings.shape().dims() else {
            bail!(
                "token embeddings must be [vocab_size, hidden_size], got {:?}",
                embeddings.shape().dims()
            );
        };
        Ok(Self {
            lm_head: if config.tie_word_embeddings() {
                Linear::tied(embeddings, p.linear_input_dtype())
            } else {
                Linear::new(p)?
            },
            vocab_size,
            context_length: config.context_length(),
            base_model,
            config,
        })
    }
}

/// The arena that a chunk records into: a fresh one, unless the caller installed a default
/// arena. A chunk in one arena runs as one program, its computations over parameters included.
fn chunk_arena<B: PortableBackend + 'static>(backend: &Arc<B>) -> Option<ArenaGuard> {
    current_arena::<B>()
        .is_none()
        .then(|| push_default_arena(GraphArena::new(Arc::clone(backend))))
}

impl<B: PortableBackend + 'static, C: DecoderConfig> CausalDecoder<B, C> {
    fn backend(&self) -> Arc<B> {
        self.lm_head.weight.backend()
    }

    /// The KV cache capacity of sequences of up to `len` tokens. Rounding up to a power of two
    /// lets sequences of different lengths reuse compiled programs.
    fn cache_capacity(&self, len: usize) -> usize {
        len.next_power_of_two().min(self.context_length)
    }

    /// The final hidden states of the token ids `tokens` (`si32 [T]`) at positions
    /// `start..start + T`. It reads and extends `caches`.
    fn chunk(
        &self,
        tokens: &DeviceTensor<B>,
        start: usize,
        caches: &mut [LayerCache<B>],
    ) -> Result<DeviceTensor<B>> {
        let positions = index_tensor(&self.backend(), start..start + tokens.size(0))?;
        Ok(self.base_model((tokens, &positions, caches)))
    }

    /// Runs `tokens` from position 0 into the empty `caches`, in chunks of at most
    /// [`PREFILL_CHUNK`]. It returns what `on_logits` makes of the logits of the last chunk. With
    /// `every_position`, those are the logits of every position of the chunk. Otherwise they are
    /// the logits of the last position only (`[1, vocab]`). With `every_position`, `on_logits`
    /// also receives the logits of every earlier chunk.
    ///
    /// Each chunk, with what `on_logits` records, records into one arena and runs as one program.
    fn prefill<T>(
        &self,
        tokens: &[usize],
        caches: &mut [LayerCache<B>],
        every_position: bool,
        mut on_logits: impl FnMut(DeviceTensor<B>) -> Result<T>,
    ) -> Result<T> {
        ensure!(!tokens.is_empty(), "token sequence must be non-empty");
        ensure!(
            tokens.len() <= self.context_length,
            "{} tokens exceed the model context length {}",
            tokens.len(),
            self.context_length
        );
        if let Some((i, token)) = tokens
            .iter()
            .enumerate()
            .find(|(_, &t)| t >= self.vocab_size)
        {
            bail!(
                "token id {token} at position {i} exceeds vocabulary size {}",
                self.vocab_size
            );
        }
        let backend = self.backend();
        let last = (tokens.len() - 1) / PREFILL_CHUNK;
        for (index, chunk) in tokens.chunks(PREFILL_CHUNK).enumerate() {
            let _arena = chunk_arena(&backend);
            let ids = index_tensor(&backend, chunk.iter().copied())?;
            let hidden = self.chunk(&ids, index * PREFILL_CHUNK, caches)?;
            if index == last {
                let rows = if every_position {
                    hidden
                } else {
                    hidden.slice(s![-1..])
                };
                return on_logits(self.logits(&rows));
            }
            if every_position {
                on_logits(self.logits(&hidden))?;
            }
            let states: Vec<_> = caches.iter().flat_map(LayerCache::tensors).collect();
            DeviceTensor::materialize_many(&states)?;
        }
        unreachable!("a non-empty sequence has a last chunk")
    }

    /// Logits for final hidden states `[N, embed_dim]`.
    fn logits(&self, hidden: &DeviceTensor<B>) -> DeviceTensor<B> {
        let logits = self.lm_head(hidden);
        capture::record("lm_head", &logits);
        logits
    }
}

impl<B: PortableBackend + 'static, C: DecoderConfig> CausalLanguageModel for CausalDecoder<B, C> {
    fn context_length(&self) -> usize {
        self.context_length
    }

    fn forward(&self, tokens: &[usize]) -> Result<HostTensor> {
        let capacity = self.cache_capacity(tokens.len());
        let mut caches = self.base_model.empty_caches(&self.backend(), capacity)?;
        let mut logits = Vec::with_capacity(tokens.len() * self.vocab_size);
        self.prefill(tokens, &mut caches, true, |chunk| {
            logits.extend_from_slice(chunk.to_host()?.as_f32()?);
            Ok(())
        })?;
        HostTensor::f32(vec![tokens.len(), self.vocab_size], logits)
    }

    fn decode(
        &self,
        prompt: &[usize],
        max_len: usize,
        sampling: &Sampling,
    ) -> Result<Box<dyn Decode + '_>> {
        let backend = self.backend();
        let capacity = self.cache_capacity(max_len.max(prompt.len()));
        let mut caches = self.base_model.empty_caches(&backend, capacity)?;
        let mut picker = Picker::new(&backend, sampling)?;
        let next = self.prefill(
            prompt,
            &mut caches,
            false,
            |logits| Ok(picker.pick(&logits)),
        )?;
        // The prefill runs here, together with the pick of the first token.
        DeviceTensor::materialize_many(&[&next])?;
        Ok(Box::new(Decoding {
            model: self,
            caches,
            position: prompt.len(),
            capacity,
            next,
            returned: false,
            picker,
        }))
    }
}

impl<B: PortableBackend + 'static, C: DecoderConfig> LoadedModel for CausalDecoder<B, C> {
    fn kind(&self) -> &str {
        C::KIND
    }

    fn as_causal_lm(&self) -> Option<&dyn CausalLanguageModel> {
        Some(self)
    }
}

/// Picks tokens from logits on the device, as a [`Sampling`] says.
struct Picker<B: PortableBackend + 'static> {
    sampling: Sampling,
    /// The generator state of random draws (`ui64 [2]`).
    state: Option<DeviceTensor<B>>,
}

impl<B: PortableBackend + 'static> Picker<B> {
    fn new(backend: &Arc<B>, sampling: &Sampling) -> Result<Self> {
        let state = if sampling.is_random() {
            // A generator state is a key and a counter.
            let bytes: Vec<u8> = [sampling.seed, 0]
                .iter()
                .flat_map(|word| word.to_le_bytes())
                .collect();
            let state = HostTensor::from_le_bytes(&state_type(), &bytes)?;
            Some(DeviceTensor::from_host(Arc::clone(backend), state)?)
        } else {
            None
        };
        Ok(Self {
            sampling: *sampling,
            state,
        })
    }

    /// The token (`si32 [1]`) that the logits `[1, vocab]` give.
    fn pick(&mut self, logits: &DeviceTensor<B>) -> DeviceTensor<B> {
        let Some(state) = &self.state else {
            return logits.argmax(-1);
        };
        let Sampling {
            temperature, top_k, ..
        } = self.sampling;
        let (token, state) = functional::sample(logits, temperature, top_k, state);
        state.export();
        self.state = Some(state);
        token
    }
}

/// The decoding state of a [`CausalDecoder`] after a prompt.
struct Decoding<'a, B: PortableBackend + 'static, C: DecoderConfig> {
    /// The model that runs each step.
    model: &'a CausalDecoder<B, C>,
    /// The decode state of each layer.
    caches: Vec<LayerCache<B>>,
    /// Tokens in the caches.
    position: usize,
    /// The positions the caches have room for.
    capacity: usize,
    /// The token after them (`si32 [1]`), which a step feeds to the model.
    next: DeviceTensor<B>,
    /// Whether [`Decode::next_token`] returned `next`.
    returned: bool,
    /// Picks the token after each step.
    picker: Picker<B>,
}

impl<B: PortableBackend + 'static, C: DecoderConfig> Decode for Decoding<'_, B, C> {
    /// Feeds the last token to the model and picks the next one, as one program. The first call
    /// reads the token that the prefill picked.
    fn next_token(&mut self) -> Result<usize> {
        if self.returned {
            ensure!(
                self.position < self.capacity,
                "the KV caches hold {} positions, all used",
                self.capacity
            );
            let _arena = chunk_arena(&self.model.backend());
            let hidden = self
                .model
                .chunk(&self.next, self.position, &mut self.caches)?;
            self.position += 1;
            self.next = self.picker.pick(&self.model.logits(&hidden));
        }
        self.returned = true;
        let token = self.next.to_host()?;
        let &[id] = token.as_si32()? else {
            bail!("a step picks one token, got {:?}", token.dims);
        };
        usize::try_from(id).map_err(|_| anyhow!("the model picked the negative token id {id}"))
    }
}
