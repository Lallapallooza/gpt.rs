//! The decoder-only causal language model that every text model uses.
//!
//! A model contributes only its configuration ([`DecoderConfig`]). This module implements
//! prefill, decoding and cache management once for all models. The modules mirror the Hugging
//! Face `*ForCausalLM` tree, which names the parameters and the module outputs that
//! [`crate::nn::capture`] records.

use std::sync::Arc;

use anyhow::{anyhow, bail, ensure, Result};
use rand::RngCore;
use serde::de::DeserializeOwned;
use serde::Serialize;

use super::{index_tensor, CausalLanguageModel, LayerCache};
use crate::backend::spec::{DecodeSampleRequest, PortableBackend};
use crate::model::ModelConfig;
use crate::module::Layer;
use crate::nn::{
    self, capture, AttentionPositions, DecoderBlock, Embedding, LayerLoader, Linear, MixerConfig,
    MlpConfig, Norm, NormConfig, RotaryEmbedding,
};
use crate::ops::functional::slice_along_axis;
use crate::ops::graph::context::{current_arena, push_default_arena};
use crate::ops::graph::GraphArena;
use crate::runtime::{LoadedModel, ModelInput, ModelOutput};
use crate::tensor::{DeviceTensor, DeviceTensorOps, Shape, Tensor};

/// Largest number of prompt tokens processed by one compiled forward program.
///
/// Prefill splits the prompt into chunks of this size plus one remainder chunk. Every chunk
/// streams all weights once, so larger chunks mean fewer weight passes. The chunk size also bounds
/// the attention scores and activations that one program holds.
pub const PREFILL_CHUNK: usize = 256;

/// Structure of a [`CausalDecoder`].
#[derive(Debug, Clone)]
pub struct DecoderLayout {
    pub vocab_size: usize,
    pub context_length: usize,
    pub embed_dim: usize,
    pub learned_positions: bool,
    /// The two norms of each layer and `model.norm`.
    pub norm: NormConfig,
    /// Token mixer of each layer. The attention layers share one rotary embedding.
    pub mixers: Vec<MixerConfig>,
    pub mlp: MlpConfig,
}

/// Checkpoint configuration of a decoder model: its model kind and the layout it describes.
pub trait DecoderConfig: Serialize + DeserializeOwned + Send + 'static {
    const KIND: &'static str;

    /// Validates the configuration and maps it to the decoder structure.
    fn layout(&self) -> Result<DecoderLayout>;
}

/// Decoder-only causal language model configured by `C`, laid out like a Hugging Face
/// `*ForCausalLM`: the decoder `model` and the `lm_head` projection.
#[nn::module]
pub struct CausalDecoder<C> {
    #[module(config)]
    pub config: C,
    #[module(config)]
    vocab_size: usize,
    #[module(config)]
    context_length: usize,
    pub model: DecoderModel,
    pub lm_head: Linear,
}

/// The decoder of a [`CausalDecoder`], the Hugging Face `model` module.
#[nn::module]
pub struct DecoderModel {
    pub embed_tokens: Embedding,
    pub embed_positions: Option<Embedding>,
    pub layers: Vec<DecoderBlock>,
    pub norm: Norm,
    #[module(config)]
    pub rotary_emb: Option<RotaryEmbedding>,
}

#[nn::module]
impl DecoderModel {
    /// Returns the final hidden states `[T, embed_dim]` of `tokens` at `position` and updates
    /// `caches`, one per layer.
    fn forward(
        &self,
        tokens: &[usize],
        position: usize,
        caches: &mut [LayerCache<B>],
    ) -> Result<Tensor> {
        let backend = self.embed_tokens.weight.backend();
        let mut hidden = self.embed(&backend, tokens, position)?;
        let rotary = self.rotary_emb.as_ref();
        let positions = AttentionPositions::new(&backend, position, tokens.len(), rotary)?;
        for (layer, (block, cache)) in self.layers.iter().zip(caches.iter_mut()).enumerate() {
            let scope = capture::scope(format_args!("layers.{layer}"));
            let (output, updated) = block.call((&hidden, cache, &positions))?;
            scope.record(&output)?;
            cache.replace(updated);
            hidden = output;
        }
        let normed = self.norm(&hidden)?;
        capture::record("norm", &normed)?;
        Ok(normed)
    }

    /// Token embeddings of `tokens` at `position`, plus learned position embeddings when the
    /// model has them.
    fn embed(&self, backend: &Arc<B>, tokens: &[usize], position: usize) -> Result<Tensor> {
        let token_indices = index_tensor(backend, tokens.iter().copied())?;
        let Some(embed_positions) = &self.embed_positions else {
            let hidden = self.embed_tokens(&token_indices)?;
            capture::record("embed_tokens", &hidden)?;
            return Ok(hidden);
        };
        let position_indices = index_tensor(backend, position..position + tokens.len())?;
        let positions = embed_positions.call(&position_indices)?;
        let hidden = self.embed_tokens(&token_indices)?;
        capture::record("embed_tokens", &hidden)?;
        capture::record("embed_positions", &positions)?;
        hidden.add(&positions)
    }
}

/// Builds the decoder of `config.kind` from checkpoint tensors.
pub(crate) fn build_from_model_config<B: PortableBackend + 'static, C: DecoderConfig>(
    backend: Arc<B>,
    cfg: &ModelConfig,
    get: &mut dyn FnMut(&str) -> Result<DeviceTensor<B>>,
) -> Result<Box<dyn LoadedModel<B>>> {
    let config: C = serde_json::from_value(cfg.config.clone())
        .map_err(|err| anyhow!("invalid {} config: {err}", C::KIND))?;
    let mut params =
        LayerLoader::new(backend, get).with_linear_input_dtype(cfg.runtime.matmul_input_dtype);
    Ok(Box::new(CausalDecoder::build(config, &mut params)?))
}

impl<B: PortableBackend + 'static, C: DecoderConfig> CausalDecoder<B, C> {
    /// Randomly initialised model (see [`LayerLoader::random`]).
    pub fn random(config: C, backend: Arc<B>, rng: &mut dyn RngCore) -> Result<Self> {
        Self::build(config, &mut LayerLoader::random(backend, rng))
    }

    pub fn build(config: C, params: &mut LayerLoader<'_, B>) -> Result<Self> {
        let DecoderLayout {
            vocab_size,
            context_length,
            embed_dim: dim,
            learned_positions,
            norm,
            mixers,
            mlp,
        } = config.layout()?;
        let mut ropes = mixers.iter().filter_map(|mixer| match mixer {
            MixerConfig::Attention(attention) => Some(attention.rope),
            MixerConfig::LinearAttention(_) => None,
        });
        let rope = ropes.next().flatten();
        ensure!(
            ropes.all(|other| other == rope),
            "attention layers must share one rotary embedding"
        );
        let layers = mixers
            .iter()
            .enumerate()
            .map(|(layer, mixer)| {
                let prefix = format!("model.layers.{layer}");
                DecoderBlock::load(params, &prefix, dim, norm, mixer, mlp)
            })
            .collect::<Result<Vec<_>>>()?;
        let embed_tokens = params.embedding("model.embed_tokens", vocab_size, dim)?;
        let embed_positions = learned_positions
            .then(|| params.embedding("model.embed_positions", context_length, dim))
            .transpose()?;
        let model = DecoderModel {
            embed_tokens,
            embed_positions,
            layers,
            norm: norm.load(params, "model.norm", dim)?,
            rotary_emb: rope.map(RotaryEmbedding::new).transpose()?,
        };
        Ok(Self {
            vocab_size,
            context_length,
            model,
            lm_head: params.linear("lm_head", dim, vocab_size, false)?,
            config,
        })
    }
}

impl<B: PortableBackend + 'static, C> CausalDecoder<B, C> {
    /// Full-sequence logits `[T, vocab]`, computed as a prefill into fresh caches.
    pub fn forward(&self, tokens: &[usize]) -> Result<Tensor> {
        let mut logits = Vec::with_capacity(tokens.len() * self.vocab_size);
        let mut slots = vec![None; self.model.layers.len()];
        let on_logits = |chunk: DeviceTensor<B>| {
            logits.extend_from_slice(chunk.to_host()?.data());
            Ok(())
        };
        self.run(tokens, 0, &mut slots, Some(tokens.len()), true, on_logits)?;
        Tensor::from_vec(Shape::new([tokens.len(), self.vocab_size]), logits)
    }

    /// Runs `tokens` at `position_offset` in chunks of at most [`PREFILL_CHUNK`], reading and
    /// extending the per-layer caches in `slots`. With `every_position`, `on_logits` receives the
    /// logits of every chunk. Otherwise it receives only the `[1, vocab]` logits of the last
    /// position. `run` returns its last result.
    ///
    /// Each chunk, including its parameter-only computations, is captured into one arena and runs
    /// as exactly one compiled program. `slots` change only after every chunk succeeds.
    fn run<T>(
        &self,
        tokens: &[usize],
        position_offset: usize,
        slots: &mut [Option<LayerCache<B>>],
        capacity: Option<usize>,
        every_position: bool,
        mut on_logits: impl FnMut(DeviceTensor<B>) -> Result<T>,
    ) -> Result<T> {
        ensure!(!tokens.is_empty(), "token sequence must be non-empty");
        let end = position_offset
            .checked_add(tokens.len())
            .ok_or_else(|| anyhow!("token position offset overflow"))?;
        ensure!(
            end <= self.context_length,
            "tokens end at position {end}, past the model context length {}",
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
        let capacity = self.cache_capacity(capacity, end)?;
        let mut caches = self.prepare_caches(slots, position_offset, capacity)?;
        let backend = self.lm_head.weight.backend();

        let mut start = 0;
        loop {
            let _arena = current_arena::<B>()
                .is_none()
                .then(|| push_default_arena(GraphArena::new(Arc::clone(&backend))));
            let len = (tokens.len() - start).min(PREFILL_CHUNK);
            let chunk = &tokens[start..start + len];
            let hidden = {
                let _scope = capture::scope(format_args!("model"));
                self.model((chunk, position_offset + start, &mut caches))?
            };
            start += len;
            if start == tokens.len() {
                let rows = if every_position {
                    hidden
                } else {
                    let last = hidden.shape().dims()[0] - 1;
                    slice_along_axis(&hidden, 0, last, 1)?
                };
                let out = on_logits(self.logits(&rows)?)?;
                for (slot, cache) in slots.iter_mut().zip(caches) {
                    *slot = Some(cache);
                }
                return Ok(out);
            }
            if every_position {
                on_logits(self.logits(&hidden)?)?;
            }
            let states: Vec<_> = caches.iter().flat_map(LayerCache::tensors).collect();
            DeviceTensor::materialize_many(&states)?;
        }
    }

    /// KV cache capacity for a call that ends at position `end`. Rounding `end` up to a power of
    /// two lets calls at different positions reuse compiled programs.
    fn cache_capacity(&self, fixed: Option<usize>, end: usize) -> Result<usize> {
        let context_length = self.context_length;
        match fixed {
            Some(capacity) => {
                ensure!(
                    end <= capacity && capacity <= context_length,
                    "KV cache capacity {capacity} must cover {end} tokens and fit the context length {context_length}"
                );
                Ok(capacity)
            }
            None => Ok(end.next_power_of_two().min(context_length)),
        }
    }

    /// Working caches for a call that writes positions `position_offset..` with room for
    /// `capacity`. Populated slots are cloned, so they stay intact if the call fails. Their KV
    /// caches must hold exactly `position_offset` tokens.
    fn prepare_caches(
        &self,
        slots: &[Option<LayerCache<B>>],
        position_offset: usize,
        capacity: usize,
    ) -> Result<Vec<LayerCache<B>>> {
        let layers = &self.model.layers;
        ensure!(
            slots.len() == layers.len(),
            "expected {} cache slots (one per layer), got {}",
            layers.len(),
            slots.len()
        );
        if slots.iter().all(Option::is_none) {
            ensure!(
                position_offset == 0,
                "empty caches require position offset 0 (got {position_offset})"
            );
            let backend = self.lm_head.weight.backend();
            return layers
                .iter()
                .map(|block| block.mixer.empty_cache(&backend, capacity))
                .collect();
        }
        slots
            .iter()
            .enumerate()
            .map(|(layer, slot)| {
                let cache = slot
                    .clone()
                    .ok_or_else(|| anyhow!("layer {layer} is missing its decode cache"))?;
                let LayerCache::Attention(kv) = &cache else {
                    return Ok(cache);
                };
                ensure!(
                    kv.len() == position_offset,
                    "layer {layer} KV cache holds {} tokens, position offset is {position_offset}",
                    kv.len()
                );
                if kv.capacity() >= capacity {
                    return Ok(cache);
                }
                let grown = LayerCache::Attention(kv.grow(capacity)?);
                cache.release_exports();
                Ok(grown)
            })
            .collect()
    }

    /// Logits for final hidden states `[N, embed_dim]`.
    fn logits(&self, hidden: &DeviceTensor<B>) -> Result<DeviceTensor<B>> {
        let logits = self.lm_head(hidden)?;
        capture::record("lm_head", &logits)?;
        Ok(logits)
    }
}

impl<B: PortableBackend + 'static, C> CausalLanguageModel<B> for CausalDecoder<B, C> {
    fn context_length(&self) -> usize {
        self.context_length
    }

    fn num_layers(&self) -> usize {
        self.model.layers.len()
    }

    fn forward(&self, tokens: &[usize]) -> Result<Tensor> {
        CausalDecoder::forward(self, tokens)
    }

    fn forward_with_decode_cache(
        &self,
        tokens: &[usize],
        position_offset: usize,
        caches: &mut [Option<LayerCache<B>>],
        capacity: Option<usize>,
    ) -> Result<Tensor> {
        let to_host = |logits: DeviceTensor<B>| logits.to_host();
        self.run(tokens, position_offset, caches, capacity, false, to_host)
    }

    fn forward_with_decode_cache_sample_next(
        &self,
        tokens: &[usize],
        position_offset: usize,
        caches: &mut [Option<LayerCache<B>>],
        capacity: Option<usize>,
        request: DecodeSampleRequest,
    ) -> Result<Option<usize>> {
        let backend = self.lm_head.weight.backend();
        if !backend.supports_decode_sampling(request) {
            return Ok(None);
        }
        let sample = |logits: DeviceTensor<B>| {
            let handle = logits.materialize()?;
            Ok(backend.sample_decode_token(&handle, &logits.tensor_spec(), request)?)
        };
        self.run(tokens, position_offset, caches, capacity, false, sample)
    }
}

impl<B: PortableBackend + 'static, C: DecoderConfig> LoadedModel<B> for CausalDecoder<B, C> {
    fn kind(&self) -> &str {
        C::KIND
    }

    fn forward(&mut self, input: ModelInput<B>) -> Result<ModelOutput> {
        match input {
            ModelInput::Tokens(tokens) => {
                Ok(ModelOutput::Tensor(CausalDecoder::forward(self, &tokens)?))
            }
            ModelInput::Vision(_) => {
                bail!("model '{}' expects token input, got vision input", C::KIND)
            }
        }
    }

    fn as_causal_lm(&self) -> Option<&dyn CausalLanguageModel<B>> {
        Some(self)
    }
}
