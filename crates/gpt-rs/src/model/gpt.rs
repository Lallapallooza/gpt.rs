//! GPT-2, laid out like the Hugging Face `GPT2LMHeadModel`.
//!
//! `wte + wpe` -> `n_layer` x `h.{i}` \[`ln_1` -> `attn` -> +residual -> `ln_2` -> `mlp` ->
//! +residual\] -> `ln_f` -> `lm_head` ([`CausalDecoder`] with the base model [`Gpt2Model`] as
//! `transformer`).
//!
//! Model-specific: learned absolute positions (`wpe`), [`LayerNorm`]s, multi-head attention whose
//! fused `c_attn` projection emits the queries, keys and values, and a non-gated MLP. Every
//! projection except `lm_head` is a [`Conv1D`] with an `[in_features, out_features]` weight.

use std::sync::Arc;

use anyhow::{bail, Result};

use crate::backend::spec::PortableBackend;
use crate::inference::decoder::{BaseModel, CausalDecoder, DecoderConfig};
use crate::inference::{index_tensor, LayerCache};
use crate::nn::{
    self, cached_attention, capture, Activation, ActivationFunction, AttentionConfig,
    AttentionPositions, Conv1D, Embedding, LayerNorm, Params,
};
use crate::ops::functional::{self, DecodeKvCache, DeviceTensorOps};

pub const KIND: &str = "gpt2";

/// GPT-2 decoder.
pub type Gpt<B> = CausalDecoder<B, GptConfig>;

/// The keys of a Hugging Face `GPT2Config` that the model reads.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct GptConfig {
    pub n_positions: usize,
    pub n_embd: usize,
    pub n_layer: usize,
    pub n_head: usize,
    pub layer_norm_epsilon: f32,
    pub activation_function: ActivationFunction,
    #[serde(default = "tied_by_default")]
    pub tie_word_embeddings: bool,
}

/// `GPT2Config` ties `lm_head` to `wte` unless the config says otherwise.
fn tied_by_default() -> bool {
    true
}

impl DecoderConfig for GptConfig {
    const KIND: &'static str = KIND;
    const BASE_MODEL_PREFIX: &'static str = "transformer";

    type Model<B: PortableBackend + 'static> = Gpt2Model<B>;

    fn context_length(&self) -> usize {
        self.n_positions
    }

    fn tie_word_embeddings(&self) -> bool {
        self.tie_word_embeddings
    }

    fn base_model<B: PortableBackend + 'static>(&self, p: &mut Params<B>) -> Result<Gpt2Model<B>> {
        Gpt2Model::new(p, self)
    }
}

/// The GPT-2 transformer (`GPT2Model`).
#[nn::module]
pub struct Gpt2Model {
    pub wte: Embedding,
    pub wpe: Embedding,
    pub h: Vec<Gpt2Block>,
    pub ln_f: LayerNorm,
}

#[nn::module]
impl Gpt2Model {
    pub fn new(p: &mut Params<B>, config: &GptConfig) -> Result<Self> {
        Ok(Self {
            wte: Embedding::new(p)?,
            wpe: Embedding::new(p)?,
            h: p.each(0..config.n_layer, |p, _| Gpt2Block::new(p, config))?,
            ln_f: LayerNorm::new(p, config.layer_norm_epsilon)?,
        })
    }

    fn forward(
        &self,
        tokens: &[usize],
        position: usize,
        caches: &mut [LayerCache<B>],
    ) -> Result<Tensor> {
        let backend = self.wte.weight.backend();
        let inputs_embeds = self.wte(&index_tensor(&backend, tokens.iter().copied())?)?;
        capture::record("wte", &inputs_embeds)?;
        let position_ids = index_tensor(&backend, position..position + tokens.len())?;
        let position_embeds = self.wpe(&position_ids)?;
        capture::record("wpe", &position_embeds)?;
        let mut hidden = inputs_embeds.add(&position_embeds)?;
        let positions = AttentionPositions::new(&backend, position, tokens.len(), None)?;
        for (layer, (block, cache)) in self.h.iter().zip(caches.iter_mut()).enumerate() {
            let scope = capture::scope(format_args!("h.{layer}"));
            let LayerCache::Attention(kv) = &*cache else {
                bail!("GPT-2 layer {layer} needs a KV cache");
            };
            let (output, kv) = block.call((&hidden, kv, &positions))?;
            scope.record(&output)?;
            cache.replace(LayerCache::Attention(kv));
            hidden = output;
        }
        let output = self.ln_f(&hidden)?;
        capture::record("ln_f", &output)?;
        Ok(output)
    }
}

impl<B: PortableBackend + 'static> BaseModel<B> for Gpt2Model<B> {
    fn input_embeddings(&self) -> &Embedding<B> {
        &self.wte
    }

    fn num_layers(&self) -> usize {
        self.h.len()
    }

    fn empty_caches(&self, backend: &Arc<B>, capacity: usize) -> Result<Vec<LayerCache<B>>> {
        self.h
            .iter()
            .map(|block| {
                let cache = block.attn.config.empty_cache(backend, capacity)?;
                Ok(LayerCache::Attention(cache))
            })
            .collect()
    }
}

/// GPT-2 layer (`GPT2Block`): `h = x + attn(ln_1(x))`, `out = h + mlp(ln_2(h))`.
#[nn::module]
pub struct Gpt2Block {
    pub ln_1: LayerNorm,
    pub attn: Gpt2Attention,
    pub ln_2: LayerNorm,
    pub mlp: Gpt2Mlp,
}

#[nn::module]
impl Gpt2Block {
    pub fn new(p: &mut Params<B>, config: &GptConfig) -> Result<Self> {
        let eps = config.layer_norm_epsilon;
        Ok(Self {
            ln_1: LayerNorm::new(p, eps)?,
            attn: Gpt2Attention::new(p, config)?,
            ln_2: LayerNorm::new(p, eps)?,
            mlp: Gpt2Mlp::new(p, config)?,
        })
    }

    /// Runs the layer on `x` (`[T, n_embd]`) at `positions`. Returns the output and the updated
    /// cache.
    fn forward(
        &self,
        x: &Tensor,
        cache: &DecodeKvCache<B>,
        positions: &AttentionPositions<B>,
    ) -> Result<(Tensor, DecodeKvCache<B>)> {
        let normed = self.ln_1(x)?;
        capture::record("ln_1", &normed)?;
        let (attn, cache) = self.attn((&normed, cache, positions))?;
        capture::record("attn", &attn)?;
        let hidden = attn.add(x)?;
        let normed = self.ln_2(&hidden)?;
        capture::record("ln_2", &normed)?;
        let mlp = self.mlp(&normed)?;
        capture::record("mlp", &mlp)?;
        Ok((hidden.add(&mlp)?, cache))
    }
}

/// GPT-2 multi-head causal self-attention (`GPT2Attention`). `c_attn` emits
/// `[queries, keys, values]`, each `n_embd` wide.
#[nn::module]
pub struct Gpt2Attention {
    #[module(config)]
    pub config: AttentionConfig,
    pub c_attn: Conv1D,
    pub c_proj: Conv1D,
}

#[nn::module]
impl Gpt2Attention {
    pub fn new(p: &mut Params<B>, config: &GptConfig) -> Result<Self> {
        Ok(Self {
            config: AttentionConfig::with_equal_heads(config.n_embd, config.n_head)?,
            c_attn: Conv1D::new(p)?,
            c_proj: Conv1D::new(p)?,
        })
    }

    /// Runs attention for `x` (`[T, n_embd]`) at `positions`, appending the keys and values of
    /// `x` to `cache`.
    fn forward(
        &self,
        x: &Tensor,
        cache: &DecodeKvCache<B>,
        positions: &AttentionPositions<B>,
    ) -> Result<(Tensor, DecodeKvCache<B>)> {
        let cfg = &self.config;
        let (t, dim) = (x.shape().dims()[0], cfg.embed_dim);
        let qkv = self.c_attn(x)?;
        let heads = |index: usize| {
            let part = functional::slice_along_axis(&qkv, 1, index * dim, dim)?;
            functional::reshape(&part, &[t, cfg.num_query_heads, cfg.head_dim])
        };
        let (output, cache) =
            cached_attention(cfg, &heads(0)?, &heads(1)?, &heads(2)?, cache, positions)?;
        Ok((self.c_proj(&output)?, cache))
    }
}

/// GPT-2 MLP (`GPT2MLP`): `c_proj(act(c_fc(x)))`.
#[nn::module]
pub struct Gpt2Mlp {
    pub c_fc: Conv1D,
    pub act: Activation,
    pub c_proj: Conv1D,
}

#[nn::module]
impl Gpt2Mlp {
    pub fn new(p: &mut Params<B>, config: &GptConfig) -> Result<Self> {
        Ok(Self {
            c_fc: Conv1D::new(p)?,
            act: config.activation_function.layer(),
            c_proj: Conv1D::new(p)?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let hidden = self.c_fc(x)?;
        let hidden = self.act(&hidden)?;
        self.c_proj(&hidden)
    }
}
