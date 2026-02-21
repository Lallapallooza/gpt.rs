use crate::backend::spec::PortableBackend;
use crate::nn::{AttentionConfig, CausalSelfAttention, Embedding, GatedFeedForward, RmsNorm};
use crate::ops::functional::{self, AttentionCache, DecodeKvCache, RopeConfig, RopeScaling};
use crate::tensor::{DeviceTensor, DeviceTensorOps, Shape, Tensor};
use anyhow::{anyhow, bail, ensure, Result};
use rand::Rng;
use std::convert::TryFrom;
use std::sync::Arc;

pub const KIND: &str = "ministral";

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MinistralConfig {
    pub vocab_size: usize,
    pub context_length: usize,
    pub embed_dim: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub mlp_hidden_dim: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub rotary_dim: usize,
    #[serde(default)]
    pub rope_scaling: RopeScaling,
}

impl Default for MinistralConfig {
    fn default() -> Self {
        Self {
            vocab_size: 32_000,
            context_length: 2_048,
            embed_dim: 512,
            num_layers: 8,
            num_heads: 8,
            num_kv_heads: 8,
            mlp_hidden_dim: 2048,
            rms_norm_eps: 1e-5,
            rope_theta: 10_000.0,
            rotary_dim: 64,
            rope_scaling: RopeScaling::None,
        }
    }
}

pub(crate) fn build_from_model_config<B: PortableBackend + 'static>(
    backend: Arc<B>,
    cfg: &super::ModelConfig,
    get: &mut dyn FnMut(&str) -> Result<DeviceTensor<B>>,
) -> Result<Box<dyn crate::runtime::LoadedModel<B>>> {
    let model_cfg: MinistralConfig = serde_json::from_value(cfg.config.clone())
        .map_err(|err| anyhow!("invalid ministral config: {err}"))?;
    Ok(Box::new(Ministral::build_from_params(
        model_cfg, backend, get,
    )?))
}

pub struct MinistralBlock<B: PortableBackend + 'static> {
    pub backend: Arc<B>,
    pub attention: CausalSelfAttention<B>,
    pub feed_forward: GatedFeedForward<B>,
    pub norm_1: RmsNorm<B>,
    pub norm_2: RmsNorm<B>,
}

impl<B: PortableBackend + 'static> MinistralBlock<B> {
    fn attention_forward(
        &self,
        x: &DeviceTensor<B>,
        cache: Option<&AttentionCache<B>>,
        cos: &DeviceTensor<B>,
        sin: &DeviceTensor<B>,
    ) -> Result<(DeviceTensor<B>, AttentionCache<B>)> {
        let qkv = self.attention.proj_qkv.forward(x)?;
        let qkv = functional::apply_rope_qkv_packed(
            self.backend.as_ref(),
            &self.attention.config,
            &qkv,
            cos,
            sin,
        )?;
        let functional::AttentionComputation { output, cache, .. } =
            functional::attention(self.backend.as_ref(), &self.attention.config, &qkv, cache)?;
        let output = self.attention.proj_out.forward(&output)?;
        Ok((output, cache))
    }

    pub fn forward(
        &self,
        x: &DeviceTensor<B>,
        cos: &DeviceTensor<B>,
        sin: &DeviceTensor<B>,
    ) -> Result<DeviceTensor<B>> {
        let normed = self.norm_1.forward(x)?;
        let (attn_output, _) = self.attention_forward(&normed, None, cos, sin)?;
        let residual = attn_output.add(x)?;
        let normed2 = self.norm_2.forward(&residual)?;
        let ff_output = self.feed_forward.forward(&normed2)?;
        ff_output.add(&residual)
    }

    pub fn forward_with_cache(
        &self,
        x: &DeviceTensor<B>,
        cache: Option<&AttentionCache<B>>,
        cos: &DeviceTensor<B>,
        sin: &DeviceTensor<B>,
    ) -> Result<(DeviceTensor<B>, AttentionCache<B>)> {
        let normed = self.norm_1.forward(x)?;
        let (attn_output, updated_cache) = self.attention_forward(&normed, cache, cos, sin)?;
        let residual = attn_output.add(x)?;
        let normed2 = self.norm_2.forward(&residual)?;
        let ff_output = self.feed_forward.forward(&normed2)?;
        let output = ff_output.add(&residual)?;
        Ok((output, updated_cache))
    }

    pub fn forward_with_decode_cache(
        &self,
        x: &DeviceTensor<B>,
        cache: &DecodeKvCache<B>,
        update_starts: &DeviceTensor<B>,
        query_start: &DeviceTensor<B>,
        cos: &DeviceTensor<B>,
        sin: &DeviceTensor<B>,
    ) -> Result<(DeviceTensor<B>, DecodeKvCache<B>)> {
        let normed = self.norm_1.forward(x)?;
        let qkv = self.attention.proj_qkv.forward(&normed)?;
        let qkv = functional::apply_rope_qkv_packed(
            self.backend.as_ref(),
            &self.attention.config,
            &qkv,
            cos,
            sin,
        )?;
        let functional::DecodeAttentionComputation {
            output: attention_context,
            cache: updated_cache,
        } = functional::attention_decode_cache(
            self.backend.as_ref(),
            &self.attention.config,
            &qkv,
            cache,
            update_starts,
            query_start,
        )?;
        let attn_output = self.attention.proj_out.forward(&attention_context)?;
        let residual = attn_output.add(x)?;
        let normed2 = self.norm_2.forward(&residual)?;
        let ff_output = self.feed_forward.forward(&normed2)?;
        let output = ff_output.add(&residual)?;
        Ok((output, updated_cache))
    }
}

pub struct Ministral<B: PortableBackend + 'static> {
    pub backend: Arc<B>,
    pub config: MinistralConfig,
    pub tok_embeddings: Embedding<B>,
    pub blocks: Vec<MinistralBlock<B>>,
    pub final_norm: RmsNorm<B>,
    pub lm_head: DeviceTensor<B>,
}

impl<B: PortableBackend + 'static> Ministral<B> {
    fn rope_config(&self) -> RopeConfig {
        RopeConfig {
            rotary_dim: self.config.rotary_dim,
            theta: self.config.rope_theta,
            scaling: self.config.rope_scaling,
        }
    }

    fn rope_tables(
        &self,
        position_offset: usize,
        seq_len: usize,
    ) -> Result<(DeviceTensor<B>, DeviceTensor<B>)> {
        let cache =
            functional::rotary_cos_sin_cache_slice(position_offset, seq_len, self.rope_config())?;
        cache.to_device(Arc::clone(&self.backend))
    }

    pub fn random(config: MinistralConfig, backend: Arc<B>, rng: &mut impl Rng) -> Result<Self> {
        ensure!(
            config.embed_dim.is_multiple_of(config.num_heads),
            "embed dim must divide query heads"
        );
        ensure!(
            config.num_heads.is_multiple_of(config.num_kv_heads),
            "query heads must be divisible by kv heads"
        );
        ensure!(
            config.rotary_dim.is_multiple_of(2)
                && config.rotary_dim <= config.embed_dim / config.num_heads,
            "rotary_dim must be even and <= head_dim"
        );

        let weight_std = 0.02;
        let tok_embeddings_weight = DeviceTensor::from_host(
            Arc::clone(&backend),
            Tensor::randn(
                Shape::new([config.vocab_size, config.embed_dim]),
                weight_std,
                rng,
            ),
        )?;
        let tok_embeddings = Embedding::new(Arc::clone(&backend), tok_embeddings_weight)?;

        let mut blocks = Vec::with_capacity(config.num_layers);
        for _ in 0..config.num_layers {
            let attn = CausalSelfAttention::new(
                Arc::clone(&backend),
                AttentionConfig::with_kv(config.embed_dim, config.num_heads, config.num_kv_heads),
                Tensor::randn(
                    Shape::new([
                        config.embed_dim,
                        config.embed_dim
                            + 2 * (config.num_kv_heads * (config.embed_dim / config.num_heads)),
                    ]),
                    weight_std,
                    rng,
                ),
                Tensor::randn(
                    Shape::new([config.embed_dim, config.embed_dim]),
                    weight_std,
                    rng,
                ),
                Option::<Tensor>::None,
                Option::<Tensor>::None,
            )?;

            let ff = GatedFeedForward::new(
                Arc::clone(&backend),
                Tensor::randn(
                    Shape::new([config.embed_dim, config.mlp_hidden_dim]),
                    weight_std,
                    rng,
                ),
                Tensor::randn(
                    Shape::new([config.embed_dim, config.mlp_hidden_dim]),
                    weight_std,
                    rng,
                ),
                Tensor::randn(
                    Shape::new([config.mlp_hidden_dim, config.embed_dim]),
                    weight_std,
                    rng,
                ),
                Option::<Tensor>::None,
                Option::<Tensor>::None,
                Option::<Tensor>::None,
            )?;

            let norm_1 = RmsNorm::new(
                Arc::clone(&backend),
                Tensor::ones(Shape::new([config.embed_dim])),
                config.rms_norm_eps,
            )?;
            let norm_2 = RmsNorm::new(
                Arc::clone(&backend),
                Tensor::ones(Shape::new([config.embed_dim])),
                config.rms_norm_eps,
            )?;

            blocks.push(MinistralBlock {
                backend: Arc::clone(&backend),
                attention: attn,
                feed_forward: ff,
                norm_1,
                norm_2,
            });
        }

        let final_norm = RmsNorm::new(
            Arc::clone(&backend),
            Tensor::ones(Shape::new([config.embed_dim])),
            config.rms_norm_eps,
        )?;
        let lm_head = DeviceTensor::from_host(
            Arc::clone(&backend),
            Tensor::randn(
                Shape::new([config.embed_dim, config.vocab_size]),
                weight_std,
                rng,
            ),
        )?;

        Ok(Self {
            backend,
            config,
            tok_embeddings,
            blocks,
            final_norm,
            lm_head,
        })
    }

    pub fn build_from_params(
        config: MinistralConfig,
        backend: Arc<B>,
        get: &mut dyn FnMut(&str) -> Result<DeviceTensor<B>>,
    ) -> Result<Self> {
        let tok_embeddings = Embedding::new(Arc::clone(&backend), get("tok_embeddings.weight")?)?;

        let mut blocks = Vec::with_capacity(config.num_layers);
        for layer in 0..config.num_layers {
            let prefix = format!("blocks.{}", layer);
            let attn = CausalSelfAttention::new(
                Arc::clone(&backend),
                AttentionConfig::with_kv(config.embed_dim, config.num_heads, config.num_kv_heads),
                get(&format!("{}.attention.w_qkv", prefix))?,
                get(&format!("{}.attention.w_out", prefix))?,
                Option::<DeviceTensor<B>>::None,
                Option::<DeviceTensor<B>>::None,
            )?;
            let ff = GatedFeedForward::new(
                Arc::clone(&backend),
                get(&format!("{}.feed_forward.w_gate", prefix))?,
                get(&format!("{}.feed_forward.w_up", prefix))?,
                get(&format!("{}.feed_forward.w_down", prefix))?,
                Option::<DeviceTensor<B>>::None,
                Option::<DeviceTensor<B>>::None,
                Option::<DeviceTensor<B>>::None,
            )?;
            let norm_1 = RmsNorm::new(
                Arc::clone(&backend),
                get(&format!("{}.norm_1.gamma", prefix))?,
                config.rms_norm_eps,
            )?;
            let norm_2 = RmsNorm::new(
                Arc::clone(&backend),
                get(&format!("{}.norm_2.gamma", prefix))?,
                config.rms_norm_eps,
            )?;
            blocks.push(MinistralBlock {
                backend: Arc::clone(&backend),
                attention: attn,
                feed_forward: ff,
                norm_1,
                norm_2,
            });
        }

        let final_norm = RmsNorm::new(
            Arc::clone(&backend),
            get("final_norm.gamma")?,
            config.rms_norm_eps,
        )?;
        let lm_head = get("lm_head")?;

        Ok(Self {
            backend,
            config,
            tok_embeddings,
            blocks,
            final_norm,
            lm_head,
        })
    }

    pub fn forward(&self, tokens: &[usize]) -> Result<Tensor> {
        self.validate_tokens(tokens)?;

        let token_indices_host = Tensor::from_i32(
            Shape::new([tokens.len()]),
            tokens
                .iter()
                .map(|&idx| {
                    i32::try_from(idx).map_err(|_| anyhow!("token index {} exceeds i32::MAX", idx))
                })
                .collect::<Result<Vec<i32>>>()?,
        )?;
        let token_indices = DeviceTensor::from_host(Arc::clone(&self.backend), token_indices_host)?;
        let mut hidden = self.tok_embeddings.forward(&token_indices)?;
        let (cos, sin) = self.rope_tables(0, tokens.len())?;

        for block in &self.blocks {
            hidden = block.forward(&hidden, &cos, &sin)?;
        }

        let normalized = self.final_norm.forward(&hidden)?;
        let logits = normalized.matmul(&self.lm_head)?;
        logits.to_host()
    }

    pub fn forward_with_cache(
        &self,
        tokens: &[usize],
        position_offset: usize,
        caches: &mut [Option<AttentionCache<B>>],
    ) -> Result<Tensor> {
        self.validate_tokens_with_offset(tokens, position_offset)?;
        ensure!(
            caches.len() == self.blocks.len(),
            "expected {} cache slots (one per layer), got {}",
            self.blocks.len(),
            caches.len()
        );

        let token_indices_host = Tensor::from_i32(
            Shape::new([tokens.len()]),
            tokens
                .iter()
                .map(|&idx| {
                    i32::try_from(idx).map_err(|_| anyhow!("token index {} exceeds i32::MAX", idx))
                })
                .collect::<Result<Vec<i32>>>()?,
        )?;
        let token_indices = DeviceTensor::from_host(Arc::clone(&self.backend), token_indices_host)?;
        let mut hidden = self.tok_embeddings.forward(&token_indices)?;
        let (cos, sin) = self.rope_tables(position_offset, tokens.len())?;

        let mut new_caches: Vec<AttentionCache<B>> = Vec::with_capacity(self.blocks.len());
        for (block, existing_cache) in self.blocks.iter().zip(caches.iter()) {
            let (block_output, updated_cache) =
                block.forward_with_cache(&hidden, existing_cache.as_ref(), &cos, &sin)?;
            new_caches.push(updated_cache);
            hidden = block_output;
        }

        let normalized = self.final_norm.forward(&hidden)?;
        let logits = normalized.matmul(&self.lm_head)?;

        for (slot, new_cache) in caches.iter_mut().zip(new_caches.into_iter()) {
            if let Some(old_cache) = slot.take() {
                if let Some((graph, value)) = old_cache.keys().graph_value() {
                    graph.unexport(value);
                }
                if let Some((graph, value)) = old_cache.values().graph_value() {
                    graph.unexport(value);
                }
            }
            *slot = Some(new_cache);
        }

        logits.to_host()
    }

    pub fn forward_with_decode_cache(
        &self,
        tokens: &[usize],
        position_offset: usize,
        caches: &mut [Option<DecodeKvCache<B>>],
    ) -> Result<Tensor> {
        self.forward_with_decode_cache_internal(tokens, position_offset, caches, None)
    }

    pub fn forward_with_decode_cache_with_capacity(
        &self,
        tokens: &[usize],
        position_offset: usize,
        caches: &mut [Option<DecodeKvCache<B>>],
        capacity: usize,
    ) -> Result<Tensor> {
        self.forward_with_decode_cache_internal(tokens, position_offset, caches, Some(capacity))
    }

    fn forward_with_decode_cache_internal(
        &self,
        tokens: &[usize],
        position_offset: usize,
        caches: &mut [Option<DecodeKvCache<B>>],
        fixed_capacity: Option<usize>,
    ) -> Result<Tensor> {
        self.validate_tokens_with_offset(tokens, position_offset)?;
        ensure!(
            caches.len() == self.blocks.len(),
            "expected {} cache slots (one per layer), got {}",
            self.blocks.len(),
            caches.len()
        );

        let total = position_offset
            .checked_add(tokens.len())
            .ok_or_else(|| anyhow!("token position offset overflow"))?;
        let required_capacity = match fixed_capacity {
            Some(capacity) => {
                ensure!(capacity > 0, "decode cache capacity must be > 0");
                ensure!(
                    capacity <= self.config.context_length,
                    "decode cache capacity {} exceeds model context length {}",
                    capacity,
                    self.config.context_length
                );
                ensure!(
                    capacity >= total,
                    "decode cache capacity {} is too small for required sequence length {}",
                    capacity,
                    total
                );
                capacity
            }
            None => decode_cache_capacity(total, self.config.context_length),
        };

        let start_zeros_host = Tensor::from_i32(Shape::new([3]), vec![0, 0, 0])?;
        let start_zeros = DeviceTensor::from_host(Arc::clone(&self.backend), start_zeros_host)?;

        if caches
            .iter()
            .filter_map(|c| c.as_ref().map(|cache| cache.capacity()))
            .next()
            .is_some_and(|cap| cap < required_capacity)
        {
            for slot in caches.iter_mut() {
                let Some(old) = slot.as_ref() else {
                    continue;
                };

                let old_dims = old.keys().shape().dims();
                let shape = Shape::new([old_dims[0], required_capacity, old_dims[2]]);
                let new_keys = DeviceTensor::zeros(Arc::clone(&self.backend), shape.clone())?;
                let new_values = DeviceTensor::zeros(Arc::clone(&self.backend), shape)?;

                let keys = functional::dynamic_update_slice_into(
                    self.backend.as_ref(),
                    &new_keys,
                    old.keys(),
                    &start_zeros,
                )?;
                let values = functional::dynamic_update_slice_into(
                    self.backend.as_ref(),
                    &new_values,
                    old.values(),
                    &start_zeros,
                )?;
                *slot = Some(DecodeKvCache::new(keys, values, old.len())?);
            }
        }

        let token_indices_host = Tensor::from_i32(
            Shape::new([tokens.len()]),
            tokens
                .iter()
                .map(|&idx| {
                    i32::try_from(idx).map_err(|_| anyhow!("token index {} exceeds i32::MAX", idx))
                })
                .collect::<Result<Vec<i32>>>()?,
        )?;
        let token_indices = DeviceTensor::from_host(Arc::clone(&self.backend), token_indices_host)?;
        let mut hidden = self.tok_embeddings.forward(&token_indices)?;
        let (cos, sin) = self.rope_tables(position_offset, tokens.len())?;

        let mut new_caches: Vec<DecodeKvCache<B>> = Vec::with_capacity(self.blocks.len());
        let pos_i32 = i32::try_from(position_offset)
            .map_err(|_| anyhow!("position offset {} exceeds i32::MAX", position_offset))?;
        let update_starts_host = Tensor::from_i32(Shape::new([3]), vec![0, pos_i32, 0])?;
        let update_starts = DeviceTensor::from_host(Arc::clone(&self.backend), update_starts_host)?;
        let query_start_host = Tensor::from_i32(Shape::new([1]), vec![pos_i32])?;
        let query_start = DeviceTensor::from_host(Arc::clone(&self.backend), query_start_host)?;

        for (block, slot) in self.blocks.iter().zip(caches.iter()) {
            if let Some(existing) = slot.as_ref() {
                ensure!(
                    existing.len() == position_offset,
                    "decode cache length {} must match position offset {}",
                    existing.len(),
                    position_offset
                );
                let (block_output, updated_cache) = block.forward_with_decode_cache(
                    &hidden,
                    existing,
                    &update_starts,
                    &query_start,
                    &cos,
                    &sin,
                )?;
                new_caches.push(updated_cache);
                hidden = block_output;
            } else {
                let (block_output, updated_cache) =
                    block.forward_with_cache(&hidden, None, &cos, &sin)?;
                let present_dims = updated_cache.keys().shape().dims();
                let shape = Shape::new([present_dims[0], required_capacity, present_dims[2]]);
                let keys_base = DeviceTensor::zeros(Arc::clone(&self.backend), shape.clone())?;
                let values_base = DeviceTensor::zeros(Arc::clone(&self.backend), shape)?;

                let keys = functional::dynamic_update_slice_into(
                    self.backend.as_ref(),
                    &keys_base,
                    updated_cache.keys(),
                    &start_zeros,
                )?;
                let values = functional::dynamic_update_slice_into(
                    self.backend.as_ref(),
                    &values_base,
                    updated_cache.values(),
                    &start_zeros,
                )?;
                new_caches.push(DecodeKvCache::new(keys, values, total)?);
                hidden = block_output;
            }
        }

        let normalized = self.final_norm.forward(&hidden)?;
        let logits = normalized.matmul(&self.lm_head)?;

        for (slot, new_cache) in caches.iter_mut().zip(new_caches.into_iter()) {
            if let Some(old_cache) = slot.take() {
                if let Some((graph, value)) = old_cache.keys().graph_value() {
                    graph.unexport(value);
                }
                if let Some((graph, value)) = old_cache.values().graph_value() {
                    graph.unexport(value);
                }
            }
            *slot = Some(new_cache);
        }

        logits.to_host()
    }

    fn validate_tokens(&self, tokens: &[usize]) -> Result<()> {
        ensure!(
            !tokens.is_empty(),
            "token sequence must contain at least one element"
        );
        ensure!(
            tokens.len() <= self.config.context_length,
            "token sequence length {} exceeds model context length {}",
            tokens.len(),
            self.config.context_length
        );
        for (position, &token) in tokens.iter().enumerate() {
            ensure!(
                token < self.config.vocab_size,
                "token id {} at position {} exceeds vocabulary size {}",
                token,
                position,
                self.config.vocab_size
            );
        }
        Ok(())
    }

    fn validate_tokens_with_offset(&self, tokens: &[usize], position_offset: usize) -> Result<()> {
        ensure!(
            !tokens.is_empty(),
            "token sequence must contain at least one element"
        );
        let total = position_offset
            .checked_add(tokens.len())
            .ok_or_else(|| anyhow!("token position offset overflow"))?;
        ensure!(
            total <= self.config.context_length,
            "token sequence length {} with offset {} exceeds model context length {}",
            tokens.len(),
            position_offset,
            self.config.context_length
        );
        for (i, &token) in tokens.iter().enumerate() {
            ensure!(
                token < self.config.vocab_size,
                "token id {} at slice position {} exceeds vocabulary size {}",
                token,
                i,
                self.config.vocab_size
            );
        }
        Ok(())
    }
}

fn decode_cache_capacity(required_len: usize, max_len: usize) -> usize {
    let bucket = required_len.next_power_of_two().max(1);
    bucket.min(max_len)
}

impl<B: PortableBackend + 'static> crate::inference::CausalLanguageModel<B> for Ministral<B> {
    fn context_length(&self) -> usize {
        self.config.context_length
    }

    fn num_layers(&self) -> usize {
        self.blocks.len()
    }

    fn forward(&self, tokens: &[usize]) -> Result<Tensor> {
        Ministral::forward(self, tokens)
    }

    fn forward_with_decode_cache(
        &self,
        tokens: &[usize],
        position_offset: usize,
        caches: &mut [Option<DecodeKvCache<B>>],
    ) -> Result<Tensor> {
        Ministral::forward_with_decode_cache(self, tokens, position_offset, caches)
    }

    fn forward_with_decode_cache_with_capacity(
        &self,
        tokens: &[usize],
        position_offset: usize,
        caches: &mut [Option<DecodeKvCache<B>>],
        capacity: usize,
    ) -> Result<Tensor> {
        Ministral::forward_with_decode_cache_with_capacity(
            self,
            tokens,
            position_offset,
            caches,
            capacity,
        )
    }
}

impl<B: PortableBackend + 'static> crate::runtime::LoadedModel<B> for Ministral<B> {
    fn kind(&self) -> &str {
        KIND
    }

    fn forward(
        &mut self,
        input: crate::runtime::ModelInput<B>,
    ) -> Result<crate::runtime::ModelOutput> {
        match input {
            crate::runtime::ModelInput::Tokens(tokens) => Ok(crate::runtime::ModelOutput::Tensor(
                Ministral::forward(self, &tokens)?,
            )),
            crate::runtime::ModelInput::Vision(_) => {
                bail!("model '{KIND}' expects token input, got vision input")
            }
        }
    }

    fn as_causal_lm(&self) -> Option<&dyn crate::inference::CausalLanguageModel<B>> {
        Some(self)
    }
}
