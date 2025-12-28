use super::GptConfig;
use crate::backend::spec::PortableBackend;
use crate::module::{Module, ParamVisitor, ParamVisitorMut, TensorRole};
use crate::nn::{
    AttentionConfig, CausalSelfAttention, CausalSelfAttentionGradients, CausalSelfAttentionState,
    Embedding, EmbeddingState, FeedForward, FeedForwardGradients, FeedForwardState, LayerNorm,
    LayerNormGradients, LayerNormState,
};
use crate::ops::functional::{
    self, build_registry, AttentionCache, DecodeKvCache, FunctionalRegistryHandle,
};
use crate::tensor::{DeviceTensor, DeviceTensorOps, Shape, Tensor};
use anyhow::{anyhow, bail, ensure, Result};
use rand::Rng;
use std::collections::HashMap;
use std::convert::TryFrom;
use std::sync::Arc;

pub struct GptBlock<B: PortableBackend + 'static> {
    pub backend: Arc<B>,
    pub attention: CausalSelfAttention<B>,
    pub feed_forward: FeedForward<B>,
    pub ln_1: LayerNorm<B>,
    pub ln_2: LayerNorm<B>,
}

pub struct GptBlockState<B: PortableBackend + 'static> {
    pub input: DeviceTensor<B>,
    pub ln1: LayerNormState<B>,
    pub attn: CausalSelfAttentionState<B>,
    pub ln2: LayerNormState<B>,
    pub ff: FeedForwardState<B>,
}

pub struct GptBlockGradients {
    pub attention: CausalSelfAttentionGradients,
    pub feed_forward: FeedForwardGradients,
    pub ln1: LayerNormGradients,
    pub ln2: LayerNormGradients,
}

impl<B: PortableBackend + 'static> GptBlock<B> {
    pub fn forward(&self, x: &DeviceTensor<B>) -> Result<DeviceTensor<B>> {
        let normed = self.ln_1.forward(x)?;
        let attn_output = self.attention.forward(&normed)?;
        let residual = attn_output.add(x)?;
        let normed2 = self.ln_2.forward(&residual)?;
        let ff_output = self.feed_forward.forward(&normed2)?;
        ff_output.add(&residual)
    }

    pub fn forward_with_cache(
        &self,
        x: &DeviceTensor<B>,
        cache: Option<&AttentionCache<B>>,
    ) -> Result<(DeviceTensor<B>, AttentionCache<B>)> {
        let normed = self.ln_1.forward(x)?;
        let (attn_output, attn_state) = self.attention.forward_with_cache(&normed, cache)?;
        let residual = attn_output.add(x)?;
        let normed2 = self.ln_2.forward(&residual)?;
        let ff_output = self.feed_forward.forward(&normed2)?;
        let output = ff_output.add(&residual)?;
        Ok((output, attn_state.cache))
    }

    pub fn forward_with_decode_cache(
        &self,
        x: &DeviceTensor<B>,
        cache: &DecodeKvCache<B>,
        update_starts: &DeviceTensor<B>,
        query_start: &DeviceTensor<B>,
    ) -> Result<(DeviceTensor<B>, DecodeKvCache<B>)> {
        let normed = self.ln_1.forward(x)?;
        let (attn_output, updated_cache) =
            self.attention
                .forward_with_decode_cache(&normed, cache, update_starts, query_start)?;
        let residual = attn_output.add(x)?;
        let normed2 = self.ln_2.forward(&residual)?;
        let ff_output = self.feed_forward.forward(&normed2)?;
        let output = ff_output.add(&residual)?;
        Ok((output, updated_cache))
    }

    pub fn forward_with_state(
        &self,
        x: &DeviceTensor<B>,
    ) -> Result<(DeviceTensor<B>, GptBlockState<B>)> {
        let (normed, ln1_state) = self.ln_1.forward_with_state(x)?;
        let (attn_output, attn_state) = self.attention.forward_with_state(&normed)?;
        let residual = attn_output.add(x)?;
        let (normed2, ln2_state) = self.ln_2.forward_with_state(&residual)?;
        let (ff_output, ff_state) = self.feed_forward.forward_with_state(&normed2)?;
        let output = ff_output.add(&residual)?;

        Ok((
            output,
            GptBlockState {
                input: x.clone(),
                ln1: ln1_state,
                attn: attn_state,
                ln2: ln2_state,
                ff: ff_state,
            },
        ))
    }

    pub fn backward(
        &self,
        state: &GptBlockState<B>,
        grad_output: &DeviceTensor<B>,
    ) -> Result<(GptBlockGradients, DeviceTensor<B>)> {
        let _ = (state, grad_output);
        bail!("GptBlock backward is not available on the portable backend yet")
    }
}

impl<B: PortableBackend + 'static> Module<B> for GptBlock<B> {
    fn visit_params(&self, v: &mut ParamVisitor<'_, B>) -> Result<()> {
        v.scoped("attention", |v| self.attention.visit_params(v))?;
        v.scoped("feed_forward", |v| self.feed_forward.visit_params(v))?;
        v.scoped("ln_1", |v| self.ln_1.visit_params(v))?;
        v.scoped("ln_2", |v| self.ln_2.visit_params(v))?;
        Ok(())
    }

    fn visit_params_mut(&mut self, v: &mut ParamVisitorMut<'_, B>) -> Result<()> {
        v.scoped("attention", |v| self.attention.visit_params_mut(v))?;
        v.scoped("feed_forward", |v| self.feed_forward.visit_params_mut(v))?;
        v.scoped("ln_1", |v| self.ln_1.visit_params_mut(v))?;
        v.scoped("ln_2", |v| self.ln_2.visit_params_mut(v))?;
        Ok(())
    }
}
pub struct Gpt<B: PortableBackend + 'static> {
    pub backend: Arc<B>,
    functional: FunctionalRegistryHandle<B>,
    pub config: GptConfig,
    pub tok_embeddings: Embedding<B>,
    pub pos_embeddings: DeviceTensor<B>,
    pub blocks: Vec<GptBlock<B>>,
    pub final_ln: LayerNorm<B>,
    pub lm_head: DeviceTensor<B>,
}

pub struct GptForwardState<B: PortableBackend + 'static> {
    pub token_state: EmbeddingState<B>,
    pub embedding_output: DeviceTensor<B>,
    pub block_states: Vec<GptBlockState<B>>,
    pub post_blocks: DeviceTensor<B>,
    pub final_ln_output: DeviceTensor<B>,
    pub final_ln_state: LayerNormState<B>,
}

impl<B: PortableBackend + 'static> Gpt<B> {
    pub fn random(config: GptConfig, backend: Arc<B>, rng: &mut impl Rng) -> Result<Self> {
        let functional: FunctionalRegistryHandle<B> = build_registry(&config.functional_overrides);
        let embed_dim = config.embed_dim;
        let hidden_dim = embed_dim * config.mlp_ratio;
        let weight_std = 0.02;

        let tok_embeddings_weight = DeviceTensor::from_host(
            Arc::clone(&backend),
            Tensor::randn(Shape::new([config.vocab_size, embed_dim]), weight_std, rng)
                .requires_grad(true),
        )?;
        let tok_embeddings = Embedding::new(Arc::clone(&backend), tok_embeddings_weight)?;
        let pos_embeddings = DeviceTensor::from_host(
            Arc::clone(&backend),
            Tensor::randn(
                Shape::new([config.context_length, embed_dim]),
                weight_std,
                rng,
            )
            .requires_grad(true),
        )?;

        let mut blocks = Vec::with_capacity(config.num_layers);
        for _ in 0..config.num_layers {
            let attn = CausalSelfAttention::new(
                Arc::clone(&backend),
                AttentionConfig::with_equal_heads(embed_dim, config.num_heads),
                Tensor::randn(Shape::new([embed_dim, 3 * embed_dim]), weight_std, rng)
                    .requires_grad(true),
                Tensor::randn(Shape::new([embed_dim, embed_dim]), weight_std, rng)
                    .requires_grad(true),
                Some(Tensor::zeros(Shape::new([3 * embed_dim])).requires_grad(true)),
                Some(Tensor::zeros(Shape::new([embed_dim])).requires_grad(true)),
            )?;

            let ff = FeedForward::new(
                Arc::clone(&backend),
                Tensor::randn(Shape::new([embed_dim, hidden_dim]), weight_std, rng)
                    .requires_grad(true),
                Tensor::randn(Shape::new([hidden_dim, embed_dim]), weight_std, rng)
                    .requires_grad(true),
                Some(Tensor::zeros(Shape::new([hidden_dim])).requires_grad(true)),
                Some(Tensor::zeros(Shape::new([embed_dim])).requires_grad(true)),
            )?;

            let ln_1 = LayerNorm::new(
                Arc::clone(&backend),
                Tensor::ones(Shape::new([embed_dim])).requires_grad(true),
                Tensor::zeros(Shape::new([embed_dim])).requires_grad(true),
                1e-5,
            )?;
            let ln_2 = LayerNorm::new(
                Arc::clone(&backend),
                Tensor::ones(Shape::new([embed_dim])).requires_grad(true),
                Tensor::zeros(Shape::new([embed_dim])).requires_grad(true),
                1e-5,
            )?;

            blocks.push(GptBlock {
                backend: Arc::clone(&backend),
                attention: attn,
                feed_forward: ff,
                ln_1,
                ln_2,
            });
        }

        let final_ln = LayerNorm::new(
            Arc::clone(&backend),
            Tensor::ones(Shape::new([embed_dim])).requires_grad(true),
            Tensor::zeros(Shape::new([embed_dim])).requires_grad(true),
            1e-5,
        )?;
        let lm_head = DeviceTensor::from_host(
            Arc::clone(&backend),
            Tensor::randn(Shape::new([embed_dim, config.vocab_size]), weight_std, rng)
                .requires_grad(true),
        )?;

        Ok(Gpt {
            backend,
            functional,
            config,
            tok_embeddings,
            pos_embeddings,
            blocks,
            final_ln,
            lm_head,
        })
    }

    pub fn from_named_tensors(
        config: GptConfig,
        backend: Arc<B>,
        mut tensors: HashMap<String, Tensor>,
    ) -> Result<Self> {
        fn take_tensor(map: &mut HashMap<String, Tensor>, name: &str) -> Result<Tensor> {
            map.remove(name)
                .ok_or_else(|| anyhow!("missing tensor {} in checkpoint", name))
        }

        let tok_embeddings_weight = DeviceTensor::from_host(
            Arc::clone(&backend),
            take_tensor(&mut tensors, "tok_embeddings.weight")?,
        )?;
        let tok_embeddings = Embedding::new(Arc::clone(&backend), tok_embeddings_weight)?;
        let pos_embeddings = DeviceTensor::from_host(
            Arc::clone(&backend),
            take_tensor(&mut tensors, "pos_embeddings")?,
        )?;

        let mut blocks = Vec::with_capacity(config.num_layers);
        let functional: FunctionalRegistryHandle<B> = build_registry(&config.functional_overrides);
        for layer in 0..config.num_layers {
            let prefix = format!("blocks.{}", layer);
            let attn = CausalSelfAttention::new(
                Arc::clone(&backend),
                AttentionConfig::with_equal_heads(config.embed_dim, config.num_heads),
                take_tensor(&mut tensors, &format!("{}.attention.w_qkv", prefix))?,
                take_tensor(&mut tensors, &format!("{}.attention.w_out", prefix))?,
                Some(take_tensor(
                    &mut tensors,
                    &format!("{}.attention.b_qkv", prefix),
                )?),
                Some(take_tensor(
                    &mut tensors,
                    &format!("{}.attention.b_out", prefix),
                )?),
            )?;

            let ff = FeedForward::new(
                Arc::clone(&backend),
                take_tensor(&mut tensors, &format!("{}.feed_forward.w_in", prefix))?,
                take_tensor(&mut tensors, &format!("{}.feed_forward.w_out", prefix))?,
                Some(take_tensor(
                    &mut tensors,
                    &format!("{}.feed_forward.b_in", prefix),
                )?),
                Some(take_tensor(
                    &mut tensors,
                    &format!("{}.feed_forward.b_out", prefix),
                )?),
            )?;

            let ln_1 = LayerNorm::new(
                Arc::clone(&backend),
                take_tensor(&mut tensors, &format!("{}.ln_1.gamma", prefix))?,
                take_tensor(&mut tensors, &format!("{}.ln_1.beta", prefix))?,
                1e-5,
            )?;
            let ln_2 = LayerNorm::new(
                Arc::clone(&backend),
                take_tensor(&mut tensors, &format!("{}.ln_2.gamma", prefix))?,
                take_tensor(&mut tensors, &format!("{}.ln_2.beta", prefix))?,
                1e-5,
            )?;

            blocks.push(GptBlock {
                backend: Arc::clone(&backend),
                attention: attn,
                feed_forward: ff,
                ln_1,
                ln_2,
            });
        }

        let final_ln = LayerNorm::new(
            Arc::clone(&backend),
            take_tensor(&mut tensors, "final_ln.gamma")?,
            take_tensor(&mut tensors, "final_ln.beta")?,
            1e-5,
        )?;
        let lm_head =
            DeviceTensor::from_host(Arc::clone(&backend), take_tensor(&mut tensors, "lm_head")?)?;

        if !tensors.is_empty() {
            let extras: Vec<String> = tensors.keys().cloned().collect();
            bail!("checkpoint contained unexpected tensors: {:?}", extras);
        }

        Ok(Gpt {
            backend,
            functional,
            config,
            tok_embeddings,
            pos_embeddings,
            blocks,
            final_ln,
            lm_head,
        })
    }

    pub fn build_from_params(
        config: GptConfig,
        backend: Arc<B>,
        mut get: impl FnMut(&str) -> Result<DeviceTensor<B>>,
    ) -> Result<Self> {
        let functional: FunctionalRegistryHandle<B> = build_registry(&config.functional_overrides);

        let tok_embeddings_weight = get("tok_embeddings.weight")?;
        let tok_embeddings = Embedding::new(Arc::clone(&backend), tok_embeddings_weight)?;
        let pos_embeddings = get("pos_embeddings")?;

        let mut blocks = Vec::with_capacity(config.num_layers);
        for layer in 0..config.num_layers {
            let prefix = format!("blocks.{}", layer);

            let attn = CausalSelfAttention::new(
                Arc::clone(&backend),
                AttentionConfig::with_equal_heads(config.embed_dim, config.num_heads),
                get(&format!("{}.attention.w_qkv", prefix))?,
                get(&format!("{}.attention.w_out", prefix))?,
                Some(get(&format!("{}.attention.b_qkv", prefix))?),
                Some(get(&format!("{}.attention.b_out", prefix))?),
            )?;

            let ff = FeedForward::new(
                Arc::clone(&backend),
                get(&format!("{}.feed_forward.w_in", prefix))?,
                get(&format!("{}.feed_forward.w_out", prefix))?,
                Some(get(&format!("{}.feed_forward.b_in", prefix))?),
                Some(get(&format!("{}.feed_forward.b_out", prefix))?),
            )?;

            let ln_1 = LayerNorm::new(
                Arc::clone(&backend),
                get(&format!("{}.ln_1.gamma", prefix))?,
                get(&format!("{}.ln_1.beta", prefix))?,
                1e-5,
            )?;
            let ln_2 = LayerNorm::new(
                Arc::clone(&backend),
                get(&format!("{}.ln_2.gamma", prefix))?,
                get(&format!("{}.ln_2.beta", prefix))?,
                1e-5,
            )?;

            blocks.push(GptBlock {
                backend: Arc::clone(&backend),
                attention: attn,
                feed_forward: ff,
                ln_1,
                ln_2,
            });
        }

        let final_ln = LayerNorm::new(
            Arc::clone(&backend),
            get("final_ln.gamma")?,
            get("final_ln.beta")?,
            1e-5,
        )?;
        let lm_head = get("lm_head")?;

        Ok(Gpt {
            backend,
            functional,
            config,
            tok_embeddings,
            pos_embeddings,
            blocks,
            final_ln,
            lm_head,
        })
    }

    pub fn forward_with_state(&self, tokens: &[usize]) -> Result<(Tensor, GptForwardState<B>)> {
        crate::ops::functional::with_registry(self.functional.clone(), || {
            self.validate_tokens(tokens)?;

            let token_indices_host = Tensor::from_i32(
                Shape::new([tokens.len()]),
                tokens
                    .iter()
                    .map(|&idx| {
                        i32::try_from(idx)
                            .map_err(|_| anyhow!("token index {} exceeds i32::MAX", idx))
                    })
                    .collect::<Result<Vec<i32>>>()?,
            )?;
            let token_indices =
                DeviceTensor::from_host(Arc::clone(&self.backend), token_indices_host)?;
            let (token_embeddings, token_state) =
                self.tok_embeddings.forward_with_state(&token_indices)?;
            let positions: Vec<usize> = (0..tokens.len()).collect();
            let position_indices_host = Tensor::from_i32(
                Shape::new([positions.len()]),
                positions.iter().map(|&idx| idx as i32).collect(),
            )?;
            let position_indices =
                DeviceTensor::from_host(Arc::clone(&self.backend), position_indices_host)?;
            let position_embeddings = functional::embedding_lookup_with_graph(
                self.backend.as_ref(),
                &self.pos_embeddings,
                &position_indices,
                token_embeddings.graph(),
            )?;

            let embedding_output = token_embeddings.add(&position_embeddings)?;
            let mut hidden = embedding_output.clone();
            let mut block_states = Vec::with_capacity(self.blocks.len());
            for block in &self.blocks {
                let (block_output, block_state) = block.forward_with_state(&hidden)?;
                block_states.push(block_state);
                hidden = block_output;
            }

            let post_blocks = hidden.clone();
            let (final_ln_output, final_ln_state) = self.final_ln.forward_with_state(&hidden)?;
            let logits_device = final_ln_output.matmul(&self.lm_head)?;
            let logits = logits_device.to_host()?;

            let state = GptForwardState {
                token_state,
                embedding_output,
                block_states,
                post_blocks,
                final_ln_output,
                final_ln_state,
            };

            Ok((logits, state))
        })
    }

    pub fn forward_hidden(&self, tokens: &[usize]) -> Result<Tensor> {
        crate::ops::functional::with_registry(self.functional.clone(), || {
            self.validate_tokens(tokens)?;

            let token_indices_host = Tensor::from_i32(
                Shape::new([tokens.len()]),
                tokens
                    .iter()
                    .map(|&idx| {
                        i32::try_from(idx)
                            .map_err(|_| anyhow!("token index {} exceeds i32::MAX", idx))
                    })
                    .collect::<Result<Vec<i32>>>()?,
            )?;
            let token_indices =
                DeviceTensor::from_host(Arc::clone(&self.backend), token_indices_host)?;
            let token_embeddings = self.tok_embeddings.forward(&token_indices)?;
            let positions: Vec<usize> = (0..tokens.len()).collect();
            let position_indices_host = Tensor::from_i32(
                Shape::new([positions.len()]),
                positions.iter().map(|&idx| idx as i32).collect(),
            )?;
            let position_indices =
                DeviceTensor::from_host(Arc::clone(&self.backend), position_indices_host)?;
            let position_embeddings = functional::embedding_lookup_with_graph(
                self.backend.as_ref(),
                &self.pos_embeddings,
                &position_indices,
                token_embeddings.graph(),
            )?;

            let mut hidden = token_embeddings.add(&position_embeddings)?;
            for block in &self.blocks {
                hidden = block.forward(&hidden)?;
            }

            let normalized = self.final_ln.forward(&hidden)?;
            normalized.to_host()
        })
    }

    pub fn forward(&self, tokens: &[usize]) -> Result<Tensor> {
        crate::ops::functional::with_registry(self.functional.clone(), || {
            self.validate_tokens(tokens)?;

            let token_indices_host = Tensor::from_i32(
                Shape::new([tokens.len()]),
                tokens
                    .iter()
                    .map(|&idx| {
                        i32::try_from(idx)
                            .map_err(|_| anyhow!("token index {} exceeds i32::MAX", idx))
                    })
                    .collect::<Result<Vec<i32>>>()?,
            )?;
            let token_indices =
                DeviceTensor::from_host(Arc::clone(&self.backend), token_indices_host)?;
            let token_embeddings = self.tok_embeddings.forward(&token_indices)?;
            let positions: Vec<usize> = (0..tokens.len()).collect();
            let position_indices_host = Tensor::from_i32(
                Shape::new([positions.len()]),
                positions.iter().map(|&idx| idx as i32).collect(),
            )?;
            let position_indices =
                DeviceTensor::from_host(Arc::clone(&self.backend), position_indices_host)?;
            let position_embeddings = functional::embedding_lookup_with_graph(
                self.backend.as_ref(),
                &self.pos_embeddings,
                &position_indices,
                token_embeddings.graph(),
            )?;

            let mut hidden = token_embeddings.add(&position_embeddings)?;
            for block in &self.blocks {
                hidden = block.forward(&hidden)?;
            }

            let normalized = self.final_ln.forward(&hidden)?;
            let logits = normalized.matmul(&self.lm_head)?;
            logits.to_host()
        })
    }

    pub fn forward_with_cache(
        &self,
        tokens: &[usize],
        position_offset: usize,
        caches: &mut [Option<AttentionCache<B>>],
    ) -> Result<Tensor> {
        crate::ops::functional::with_registry(self.functional.clone(), || {
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
                        i32::try_from(idx)
                            .map_err(|_| anyhow!("token index {} exceeds i32::MAX", idx))
                    })
                    .collect::<Result<Vec<i32>>>()?,
            )?;
            let token_indices =
                DeviceTensor::from_host(Arc::clone(&self.backend), token_indices_host)?;
            let token_embeddings = self.tok_embeddings.forward(&token_indices)?;
            let positions: Vec<usize> = (0..tokens.len()).map(|i| position_offset + i).collect();
            let position_indices_host = Tensor::from_i32(
                Shape::new([positions.len()]),
                positions.iter().map(|&idx| idx as i32).collect(),
            )?;
            let position_indices =
                DeviceTensor::from_host(Arc::clone(&self.backend), position_indices_host)?;
            let position_embeddings = functional::embedding_lookup_with_graph(
                self.backend.as_ref(),
                &self.pos_embeddings,
                &position_indices,
                token_embeddings.graph(),
            )?;

            let mut hidden = token_embeddings.add(&position_embeddings)?;
            let mut new_caches: Vec<AttentionCache<B>> = Vec::with_capacity(self.blocks.len());
            for (block, existing_cache) in self.blocks.iter().zip(caches.iter()) {
                let (block_output, updated_cache) =
                    block.forward_with_cache(&hidden, existing_cache.as_ref())?;
                new_caches.push(updated_cache);
                hidden = block_output;
            }

            let normalized = self.final_ln.forward(&hidden)?;
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
        })
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
        crate::ops::functional::with_registry(self.functional.clone(), || {
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

            // If we already have caches and the bucket grew, upsize all layers in one pass.
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
                        i32::try_from(idx)
                            .map_err(|_| anyhow!("token index {} exceeds i32::MAX", idx))
                    })
                    .collect::<Result<Vec<i32>>>()?,
            )?;
            let token_indices =
                DeviceTensor::from_host(Arc::clone(&self.backend), token_indices_host)?;
            let token_embeddings = self.tok_embeddings.forward(&token_indices)?;

            let positions: Vec<usize> = (0..tokens.len()).map(|i| position_offset + i).collect();
            let position_indices_host = Tensor::from_i32(
                Shape::new([positions.len()]),
                positions.iter().map(|&idx| idx as i32).collect(),
            )?;
            let position_indices =
                DeviceTensor::from_host(Arc::clone(&self.backend), position_indices_host)?;
            let position_embeddings = functional::embedding_lookup_with_graph(
                self.backend.as_ref(),
                &self.pos_embeddings,
                &position_indices,
                token_embeddings.graph(),
            )?;

            let mut hidden = token_embeddings.add(&position_embeddings)?;
            let mut new_caches: Vec<DecodeKvCache<B>> = Vec::with_capacity(self.blocks.len());

            let pos_i32 = i32::try_from(position_offset)
                .map_err(|_| anyhow!("position offset {} exceeds i32::MAX", position_offset))?;
            let update_starts_host = Tensor::from_i32(Shape::new([3]), vec![0, pos_i32, 0])?;
            let update_starts =
                DeviceTensor::from_host(Arc::clone(&self.backend), update_starts_host)?;
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
                    )?;
                    new_caches.push(updated_cache);
                    hidden = block_output;
                } else {
                    let (block_output, updated_cache) = block.forward_with_cache(&hidden, None)?;
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

            let normalized = self.final_ln.forward(&hidden)?;
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
        })
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

    pub fn for_each_parameter<F>(&self, mut f: F) -> Result<()>
    where
        F: FnMut(&str, &DeviceTensor<B>) -> Result<()>,
    {
        f("tok_embeddings.weight", &self.tok_embeddings.weight)?;
        f("pos_embeddings", &self.pos_embeddings)?;
        for (i, block) in self.blocks.iter().enumerate() {
            let prefix = format!("blocks.{}", i);
            f(
                &format!("{}.attention.w_qkv", prefix),
                &block.attention.proj_qkv.weight,
            )?;
            if let Some(bias) = &block.attention.proj_qkv.bias {
                f(&format!("{}.attention.b_qkv", prefix), bias)?;
            }
            f(
                &format!("{}.attention.w_out", prefix),
                &block.attention.proj_out.weight,
            )?;
            if let Some(bias) = &block.attention.proj_out.bias {
                f(&format!("{}.attention.b_out", prefix), bias)?;
            }
            f(
                &format!("{}.feed_forward.w_in", prefix),
                &block.feed_forward.w_in.weight,
            )?;
            if let Some(bias) = &block.feed_forward.w_in.bias {
                f(&format!("{}.feed_forward.b_in", prefix), bias)?;
            }
            f(
                &format!("{}.feed_forward.w_out", prefix),
                &block.feed_forward.w_out.weight,
            )?;
            if let Some(bias) = &block.feed_forward.w_out.bias {
                f(&format!("{}.feed_forward.b_out", prefix), bias)?;
            }
            f(&format!("{}.ln_1.gamma", prefix), &block.ln_1.gamma)?;
            f(&format!("{}.ln_1.beta", prefix), &block.ln_1.beta)?;
            f(&format!("{}.ln_2.gamma", prefix), &block.ln_2.gamma)?;
            f(&format!("{}.ln_2.beta", prefix), &block.ln_2.beta)?;
        }
        f("final_ln.gamma", &self.final_ln.gamma)?;
        f("final_ln.beta", &self.final_ln.beta)?;
        f("lm_head", &self.lm_head)?;
        Ok(())
    }

    pub fn for_each_parameter_mut<F>(&mut self, mut f: F) -> Result<()>
    where
        F: FnMut(&str, &mut DeviceTensor<B>) -> Result<()>,
    {
        f("tok_embeddings.weight", &mut self.tok_embeddings.weight)?;
        f("pos_embeddings", &mut self.pos_embeddings)?;
        for (i, block) in self.blocks.iter_mut().enumerate() {
            let prefix = format!("blocks.{}", i);
            f(
                &format!("{}.attention.w_qkv", prefix),
                &mut block.attention.proj_qkv.weight,
            )?;
            if let Some(bias) = &mut block.attention.proj_qkv.bias {
                f(&format!("{}.attention.b_qkv", prefix), bias)?;
            }
            f(
                &format!("{}.attention.w_out", prefix),
                &mut block.attention.proj_out.weight,
            )?;
            if let Some(bias) = &mut block.attention.proj_out.bias {
                f(&format!("{}.attention.b_out", prefix), bias)?;
            }
            f(
                &format!("{}.feed_forward.w_in", prefix),
                &mut block.feed_forward.w_in.weight,
            )?;
            if let Some(bias) = &mut block.feed_forward.w_in.bias {
                f(&format!("{}.feed_forward.b_in", prefix), bias)?;
            }
            f(
                &format!("{}.feed_forward.w_out", prefix),
                &mut block.feed_forward.w_out.weight,
            )?;
            if let Some(bias) = &mut block.feed_forward.w_out.bias {
                f(&format!("{}.feed_forward.b_out", prefix), bias)?;
            }
            f(&format!("{}.ln_1.gamma", prefix), &mut block.ln_1.gamma)?;
            f(&format!("{}.ln_1.beta", prefix), &mut block.ln_1.beta)?;
            f(&format!("{}.ln_2.gamma", prefix), &mut block.ln_2.gamma)?;
            f(&format!("{}.ln_2.beta", prefix), &mut block.ln_2.beta)?;
        }
        f("final_ln.gamma", &mut self.final_ln.gamma)?;
        f("final_ln.beta", &mut self.final_ln.beta)?;
        f("lm_head", &mut self.lm_head)?;
        Ok(())
    }
}

impl<B: PortableBackend + 'static> Module<B> for Gpt<B> {
    fn visit_params(&self, v: &mut ParamVisitor<'_, B>) -> Result<()> {
        v.scoped("tok_embeddings", |v| self.tok_embeddings.visit_params(v))?;
        v.param(
            "pos_embeddings",
            TensorRole::Parameter,
            &self.pos_embeddings,
        )?;
        v.scoped("blocks", |v| {
            for (i, block) in self.blocks.iter().enumerate() {
                let idx = i.to_string();
                v.scoped(&idx, |v| block.visit_params(v))?;
            }
            Ok(())
        })?;
        v.scoped("final_ln", |v| self.final_ln.visit_params(v))?;
        v.param("lm_head", TensorRole::Parameter, &self.lm_head)?;
        Ok(())
    }

    fn visit_params_mut(&mut self, v: &mut ParamVisitorMut<'_, B>) -> Result<()> {
        v.scoped("tok_embeddings", |v| {
            self.tok_embeddings.visit_params_mut(v)
        })?;
        v.param(
            "pos_embeddings",
            TensorRole::Parameter,
            &mut self.pos_embeddings,
        )?;
        v.scoped("blocks", |v| {
            for (i, block) in self.blocks.iter_mut().enumerate() {
                let idx = i.to_string();
                v.scoped(&idx, |v| block.visit_params_mut(v))?;
            }
            Ok(())
        })?;
        v.scoped("final_ln", |v| self.final_ln.visit_params_mut(v))?;
        v.param("lm_head", TensorRole::Parameter, &mut self.lm_head)?;
        Ok(())
    }
}

fn decode_cache_capacity(required_len: usize, max_len: usize) -> usize {
    let bucket = required_len.next_power_of_two().max(1);
    bucket.min(max_len)
}

impl<B: PortableBackend + 'static> crate::inference::CausalLanguageModel<B> for Gpt<B> {
    fn context_length(&self) -> usize {
        self.config.context_length
    }

    fn num_layers(&self) -> usize {
        self.blocks.len()
    }

    fn forward(&self, tokens: &[usize]) -> Result<Tensor> {
        Gpt::forward(self, tokens)
    }

    fn forward_with_decode_cache(
        &self,
        tokens: &[usize],
        position_offset: usize,
        caches: &mut [Option<DecodeKvCache<B>>],
    ) -> Result<Tensor> {
        Gpt::forward_with_decode_cache(self, tokens, position_offset, caches)
    }

    fn forward_with_decode_cache_with_capacity(
        &self,
        tokens: &[usize],
        position_offset: usize,
        caches: &mut [Option<DecodeKvCache<B>>],
        capacity: usize,
    ) -> Result<Tensor> {
        Gpt::forward_with_decode_cache_with_capacity(
            self,
            tokens,
            position_offset,
            caches,
            capacity,
        )
    }
}
