use crate::backend::spec::PortableBackend;
use crate::inference::sampler::Sampler;
use crate::model::Gpt;
use crate::ops::functional::DecodeKvCache;
use crate::tensor::{Shape, Tensor};
use anyhow::{ensure, Result};

fn last_logits_row(logits: &Tensor) -> Result<&[f32]> {
    let dims = logits.shape().dims();
    ensure!(
        dims.len() == 2,
        "expected logits [T, V], got shape {:?}",
        dims
    );
    let seq_len = dims[0];
    let vocab = dims[1];
    ensure!(seq_len > 0 && vocab > 0, "logits must be non-empty");
    let data = logits.data();
    let start = (seq_len - 1) * vocab;
    Ok(&data[start..start + vocab])
}

pub struct GenerateConfig {
    pub max_new_tokens: usize,
    pub kv_cache: bool,
}

impl Default for GenerateConfig {
    fn default() -> Self {
        Self {
            max_new_tokens: 0,
            kv_cache: true,
        }
    }
}

pub struct Generator<'a, B: PortableBackend + 'static> {
    model: &'a Gpt<B>,
    sampler: &'a Sampler,
    caches: Option<Vec<Option<DecodeKvCache<B>>>>,
    processed_len: usize,
    window_start: usize,
    tokens: Vec<usize>,
    logits_row: Vec<f32>,
    kv_cache_capacity: Option<usize>,
}

impl<'a, B: PortableBackend + 'static> Generator<'a, B> {
    pub fn new(
        model: &'a Gpt<B>,
        sampler: &'a Sampler,
        prompt: &[usize],
        kv_cache: bool,
    ) -> Result<Self> {
        Self::new_with_kv_cache_capacity(model, sampler, prompt, kv_cache, None)
    }

    pub fn new_with_kv_cache_capacity(
        model: &'a Gpt<B>,
        sampler: &'a Sampler,
        prompt: &[usize],
        kv_cache: bool,
        kv_cache_capacity: Option<usize>,
    ) -> Result<Self> {
        ensure!(!prompt.is_empty(), "prompt must be non-empty");

        let context_length = model.config.context_length;
        let tokens = prompt.to_vec();
        let window_start = tokens.len().saturating_sub(context_length);

        let mut caches = kv_cache.then(|| vec![None; model.blocks.len()]);
        let mut processed_len = 0usize;

        let context = &tokens[window_start..];
        ensure!(!context.is_empty(), "context window must be non-empty");

        let logits_row = if let Some(caches_vec) = caches.as_mut() {
            let logits = if let Some(capacity) = kv_cache_capacity {
                model.forward_with_decode_cache_with_capacity(context, 0, caches_vec, capacity)?
            } else {
                model.forward_with_decode_cache(context, 0, caches_vec)?
            };
            processed_len = context.len();
            last_logits_row(&logits)?.to_vec()
        } else {
            let logits = model.forward(context)?;
            last_logits_row(&logits)?.to_vec()
        };

        Ok(Self {
            model,
            sampler,
            caches,
            processed_len,
            window_start,
            tokens,
            logits_row,
            kv_cache_capacity,
        })
    }

    pub fn tokens(&self) -> &[usize] {
        &self.tokens
    }

    pub fn into_tokens(self) -> Vec<usize> {
        self.tokens
    }

    /// Samples the next token from the current logits row and appends it to the generated
    /// sequence without computing the next logits.
    ///
    /// This is intended for the final generation step: once the token is emitted, the caller can
    /// stop without paying for an unused forward pass.
    pub fn step_final(&mut self) -> Result<usize> {
        let vocab = self.logits_row.len();
        ensure!(vocab > 0, "logits row is empty");

        let row_tensor =
            Tensor::from_vec(Shape::new([vocab]), std::mem::take(&mut self.logits_row))?;
        let next = self.sampler.sample(&row_tensor);
        self.tokens.push(next);
        Ok(next)
    }

    pub fn step(&mut self) -> Result<usize> {
        let vocab = self.logits_row.len();
        ensure!(vocab > 0, "logits row is empty");

        let row_tensor =
            Tensor::from_vec(Shape::new([vocab]), std::mem::take(&mut self.logits_row))?;
        let next = self.sampler.sample(&row_tensor);
        self.tokens.push(next);

        let context_length = self.model.config.context_length;
        if self.tokens.len() - self.window_start > context_length {
            self.window_start = self.tokens.len().saturating_sub(context_length);
            if let Some(caches_vec) = self.caches.as_mut() {
                caches_vec.iter_mut().for_each(|slot| *slot = None);
                self.processed_len = 0;
            }
        }

        if let Some(caches_vec) = self.caches.as_mut() {
            let context = &self.tokens[self.window_start..];
            let offset = self.processed_len.min(context.len());
            let chunk = &context[offset..];
            ensure!(!chunk.is_empty(), "decode chunk must be non-empty");
            let logits = if let Some(capacity) = self.kv_cache_capacity {
                self.model
                    .forward_with_decode_cache_with_capacity(chunk, offset, caches_vec, capacity)?
            } else {
                self.model
                    .forward_with_decode_cache(chunk, offset, caches_vec)?
            };
            self.processed_len = context.len();
            self.logits_row = last_logits_row(&logits)?.to_vec();
        } else {
            let context_len = context_length.min(self.tokens.len());
            let start = self.tokens.len() - context_len;
            let logits = self.model.forward(&self.tokens[start..])?;
            self.logits_row = last_logits_row(&logits)?.to_vec();
        }

        Ok(next)
    }
}

pub fn generate_tokens<B: PortableBackend + 'static>(
    model: &Gpt<B>,
    prompt: &[usize],
    sampler: &Sampler,
    cfg: GenerateConfig,
) -> Result<Vec<usize>> {
    if cfg.max_new_tokens == 0 {
        return Ok(prompt.to_vec());
    }

    let mut gen = Generator::new(model, sampler, prompt, cfg.kv_cache)?;
    for step in 0..cfg.max_new_tokens {
        if step + 1 == cfg.max_new_tokens {
            gen.step_final()?;
        } else {
            gen.step()?;
        }
    }
    Ok(gen.into_tokens())
}
