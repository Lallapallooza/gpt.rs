//! Token generation: [`Generator`] picks the tokens after a prompt one at a time.

use anyhow::{ensure, Result};

use crate::inference::{CausalLanguageModel, Decode};

/// How a generation picks each token: the most likely one, or a random draw from the softmax of
/// the logits divided by `temperature`, over the `top_k` most likely tokens when it is set. The
/// model picks on the device, so a step downloads one token id.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Sampling {
    /// Zero or less picks the most likely token.
    pub temperature: f32,
    pub top_k: Option<usize>,
    /// The seed of the random draws.
    pub seed: u64,
}

impl Sampling {
    /// Picks the most likely token.
    pub fn greedy() -> Self {
        Self {
            temperature: 0.0,
            top_k: None,
            seed: 0,
        }
    }

    /// Draws each token at `temperature`, with the random draws of `seed`.
    pub fn random(temperature: f32, seed: u64) -> Self {
        Self {
            temperature,
            top_k: None,
            seed,
        }
    }

    /// Draws only among the `k` most likely tokens.
    pub fn with_top_k(mut self, k: usize) -> Self {
        self.top_k = Some(k.max(1));
        self
    }

    /// Whether the tokens are random draws rather than the most likely ones.
    pub fn is_random(&self) -> bool {
        self.temperature > 0.0
    }
}

/// Generates the tokens after a prompt, one per [`Generator::step`].
pub struct Generator<'a> {
    decode: Box<dyn Decode + 'a>,
    tokens: Vec<usize>,
    /// The most tokens, the prompt included, that the generation holds.
    max_len: usize,
    stop_tokens: Vec<usize>,
    finished: bool,
}

impl<'a> Generator<'a> {
    /// Prefills `prompt` to generate up to `max_new_tokens` after it. The generation ends early
    /// when it fills the context length of `model`.
    pub fn new(
        model: &'a dyn CausalLanguageModel,
        prompt: &[usize],
        max_new_tokens: usize,
        sampling: &Sampling,
    ) -> Result<Self> {
        ensure!(!prompt.is_empty(), "prompt must be non-empty");
        ensure!(
            max_new_tokens > 0,
            "a generation needs at least one new token"
        );
        let context_length = model.context_length();
        ensure!(
            prompt.len() < context_length,
            "a prompt of {} tokens leaves no room in the context length {context_length}",
            prompt.len()
        );
        let max_len = (prompt.len() + max_new_tokens).min(context_length);
        Ok(Self {
            decode: model.decode(prompt, max_len, sampling)?,
            tokens: prompt.to_vec(),
            max_len,
            stop_tokens: Vec::new(),
            finished: false,
        })
    }

    /// Ends the generation right after one of `tokens`, which the output keeps.
    pub fn with_stop_tokens(mut self, tokens: &[usize]) -> Self {
        self.stop_tokens = tokens.to_vec();
        self
    }

    /// Whether the generation ended: after a stop token, or with every token it may hold. Further
    /// steps return an error.
    pub fn is_finished(&self) -> bool {
        self.finished
    }

    pub fn tokens(&self) -> &[usize] {
        &self.tokens
    }

    pub fn into_tokens(self) -> Vec<usize> {
        self.tokens
    }

    /// The next token, which the output keeps.
    pub fn step(&mut self) -> Result<usize> {
        ensure!(!self.finished, "generation already finished");
        let token = self.decode.next_token()?;
        self.tokens.push(token);
        self.finished = self.stop_tokens.contains(&token) || self.tokens.len() == self.max_len;
        Ok(token)
    }
}

/// `prompt` followed by up to `max_new_tokens` generated tokens. The generation ends after one of
/// `stop_tokens`, which the output keeps, or when it fills the context length.
pub fn generate_tokens(
    model: &dyn CausalLanguageModel,
    prompt: &[usize],
    max_new_tokens: usize,
    sampling: &Sampling,
    stop_tokens: &[usize],
) -> Result<Vec<usize>> {
    if max_new_tokens == 0 {
        return Ok(prompt.to_vec());
    }
    let mut generator =
        Generator::new(model, prompt, max_new_tokens, sampling)?.with_stop_tokens(stop_tokens);
    while !generator.is_finished() {
        generator.step()?;
    }
    Ok(generator.into_tokens())
}
