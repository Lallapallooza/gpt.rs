//! A generation ends on a stop token, at its token budget or at the context length, without
//! decoding another token.

use std::cell::Cell;

use anyhow::{bail, Result};
use gpt_rs::inference::generate::{generate_tokens, Generator, Sampling};
use gpt_rs::inference::{CausalLanguageModel, Decode};
use gpt_rs::tensor::HostTensor;

/// A model with context length `context_length` that decodes 1, 2, 3, ... after any prompt.
struct Counting {
    context_length: usize,
    decoded: Cell<usize>,
}

impl Counting {
    fn new(context_length: usize) -> Self {
        Self {
            context_length,
            decoded: Cell::new(0),
        }
    }
}

impl CausalLanguageModel for Counting {
    fn context_length(&self) -> usize {
        self.context_length
    }

    fn forward(&self, _tokens: &[usize]) -> Result<HostTensor> {
        bail!("generation does not run a full forward")
    }

    fn decode(
        &self,
        _prompt: &[usize],
        _max_len: usize,
        _sampling: &Sampling,
    ) -> Result<Box<dyn Decode + '_>> {
        Ok(Box::new(Counter { model: self }))
    }
}

struct Counter<'a> {
    model: &'a Counting,
}

impl Decode for Counter<'_> {
    fn next_token(&mut self) -> Result<usize> {
        let decoded = &self.model.decoded;
        decoded.set(decoded.get() + 1);
        Ok(decoded.get())
    }
}

#[test]
fn generator_stops_after_a_stop_token_without_another_step() -> Result<()> {
    let model = Counting::new(16);
    let mut generator =
        Generator::new(&model, &[0], 8, &Sampling::greedy())?.with_stop_tokens(&[2]);
    assert_eq!(generator.step()?, 1);
    assert_eq!(generator.step()?, 2);
    assert!(generator.is_finished());
    assert_eq!(model.decoded.get(), 2);
    let err = generator
        .step()
        .expect_err("a step after a stop token must fail");
    assert_eq!(err.to_string(), "generation already finished");
    assert_eq!(model.decoded.get(), 2);
    Ok(())
}

#[test]
fn generate_tokens_keeps_the_stop_token_and_ends() -> Result<()> {
    let model = Counting::new(16);
    let tokens = generate_tokens(&model, &[0], 5, &Sampling::greedy(), &[2])?;
    assert_eq!(tokens, [0, 1, 2]);
    Ok(())
}

#[test]
fn generation_ends_when_it_fills_the_context_length() -> Result<()> {
    let model = Counting::new(4);
    let tokens = generate_tokens(&model, &[0], 10, &Sampling::greedy(), &[])?;
    assert_eq!(tokens, [0, 1, 2, 3]);
    assert_eq!(model.decoded.get(), 3);
    Ok(())
}

#[test]
fn a_prompt_that_fills_the_context_length_is_rejected() {
    let model = Counting::new(4);
    let err = Generator::new(&model, &[0, 1, 2, 3], 1, &Sampling::greedy())
        .err()
        .expect("no room for a new token");
    assert_eq!(
        err.to_string(),
        "a prompt of 4 tokens leaves no room in the context length 4"
    );
}
