//! Building blocks for GPT-style tokenization.
//!
//! The tokenizer module wires together byte pair encoding (BPE) utilities and the
//! higher level [`Tokenizer`] implementation that mirrors the GPT-2/GPT-3 family of
//! vocabularies. The public surface offers a serde-friendly [`TokenizerConfig`] for
//! loading vocabularies exported from Python tokenizers while keeping the runtime
//! focused on device-friendly text processing.

pub mod bpe;
pub mod model;

pub use model::{Tokenizer, TokenizerConfig};
