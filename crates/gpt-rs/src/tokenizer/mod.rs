//! Building blocks for GPT-style tokenization.
//!
//! The tokenizer module wires together byte pair encoding (BPE) utilities and the
//! byte-level BPE [`Tokenizer`] used by GPT-2 and Qwen vocabularies. [`TokenizerConfig`]
//! loads the flat gpt-rs schema or a Hugging Face `tokenizer.json`.

pub mod bpe;
pub mod model;

pub use model::{AddedToken, StreamDecoder, Tokenizer, TokenizerConfig};
