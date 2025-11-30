//! High-level GPT-style tokenizer backed by byte pair encoding (BPE).
//!
//! The model exposes a thin `Tokenizer` wrapper around the pre-trained vocabulary and merge
//! tables shipped with GPT-2/GPT-3 style models. It mirrors the semantics of the original
//! Python implementation while remaining dependency-free at runtime so the tokenizer can run
//! inside inference and training binaries without Python bindings.

use super::bpe::{get_pairs, BpeMerges};
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Mutex;

/// Regular expression used by GPT-2 to chunk input text prior to BPE merges.
const GPT2_PATTERN: &str = r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+";

/// Serializable tokenizer definition that mirrors the JSON schema exported by Python tools.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenizerConfig {
    /// Maps string tokens to their integer ids as learned during tokenizer training.
    pub vocab: HashMap<String, usize>,
    /// Ordered list of merge operations; earlier entries represent higher merge priority.
    pub merges: Vec<(String, String)>,
    /// Symbol to substitute when the tokenizer observes an unknown token at runtime.
    #[serde(default = "default_unk_token")]
    pub unk_token: String,
}

/// Provides the default unknown token marker when a configuration omits the field.
fn default_unk_token() -> String {
    "<unk>".to_string()
}

/// Runtime tokenizer capable of encoding strings to ids and decoding ids back to text.
#[derive(Debug)]
pub struct Tokenizer {
    /// Forward mapping from token string to id.
    encoder: HashMap<String, usize>,
    /// Reverse mapping from id to token string; preserves GPT-2 byte-level encoding.
    decoder: Vec<String>,
    /// Learned merge priorities wrapped in a `BpeMerges` helper.
    merges: BpeMerges,
    /// Identifier of the unknown token used as a fallback during encoding and decoding.
    unk_id: usize,
    /// Lookup table translating raw bytes into printable Unicode code points.
    byte_encoder: HashMap<u8, char>,
    /// Reverse lookup used when reconstructing raw UTF-8 output.
    byte_decoder: HashMap<char, u8>,
    /// Cache for memoizing intermediate BPE results to avoid recomputing merges.
    cache: Mutex<HashMap<String, String>>,
    /// Compiled tokenization pattern applied before the BPE merge loop runs.
    pattern: Regex,
}

impl Tokenizer {
    /// Builds a tokenizer directly from a serialized [`TokenizerConfig`].
    ///
    /// The constructor assembles both encoder and decoder tables, resolves the unknown token
    /// id, and prepares byte-level Unicode shims so tokenization stays faithful to the original
    /// GPT-2 rules regardless of input UTF-8 sequences.
    pub fn from_config(config: TokenizerConfig) -> Self {
        let TokenizerConfig {
            vocab,
            merges,
            unk_token,
        } = config;

        let mut decoder = vec![String::new(); vocab.len()];
        for (token, &idx) in &vocab {
            if idx < decoder.len() {
                decoder[idx] = token.clone();
            }
        }

        let unk_id = vocab
            .get(&unk_token)
            .copied()
            .or_else(|| vocab.get("<unk>").copied())
            .expect("missing unknown token in vocabulary");

        let ranks = merges
            .into_iter()
            .enumerate()
            .map(|(rank, (a, b))| ((a, b), rank))
            .collect();

        let (byte_encoder, byte_decoder) = bytes_to_unicode();
        let pattern = Regex::new(GPT2_PATTERN).expect("invalid GPT-2 tokenizer regex pattern");

        Tokenizer {
            encoder: vocab,
            decoder,
            merges: BpeMerges::new(ranks),
            unk_id,
            byte_encoder,
            byte_decoder,
            cache: Mutex::new(HashMap::new()),
            pattern,
        }
    }

    /// Encodes UTF-8 text into a sequence of token ids using GPT-2 compatible rules.
    ///
    /// Input is first segmented with the GPT-2 regex, then each segment is byte-encoded and
    /// progressively merged using the stored BPE ranks. Unknown tokens fall back to the
    /// configured `unk_id`, mirroring the behaviour of the reference implementation.
    pub fn encode(&self, text: &str) -> Vec<usize> {
        if text.is_empty() {
            return Vec::new();
        }

        let mut ids = Vec::new();
        for mat in self.pattern.find_iter(text) {
            let piece = mat.as_str();
            if piece.is_empty() {
                continue;
            }
            let transformed = self.byte_encode(piece);
            let bpe_tokens = self.bpe(&transformed);
            for token in bpe_tokens.split(' ') {
                let id = self.encoder.get(token).copied().unwrap_or(self.unk_id);
                ids.push(id);
            }
        }
        ids
    }

    /// Decodes token ids back into human-readable text.
    ///
    /// The decoder mirrors the Python pipeline by concatenating token strings, mapping the
    /// intermediate representation through the inverse byte lookup, and finally materializing
    /// UTF-8 output. Any id outside the vocabulary produces the unknown token string.
    pub fn decode(&self, tokens: &[usize]) -> String {
        if tokens.is_empty() {
            return String::new();
        }

        let mut text = String::new();
        for &token in tokens {
            let piece = self
                .decoder
                .get(token)
                .cloned()
                .unwrap_or_else(|| self.decoder[self.unk_id].clone());
            text.push_str(&piece);
        }

        let mut bytes = Vec::with_capacity(text.len());
        for ch in text.chars() {
            if let Some(&b) = self.byte_decoder.get(&ch) {
                bytes.push(b);
            } else {
                let mut buf = [0u8; 4];
                let encoded = ch.encode_utf8(&mut buf);
                bytes.extend_from_slice(encoded.as_bytes());
            }
        }

        String::from_utf8(bytes).unwrap_or_default()
    }

    /// Applies the BPE merge loop to a byte-encoded token segment.
    ///
    /// Results are cached per input segment to amortize repeated work during long prompts.
    /// The output mirrors the Python implementation by separating merged tokens with spaces
    /// so the caller can map each piece into a vocabulary id.
    fn bpe(&self, token: &str) -> String {
        if token.is_empty() {
            return String::new();
        }

        if let Some(cached) = self.cache.lock().unwrap().get(token).cloned() {
            return cached;
        }

        let mut word: Vec<String> = token.chars().map(|ch| ch.to_string()).collect();
        if word.len() <= 1 {
            let result = token.to_string();
            self.cache
                .lock()
                .unwrap()
                .insert(token.to_string(), result.clone());
            return result;
        }

        let mut pairs = get_pairs(&word);
        while !pairs.is_empty() {
            let mut min_rank = usize::MAX;
            let mut best_pair: Option<(String, String)> = None;

            for pair in &pairs {
                if let Some(rank) = self.merges.rank(pair) {
                    if rank < min_rank {
                        min_rank = rank;
                        best_pair = Some(pair.clone());
                    }
                }
            }

            let Some(best_pair) = best_pair else {
                break;
            };

            let first = best_pair.0;
            let second = best_pair.1;
            let first_ref = first.as_str();
            let second_ref = second.as_str();

            let mut new_word = Vec::with_capacity(word.len());
            let mut i = 0;
            while i < word.len() {
                if i + 1 < word.len()
                    && word[i].as_str() == first_ref
                    && word[i + 1].as_str() == second_ref
                {
                    let merged = format!("{}{}", word[i], word[i + 1]);
                    new_word.push(merged);
                    i += 2;
                } else {
                    new_word.push(word[i].clone());
                    i += 1;
                }
            }

            word = new_word;
            if word.len() == 1 {
                break;
            }
            pairs = get_pairs(&word);
        }

        let result = word.join(" ");
        self.cache
            .lock()
            .unwrap()
            .insert(token.to_string(), result.clone());
        result
    }

    /// Converts a chunk of text into the intermediate Unicode alphabet used by GPT-2.
    ///
    /// This mapping ensures byte stability by projecting raw bytes into the 256..512
    /// code-point range for bytes that do not map to printable ASCII characters.
    fn byte_encode(&self, text: &str) -> String {
        text.as_bytes()
            .iter()
            .map(|b| self.byte_encoder.get(b).copied().unwrap_or(char::from(*b)))
            .collect()
    }
}

/// Builds two synchronized lookup tables that map between raw bytes and the unicode alphabet
/// expected by GPT-2 tokenization.
fn bytes_to_unicode() -> (HashMap<u8, char>, HashMap<char, u8>) {
    let mut bs: Vec<u8> = (33u8..=126).chain(161..=172).chain(174..=255).collect();
    let mut cs: Vec<char> = bs.iter().map(|&b| b as char).collect();
    let mut n: u32 = 0;
    for b in 0u8..=255 {
        if !bs.contains(&b) {
            bs.push(b);
            let ch = char::from_u32(256 + n).expect("unable to map byte to unicode");
            cs.push(ch);
            n += 1;
        }
    }

    let mut encoder = HashMap::new();
    let mut decoder = HashMap::new();
    for (b, c) in bs.into_iter().zip(cs.into_iter()) {
        encoder.insert(b, c);
        decoder.insert(c, b);
    }
    (encoder, decoder)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::PathBuf;

    fn load_gpt2_tokenizer() -> Tokenizer {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join("configs")
            .join("gpt2_tokenizer.json");
        let data = fs::read_to_string(path).expect("failed to read GPT-2 tokenizer config");
        let config: TokenizerConfig =
            serde_json::from_str(&data).expect("invalid tokenizer config JSON");
        Tokenizer::from_config(config)
    }

    #[test]
    fn gpt2_encode_matches_python() {
        let tokenizer = load_gpt2_tokenizer();
        assert_eq!(tokenizer.encode("Hello world"), vec![15496, 995]);
        assert_eq!(tokenizer.encode(" Hello world"), vec![18435, 995]);
    }

    #[test]
    fn gpt2_decode_matches_python() {
        let tokenizer = load_gpt2_tokenizer();
        let hello_world = vec![15496usize, 995];
        assert_eq!(tokenizer.decode(&hello_world), "Hello world");

        let leading_space = vec![18435usize, 995];
        assert_eq!(tokenizer.decode(&leading_space), " Hello world");
    }
}
