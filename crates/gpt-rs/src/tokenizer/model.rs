//! Byte-level byte pair encoding (BPE) tokenizer.
//!
//! [`Tokenizer`] implements the byte-level BPE scheme of GPT-2 and the Qwen family, with added
//! tokens such as `<|im_start|>` matched verbatim. At load time, the tokenizer rejects a Hugging
//! Face `tokenizer.json` that uses options outside this scheme.

use super::bpe::{get_pairs, BpeMerges};
use aho_corasick::{AhoCorasick, MatchKind};
use anyhow::{anyhow, bail, ensure, Context, Result};
use fancy_regex::Regex;
use serde::{Deserialize, Serialize};
use std::borrow::Cow;
use std::collections::HashMap;
use std::sync::Mutex;
use unicode_normalization::{is_nfc_quick, IsNormalized, UnicodeNormalization};

/// Regular expression used by GPT-2 to chunk input text prior to BPE merges.
const GPT2_PATTERN: &str =
    r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+";

/// Token that the tokenizer matches verbatim in the input text. Pre-tokenization and merges never
/// split it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AddedToken {
    pub content: String,
    pub id: usize,
}

/// Serializable tokenizer definition that mirrors the JSON schema exported by Python tools.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenizerConfig {
    /// Maps string tokens to their integer ids as learned during tokenizer training.
    pub vocab: HashMap<String, usize>,
    /// Ordered list of merge operations; earlier entries represent higher merge priority.
    pub merges: Vec<(String, String)>,
    /// Symbol that replaces pieces missing from the vocabulary. Byte-level vocabularies cover
    /// every byte, so they never emit it.
    #[serde(default = "default_unk_token")]
    pub unk_token: String,
    /// Pre-tokenization regex. The default is the GPT-2 pattern.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pattern: Option<String>,
    /// When true, the tokenizer applies Unicode NFC normalization before pre-tokenization.
    #[serde(default)]
    pub normalize_nfc: bool,
    /// Tokens matched verbatim before normalization and pre-tokenization.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub added_tokens: Vec<AddedToken>,
}

/// Provides the default unknown token marker when a configuration omits the field.
fn default_unk_token() -> String {
    "<unk>".to_string()
}

/// The subset of a Hugging Face `tokenizer.json` that byte-level BPE needs. Serde rejects unknown
/// normalizer, pre-tokenizer, and decoder types. [`HfTokenizerJson::into_config`] rejects options
/// that change the encoding in ways that [`Tokenizer`] does not implement.
#[derive(Debug, Deserialize)]
struct HfTokenizerJson {
    model: HfBpe,
    #[serde(default)]
    added_tokens: Vec<HfAddedToken>,
    normalizer: Option<HfNormalizer>,
    pre_tokenizer: HfPreTokenizer,
    decoder: HfDecoder,
}

#[derive(Debug, Deserialize)]
struct HfBpe {
    #[serde(rename = "type")]
    kind: Option<String>,
    vocab: HashMap<String, usize>,
    merges: Vec<HfMergeEntry>,
    unk_token: Option<String>,
    dropout: Option<f64>,
    continuing_subword_prefix: Option<String>,
    end_of_word_suffix: Option<String>,
    #[serde(default)]
    byte_fallback: bool,
    #[serde(default)]
    ignore_merges: bool,
}

#[derive(Debug, Deserialize)]
struct HfAddedToken {
    id: usize,
    content: String,
    #[serde(default)]
    normalized: bool,
    #[serde(default)]
    lstrip: bool,
    #[serde(default)]
    rstrip: bool,
    #[serde(default)]
    single_word: bool,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
enum HfNormalizer {
    #[serde(rename = "NFC")]
    Nfc,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
enum HfPreTokenizer {
    ByteLevel {
        #[serde(default)]
        add_prefix_space: bool,
        #[serde(default = "default_true")]
        use_regex: bool,
    },
    Split {
        pattern: HfPattern,
        behavior: String,
        #[serde(default)]
        invert: bool,
    },
    Sequence {
        pretokenizers: Vec<HfPreTokenizer>,
    },
}

#[derive(Debug, Deserialize)]
enum HfPattern {
    Regex(String),
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
enum HfDecoder {
    ByteLevel {},
}

fn default_true() -> bool {
    true
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum HfMergeEntry {
    Pair([String; 2]),
    SpaceSeparated(String),
}

impl HfTokenizerJson {
    fn into_config(self) -> Result<TokenizerConfig> {
        let HfTokenizerJson {
            model,
            added_tokens,
            normalizer,
            pre_tokenizer,
            decoder: HfDecoder::ByteLevel {},
        } = self;
        if let Some(kind) = model.kind.as_deref() {
            ensure!(kind == "BPE", "unsupported tokenizer model type '{kind}'");
        }
        ensure!(
            model.dropout.is_none()
                && !model.byte_fallback
                && !model.ignore_merges
                && model
                    .continuing_subword_prefix
                    .as_deref()
                    .unwrap_or("")
                    .is_empty()
                && model.end_of_word_suffix.as_deref().unwrap_or("").is_empty(),
            "BPE dropout, byte_fallback, ignore_merges, and subword prefixes or suffixes are not supported"
        );
        // The tokenizer matches added tokens on the raw text. The raw text equals the normalized
        // text only when there is no normalizer.
        for token in &added_tokens {
            ensure!(
                !(token.lstrip
                    || token.rstrip
                    || token.single_word
                    || (token.normalized && normalizer.is_some())),
                "added token {:?} uses normalized, lstrip, rstrip, or single_word matching, which is not supported",
                token.content
            );
        }
        // Accept only two layouts. The GPT-2 layout is a ByteLevel with its own regex. The Qwen
        // layout is a Split regex followed by a ByteLevel without a regex.
        let pattern = match pre_tokenizer {
            HfPreTokenizer::ByteLevel {
                add_prefix_space: false,
                use_regex: true,
            } => None,
            HfPreTokenizer::Sequence { pretokenizers } => match pretokenizers.as_slice() {
                [HfPreTokenizer::Split {
                    pattern: HfPattern::Regex(regex),
                    behavior,
                    invert: false,
                }, HfPreTokenizer::ByteLevel {
                    add_prefix_space: false,
                    use_regex: false,
                }] if behavior == "Isolated" => Some(regex.clone()),
                other => bail!("unsupported pre-tokenizer sequence {other:?}"),
            },
            other => bail!("unsupported pre-tokenizer {other:?}"),
        };
        let merges = model
            .merges
            .into_iter()
            .map(|entry| match entry {
                HfMergeEntry::Pair([left, right]) => Ok((left, right)),
                HfMergeEntry::SpaceSeparated(raw) => raw
                    .split_once(' ')
                    .map(|(left, right)| (left.to_string(), right.to_string()))
                    .ok_or_else(|| anyhow!("invalid BPE merge entry '{raw}'")),
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(TokenizerConfig {
            vocab: model.vocab,
            merges,
            unk_token: model
                .unk_token
                .filter(|token| !token.is_empty())
                .unwrap_or_else(default_unk_token),
            pattern,
            normalize_nfc: matches!(normalizer, Some(HfNormalizer::Nfc)),
            added_tokens: added_tokens
                .into_iter()
                .map(|token| AddedToken {
                    content: token.content,
                    id: token.id,
                })
                .collect(),
        })
    }
}

impl TokenizerConfig {
    /// Parses the flat gpt-rs tokenizer schema, or a Hugging Face `tokenizer.json` of a byte-level
    /// BPE model.
    pub fn from_json_str(data: &str) -> Result<Self> {
        if let Ok(cfg) = serde_json::from_str::<TokenizerConfig>(data) {
            return Ok(cfg);
        }
        serde_json::from_str::<HfTokenizerJson>(data)
            .context("failed to parse Hugging Face tokenizer.json")?
            .into_config()
    }
}

/// Runtime tokenizer capable of encoding strings to ids and decoding ids back to text.
#[derive(Debug)]
pub struct Tokenizer {
    /// Forward mapping from token string to id.
    encoder: HashMap<String, usize>,
    /// Raw bytes of every token id. For an added token, the bytes are its UTF-8 content.
    token_bytes: Vec<Vec<u8>>,
    /// Learned merge priorities wrapped in a `BpeMerges` helper.
    merges: BpeMerges,
    /// Id that replaces pieces missing from the vocabulary.
    unk_id: Option<usize>,
    /// Printable Unicode code point standing in for each raw byte.
    byte_encoder: [char; 256],
    /// Cache for memoizing intermediate BPE results to avoid recomputing merges.
    cache: Mutex<HashMap<String, String>>,
    /// Compiled tokenization pattern applied before the BPE merge loop runs.
    pattern: Regex,
    normalize_nfc: bool,
    /// Leftmost-longest matcher over the added token contents.
    added: AhoCorasick,
    /// Token id of each `added` pattern.
    added_ids: Vec<usize>,
}

impl Tokenizer {
    /// Loads a tokenizer from a gpt-rs flat config or a Hugging Face `tokenizer.json`.
    pub fn from_file(path: impl AsRef<std::path::Path>) -> Result<Self> {
        let path = path.as_ref();
        let data = std::fs::read_to_string(path)
            .with_context(|| format!("failed to read tokenizer from {}", path.display()))?;
        let config = TokenizerConfig::from_json_str(&data).with_context(|| {
            format!(
                "invalid tokenizer config in {}: expected the flat gpt-rs schema or a Hugging Face tokenizer.json",
                path.display()
            )
        })?;
        Self::from_config(config)
    }

    /// Builds a tokenizer from a serialized [`TokenizerConfig`].
    pub fn from_config(config: TokenizerConfig) -> Result<Self> {
        let TokenizerConfig {
            vocab,
            merges,
            unk_token,
            pattern,
            normalize_nfc,
            added_tokens,
        } = config;

        let max_id = vocab
            .values()
            .chain(added_tokens.iter().map(|token| &token.id))
            .copied()
            .max()
            .ok_or_else(|| anyhow!("tokenizer vocabulary is empty"))?;
        let byte_encoder = bytes_to_unicode();
        let byte_decoder: HashMap<char, u8> = (0..=255u8)
            .map(|b| (byte_encoder[usize::from(b)], b))
            .collect();
        let mut token_bytes = vec![Vec::new(); max_id + 1];
        for (token, &idx) in &vocab {
            // Symbols outside the byte alphabet, such as a literal `<unk>`, decode as their UTF-8
            // bytes.
            let mut bytes = Vec::with_capacity(token.len());
            for ch in token.chars() {
                match byte_decoder.get(&ch) {
                    Some(&b) => bytes.push(b),
                    None => bytes.extend_from_slice(ch.encode_utf8(&mut [0u8; 4]).as_bytes()),
                }
            }
            token_bytes[idx] = bytes;
        }
        let added_tokens: Vec<AddedToken> = added_tokens
            .into_iter()
            .filter(|token| !token.content.is_empty())
            .collect();
        for token in &added_tokens {
            token_bytes[token.id] = token.content.as_bytes().to_vec();
        }
        let added = AhoCorasick::builder()
            .match_kind(MatchKind::LeftmostLongest)
            .build(added_tokens.iter().map(|token| &token.content))
            .context("failed to build the added-token matcher")?;
        let added_ids = added_tokens.iter().map(|token| token.id).collect();

        let unk_id = vocab
            .get(&unk_token)
            .copied()
            .or_else(|| vocab.get("<unk>").copied());

        let ranks = merges
            .into_iter()
            .enumerate()
            .map(|(rank, (a, b))| ((a, b), rank))
            .collect();

        let pattern_src = pattern.as_deref().unwrap_or(GPT2_PATTERN);
        let pattern = Regex::new(pattern_src)
            .with_context(|| format!("invalid pre-tokenization pattern {pattern_src:?}"))?;

        Ok(Tokenizer {
            encoder: vocab,
            token_bytes,
            merges: BpeMerges::new(ranks),
            unk_id,
            byte_encoder,
            cache: Mutex::new(HashMap::new()),
            pattern,
            normalize_nfc,
            added,
            added_ids,
        })
    }

    /// Encodes UTF-8 text into token ids, like Hugging Face
    /// `encode(text, add_special_tokens=False)`. It does not apply post-processor templates, such
    /// as BOS or EOS insertion.
    ///
    /// The tokenizer first matches added tokens, leftmost and longest first. Then it processes each
    /// remaining span in these steps: NFC normalization when configured, a split with the
    /// pre-tokenization pattern, byte encoding, and merges.
    pub fn encode(&self, text: &str) -> Result<Vec<usize>> {
        let mut ids = Vec::new();
        let mut last = 0;
        for mat in self.added.find_iter(text) {
            self.encode_span(&text[last..mat.start()], &mut ids)?;
            ids.push(self.added_ids[mat.pattern().as_usize()]);
            last = mat.end();
        }
        self.encode_span(&text[last..], &mut ids)?;
        Ok(ids)
    }

    fn encode_span(&self, span: &str, ids: &mut Vec<usize>) -> Result<()> {
        let normalized: Cow<'_, str> =
            if self.normalize_nfc && is_nfc_quick(span.chars()) != IsNormalized::Yes {
                Cow::Owned(span.nfc().collect())
            } else {
                Cow::Borrowed(span)
            };
        // As in Hugging Face's `Isolated` split, text between matches is a piece of its own.
        let mut last = 0;
        for mat in self.pattern.find_iter(&normalized) {
            let mat = mat.context("pre-tokenization regex failed")?;
            self.encode_piece(&normalized[last..mat.start()], ids)?;
            self.encode_piece(mat.as_str(), ids)?;
            last = mat.end();
        }
        self.encode_piece(&normalized[last..], ids)
    }

    fn encode_piece(&self, piece: &str, ids: &mut Vec<usize>) -> Result<()> {
        if piece.is_empty() {
            return Ok(());
        }
        for token in self.bpe(&self.byte_encode(piece)).split(' ') {
            let id = match self.encoder.get(token) {
                Some(&id) => id,
                None => self.unk_id.ok_or_else(|| {
                    anyhow!("token {token:?} is not in the vocabulary and no unk token is set")
                })?,
            };
            ids.push(id);
        }
        Ok(())
    }

    /// Returns the raw bytes that `token` represents. An added token yields its UTF-8 content. An
    /// unknown id yields nothing.
    pub fn token_bytes(&self, token: usize) -> &[u8] {
        self.token_bytes.get(token).map_or(&[], Vec::as_slice)
    }

    /// Decodes token ids into text. Invalid UTF-8 sequences become U+FFFD.
    pub fn decode(&self, tokens: &[usize]) -> String {
        let bytes: Vec<u8> = tokens
            .iter()
            .flat_map(|&t| self.token_bytes(t))
            .copied()
            .collect();
        String::from_utf8_lossy(&bytes).into_owned()
    }

    /// Returns the number of token ids (vocabulary plus added tokens).
    pub fn vocab_size(&self) -> usize {
        self.token_bytes.len()
    }

    /// Applies the BPE merge loop to a byte-encoded token segment.
    ///
    /// Results are cached per input segment to amortize repeated work during long prompts.
    /// The output separates merged tokens with spaces so the caller can map each piece into a
    /// vocabulary id.
    fn bpe(&self, token: &str) -> String {
        if token.is_empty() {
            return String::new();
        }

        if let Some(cached) = self
            .cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .get(token)
            .cloned()
        {
            return cached;
        }

        let mut word: Vec<String> = token.chars().map(|ch| ch.to_string()).collect();
        if word.len() <= 1 {
            let result = token.to_string();
            self.cache
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
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
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .insert(token.to_string(), result.clone());
        result
    }

    /// Converts a chunk of text into the intermediate Unicode alphabet used by byte-level BPE.
    fn byte_encode(&self, text: &str) -> String {
        text.bytes()
            .map(|b| self.byte_encoder[usize::from(b)])
            .collect()
    }
}

/// Incremental detokenizer for streaming generation.
///
/// Byte-level tokens can end in the middle of a UTF-8 character. [`StreamDecoder::push`] returns
/// only complete characters. It keeps the trailing partial sequence until later tokens complete it.
#[derive(Debug, Default)]
pub struct StreamDecoder {
    pending: Vec<u8>,
}

impl StreamDecoder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Appends `token` and returns the text that became complete.
    pub fn push(&mut self, tokenizer: &Tokenizer, token: usize) -> String {
        self.pending.extend_from_slice(tokenizer.token_bytes(token));
        let mut out = String::new();
        loop {
            match std::str::from_utf8(&self.pending) {
                Ok(text) => {
                    out.push_str(text);
                    self.pending.clear();
                    return out;
                }
                Err(err) => {
                    let valid = err.valid_up_to();
                    out.push_str(
                        std::str::from_utf8(&self.pending[..valid]).expect("prefix is valid UTF-8"),
                    );
                    match err.error_len() {
                        // Incomplete trailing sequence: wait for more bytes.
                        None => {
                            self.pending.drain(..valid);
                            return out;
                        }
                        // Invalid sequence: emit a replacement character and continue.
                        Some(bad) => {
                            out.push(char::REPLACEMENT_CHARACTER);
                            self.pending.drain(..valid + bad);
                        }
                    }
                }
            }
        }
    }

    /// Flushes the bytes that are still pending at the end of generation. Invalid bytes become
    /// U+FFFD.
    pub fn finish(&mut self) -> String {
        let out = String::from_utf8_lossy(&self.pending).into_owned();
        self.pending.clear();
        out
    }
}

/// Maps every byte to the printable Unicode alphabet of byte-level BPE. Printable Latin-1 bytes map
/// to themselves. The other bytes map to consecutive code points from U+0100.
fn bytes_to_unicode() -> [char; 256] {
    let printable = |b: u8| matches!(b, 33..=126 | 161..=172 | 174..=255);
    let mut table = ['\0'; 256];
    let mut shifted = 0u32;
    for b in 0..=255u8 {
        table[usize::from(b)] = if printable(b) {
            char::from(b)
        } else {
            shifted += 1;
            char::from_u32(255 + shifted).expect("code points below U+0200 are valid")
        };
    }
    table
}
