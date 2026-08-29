//! Loading Hugging Face `tokenizer.json` files and decoding tokens as a stream.

use gpt_rs::tokenizer::{StreamDecoder, Tokenizer, TokenizerConfig};
use serde_json::{json, Value};

/// The Qwen2/3-family split pattern.
const QWEN_PATTERN: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

/// A byte-level BPE `tokenizer.json` in the Qwen layout, with a few merges.
fn fixture() -> Value {
    let mut vocab = serde_json::Map::new();
    let mut add = |token: String| {
        let id = vocab.len();
        vocab.entry(token).or_insert(Value::from(id));
    };
    // GPT-2 byte-to-unicode alphabet.
    let mut shifted = 0u32;
    for b in 0u32..=255 {
        let printable = matches!(b, 33..=126 | 161..=172 | 174..=255);
        let cp = if printable {
            b
        } else {
            shifted += 1;
            255 + shifted
        };
        add(char::from_u32(cp).unwrap().to_string());
    }
    let merges = [
        "h e", "l l", "he ll", "hell o", "Ġ w", "Ġw o", "Ġwo r", "l d", "Ġwor ld", "Ġ Ġ", "1 2",
        "b c", "a b",
    ];
    for merge in merges {
        add(merge.replace(' ', ""));
    }
    let base = vocab.len();
    json!({
        "added_tokens": [
            {"id": base, "content": "<|im_start|>", "special": true, "normalized": false},
            {"id": base + 1, "content": "<|im_end|>", "special": true, "normalized": false}
        ],
        "normalizer": {"type": "NFC"},
        "pre_tokenizer": {"type": "Sequence", "pretokenizers": [
            {"type": "Split", "pattern": {"Regex": QWEN_PATTERN}, "behavior": "Isolated", "invert": false},
            {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false, "use_regex": false}
        ]},
        "decoder": {"type": "ByteLevel"},
        "model": {"type": "BPE", "unk_token": null, "vocab": vocab, "merges": merges}
    })
}

fn load(json: &Value) -> anyhow::Result<Tokenizer> {
    Tokenizer::from_config(TokenizerConfig::from_json_str(&json.to_string())?)
}

fn pieces(tok: &Tokenizer, text: &str) -> Vec<String> {
    tok.encode(text)
        .unwrap()
        .into_iter()
        .map(|id| tok.decode(&[id]))
        .collect()
}

#[test]
fn merges_apply_lowest_rank_first() {
    let tok = load(&fixture()).unwrap();
    assert_eq!(pieces(&tok, "hello world"), ["hello", " world"]);
    // "b c" ranks before "a b", so it wins even though "a b" is leftmost.
    assert_eq!(pieces(&tok, "abc"), ["a", "bc"]);
}

#[test]
fn split_regex_replaces_the_gpt2_pattern() {
    let tok = load(&fixture()).unwrap();
    // Qwen splits digits one by one, so the "1 2" merge never applies. GPT-2 keeps "123" together.
    assert_eq!(pieces(&tok, "x123"), ["x", "1", "2", "3"]);
    let mut gpt2 = fixture();
    gpt2["pre_tokenizer"] = json!({"type": "ByteLevel", "add_prefix_space": false});
    assert_eq!(pieces(&load(&gpt2).unwrap(), "x123"), ["x", "12", "3"]);
}

#[test]
fn whitespace_lookahead_keeps_last_space_with_following_word() {
    let tok = load(&fixture()).unwrap();
    assert_eq!(pieces(&tok, "hello  world"), ["hello", " ", " world"]);
    // Trailing whitespace has nothing after it and stays one piece.
    assert_eq!(pieces(&tok, "hello  "), ["hello", "  "]);
}

#[test]
fn nfc_normalization_composes_before_byte_encoding() {
    let tok = load(&fixture()).unwrap();
    let decomposed = tok.encode("e\u{301}").unwrap();
    assert_eq!(tok.encode("\u{e9}").unwrap(), decomposed);
    assert_eq!(tok.decode(&decomposed), "\u{e9}");
}

#[test]
fn added_tokens_match_leftmost_longest() {
    let mut json = fixture();
    // "<|im" comes before the longer "<|im_end|>" in the list. Where both match, only
    // longest-match semantics pick "<|im_end|>".
    json["added_tokens"].as_array_mut().unwrap().insert(
        0,
        json!({"id": 999, "content": "<|im", "normalized": false}),
    );
    let tok = load(&json).unwrap();
    assert_eq!(
        pieces(&tok, "<|im_start|>hello<|im_end|><|im"),
        ["<|im_start|>", "hello", "<|im_end|>", "<|im"]
    );
}

#[test]
fn stream_decoder_waits_for_complete_characters() {
    let tok = load(&fixture()).unwrap();
    // "é" is two byte-level tokens (0xC3, 0xA9). The first token alone is not valid UTF-8.
    let ids = tok.encode("\u{e9}!").unwrap();
    assert_eq!(ids.len(), 3);
    let mut stream = StreamDecoder::new();
    assert_eq!(stream.push(&tok, ids[0]), "");
    assert_eq!(stream.push(&tok, ids[1]), "\u{e9}");
    assert_eq!(stream.push(&tok, ids[2]), "!");
    assert_eq!(stream.finish(), "");
}

#[test]
fn stream_decoder_replaces_invalid_bytes_and_flushes_partial_ones() {
    let tok = load(&fixture()).unwrap();
    let continuation = tok.encode("\u{e9}").unwrap()[1]; // 0xA9 alone is invalid UTF-8
    let lead = tok.encode("\u{e9}").unwrap()[0]; // 0xC3 starts a two-byte sequence
    let mut stream = StreamDecoder::new();
    assert_eq!(stream.push(&tok, continuation), "\u{fffd}");
    assert_eq!(stream.push(&tok, tok.encode("a").unwrap()[0]), "a");
    assert_eq!(stream.push(&tok, lead), "");
    assert_eq!(stream.finish(), "\u{fffd}");
}

#[test]
fn unsupported_options_are_rejected() {
    type Mutation = fn(&mut Value);
    let cases: [(Mutation, &str); 4] = [
        (|j| j["model"]["type"] = json!("WordPiece"), "WordPiece"),
        (
            |j| j["model"]["ignore_merges"] = json!(true),
            "ignore_merges",
        ),
        (|j| j["added_tokens"][0]["lstrip"] = json!(true), "lstrip"),
        (
            |j| j["pre_tokenizer"]["pretokenizers"][0]["invert"] = json!(true),
            "pre-tokenizer",
        ),
    ];
    for (mutate, expected) in cases {
        let mut json = fixture();
        mutate(&mut json);
        let err = TokenizerConfig::from_json_str(&json.to_string()).unwrap_err();
        assert!(format!("{err:#}").contains(expected), "{expected}: {err:#}");
    }
}
