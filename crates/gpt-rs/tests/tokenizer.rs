use std::fs;
use std::path::Path;

use gpt_rs::tokenizer::{Tokenizer, TokenizerConfig};

fn load_gpt2_tokenizer() -> Tokenizer {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("configs")
        .join("gpt2_tokenizer.json");
    let data = fs::read_to_string(path).expect("failed to read tokenizer config");
    let config = TokenizerConfig::from_json_str(&data).expect("invalid tokenizer config");
    Tokenizer::from_config(config).expect("valid tokenizer config")
}

#[test]
fn tokenizer_roundtrip() {
    let tokenizer = load_gpt2_tokenizer();

    let text = "Hello rust";
    let tokens = tokenizer.encode(text).expect("encode");
    let decoded = tokenizer.decode(&tokens);

    assert_eq!(decoded, text);
    assert!(!tokens.is_empty());
}

#[test]
fn tokenizer_config_parses_flat_schema() {
    let json = r#"
    {
        "vocab": {"<unk>": 0, "H": 1, "i": 2},
        "merges": [["H", "i"]],
        "unk_token": "<unk>"
    }
    "#;
    let cfg = TokenizerConfig::from_json_str(json).expect("flat schema should parse");
    assert_eq!(cfg.vocab.get("H"), Some(&1));
    assert_eq!(cfg.merges, vec![("H".to_string(), "i".to_string())]);
    assert_eq!(cfg.unk_token, "<unk>");
}

#[test]
fn tokenizer_config_parses_hf_tokenizer_json_schema() {
    let json = r#"
    {
        "added_tokens": [
            {"id": 3, "content": "<|endoftext|>", "single_word": false, "lstrip": false,
             "rstrip": false, "normalized": true, "special": true}
        ],
        "normalizer": null,
        "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true,
                          "use_regex": true},
        "post_processor": {"type": "ByteLevel", "add_prefix_space": true, "trim_offsets": false,
                           "use_regex": true},
        "decoder": {"type": "ByteLevel", "add_prefix_space": true, "trim_offsets": true,
                    "use_regex": true},
        "model": {"dropout": null, "unk_token": null, "continuing_subword_prefix": "",
                  "end_of_word_suffix": "", "fuse_unk": false,
                  "vocab": {"H": 0, "i": 1, "Hi": 2}, "merges": ["H i"]}
    }
    "#;
    let cfg = TokenizerConfig::from_json_str(json).expect("hf schema should parse");
    assert_eq!(cfg.merges, vec![("H".to_string(), "i".to_string())]);
    let tokenizer = Tokenizer::from_config(cfg).unwrap();
    // `normalized` has no effect without a normalizer, so the added token still matches.
    assert_eq!(tokenizer.encode("Hi<|endoftext|>").unwrap(), vec![2, 3]);
}

#[test]
fn gpt2_encode_matches_python() {
    let tokenizer = load_gpt2_tokenizer();
    assert_eq!(tokenizer.encode("Hello world").unwrap(), vec![15496, 995]);
    assert_eq!(tokenizer.encode(" Hello world").unwrap(), vec![18435, 995]);
    // Whitespace runs leave their last space to the following word (`\s+(?!\S)`).
    assert_eq!(
        tokenizer.encode("Hello  world").unwrap(),
        vec![15496, 220, 995]
    );
    assert_eq!(
        tokenizer.encode("a   b\n\nc  ").unwrap(),
        vec![64, 220, 220, 275, 198, 198, 66, 220, 220]
    );
    assert_eq!(
        tokenizer.encode("it's  2026!").unwrap(),
        vec![270, 338, 220, 1160, 2075, 0]
    );
}

#[test]
fn gpt2_decode_matches_python() {
    let tokenizer = load_gpt2_tokenizer();
    let hello_world = vec![15496usize, 995];
    assert_eq!(tokenizer.decode(&hello_world), "Hello world");

    let leading_space = vec![18435usize, 995];
    assert_eq!(tokenizer.decode(&leading_space), " Hello world");
}
