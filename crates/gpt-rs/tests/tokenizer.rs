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
    Tokenizer::from_config(config)
}

#[test]
fn tokenizer_roundtrip() {
    let tokenizer = load_gpt2_tokenizer();

    let text = "Hello rust";
    let tokens = tokenizer.encode(text);
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
        "model": {
            "type": "BPE",
            "vocab": {"<unk>": 0, "H": 1, "i": 2},
            "merges": [["H", "i"]],
            "unk_token": null
        },
        "added_tokens": [
            {"id": 0, "content": "<unk>", "special": true}
        ]
    }
    "#;
    let cfg = TokenizerConfig::from_json_str(json).expect("hf schema should parse");
    assert_eq!(cfg.vocab.get("i"), Some(&2));
    assert_eq!(cfg.merges, vec![("H".to_string(), "i".to_string())]);
    assert_eq!(cfg.unk_token, "<unk>");
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
