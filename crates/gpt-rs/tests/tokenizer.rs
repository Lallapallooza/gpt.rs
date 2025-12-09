use std::fs;
use std::path::Path;

use gpt_rs::tokenizer::{Tokenizer, TokenizerConfig};

#[test]
fn tokenizer_roundtrip() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("configs")
        .join("gpt2_tokenizer.json");
    let data = fs::read_to_string(path).expect("failed to read tokenizer config");
    let config: TokenizerConfig = serde_json::from_str(&data).expect("invalid tokenizer config");
    let tokenizer = Tokenizer::from_config(config);

    let text = "Hello rust";
    let tokens = tokenizer.encode(text);
    let decoded = tokenizer.decode(&tokens);

    assert_eq!(decoded, text);
    assert!(!tokens.is_empty());
}
