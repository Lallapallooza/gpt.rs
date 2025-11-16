use std::env;
use std::sync::OnceLock;

static GPTRS_EAGER: OnceLock<bool> = OnceLock::new();

fn parse_bool(value: &str) -> bool {
    let normalized = value.trim().to_ascii_lowercase();
    matches!(normalized.as_str(), "1" | "true" | "yes" | "on")
}

pub(crate) fn eager_enabled() -> bool {
    *GPTRS_EAGER.get_or_init(|| match env::var("GPTRS_EAGER") {
        Ok(value) if !value.trim().is_empty() => parse_bool(&value),
        _ => false,
    })
}
