use std::env;
use std::sync::OnceLock;

static GPTRS_EAGER: OnceLock<bool> = OnceLock::new();
static GPTRS_PASS_STATS: OnceLock<bool> = OnceLock::new();
static GPTRS_TRACE_CHECK: OnceLock<bool> = OnceLock::new();

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

pub(crate) fn pass_stats_enabled() -> bool {
    *GPTRS_PASS_STATS.get_or_init(|| match env::var("GPTRS_PASS_STATS") {
        Ok(value) if !value.trim().is_empty() => parse_bool(&value),
        _ => false,
    })
}

/// Whether a traced function records every call again, and fails when the graph it records is
/// not the one its trace replays.
pub(crate) fn trace_check_enabled() -> bool {
    *GPTRS_TRACE_CHECK.get_or_init(|| match env::var("GPTRS_TRACE_CHECK") {
        Ok(value) if !value.trim().is_empty() => parse_bool(&value),
        _ => false,
    })
}
