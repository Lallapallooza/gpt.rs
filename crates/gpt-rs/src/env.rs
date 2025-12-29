use std::env;
use std::sync::OnceLock;

static GPTRS_EAGER: OnceLock<bool> = OnceLock::new();
static GPTRS_OPT_PRE_ITERS: OnceLock<usize> = OnceLock::new();
static GPTRS_OPT_POST_ITERS: OnceLock<usize> = OnceLock::new();
static GPTRS_PASS_STATS: OnceLock<bool> = OnceLock::new();

fn parse_bool(value: &str) -> bool {
    let normalized = value.trim().to_ascii_lowercase();
    matches!(normalized.as_str(), "1" | "true" | "yes" | "on")
}

fn parse_usize(value: &str) -> Option<usize> {
    value.trim().parse::<usize>().ok()
}

pub(crate) fn eager_enabled() -> bool {
    *GPTRS_EAGER.get_or_init(|| match env::var("GPTRS_EAGER") {
        Ok(value) if !value.trim().is_empty() => parse_bool(&value),
        _ => false,
    })
}

pub(crate) fn optimizer_pre_iters() -> usize {
    *GPTRS_OPT_PRE_ITERS.get_or_init(|| {
        env::var("GPTRS_OPT_PRE_ITERS")
            .ok()
            .and_then(|v| parse_usize(&v))
            .unwrap_or(2)
    })
}

pub(crate) fn optimizer_post_iters() -> usize {
    *GPTRS_OPT_POST_ITERS.get_or_init(|| {
        env::var("GPTRS_OPT_POST_ITERS")
            .ok()
            .and_then(|v| parse_usize(&v))
            .unwrap_or(4)
    })
}

pub(crate) fn pass_stats_enabled() -> bool {
    *GPTRS_PASS_STATS.get_or_init(|| match env::var("GPTRS_PASS_STATS") {
        Ok(value) if !value.trim().is_empty() => parse_bool(&value),
        _ => false,
    })
}
