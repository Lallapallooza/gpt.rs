#![cfg(feature = "profiler")]

use std::sync::Mutex;
use std::time::Duration;

use gpt_rs::profiling;
use gpt_rs::profiling::{ProfileFormatOptions, WorkStats};

/// The profiler is global, so the tests take turns.
static PROFILER: Mutex<()> = Mutex::new(());

fn record_common_backend_metrics() {
    profiling::record_backend_aggregate(
        "backend.elementwise_fused",
        1,
        Duration::from_millis(12),
        WorkStats::default(),
    );
    profiling::record_backend_aggregate(
        "backend.reshape",
        24,
        Duration::ZERO,
        WorkStats::default(),
    );
    profiling::record_backend_aggregate(
        "backend.dot_general",
        24,
        Duration::from_millis(18),
        WorkStats::default(),
    );
    profiling::record_backend_aggregate(
        "backend.reduce.sum",
        24,
        Duration::from_millis(20),
        WorkStats::default(),
    );
    profiling::record_backend_aggregate(
        "backend.transpose",
        24,
        Duration::from_millis(15),
        WorkStats::default(),
    );
    profiling::record_backend_aggregate(
        "backend.take",
        3,
        Duration::from_millis(7),
        WorkStats::default(),
    );
}

fn seed_many_backend_rows() {
    for idx in 0..40u32 {
        let signature = profiling::signature_id(&format!("sig-{idx}"));
        profiling::record_backend_aggregate_with_signature(
            "backend.dot_general",
            signature,
            1,
            Duration::from_micros(1_000 + u64::from(idx)),
            WorkStats::default(),
        );
    }
}

#[test]
fn kpis_divide_the_backend_totals_by_the_measured_units() {
    let _profiler = PROFILER.lock().unwrap();
    profiling::reset();
    record_common_backend_metrics();

    let options = ProfileFormatOptions {
        measured_units: Some(8.0),
        unit_label: Some("token".to_string()),
        ..ProfileFormatOptions::default()
    };
    let snapshot = profiling::take_snapshot(false, &options).expect("expected a profile snapshot");
    let kpi = |name: &str| {
        snapshot
            .profile_jsonl
            .lines()
            .map(|line| serde_json::from_str::<serde_json::Value>(line).unwrap())
            .find(|line| line["type"] == "kpi" && line["name"] == name)
            .map(|line| line["value"].clone())
    };
    // 72 ms and 100 calls over 8 tokens.
    assert_eq!(kpi("backend_ms/token"), Some("9.000000".into()));
    assert_eq!(kpi("backend_calls/token"), Some("12.500000".into()));
}

#[test]
fn compact_backend_view_can_be_expanded_with_profile_full() {
    let _profiler = PROFILER.lock().unwrap();
    profiling::reset();
    seed_many_backend_rows();
    let compact = profiling::take_snapshot(
        false,
        &ProfileFormatOptions {
            backend_min_percent: 100.0,
            ..ProfileFormatOptions::default()
        },
    )
    .expect("expected compact profiler report")
    .formatted;

    profiling::reset();
    seed_many_backend_rows();
    let full = profiling::take_snapshot(
        false,
        &ProfileFormatOptions {
            profile_full: true,
            ..ProfileFormatOptions::default()
        },
    )
    .expect("expected full profiler report")
    .formatted;

    // The compact table keeps the `backend_top_n` (24) slowest of the 40 rows.
    assert_eq!(compact.matches("backend.dot_general [").count(), 24);
    assert_eq!(full.matches("backend.dot_general [").count(), 40);
}
