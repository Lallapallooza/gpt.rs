#![cfg(feature = "profiler")]

use std::time::Duration;

use gpt_rs::profiling;
use gpt_rs::profiling::{ProfileFormatOptions, WorkStats};

fn record_common_backend_metrics() {
    profiling::record_backend_aggregate(
        "backend.triton.exec.dispatch",
        1,
        Duration::from_millis(12),
        WorkStats::default(),
    );
    profiling::record_backend_aggregate(
        "backend.triton.exec.launches",
        24,
        Duration::ZERO,
        WorkStats::default(),
    );
    profiling::record_backend_aggregate(
        "backend.triton.kernel.total",
        24,
        Duration::from_millis(18),
        WorkStats::default(),
    );
    profiling::record_backend_aggregate(
        "backend.triton.kernel.host",
        24,
        Duration::from_millis(20),
        WorkStats::default(),
    );
    profiling::record_backend_aggregate(
        "backend.triton.kernel.gpu",
        24,
        Duration::from_millis(15),
        WorkStats::default(),
    );
    profiling::record_backend_aggregate(
        "backend.triton.cublas_sgemm",
        3,
        Duration::from_millis(7),
        WorkStats::default(),
    );
    profiling::cache_event("triton_backend.kernel_hit_mem");
    profiling::cache_event("triton_backend.kernel_hit_mem");
}

fn seed_many_backend_rows() {
    for idx in 0..40u32 {
        let signature = profiling::signature_id(&format!("sig-{idx}"));
        profiling::record_backend_aggregate_with_signature(
            "backend.triton.kernel.total",
            signature,
            1,
            Duration::from_micros(1_000 + u64::from(idx)),
            WorkStats::default(),
        );
    }
}

#[test]
fn format_includes_kpis_and_per_unit_columns() {
    profiling::reset();
    record_common_backend_metrics();

    let options = ProfileFormatOptions {
        measured_units: Some(8.0),
        unit_label: Some("token".to_string()),
        ..ProfileFormatOptions::default()
    };
    let formatted = profiling::take_formatted_tables_with_options(&options)
        .expect("expected formatted profiler report");

    assert!(formatted.contains("Execution KPIs"));
    assert!(formatted.contains("calls/unit"));
    assert!(formatted.contains("ms/unit"));
    assert!(formatted.contains("backend_ms/token"));
    assert!(formatted.contains("backend_calls/token"));
}

#[test]
fn compact_backend_view_can_be_expanded_with_profile_full() {
    profiling::reset();
    seed_many_backend_rows();
    let compact = profiling::take_formatted_tables_with_options(&ProfileFormatOptions {
        backend_min_percent: 100.0,
        ..ProfileFormatOptions::default()
    })
    .expect("expected compact profiler report");

    profiling::reset();
    seed_many_backend_rows();
    let full = profiling::take_formatted_tables_with_options(&ProfileFormatOptions {
        profile_full: true,
        ..ProfileFormatOptions::default()
    })
    .expect("expected full profiler report");

    let compact_rows = compact.matches("backend.triton.kernel.total [").count();
    let full_rows = full.matches("backend.triton.kernel.total [").count();
    assert!(compact_rows < full_rows);
}

#[test]
fn profiler_preserves_backend_event_names_as_emitted() {
    profiling::reset();
    profiling::record_backend_aggregate(
        "backend.unique.not_rewritten",
        1,
        Duration::from_millis(1),
        WorkStats::default(),
    );
    profiling::record_backend_aggregate(
        "backend.custom.metric",
        1,
        Duration::from_millis(1),
        WorkStats::default(),
    );

    let tables = profiling::take_tables().expect("expected profiler tables");
    let names = tables
        .backend
        .into_iter()
        .map(|row| row.name)
        .collect::<Vec<_>>();
    assert!(names
        .iter()
        .any(|name| name == "backend.unique.not_rewritten"));
    assert!(names.iter().any(|name| name == "backend.custom.metric"));
}
