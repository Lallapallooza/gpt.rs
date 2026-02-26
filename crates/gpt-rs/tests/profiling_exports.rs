#![cfg(feature = "profiler")]

use std::time::Duration;

use gpt_rs::profiling;
use gpt_rs::profiling::{ProfileFormatOptions, WorkStats};

#[test]
fn profile_snapshot_includes_deterministic_jsonl_lines() {
    profiling::reset();
    profiling::record_backend_aggregate(
        "backend.test.metric",
        3,
        Duration::from_millis(9),
        WorkStats::default(),
    );
    profiling::cache_event("cache.hit");

    let options = ProfileFormatOptions {
        measured_units: Some(3.0),
        unit_label: Some("token".to_string()),
        ..ProfileFormatOptions::default()
    };
    let snapshot = profiling::take_profile_snapshot_with_options(false, &options)
        .expect("expected profile snapshot");

    let lines = snapshot
        .profile_jsonl
        .lines()
        .filter(|line| !line.trim().is_empty())
        .collect::<Vec<_>>();
    assert!(!lines.is_empty());
    let first: serde_json::Value = serde_json::from_str(lines[0]).expect("meta line must be json");
    assert_eq!(first["type"], "meta");
    assert_eq!(first["schema"], "gpt-rs.profile.v1");

    let has_backend_row = lines.iter().any(|line| {
        let value: serde_json::Value = serde_json::from_str(line).expect("json line");
        value["type"] == "row"
            && value["table"] == "backend"
            && value["name"] == "backend.test.metric"
    });
    assert!(has_backend_row);

    let has_unit_kpi = lines.iter().any(|line| {
        let value: serde_json::Value = serde_json::from_str(line).expect("json line");
        value["type"] == "kpi" && value["name"] == "backend_ms/token"
    });
    assert!(has_unit_kpi);
}

#[test]
fn chrome_trace_converts_to_folded_and_speedscope() {
    let trace = r#"{
      "traceEvents":[
        {"name":"outer","cat":"backend","ph":"X","ts":0,"dur":100,"pid":1,"tid":1,"args":{"excl_us":40}},
        {"name":"inner","cat":"backend","ph":"X","ts":20,"dur":40,"pid":1,"tid":1,"args":{"excl_us":40}}
      ],
      "displayTimeUnit":"ms",
      "otherData":{"sections":[],"signatures":[]}
    }"#;

    let folded = profiling::chrome_trace_to_folded(trace).expect("folded output");
    assert!(folded.contains("backend::outer 40"));
    assert!(folded.contains("backend::outer;backend::inner 40"));

    let speedscope = profiling::chrome_trace_to_speedscope(trace).expect("speedscope output");
    let value: serde_json::Value = serde_json::from_str(&speedscope).expect("valid speedscope");
    assert_eq!(
        value["$schema"],
        "https://www.speedscope.app/file-format-schema.json"
    );
    assert!(value["profiles"][0]["samples"]
        .as_array()
        .is_some_and(|samples| !samples.is_empty()));
}
