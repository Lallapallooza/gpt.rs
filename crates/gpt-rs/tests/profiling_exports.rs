#![cfg(feature = "profiler")]

use std::time::Duration;

use gpt_rs::profiling;
use gpt_rs::profiling::{ProfileFormatOptions, WorkStats};

#[test]
fn profile_snapshot_writes_the_backend_row_per_unit() {
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
    let snapshot = profiling::take_snapshot(false, &options).expect("expected profile snapshot");

    let lines = snapshot
        .profile_jsonl
        .lines()
        .filter(|line| !line.trim().is_empty())
        .collect::<Vec<_>>();
    assert!(!lines.is_empty());
    let first: serde_json::Value = serde_json::from_str(lines[0]).expect("meta line must be json");
    assert_eq!(first["type"], "meta");
    assert_eq!(first["schema"], "gpt-rs.profile.v1");

    let row = lines
        .iter()
        .map(|line| serde_json::from_str::<serde_json::Value>(line).expect("json line"))
        .find(|value| value["type"] == "row" && value["name"] == "backend.test.metric")
        .expect("a row for the backend aggregate");
    // Three calls of 9 ms in all, over three tokens.
    assert_eq!(row["table"], "backend");
    assert_eq!(row["calls"], 3);
    for (field, value) in [
        ("ms_per_call", "3.000000"),
        ("self_ms", "9.000000"),
        ("total_ms", "9.000000"),
        ("pct_section", "100.000000"),
        ("pct_global", "100.000000"),
        ("calls_per_unit", "1.000000"),
        ("ms_per_unit", "3.000000"),
    ] {
        assert_eq!(row[field], value, "{field}");
    }
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
    assert_eq!(
        folded,
        "backend::outer 40\nbackend::outer;backend::inner 40\n"
    );

    let speedscope = profiling::chrome_trace_to_speedscope(trace).expect("speedscope output");
    let value: serde_json::Value = serde_json::from_str(&speedscope).expect("valid speedscope");
    assert_eq!(
        value["$schema"],
        "https://www.speedscope.app/file-format-schema.json"
    );
    // Frame 0 is `inner` and frame 1 `outer`. Each stack lists its frames from the root.
    let frames: Vec<&str> = value["shared"]["frames"]
        .as_array()
        .expect("frames")
        .iter()
        .map(|frame| frame["name"].as_str().expect("frame name"))
        .collect();
    assert_eq!(frames, ["backend::inner", "backend::outer"]);
    let profile = &value["profiles"][0];
    assert_eq!(profile["samples"], serde_json::json!([[1], [1, 0]]));
    assert_eq!(profile["weights"], serde_json::json!([40, 40]));
}
