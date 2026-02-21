#![cfg(feature = "profiler")]

use std::thread;
use std::time::Duration;

use gpt_rs::profiling;

fn sleep_for(duration: Duration) {
    thread::sleep(duration);
}

#[test]
fn nested_layer_scopes_accumulate_expected_timings() {
    profiling::reset();

    let label = "nested-layer";
    let outer_before = Duration::from_millis(15);
    let inner_duration = Duration::from_millis(30);
    let outer_after = Duration::from_millis(20);

    {
        let _outer = profiling::layer_scope(label);
        sleep_for(outer_before);
        {
            let _inner = profiling::layer_scope(label);
            sleep_for(inner_duration);
        }
        sleep_for(outer_after);
    }

    let summary = profiling::take_tables().expect("expected profiler tables");
    let layer_row = summary
        .layers
        .iter()
        .find(|row| row.name == label)
        .expect("expected layer row for nested guard");

    assert_eq!(layer_row.calls, 2);

    let expected_exclusive_ns = (outer_before + inner_duration + outer_after).as_nanos();
    let expected_inclusive_ns =
        (outer_before + outer_after + inner_duration + inner_duration).as_nanos();
    let tolerance_ns = Duration::from_millis(15).as_nanos();
    let observed_exclusive_ns = (layer_row.excl_ms * 1_000_000.0) as u128;
    let observed_inclusive_ns = (layer_row.incl_ms * 1_000_000.0) as u128;

    assert!(
        observed_exclusive_ns >= expected_exclusive_ns.saturating_sub(tolerance_ns),
        "expected exclusive_ns >= {}ns, got {}",
        expected_exclusive_ns.saturating_sub(tolerance_ns),
        observed_exclusive_ns
    );
    assert!(
        observed_exclusive_ns <= expected_exclusive_ns.saturating_add(tolerance_ns),
        "expected exclusive_ns <= {}ns, got {}",
        expected_exclusive_ns.saturating_add(tolerance_ns),
        observed_exclusive_ns
    );
    assert!(
        observed_inclusive_ns >= expected_inclusive_ns.saturating_sub(tolerance_ns),
        "expected inclusive_ns >= {}ns, got {}",
        expected_inclusive_ns.saturating_sub(tolerance_ns),
        observed_inclusive_ns
    );
    assert!(
        observed_inclusive_ns <= expected_inclusive_ns.saturating_add(tolerance_ns),
        "expected inclusive_ns <= {}ns, got {}",
        expected_inclusive_ns.saturating_add(tolerance_ns),
        observed_inclusive_ns
    );

    assert!((layer_row.percent - 100.0).abs() < 1e-6);
    assert!((layer_row.percent_global - 100.0).abs() < 1e-6);
    assert!(layer_row.incl_ms >= layer_row.excl_ms);
}
