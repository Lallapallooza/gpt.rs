#![cfg(feature = "profiler")]

use std::thread;
use std::time::{Duration, Instant};

use gpt_rs::profiling;

#[test]
fn nested_layer_scopes_count_the_inner_scope_once_in_exclusive_time() {
    profiling::reset();

    let label = "nested-layer";
    let sleep = Duration::from_millis(5);
    let outer_start = Instant::now();
    let inner_wall;
    {
        let _outer = profiling::layer_scope(label);
        thread::sleep(sleep);
        let inner_start = Instant::now();
        {
            let _inner = profiling::layer_scope(label);
            thread::sleep(sleep);
        }
        inner_wall = inner_start.elapsed();
        thread::sleep(sleep);
    }
    let outer_wall = outer_start.elapsed();

    let report = profiling::take_report().expect("expected a profiler report");
    let row = report
        .sections
        .iter()
        .flat_map(|section| &section.tables.layers)
        .find(|row| row.name == label)
        .expect("expected layer row for nested guard");
    assert_eq!(row.calls, 2);

    // Each scope's time lies within the wall time measured around it, and holds what ran inside
    // it. Exclusive time counts every moment once, so it is the outer scope's time. Inclusive time
    // adds the inner scope's time again. A microsecond covers the conversion from milliseconds.
    let slack = 1_000;
    let excl = (row.excl_ms * 1e6) as u128;
    let inner = ((row.incl_ms - row.excl_ms) * 1e6) as u128;
    assert!(
        excl <= outer_wall.as_nanos() + slack
            && excl + slack >= (sleep * 2 + inner_wall).as_nanos(),
        "exclusive {excl} ns outside [{:?}, {outer_wall:?}]",
        sleep * 2 + inner_wall
    );
    assert!(
        inner <= inner_wall.as_nanos() + slack && inner + slack >= sleep.as_nanos(),
        "inner {inner} ns outside [{sleep:?}, {inner_wall:?}]"
    );
    assert!((row.percent - 100.0).abs() < 1e-6);
    assert!((row.percent_global - 100.0).abs() < 1e-6);
}
