use std::cell::Cell;
use std::time::Duration;

thread_local! {
    static COMPILE_TIME_NS: Cell<u64> = const { Cell::new(0) };
}

pub fn reset_compile_time() {
    COMPILE_TIME_NS.with(|cell| cell.set(0));
}

pub fn add_compile_time(duration: Duration) {
    let nanos = duration.as_nanos();
    let nanos = nanos.min(u128::from(u64::MAX)) as u64;
    COMPILE_TIME_NS.with(|cell| {
        let current = cell.get();
        let next = current.saturating_add(nanos);
        cell.set(next);
    });
}

pub fn take_compile_time() -> Duration {
    COMPILE_TIME_NS.with(|cell| {
        let nanos = cell.get();
        cell.set(0);
        Duration::from_nanos(nanos)
    })
}
