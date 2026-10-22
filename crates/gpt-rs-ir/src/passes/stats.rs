//! The counters that passes declare with `#[pass(stats(...))]`.

use std::cell::RefCell;

thread_local! {
    /// The counters of the pass running on this thread, in the order they were first added.
    static COUNTS: RefCell<Vec<(&'static str, u64)>> = const { RefCell::new(Vec::new()) };
}

/// A counter of a pass. `#[pass(stats(name = "doc"))]` declares it as a constant of the pass.
pub struct Stat {
    name: &'static str,
}

impl Stat {
    pub const fn new(name: &'static str) -> Self {
        Self { name }
    }

    /// Adds `amount` to this counter of the pass running on this thread.
    pub fn add(&self, amount: usize) {
        if amount == 0 {
            return;
        }
        COUNTS.with_borrow_mut(|counts| {
            match counts.iter_mut().find(|(name, _)| *name == self.name) {
                Some((_, count)) => *count += amount as u64,
                None => counts.push((self.name, amount as u64)),
            }
        });
    }
}

/// The counters added since the last call, and resets them.
pub fn take_stats() -> Vec<(&'static str, u64)> {
    COUNTS.with_borrow_mut(std::mem::take)
}
