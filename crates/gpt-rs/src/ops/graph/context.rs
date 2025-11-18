//! Thread-local default arena context for implicit graph capture.
//!
//! This module provides a thread-local stack of graph arenas that allows functional operators
//! to implicitly capture operations into a shared graph without explicit arena passing.
//!
//! ## Usage Pattern
//!
//! The `capture_ptir!` macro can automatically resolve a graph arena from:
//! 1. Explicit `graph = expr` parameter (highest priority)
//! 2. Graph attached to input tensors
//! 3. Thread-local default arena (via this module)
//! 4. Create a new arena (lowest priority)
//!
//! ## Thread-Local Stack
//!
//! The arena stack is thread-local and supports nesting. When multiple arenas are pushed,
//! `current_arena()` returns the most recently pushed one that matches the requested backend type.
//!
//! This enables patterns like:
//! ```rust,ignore
//! let arena = GraphArena::new(backend);
//! with_default_arena(arena, || {
//!     // All operations in this scope implicitly use `arena`
//!     let result = some_functional_op(x, y)?;
//!     Ok(result)
//! })
//! ```

use std::any::Any;
use std::cell::RefCell;
use std::sync::Arc;

use crate::backend::spec::PortableBackend;

use super::GraphArena;

thread_local! {
    static ARENA_STACK: RefCell<Vec<Arc<dyn Any + Send + Sync>>> = RefCell::new(Vec::new());
}

/// RAII guard that restores the previous default arena when dropped.
///
/// Created by [`push_default_arena`] and [`with_default_arena`]. Ensures the arena
/// stack is properly unwound even if the guarded scope panics.
pub struct ArenaGuard {
    active: bool,
}

impl Drop for ArenaGuard {
    fn drop(&mut self) {
        if self.active {
            ARENA_STACK.with(|stack| {
                stack.borrow_mut().pop();
            });
            self.active = false;
        }
    }
}

/// Pushes a default arena for the current thread and returns a guard that restores it on drop.
pub fn push_default_arena<B: PortableBackend + 'static>(arena: Arc<GraphArena<B>>) -> ArenaGuard {
    ARENA_STACK.with(|stack| {
        stack.borrow_mut().push(arena as Arc<dyn Any + Send + Sync>);
    });
    ArenaGuard { active: true }
}

/// Runs `f` with the provided arena installed as the current default.
pub fn with_default_arena<B, F, R>(arena: Arc<GraphArena<B>>, f: F) -> R
where
    B: PortableBackend + 'static,
    F: FnOnce() -> R,
{
    let guard = push_default_arena(arena);
    let result = f();
    drop(guard);
    result
}

/// Retrieves the currently installed default arena, if its backend matches `B`.
pub fn current_arena<B: PortableBackend + 'static>() -> Option<Arc<GraphArena<B>>> {
    ARENA_STACK.with(|stack| {
        let stack = stack.borrow();
        for entry in stack.iter().rev() {
            if let Ok(arena) = entry.clone().downcast::<GraphArena<B>>() {
                return Some(arena);
            }
        }
        None
    })
}
