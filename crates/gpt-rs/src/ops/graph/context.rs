//! Thread-local default arena, the graph that tensor ops record into.
//!
//! This module provides a thread-local stack of graph arenas, so tensor ops record into one shared
//! graph without passing the arena around.
//!
//! ## Usage Pattern
//!
//! Every op of [`DeviceTensor`](crate::DeviceTensor) records into the default arena when one is
//! installed for its backend.
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
//!     // Every op in this scope records into `arena`.
//!     functional::gelu(&(&x + &y))
//! })
//! ```

use std::any::Any;
use std::cell::RefCell;
use std::sync::Arc;

use super::GraphArena;
use crate::backend::PortableBackend;

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

/// Runs `body` with the provided arena installed as the current default.
pub fn with_default_arena<B, F, R>(arena: Arc<GraphArena<B>>, body: F) -> R
where
    B: PortableBackend + 'static,
    F: FnOnce() -> R,
{
    let guard = push_default_arena(arena);
    let result = body();
    drop(guard);
    result
}

/// Retrieves the currently installed default arena, if its backend matches `B`.
pub fn current_arena<B: PortableBackend + 'static>() -> Option<Arc<GraphArena<B>>> {
    ARENA_STACK.with(|arenas| {
        let stack = arenas.borrow();
        for entry in stack.iter().rev() {
            if let Ok(arena) = entry.clone().downcast::<GraphArena<B>>() {
                return Some(arena);
            }
        }
        None
    })
}
