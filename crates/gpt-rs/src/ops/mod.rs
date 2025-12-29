//! High-level operations layered on top of the portable tensor and backend abstractions.
//!
//! The `ops` tree houses functional kernels and the lazy graph
//! machinery used to lower composite operations into backend programs. Modules are designed to
//! stay backend-agnostic while providing enough hooks for custom kernels and targeted overrides.
pub mod functional;
pub mod graph;
pub mod ptir;
pub mod trace;

pub use functional::*;
