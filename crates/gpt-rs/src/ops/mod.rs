//! The functionals, the graph arena that records their ops and compiles them into backend
//! programs, and the trace sinks that dumps write through. Everything here is backend-agnostic.
//! Backends speed up the recorded ops when they compile them.
pub mod functional;
pub mod graph;
pub mod trace;

pub use functional::*;
