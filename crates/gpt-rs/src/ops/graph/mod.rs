//! Lazy graph infrastructure for composing backend programs on demand.
//!
//! The graph layer records tensor operations in arena-backed graphs, allowing higher-level
//! functionals to stitch complex programs without eagerly materialising every intermediate.
//! Builders import tensors, emit operations, and the arena flushes pending nodes when material
//! results are required.
mod arena;
mod builder;
pub mod context;
mod plan;
mod state;
pub mod timing;

pub use arena::{CachePolicy, CompiledGraph, GraphArena};
pub use builder::GraphBuilder;
