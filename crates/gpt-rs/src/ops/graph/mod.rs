//! Lazy graph infrastructure for composing backend programs on demand.
//!
//! The graph layer records tensor operations in arena-backed graphs, allowing higher-level
//! functionals to stitch complex programs without eagerly materializing every intermediate.
//! Builders import tensors, emit operations, and the arena flushes pending nodes when material
//! results are required.
mod arena;
mod builder;
pub mod context;
mod plan;
mod state;

pub use self::arena::GraphArena;
pub(crate) use self::builder::GraphBuilder;

/// The identifier of a value in a graph arena.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct ValueId(pub u32);
