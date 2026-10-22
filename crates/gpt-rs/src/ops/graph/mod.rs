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
mod traced;

pub use self::arena::GraphArena;
pub(crate) use self::builder::GraphBuilder;
pub(crate) use self::state::NodeToken;
pub(crate) use self::traced::note_materialized;
pub use self::traced::{Traced, TracedFn, Traces, TracesFn};

/// The identifier of a value in a graph arena.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct ValueId(pub u32);
