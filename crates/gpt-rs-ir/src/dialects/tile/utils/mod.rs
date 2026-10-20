//! Helpers of the `tile` ops: building blocks, and reading and rewriting their attributes. One file
//! per section of `ir.rs`, and `build.rs` holds the block builders.

mod block;
pub(crate) mod build;
mod combine;
mod dispatch;
mod schedule;

pub(crate) use self::block::in_function_outside_groups;
pub(super) use self::block::SourceText;
pub use self::block::{PadSkip, UnscheduledBlock, VectorWork};
pub use self::combine::{CombineFold, Ops, Scalars, SlotBody, WhenBody};
pub use self::schedule::{
    AtomPlan, Clips, CopyAt, CopyAxis, Loop, Parallelism, PrefetchAt, RegDim, StatePlace, Step,
    Tail, Walk,
};
