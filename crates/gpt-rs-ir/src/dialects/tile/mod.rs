#![doc = include_str!("README.md")]

mod cost;
mod interp;
mod ir;
mod passes;
mod utils;
mod verify;

pub use self::interp::Acc;
pub use self::ir::*;
pub use self::passes::{
    FormGroups, InlineStaticMaps, NameTarget, PadWithIdentity, PlaceEpilogues, PlanLayouts,
    Schedule, SealDispatches, SelectAtoms, SplitLaneFolds, WalkOrderLayouts, WidenReusedMaps,
    WinogradConv,
};
pub use self::utils::build::{copy_body_ops, par_iters, push_iter, Inputs};
pub use self::utils::{
    AtomPlan, Clips, CombineFold, CopyAt, CopyAxis, Loop, Ops, PadSkip, Parallelism, PrefetchAt,
    RegDim, Scalars, SlotBody, StatePlace, Step, Tail, UnscheduledBlock, VectorWork, Walk,
    WhenBody,
};
