//! The type rules of the `tile` ops: one `OpDef` impl per op.

mod block;
mod dispatch;
mod schedule;

pub(crate) use self::dispatch::check_placements;
pub(crate) use self::schedule::{check_lanes, check_schedule};
