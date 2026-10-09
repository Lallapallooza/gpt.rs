//! Rewrites of `tensor` ops.

mod broadcast;
mod canonicalize;
mod contract;
mod dot_scale;
mod reduce;

pub use contract::{populate_merge_dot_axes_patterns, MergeContractedAxes};
pub use dot_scale::{populate_dot_scale_patterns, DotScaleFold};
