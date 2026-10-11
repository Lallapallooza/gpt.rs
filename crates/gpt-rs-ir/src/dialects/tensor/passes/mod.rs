//! The passes of the `tensor` dialect: its canonical patterns, and one file per pass.

mod canonicalize;
mod dot_scale_fold;
mod merge_contracted_axes;

pub use self::dot_scale_fold::DotScaleFold;
pub use self::merge_contracted_axes::MergeContractedAxes;
