//! Pass infrastructure for PTIR backend optimizations.

mod broadcast_canonicalization;
mod cast_canonicalization;
mod cse;
mod dce;
mod elementwise_simplification;
mod param_only_fold_to_param;
mod reshape_canonicalization;
mod slice_canonicalization;
mod transpose_canonicalization;

pub use broadcast_canonicalization::{
    BroadcastCanonicalizationPass, CollapseBroadcastChain, EliminateIdentityBroadcast,
};
pub use cast_canonicalization::{CastCanonicalizationPass, EliminateRedundantCast};
pub use cse::CommonSubexpressionEliminationPass;
pub use dce::DeadCodeEliminationPass;
pub use elementwise_simplification::ElementwiseSimplificationPass;
pub use param_only_fold_to_param::ParamOnlyFoldToParamPass;
pub use reshape_canonicalization::{
    CollapseReshapeChain, EliminateIdentityReshape, ReshapeCanonicalizationPass,
};
pub use slice_canonicalization::SliceCanonicalizationPass;
pub use transpose_canonicalization::{
    CollapseTransposeChain, EliminateIdentityTranspose, TransposeCanonicalizationPass,
};

pub use crate::backend::optimizer::{FunctionPass, PassResult as FunctionPassResult};
