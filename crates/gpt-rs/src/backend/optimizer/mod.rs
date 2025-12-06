//! Mode-less PTIR optimization infrastructure.
//!
//! This module implements the optimizer redesign described in `new_optimize.md`:
//! - a single context-aware pass trait,
//! - an implicit pipeline builder with bounded fixed points,
//! - graph-only plan inputs (role + stable id),
//! - param-only hoisting that materializes derived Params via a ParamResolver.

mod context;
mod entry;

use std::sync::Arc;

use crate::backend::pipeline::PipelineOptimizer;
use crate::backend::spec::{Function, PortableBackend};

pub use context::{OptimizeConfig, OptimizeContext, OptimizeServices};
pub use entry::{EntryParam, EntrySignature, PlanInputs};

/// Result returned by a [`FunctionPass`] after it runs.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct PassResult {
    /// Whether the pass changed the IR.
    pub changed: bool,
    /// Number of rewrite iterations executed while applying the pass.
    pub iterations: usize,
    /// Total number of patterns applied by the pass.
    pub rewrites_applied: usize,
    /// Instructions removed by the pass (typically via DCE).
    pub erased_insts: usize,
}

impl PassResult {
    /// Merges two run results, accumulating statistics.
    pub fn merge(self, other: PassResult) -> PassResult {
        PassResult {
            changed: self.changed || other.changed,
            iterations: self.iterations + other.iterations,
            rewrites_applied: self.rewrites_applied + other.rewrites_applied,
            erased_insts: self.erased_insts + other.erased_insts,
        }
    }
}

/// Canonical interface implemented by optimization passes that operate on a single function.
pub trait FunctionPass<B: PortableBackend + 'static>: Send + Sync {
    fn name(&self) -> &'static str;
    fn run(&self, function: &mut Function, cx: &mut OptimizeContext<B>) -> PassResult;
}

/// Trait implemented by PTIR optimizers invoked before caching programs.
pub trait Optimizer<B: PortableBackend + 'static>: Send + Sync {
    fn optimize(&self, function: &mut Function, cx: &mut OptimizeContext<B>) -> PassResult;
}

/// Builds the default optimizer pipeline, optionally extended with backend hooks.
pub fn default_optimizer<B: PortableBackend + 'static>(
    backend_pipeline: Option<Arc<dyn crate::backend::pipeline::BackendPipeline<B>>>,
) -> Arc<dyn Optimizer<B>> {
    Arc::new(PipelineOptimizer::new(backend_pipeline))
}
