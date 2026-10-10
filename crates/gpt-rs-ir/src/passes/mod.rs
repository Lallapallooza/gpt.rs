#![doc = include_str!("README.md")]

mod canonicalize;
mod cse;
mod greedy;
mod outline_static;
mod prune_params;
mod registry;
mod stats;

use anyhow::{anyhow, Context, Result};
pub use gpt_rs_ir_macros::pass;

pub use self::canonicalize::{Canonicalize, Dce, Fold};
pub use self::cse::Cse;
pub use self::greedy::{Greedy, GreedyReport, Rewrites};
pub use self::outline_static::OutlineStatic;
pub use self::prune_params::PruneParams;
pub use self::registry::{parse, registered, OptionInfo, PassEntry, StatInfo, PASSES};
pub use self::stats::{take_stats, Stat};
pub use crate::dialects::tensor::passes::{
    populate_dot_scale_patterns, populate_merge_dot_axes_patterns, DotScaleFold,
    MergeContractedAxes,
};
use crate::module::Module;
use crate::verify::verify;

/// A transformation of a module. `#[pass]` on the struct implements `Registered`.
pub trait Pass: Registered {
    /// Runs the pass and returns whether the module changed.
    fn run(&self, module: &mut Module) -> Result<bool>;

    /// The folds and patterns the pass applies. A pass that runs the greedy driver runs these,
    /// and the reference pages list them under the pass.
    fn rewrites(&self) -> Rewrites {
        Rewrites::default()
    }
}

/// The name and options of a pass, which `#[pass]` implements.
pub trait Registered {
    fn name(&self) -> &'static str;

    /// The options that differ from their defaults, as `{name = value, ...}`, or empty.
    fn options(&self) -> String;

    /// The name, followed by the options that differ from their defaults.
    fn label(&self) -> String {
        match self.options() {
            options if options.is_empty() => self.name().to_string(),
            options => format!("{} {options}", self.name()),
        }
    }
}

/// Passes in the order they run.
#[derive(Default)]
pub struct Pipeline {
    passes: Vec<Box<dyn Pass>>,
}

impl Pipeline {
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds `pass` at the end.
    pub fn add(&mut self, pass: impl Pass + 'static) -> &mut Self {
        self.passes.push(Box::new(pass));
        self
    }

    /// The passes, in the order they run.
    pub fn passes(&self) -> &[Box<dyn Pass>] {
        &self.passes
    }

    /// Runs the passes in order, verifying the module after each, and returns whether any changed
    /// it.
    pub fn run(&self, module: &mut Module) -> Result<bool> {
        let mut changed = false;
        for pass in &self.passes {
            changed |= run_pass(pass.as_ref(), module)?;
        }
        Ok(changed)
    }
}

/// Runs `pass` and verifies the module after it. The statistics that `Stat::add` counts from here
/// on belong to `pass`, until `take_stats` reads them.
pub fn run_pass(pass: &dyn Pass, module: &mut Module) -> Result<bool> {
    take_stats();
    let changed = pass.run(module).with_context(|| pass.label())?;
    verify(module).map_err(|err| anyhow!("after {}: {err}", pass.label()))?;
    Ok(changed)
}

/// Adds `canonicalize` and `cse`: folds, algebraic identities, the dialects' canonical forms, and
/// the removal of dead and duplicate ops.
pub fn populate_simplification_passes(pm: &mut Pipeline) {
    pm.add(Canonicalize::default()).add(Cse);
}

/// Adds the `tensor` rewrites that change where work happens: `dot-scale-fold` moves the scales of
/// dot results into the weights, `canonicalize` simplifies what that exposes, and
/// `merge-contracted-axes` puts static dot operands in the layout the dot reads.
pub fn populate_tensor_passes(pm: &mut Pipeline) {
    pm.add(DotScaleFold)
        .add(Canonicalize::default())
        .add(MergeContractedAxes);
}

/// Adds `outline-static`, which moves the computation over parameters into load-time functions,
/// then the simplification passes on what remains. It goes after every pass whose work should
/// move to load time, such as a backend's passes that put parameters in its kernels' layouts.
pub fn populate_load_time_passes(pm: &mut Pipeline) {
    pm.add(OutlineStatic);
    populate_simplification_passes(pm);
}
