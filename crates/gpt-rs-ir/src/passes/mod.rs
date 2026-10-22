#![doc = include_str!("README.md")]

mod canonicalize;
mod cse;
mod dce;
mod fold;
mod outline_static;
mod prune_params;
mod registry;
mod stats;
mod widen_narrow_floats;

use anyhow::{anyhow, Context, Result};
pub use gpt_rs_ir_macros::pass;

pub use self::canonicalize::Canonicalize;
pub use self::cse::Cse;
pub use self::dce::Dce;
pub use self::fold::Fold;
pub use self::outline_static::OutlineStatic;
pub use self::prune_params::PruneParams;
pub use self::registry::{parse, registered, OptionInfo, PassEntry, StatInfo, PASSES};
pub use self::stats::{take_stats, Stat};
pub use self::widen_narrow_floats::WidenNarrowFloats;
use crate::conversion::{LoopToIsa, TileToLoop};
use crate::dialects::loops::Contract;
use crate::dialects::tensor::DotScaleFold;
use crate::dialects::tile::Numerics;
use crate::ir::{verify, Module};
use crate::rewrite::{Greedy, Rewrites, Rewriting};
use crate::targets::Target;

/// Runs `rewrites` to a fixpoint with the greedy driver, erasing dead ops, adds the rewrites to
/// `stat`, and returns whether the module changed: the `run` of a pass that applies patterns.
pub fn run_patterns(module: &mut Rewriting, rewrites: Rewrites, stat: &Stat) -> Result<bool> {
    let report = Greedy::new(rewrites).erase_dead().run(module)?;
    stat.add(report.rewritten);
    Ok(report.changed())
}

/// A transformation of a module. `#[pass]` on the struct implements `Registered`.
pub trait Pass: Registered {
    /// Runs the pass on `module`, which it reads through `Deref` and changes only through the
    /// drivers, and returns whether the module changed.
    fn run(&self, module: &mut Rewriting) -> Result<bool>;

    /// The folds, patterns and once-rewrites the pass applies. A pass that runs the greedy driver
    /// runs these, and the reference pages list them under the pass.
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
    let changed = pass
        .run(&mut Rewriting::new(module, pass.name()))
        .with_context(|| pass.label())?;
    verify(module).map_err(|err| anyhow!("after {}: {err}", pass.label()))?;
    Ok(changed)
}

/// Adds `canonicalize` and `cse`: folds, algebraic identities, the dialects' canonical forms, and
/// the removal of dead and duplicate ops.
pub fn populate_simplification_passes(pm: &mut Pipeline) {
    pm.add(Canonicalize::default()).add(Cse);
}

/// Adds the `tensor` rewrites that change where work happens, as `numerics` allows them:
/// `dot-scale-fold` moves the scales of dot results into the weights (`scale_fold`), and
/// `canonicalize` simplifies what that exposes. A backend whose pipeline plans no layout of static
/// operands at the `tile` level adds `merge-contracted-axes` after them, which puts static dot
/// operands in the layout the dot reads.
pub fn populate_tensor_passes(pm: &mut Pipeline, numerics: &Numerics) {
    if numerics.scale_fold {
        pm.add(DotScaleFold);
    }
    pm.add(Canonicalize::default());
}

/// Adds the lowering of each dispatch through `loop` to `isa` for `target`, in this order:
///
/// 1. `tile-to-loop`;
/// 2. `canonicalize`, which simplifies the index arithmetic and the loops that `tile-to-loop`
///    writes, so that `loop-to-isa` can compute each atom in vectors;
/// 3. `contract`, which writes a product and the sum it enters as one `arith.fma` where the
///    dispatch contracts;
/// 4. `loop-to-isa`.
///
/// A backend that runs `isa` code lowers with it.
pub fn populate_isa_lowering_passes(pm: &mut Pipeline, target: Option<Target>) {
    pm.add(TileToLoop)
        .add(Canonicalize::default())
        .add(Contract)
        .add(LoopToIsa { target });
}

/// Adds `outline-static`, which moves the computation over parameters into load-time functions,
/// then the simplification passes on what remains. It goes after every pass whose work should
/// move to load time, such as a backend's passes that put parameters in its kernels' layouts.
pub fn populate_load_time_passes(pm: &mut Pipeline) {
    pm.add(OutlineStatic);
    populate_simplification_passes(pm);
}
