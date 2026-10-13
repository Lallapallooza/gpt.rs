//! Conversions between dialects: a [`ConversionTarget`] says which ops are legal, patterns
//! rewrite the illegal ones, and [`apply_conversion`] runs them until no illegal op remains.
//!
//! A conversion that covers a whole dialect decides every op of it through [`ConvertOp`]: legal,
//! or illegal with the pattern that rewrites it. The dialect's `convert` (from
//! `#[derive(Dialect)]`) adds every decision to the target and the pattern set, and it requires
//! one per op, so an op without a decision does not compile. A conversion pattern that does not
//! apply says why, and the conversion fails with that reason.

mod tensor_to_tile;
mod tile_to_loop;

use anyhow::{bail, Result};

pub use self::tensor_to_tile::TensorToTile;
pub use self::tile_to_loop::TileToLoop;
use crate::dialects::Dialect;
use crate::ir::{InOp, Module};
use crate::rewrite::{AnyPattern, Applied, PatternSet};
use crate::Op;

/// What a conversion does with one op.
pub enum OpForm {
    /// The op stays: it is legal after the conversion.
    Legal,
    /// The op is illegal, and this pattern rewrites it into legal ops.
    Rewrite(Box<dyn AnyPattern>),
    /// The op is illegal, and nothing rewrites it: the conversion fails on it, for this reason.
    Illegal(&'static str),
}

impl OpForm {
    /// The op is illegal, and `pattern` rewrites it.
    pub fn rewrite(pattern: impl AnyPattern + 'static) -> Self {
        OpForm::Rewrite(Box::new(pattern))
    }
}

/// The decision of the conversion `C` for the op `Self`.
pub trait ConvertOp<C>: InOp {
    fn form() -> OpForm;
}

/// Whether an op may remain after a conversion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Legality {
    Legal,
    /// The op must be rewritten. With a reason, nothing can rewrite it.
    Illegal(Option<&'static str>),
}

/// Whether an op is the op of one struct.
type IsOp = fn(&Op) -> bool;

/// Which ops are legal after a conversion: by op, else by dialect. An op that neither names is
/// unknown.
#[derive(Default)]
pub struct ConversionTarget {
    ops: Vec<(IsOp, Legality)>,
    dialects: Vec<(Dialect, Legality)>,
}

impl ConversionTarget {
    pub fn new() -> Self {
        Self::default()
    }

    /// Makes every op of `dialect` legal, unless an op decision says otherwise.
    pub fn legal_dialect(&mut self, dialect: Dialect) -> &mut Self {
        self.dialects.push((dialect, Legality::Legal));
        self
    }

    /// Makes every op of `dialect` illegal, unless an op decision says otherwise.
    pub fn illegal_dialect(&mut self, dialect: Dialect) -> &mut Self {
        self.dialects.push((dialect, Legality::Illegal(None)));
        self
    }

    /// Makes the op `T` legal.
    pub fn legal_op<T: InOp>(&mut self) -> &mut Self {
        self.ops.push((|op| T::cast(op).is_some(), Legality::Legal));
        self
    }

    /// Makes the op `T` illegal.
    pub fn illegal_op<T: InOp>(&mut self) -> &mut Self {
        self.ops
            .push((|op| T::cast(op).is_some(), Legality::Illegal(None)));
        self
    }

    /// Adds the decision `form` for the op `T`: legal, or illegal with its pattern added to
    /// `patterns`.
    pub fn decide<T: InOp>(&mut self, form: OpForm, patterns: &mut PatternSet) {
        match form {
            OpForm::Legal => {
                self.legal_op::<T>();
            }
            OpForm::Rewrite(pattern) => {
                self.illegal_op::<T>();
                patterns.add_boxed(pattern);
            }
            OpForm::Illegal(reason) => {
                self.ops
                    .push((|op| T::cast(op).is_some(), Legality::Illegal(Some(reason))));
            }
        }
    }

    /// A target where the ops of `dialects` are legal, and no other.
    pub fn of_dialects(dialects: &[Dialect]) -> Self {
        let mut target = Self::new();
        for &dialect in dialects {
            target.legal_dialect(dialect);
        }
        target
    }

    /// Fails on the first op of `module` that the target does not make legal.
    pub fn check(&self, module: &Module) -> Result<()> {
        for inst in module.walk(module.body()) {
            let op = module.op(inst);
            if self.legality(op) != Some(Legality::Legal) {
                bail!(
                    "{}: the {:?} dialect is not legal here",
                    op.name(),
                    op.dialect()
                );
            }
        }
        Ok(())
    }

    /// The legality of `op`, or `None` when the target does not say.
    pub fn legality(&self, op: &Op) -> Option<Legality> {
        if let Some(&(_, legality)) = self.ops.iter().find(|(matches, _)| matches(op)) {
            return Some(legality);
        }
        self.dialects
            .iter()
            .find(|(dialect, _)| *dialect == op.dialect())
            .map(|&(_, legality)| legality)
    }
}

/// Which ops a conversion must leave legal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    /// Every op ends legal: an unknown op is an error.
    Full,
    /// Every illegal op is rewritten; unknown ops stay.
    Partial,
}

/// Rewrites each illegal op of `module` with the first of `patterns` that applies to it, and
/// returns how many it rewrote. Ops that a rewrite creates are checked too. Fails on an illegal
/// op that no pattern rewrites, with the reasons the patterns gave, and in [`Mode::Full`] on an op
/// that `target` does not make legal.
pub fn apply_conversion(
    module: &mut Module,
    target: &ConversionTarget,
    patterns: &PatternSet,
    mode: Mode,
) -> Result<usize> {
    let mut rewritten = 0;
    let mut worklist = module.walk(module.body());
    worklist.reverse();
    while let Some(inst) = worklist.pop() {
        if module.is_erased(inst) {
            continue;
        }
        let op = module.op(inst);
        let reason = match (target.legality(op), mode) {
            (Some(Legality::Legal), _) | (None, Mode::Partial) => continue,
            (Some(Legality::Illegal(Some(reason))), _) => {
                bail!("{}: failed to legalize: {reason}", op.name())
            }
            (Some(Legality::Illegal(None)), _) | (None, Mode::Full) => op.name(),
        };
        let touched = match patterns.apply(module, inst, true)? {
            Applied::Rewritten { touched, .. } => touched,
            Applied::NoMatch(reasons) if reasons.is_empty() => {
                bail!("{reason}: failed to legalize: no pattern applies")
            }
            Applied::NoMatch(reasons) => {
                bail!("{reason}: failed to legalize: {}", reasons.join("; "))
            }
        };
        rewritten += 1;
        for inst in touched.into_iter().rev() {
            if !module.is_erased(inst) {
                worklist.extend(module.walk_inst(inst).into_iter().rev());
            }
        }
    }
    if mode == Mode::Full {
        target.check(module)?;
    }
    Ok(rewritten)
}
