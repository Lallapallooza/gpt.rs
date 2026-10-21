//! `loop-to-isa`: each `loop.atom` as vector code of the `isa` dialect. `control.rs` holds the
//! pattern of `loop.atom`, which writes the vector code, and `shape.rs` how that code computes
//! each op of the atom's body, from how the op's values vary over its points.

mod control;
mod shape;

use anyhow::{anyhow, ensure, Context, Result};

use self::control::*;
use super::{apply_conversion, ConversionTarget, ConvertOp, Mode, OpForm};
use crate::dialects::loops::{self, AffineForms, FuncOp, LoopOps};
use crate::dialects::{isa, Dialect};
use crate::ir::{Inst, Region, Space, Value};
use crate::passes::{pass, Pass};
use crate::rewrite::{PatternSet, Rewrites, Rewriting};
use crate::targets::Target;

/// Computes the points of each `loop.atom` at once, in the vectors of its lanes, and checks the
/// vector code against the target. The rules of the target's architecture must have code for each
/// `isa` op, and the target must have the extensions that code needs. A register tile's code must
/// hold at once no more vector and mask registers than the target has. The count goes register by
/// register, in the order every backend follows ([`isa::RegisterCode::order`]). The
/// `tile.dispatch` verifier checks the rules for the lane-wise ops and the register widths of each
/// block that computes in vectors, but counts no registers. The code of a verified schedule may
/// hold more registers than the target has, which fails here. A `loop.func` decided for another
/// target is an error. The lowering decides nothing:
///
/// - The atom's tile is a group of registers: one axis per tile axis, the last one counting the
///   vectors of lanes along the tile's last axis. A value of the body takes extent 1 along the
///   axes it does not vary on, and `isa.expand` repeats it where an op needs it whole.
/// - An index of the body is a base plus a multiple of each coordinate of the point. A load whose
///   last index moves with the lanes by 1 is `isa.load`, one whose indices do not move with them
///   is `isa.broadcast`, and one at the same element for every point stays a `loop.load`, before
///   the atom. `isa.load` and `isa.broadcast` keep the load's pad where a lane may fall outside
///   the buffer. A store is `isa.store`, contiguous along the lanes.
/// - An op on values the same at every point stays as it is, before the atom, when it has no
///   effects and no regions and gives one value. The atom runs at least one point, so the op
///   computes what the body would. Each such value enters the registers as one `isa.splat`. Each
///   other `arith` op that has a lane-wise `isa` op is that op. Any other op is an error.
/// - An access of memory that moves along a tile axis whose extent may stop short of the tile
///   takes an `isa.tile_mask` that varies along those axes alone. An access of a `reg` buffer,
///   which holds the whole tile, takes none.
///
/// An atom whose body this cannot compute, such as one that reads with a stride along the lanes,
/// is an error.
#[pass(name = "loop-to-isa", stats(realized = "atoms computed in vectors"))]
#[derive(Default)]
pub struct LoopToIsa {
    /// The target the vector code is for.
    pub target: Option<Target>,
}

impl LoopToIsa {
    /// The target and the patterns of the conversion: every dialect but `loop` is legal, and each
    /// `loop` op is as its [`ConvertOp`] decision says.
    pub fn conversion() -> (ConversionTarget, PatternSet) {
        let mut target = ConversionTarget::of_dialects(&[
            Dialect::Func,
            Dialect::Tensor,
            Dialect::Arith,
            Dialect::Host,
            Dialect::Isa,
        ]);
        let mut patterns = PatternSet::new();
        LoopOps::convert::<LoopToIsa>(&mut target, &mut patterns);
        (target, patterns)
    }
}

impl Pass for LoopToIsa {
    fn rewrites(&self) -> Rewrites {
        Rewrites {
            patterns: Self::conversion().1,
            ..Rewrites::default()
        }
    }

    fn run(&self, module: &mut Rewriting) -> Result<bool> {
        let (conversion_target, patterns) = Self::conversion();
        let realized = apply_conversion(module, &conversion_target, &patterns, Mode::Full)?;
        LoopToIsa::REALIZED.add(realized);
        for inst in module.insts(module.body()).collect::<Vec<_>>() {
            let Some(func) = module.op_as::<loops::Func>(inst) else {
                continue;
            };
            let func_target = target_of(&func, self.target.as_ref())?;
            let code = FuncCode::of(&func);
            check_reg_buffers(&func, &code.reg_buffers)?;
            let Some(target) = func_target else {
                continue;
            };
            if let Some(rules) = check_rules(&func, &code.vector_insts, target)? {
                check_registers(&func, &code.regions, target, rules)?;
            }
        }
        Ok(realized > 0)
    }
}

/// What the checks of a `loop.func` read of its code after the conversion, from one walk of its
/// body: its `reg` buffers, its `isa` ops, and its regions, the body first.
struct FuncCode {
    reg_buffers: Vec<Value>,
    vector_insts: Vec<Inst>,
    regions: Vec<Region>,
}

impl FuncCode {
    fn of(func: &FuncOp) -> FuncCode {
        let module = func.module();
        let mut code = FuncCode {
            reg_buffers: Vec::new(),
            vector_insts: Vec::new(),
            regions: vec![func.body().id()],
        };
        for inst in module.walk(func.body().id()) {
            code.regions.extend(module.regions(inst).iter().copied());
            if module.op(inst).dialect() == Dialect::Isa {
                code.vector_insts.push(inst);
            }
            let Some(alloc) = module.op_as::<loops::Alloc>(inst) else {
                continue;
            };
            if alloc.space == Space::Reg {
                code.reg_buffers.push(alloc.output().id());
            }
        }
        code
    }
}

/// The target of the `loop.func` of `attrs`, which must be `pass_target`, the one the pass is for,
/// or none when its record names none.
fn target_of<'t>(
    attrs: &loops::Func,
    pass_target: Option<&'t Target>,
) -> Result<Option<&'t Target>> {
    let Some(reference) = attrs.target.as_ref() else {
        return Ok(None);
    };
    let target = pass_target.ok_or_else(|| {
        anyhow!(
            "loop.func @{} was decided for {}, and the pass has no target",
            attrs.name,
            String::from(reference)
        )
    })?;
    ensure!(
        reference == target.reference(),
        "loop.func @{} was decided for {}, and the pass is for {}",
        attrs.name,
        String::from(reference),
        String::from(target.reference())
    );
    Ok(Some(target))
}

/// Checks that the vector code of the `loop.func` `func` reaches each of `reg_buffers` only in
/// whole registers, as [`isa::reg_registers`] says, so that it holds the buffer in vector
/// variables.
fn check_reg_buffers(func: &FuncOp, reg_buffers: &[Value]) -> Result<()> {
    let module = func.module();
    let mut forms = AffineForms::new(module);
    for &buffer in reg_buffers {
        for used in module.uses(buffer) {
            let at = match module.location(used.inst) {
                Some(location) => format!(" at {location}"),
                None => String::new(),
            };
            let reached = |why: &str| {
                anyhow!(
                    "loop.func @{}: {}{at} reaches a reg buffer of {}, {why}",
                    func.name,
                    module.op(used.inst).name(),
                    module.ty(buffer)
                )
            };
            let access = isa::Access::of(module, used.inst)
                .filter(|access| access.buffer == buffer)
                .ok_or_else(|| reached("which only whole registers reach"))?;
            isa::reg_registers(module, &access, &mut forms).map_err(reached)?;
        }
    }
    Ok(())
}

/// Checks that `target`, the target of the `loop.func` `func`, computes each of `vector_insts`,
/// its `isa` ops: the rules of its architecture have code for the op, and the target has the
/// extensions the code needs. Gives the rules, or none when the function computes no vectors.
fn check_rules(
    func: &FuncOp,
    vector_insts: &[Inst],
    target: &Target,
) -> Result<Option<&'static dyn isa::Rules>> {
    let module = func.module();
    let Some(first) = vector_insts.first() else {
        return Ok(None);
    };
    let rules = target.isa_rules().ok_or_else(|| {
        anyhow!(
            "loop.func @{}: {} computes no {} code: gpt.rs has no rules for its instruction set",
            func.name,
            target.name,
            module.op(*first).name()
        )
    })?;
    for &inst in vector_insts {
        let op = module.op(inst);
        let at = match module.location(inst) {
            Some(location) => format!(" at {location}"),
            None => String::new(),
        };
        let what = format!("loop.func @{}: {}{at}", func.name, op.name());
        let needs = rules.needs(module, inst).with_context(|| what.clone())?;
        target.has_needs(needs, &what)?;
    }
    Ok(Some(rules))
}

/// Checks that `target`, the target of the `loop.func` `func`, holds in registers every register
/// the vector code of `regions`, its regions, holds at once: per run of a register tile's code, as
/// the register model of [`isa::RegisterCode`] counts them with what `rules` say the code of each
/// op needs, at most the target's vector registers, and at most the mask registers that can mask
/// an op. A target without mask registers holds masks in vector registers. The error names the op
/// where the run that holds the most holds them.
fn check_registers(
    func: &FuncOp,
    regions: &[Region],
    target: &Target,
    rules: &dyn isa::Rules,
) -> Result<()> {
    let module = func.module();
    // The most vector registers and masks that a run holds at once, and the op where it does.
    let mut most_vectors = (0, None);
    let mut most_masks = (0, None);
    for &region in regions {
        for run in isa::vector_runs(module, region) {
            let mut vector_insts = Vec::new();
            for inst in run {
                if module.op(inst).dialect() == Dialect::Isa {
                    vector_insts.push(inst);
                }
            }
            let pressure = isa::RegisterCode::of_isa(module, &vector_insts, rules)?.pressure();
            let (vectors, masks) = pressure.held(&target.isa);
            if vectors > most_vectors.0 {
                most_vectors = (vectors, pressure.vectors_at.map(|op| vector_insts[op]));
            }
            if masks > most_masks.0 {
                most_masks = (masks, pressure.masks_at.map(|op| vector_insts[op]));
            }
        }
    }
    let at = |inst: Option<Inst>| match inst.and_then(|held_at| module.location(held_at)) {
        Some(location) => format!(" at {location}"),
        None => String::new(),
    };
    let vector_budget = target.isa.vector_registers as usize;
    ensure!(
        most_vectors.0 <= vector_budget,
        "loop.func @{}: its vector code holds {} vector registers at once{}, and {} has \
         {vector_budget}",
        func.name,
        most_vectors.0,
        at(most_vectors.1),
        target.name
    );
    let mask_budget = target.isa.write_masks as usize;
    ensure!(
        most_masks.0 <= mask_budget,
        "loop.func @{}: its vector code holds {} masks at once{}, and {} has {mask_budget} mask \
         registers that can mask an op",
        func.name,
        most_masks.0,
        at(most_masks.1),
        target.name
    );
    Ok(())
}

/// The decision of `loop-to-isa` for each `loop` op: a `loop.atom` becomes vector code, and the
/// ops of its body with it, as [`AtomToIsa`] computes each. Every op outside an atom stays.
macro_rules! decide {
    ($($op:ident => $form:expr,)*) => {$(
        impl ConvertOp<LoopToIsa> for loops::$op {
            fn form() -> OpForm {
                $form
            }
        }
    )*};
}

decide! {
    Func => OpForm::Legal,
    Return => OpForm::Legal,
    For => OpForm::Legal,
    Forall => OpForm::Legal,
    If => OpForm::Legal,
    Atom => OpForm::rewrite(AtomToIsa),
    Yield => OpForm::Legal,
    Index => OpForm::Legal,
    Predicate => OpForm::Legal,
    Alloc => OpForm::Legal,
    Pick => OpForm::Legal,
    Load => OpForm::Legal,
    Store => OpForm::Legal,
    Prefetch => OpForm::Legal,
}
