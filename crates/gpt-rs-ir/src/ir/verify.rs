//! The verifier, run after every pass.
//!
//! It checks that:
//! - the module's body holds only symbol ops: `func.func`, `tile.dispatch` and `loop.func`
//! - each value is defined before it is used, and isolated regions read nothing from outside
//! - an op with the trait `OnlyUse` is the only use of each of its operands
//! - each op gets the operand counts and kinds its groups declare, and its `result_types` agree
//!   with the stored result types and result segments
//! - a terminator is the last op of its region
//! - each region has the block arguments its op's `region_signatures` give, and ends with the
//!   op's terminator, whose operands have the yielded types

use anyhow::{anyhow, bail, ensure, Result};

use crate::ir::{type_list, Inst, Module, OpDyn, Region, RegionSig, Type};
use crate::traits::{OnlyUse, Terminator};
use crate::{Dialect, Op};

struct Verifier<'m> {
    module: &'m Module,
}

impl Verifier<'_> {
    fn region(&self, region: Region) -> Result<()> {
        let module = self.module;
        let top = module.parent_op(region).is_none();
        let mut insts = module.insts(region).peekable();
        while let Some(inst) = insts.next() {
            let op = module.op(inst);
            self.check(inst, insts.peek().is_none(), top)
                .map_err(|err| anyhow!("{}: {err:#}", op.name()))?;
            for &inner in module.regions(inst) {
                self.region(inner)?;
            }
        }
        Ok(())
    }

    /// The first problem with `inst` itself, not counting the ops in its regions.
    fn check(&self, inst: Inst, last: bool, top: bool) -> Result<()> {
        let module = self.module;
        let op = module.op(inst);
        ensure!(
            !top || op.symbol_name().is_some(),
            "only func.func, tile.dispatch and loop.func may appear at the top level"
        );
        let operands = module.operands(inst);
        if let Some(position) = operands
            .iter()
            .position(|&operand| !module.dominates(operand, inst))
        {
            bail!("operand {position} is not defined before this op");
        }
        let only_use = op.has_trait::<OnlyUse>();
        if let Some(position) = operands
            .iter()
            .position(|&operand| only_use && module.uses(operand).len() > 1)
        {
            bail!("operand {position} has another use, and this op must be its only one");
        }
        ensure!(
            last || !op.has_trait::<Terminator>(),
            "a terminator must be the last op of its region"
        );

        let segments = module.operand_segments(inst);
        let regions = module.regions(inst);
        let (types, result_segments) =
            module.infer_results(op, module.operands(inst), segments, regions, None)?;
        let stored = module.result_types(inst);
        let inferred: Vec<&Type> = types.iter().collect();
        ensure!(
            stored == inferred,
            "result types are {}, but the op gives {}",
            type_list(&stored),
            type_list(&inferred)
        );
        ensure!(
            result_segments.as_deref() == module.result_segments(inst),
            "result segments do not match the result groups"
        );

        let sigs = op.dyn_region_signatures(&module.operand_types(inst), segments)?;
        for (index, (&region, sig)) in regions.iter().zip(&sigs).enumerate() {
            self.check_region(op, index, region, sig)?;
        }
        Ok(())
    }

    /// Region `index` of `op` has the block arguments of `sig` and ends with its terminator,
    /// which yields the types of `sig`.
    fn check_region(&self, op: &Op, index: usize, region: Region, sig: &RegionSig) -> Result<()> {
        let module = self.module;
        let args: Vec<&Type> = module
            .args(region)
            .iter()
            .map(|&arg| module.ty(arg))
            .collect();
        let expected_args: Vec<&Type> = sig.args.iter().collect();
        ensure!(
            args == expected_args,
            "region {index} has arguments {}, expected {}",
            type_list(&args),
            type_list(&expected_args)
        );
        let terminator = module
            .last_inst(region)
            .filter(|&last| op.is_region_terminator(index, module.op(last)))
            .ok_or_else(|| anyhow!("region {index} does not end with its terminator"))?;
        let Some(yield_types) = &sig.yields else {
            return Ok(());
        };
        let yields = module.operand_types(terminator);
        let expected_yields: Vec<&Type> = yield_types.iter().collect();
        ensure!(
            yields == expected_yields,
            "region {index} yields {}, expected {}",
            type_list(&yields),
            type_list(&expected_yields)
        );
        Ok(())
    }
}

/// The first verifier error in `module`, outer ops first, as `<op name>: <message>`.
pub fn verify(module: &Module) -> Result<()> {
    Verifier { module }.region(module.body())
}

/// Fails when an op of a dialect outside `allowed` remains.
pub fn check_legal(module: &Module, allowed: &[Dialect]) -> Result<()> {
    crate::conversion::ConversionTarget::of_dialects(allowed).check(module)
}
