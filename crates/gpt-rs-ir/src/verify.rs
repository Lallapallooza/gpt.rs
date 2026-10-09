//! The verifier, run after every pass.
//!
//! It checks that:
//! - the module's body holds only `func.func` ops
//! - each value is defined before it is used, and isolated regions read nothing from outside
//! - each op gets the operand counts and kinds its groups declare, and its `result_types` agree
//!   with the stored result types and result segments
//! - a terminator is the last op of its region
//! - each region has the block arguments its op's `region_signatures` give, and ends with the
//!   op's terminator, whose operands have the yielded types

use anyhow::{anyhow, bail, ensure, Result};

use crate::module::{Inst, Module, Region, Value};
use crate::op::{OpDyn, RegionSig};
use crate::scope::ScopedMap;
use crate::traits::{IsolatedFromAbove, Terminator};
use crate::types::{type_list, Type};
use crate::{Dialect, Op};

struct Verifier<'m> {
    module: &'m Module,
    /// The values defined so far, in the scopes that see them.
    defined: ScopedMap<Value, ()>,
}

impl Verifier<'_> {
    fn region(&mut self, region: Region, isolated: bool) -> Result<()> {
        let module = self.module;
        self.defined.push(isolated);
        for &arg in module.args(region) {
            self.defined.insert(arg, ());
        }
        let top = module.parent_op(region).is_none();
        let mut insts = module.insts(region).peekable();
        while let Some(inst) = insts.next() {
            let op = module.op(inst);
            self.check(inst, insts.peek().is_none(), top)
                .map_err(|err| anyhow!("{}: {err:#}", op.name()))?;
            for &inner in module.regions(inst) {
                self.region(inner, op.has_trait::<IsolatedFromAbove>())?;
            }
            for &result in module.results(inst) {
                self.defined.insert(result, ());
            }
        }
        self.defined.pop();
        Ok(())
    }

    /// The first problem with `inst` itself, not counting the ops in its regions.
    fn check(&self, inst: Inst, last: bool, top: bool) -> Result<()> {
        let module = self.module;
        let op = module.op(inst);
        ensure!(
            !top || op.as_func().is_some(),
            "only func.func may appear at the top level"
        );
        let defined = |&v: &Value| self.defined.get(&v).is_some();
        if let Some(i) = module.operands(inst).iter().position(|v| !defined(v)) {
            bail!("operand {i} is not defined before this op");
        }
        ensure!(
            last || !op.has_trait::<Terminator>(),
            "a terminator must be the last op of its region"
        );

        let operand_types = module.operand_types(inst);
        let segments = module.operand_segments(inst);
        let regions = module.regions(inst);
        let (types, result_segments) =
            op.dyn_result_types(&operand_types, segments, regions, module)?;
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

        let sigs = op.dyn_region_signatures(&operand_types, segments)?;
        for (index, (&region, sig)) in regions.iter().zip(&sigs).enumerate() {
            self.check_region(op, index, region, sig)?;
        }
        Ok(())
    }

    /// Region `index` of `op` has the block arguments of `sig` and ends with its terminator,
    /// which yields the types of `sig`.
    fn check_region(&self, op: &Op, index: usize, region: Region, sig: &RegionSig) -> Result<()> {
        let module = self.module;
        let args: Vec<&Type> = module.args(region).iter().map(|&v| module.ty(v)).collect();
        let expected: Vec<&Type> = sig.args.iter().collect();
        ensure!(
            args == expected,
            "region {index} has arguments {}, expected {}",
            type_list(&args),
            type_list(&expected)
        );
        let terminator = module
            .last_inst(region)
            .filter(|&last| op.is_region_terminator(index, module.op(last)))
            .ok_or_else(|| anyhow!("region {index} does not end with its terminator"))?;
        let yields = module.operand_types(terminator);
        let expected: Vec<&Type> = sig.yields.iter().collect();
        ensure!(
            yields == expected,
            "region {index} yields {}, expected {}",
            type_list(&yields),
            type_list(&expected)
        );
        Ok(())
    }
}

/// The first verifier error in `module`, outer ops first, as `<op name>: <message>`.
pub fn verify(module: &Module) -> Result<()> {
    Verifier {
        module,
        defined: ScopedMap::new(),
    }
    .region(module.body(), true)
}

/// Fails when an op of a dialect outside `allowed` remains.
pub fn check_legal(module: &Module, allowed: &[Dialect]) -> Result<()> {
    for inst in module.walk(module.body()) {
        let op = module.op(inst);
        if !allowed.contains(&op.dialect()) {
            bail!(
                "{}: the {:?} dialect is not allowed here",
                op.name(),
                op.dialect()
            );
        }
    }
    Ok(())
}
