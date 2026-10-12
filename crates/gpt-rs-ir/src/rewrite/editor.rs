//! [`Editor`]: the edits the drivers make to a module, each recording the instructions it touched
//! so the greedy driver visits them again. It applies a pattern's staged changes, and the folds,
//! dead-op erasures and CSE replacements the drivers make themselves.

use std::collections::HashMap;

use anyhow::{bail, Result};

use crate::fold::FoldResult;
use crate::module::{Inst, Module, Region, Value};
use crate::op::Folded;
use crate::rewrite::rewriter::{Edit, Staged, Staging};
use crate::rewrite::Builder;
use crate::types::Type;
use crate::Op;

pub(crate) struct Editor<'m> {
    pub(crate) module: &'m mut Module,
    /// Instructions to visit again: those an edit created or changed, and those that lost a use.
    pub(crate) touched: Vec<Inst>,
}

impl<'m> Editor<'m> {
    pub(crate) fn new(module: &'m mut Module) -> Self {
        Self {
            module,
            touched: Vec::new(),
        }
    }

    /// Applies what a pattern staged: its new ops, each before its anchor, then its edits in
    /// order.
    pub(crate) fn apply(&mut self, staged: Staged) -> Result<()> {
        let mut map: HashMap<Value, Value> = HashMap::new();
        if let Some(staging) = staged.staging {
            let Staging {
                scratch,
                imports,
                anchors,
                ..
            } = *staging;
            map.extend(imports.into_iter().map(|(real, scratch)| (scratch, real)));
            for (inst, anchor) in anchors {
                let copy = self.copy(&scratch, inst, &mut map)?;
                self.module.insert_before(anchor, copy);
            }
        }
        let resolve = |value: Value| match value.staged_index() {
            Some(index) => map[&Value::at(index)],
            None => value,
        };
        let resolve_all = |values: Vec<Value>| values.into_iter().map(resolve).collect::<Vec<_>>();
        for edit in staged.edits {
            match edit {
                Edit::ReplaceOp { inst, values } => self.replace(inst, &resolve_all(values))?,
                Edit::ReplaceAllUses { from, to } => self.replace_all_uses(from, resolve(to)),
                Edit::SetOp { inst, op } => self.set_op(inst, *op),
                Edit::SetOperands { inst, operands } => {
                    self.set_operands(inst, resolve_all(operands))
                }
                Edit::Erase(inst) => self.erase(inst)?,
                Edit::RemoveArg { region, index } => self.remove_arg(region, index)?,
            }
        }
        Ok(())
    }

    /// A copy of `inst` of `source`, with its regions, unplaced, as [`Module::copy_inst`] makes it.
    /// It and the ops of its regions are touched.
    fn copy(
        &mut self,
        source: &Module,
        inst: Inst,
        map: &mut HashMap<Value, Value>,
    ) -> Result<Inst> {
        let copy = self.module.copy_inst(source, inst, map)?;
        self.touched.extend(self.module.walk_inst(copy));
        Ok(copy)
    }

    /// Replaces every use of `inst`'s results, terminators included, then erases it.
    pub(crate) fn replace(&mut self, inst: Inst, with: &[Value]) -> Result<()> {
        let results = self.module.results(inst).to_vec();
        if results.len() != with.len() {
            bail!(
                "{} has {} results, got {} replacements",
                self.module.op(inst).name(),
                results.len(),
                with.len()
            );
        }
        for (&from, &to) in results.iter().zip(with) {
            self.replace_all_uses(from, to);
        }
        self.erase(inst)
    }

    /// Makes every use of `from` use `to`. The users change, so the driver visits them again.
    pub(crate) fn replace_all_uses(&mut self, from: Value, to: Value) {
        self.touched
            .extend(self.module.uses(from).iter().map(|u| u.inst));
        self.module.replace_all_uses(from, to);
    }

    /// Applies what `inst` folded to: existing values or constants created before it, or new
    /// operands.
    pub(crate) fn apply_fold(&mut self, inst: Inst, fold: FoldResult) -> Result<()> {
        match fold {
            FoldResult::Results(folded) => self.replace_by_folded(inst, folded),
            FoldResult::Operands(operands) => {
                self.set_operands(inst, operands);
                Ok(())
            }
        }
    }

    fn replace_by_folded(&mut self, inst: Inst, folded: Vec<Folded>) -> Result<()> {
        let types: Vec<Type> = self
            .module
            .result_types(inst)
            .into_iter()
            .cloned()
            .collect();
        let location = self.module.location(inst).cloned();
        let mut builder = Builder::before(self.module, inst);
        builder.set_location(location);
        let values = builder.materialize(folded, &types);
        self.touched.extend(builder.into_created());
        self.replace(inst, &values)
    }

    /// Gives `inst` the operands `operands`. The ops that defined the old operands may have lost
    /// their last use, so the driver visits them again with `inst`.
    pub(crate) fn set_operands(&mut self, inst: Inst, operands: Vec<Value>) {
        let defs: Vec<Inst> = self
            .module
            .operands(inst)
            .iter()
            .filter_map(|&v| self.module.def_inst(v))
            .collect();
        self.module.set_operands(inst, operands);
        self.touched.push(inst);
        self.touched.extend(defs);
    }

    /// Gives `inst` the op `op`, which computes the same results. The facts of its results may
    /// change, so the driver visits their users again with `inst`.
    pub(crate) fn set_op(&mut self, inst: Inst, op: Op) {
        self.module.set_op(inst, op);
        self.touched.push(inst);
        for &result in self.module.results(inst) {
            self.touched
                .extend(self.module.uses(result).iter().map(|u| u.inst));
        }
    }

    /// Erases `inst`, and fails while its results are used. The ops that defined its operands
    /// may have lost their last use, so the driver visits them again.
    pub(crate) fn erase(&mut self, inst: Inst) -> Result<()> {
        let defs: Vec<Inst> = self
            .module
            .operands(inst)
            .iter()
            .filter_map(|&v| self.module.def_inst(v))
            .collect();
        self.module.erase(inst)?;
        self.touched.extend(defs);
        Ok(())
    }

    /// Removes block argument `index` of `region`, which nothing may use. The op that holds the
    /// region changes.
    pub(crate) fn remove_arg(&mut self, region: Region, index: usize) -> Result<()> {
        self.touched.extend(self.module.parent_op(region));
        self.module.remove_region_arg(region, index)
    }
}
