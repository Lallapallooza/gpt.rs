//! [`Editor`]: the edits the drivers make to a module, each recording the instructions it touched
//! so the greedy driver visits them again. It applies a pattern's staged changes, and the folds,
//! dead-op erasures and CSE replacements the drivers make themselves.

use std::collections::HashMap;

use anyhow::{bail, Result};

use crate::dialects::func::{Func, ParamId};
use crate::ir::{Folded, Inst, Module, Region, Type, Value};
use crate::rewrite::rewriter::{Edit, Move, Staged, Staging};
use crate::rewrite::{Builder, FoldResult, KEY};
use crate::text::any_items;
use crate::traits::Terminator;
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

    /// Applies what a pattern staged: the arguments functions gain, which its new ops may read,
    /// then its new ops, each before its anchor, with the ops that move into their regions, then
    /// its other edits in order.
    pub(crate) fn apply(&mut self, staged: Staged) -> Result<()> {
        let mut map: HashMap<Value, Value> = HashMap::new();
        let mut edits = Vec::with_capacity(staged.edits.len());
        for edit in staged.edits {
            let Edit::AddArg {
                func,
                ty,
                param,
                placeholder,
            } = edit
            else {
                edits.push(edit);
                continue;
            };
            map.insert(placeholder, self.add_arg(func, ty, param));
        }
        if let Some(staging) = staged.staging {
            let Staging {
                scratch,
                imports,
                anchors,
                moves,
                ..
            } = *staging;
            map.extend(imports.into_iter().map(|(real, imported)| (imported, real)));
            let mut copies = HashMap::new();
            for (inst, anchor) in anchors {
                let copy = self.copy(&scratch, inst, &mut map)?;
                self.module.insert_before(&KEY, anchor, copy);
                if moves.is_empty() {
                    continue;
                }
                for (original, copied) in scratch
                    .walk_inst(inst)
                    .into_iter()
                    .zip(self.module.walk_inst(copy))
                {
                    copies.insert(original, copied);
                }
            }
            self.apply_moves(&scratch, &copies, moves);
        }
        let resolve = |value: Value| match value.staged_index() {
            Some(index) => map[&Value::at(index)],
            None => value,
        };
        let resolve_all = |values: Vec<Value>| values.into_iter().map(resolve).collect::<Vec<_>>();
        for edit in edits {
            match edit {
                Edit::ReplaceOp { inst, values } => self.replace(inst, &resolve_all(values))?,
                Edit::ReplaceAllUses { from, to } => self.replace_all_uses(from, resolve(to)),
                Edit::SetOp { inst, op } => self.set_op(inst, *op),
                Edit::SetOperands { inst, operands } => {
                    self.set_operands(inst, resolve_all(operands))
                }
                Edit::SetOperand { inst, index, value } => {
                    self.set_operand(inst, index, resolve(value))
                }
                Edit::Erase(inst) => self.erase(inst)?,
                Edit::Inline {
                    region,
                    before,
                    args,
                } => self.inline(region, before, &resolve_all(args)),
                Edit::RemoveArg { region, index } => self.remove_arg(region, index)?,
                Edit::AddArg { .. } => unreachable!("added arguments apply first"),
            }
        }
        Ok(())
    }

    /// Moves each op of `moves` into the copy of its region of `scratch`, before the copy of the
    /// staged op it came before. `copies` gives the copy of each staged op.
    fn apply_moves(&mut self, scratch: &Module, copies: &HashMap<Inst, Inst>, moves: Vec<Move>) {
        for Move {
            op,
            region,
            position,
        } in moves
        {
            let next = scratch
                .insts(region)
                .nth(position)
                .expect("a moved op goes before the terminator of its region");
            self.module.move_before(&KEY, op, copies[&next]);
            self.touched.extend(self.module.walk_inst(op));
        }
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
            .extend(self.module.uses(from).iter().map(|use_site| use_site.inst));
        self.module.replace_all_uses(&KEY, from, to);
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
            .filter_map(|&operand| self.module.def_inst(operand))
            .collect();
        self.module.set_operands(&KEY, inst, operands);
        self.touched.push(inst);
        self.touched.extend(defs);
    }

    /// Gives `inst` the op `op`, which computes the same results. The facts of its results may
    /// change, so the driver visits their users again with `inst`.
    pub(crate) fn set_op(&mut self, inst: Inst, op: Op) {
        self.module.set_op(&KEY, inst, op);
        self.touched.push(inst);
        for &result in self.module.results(inst) {
            self.touched.extend(
                self.module
                    .uses(result)
                    .iter()
                    .map(|use_site| use_site.inst),
            );
        }
    }

    /// Erases `inst`, and fails while its results are used. The ops that defined its operands
    /// may have lost their last use, so the driver visits them again.
    pub(crate) fn erase(&mut self, inst: Inst) -> Result<()> {
        let defs: Vec<Inst> = self
            .module
            .operands(inst)
            .iter()
            .filter_map(|&operand| self.module.def_inst(operand))
            .collect();
        self.module.erase(&KEY, inst)?;
        self.touched.extend(defs);
        Ok(())
    }

    /// Gives operand `index` of `inst` the value `value`. The op that defined the old operand may
    /// have lost its last use, so the driver visits it again with `inst`.
    pub(crate) fn set_operand(&mut self, inst: Inst, index: usize, value: Value) {
        let old = self.module.operands(inst)[index];
        self.touched.extend(self.module.def_inst(old));
        self.module.set_operand(&KEY, inst, index, value);
        self.touched.push(inst);
    }

    /// Moves the ops of `region`, all but its terminator, before `before`, each use of an argument
    /// of the region using the value at its position in `args`. The moved ops are touched.
    fn inline(&mut self, region: Region, before: Inst, args: &[Value]) {
        let region_args = self.module.args(region).to_vec();
        for (&arg, &value) in region_args.iter().zip(args) {
            self.replace_all_uses(arg, value);
        }
        let insts: Vec<Inst> = self.module.insts(region).collect();
        for inst in insts {
            if self.module.op(inst).has_trait::<Terminator>() {
                continue;
            }
            self.module.move_before(&KEY, inst, before);
            self.touched.extend(self.module.walk_inst(inst));
        }
    }

    /// Removes block argument `index` of `region`, which nothing may use. The op that holds the
    /// region changes. A function loses the argument from its signature, with its parameter.
    pub(crate) fn remove_arg(&mut self, region: Region, index: usize) -> Result<()> {
        self.module.remove_region_arg(&KEY, region, index)?;
        let Some(parent) = self.module.parent_op(region) else {
            return Ok(());
        };
        self.touched.push(parent);
        let Some(func) = self.module.op_as::<Func>(parent) else {
            return Ok(());
        };
        let mut attrs = func.attrs().clone();
        attrs.args.remove(index);
        if !attrs.params.is_empty() {
            attrs.params.remove(index);
        }
        attrs.params = any_items(attrs.params);
        self.module.set_op(&KEY, parent, attrs.into());
        Ok(())
    }

    /// Adds an argument of type `ty` after the arguments of the function `func`, bound to the
    /// parameter `param`, and gives it.
    fn add_arg(&mut self, func: Inst, ty: Type, param: Option<ParamId>) -> Value {
        let body = self.module.regions(func)[0];
        let arg = self.module.append_arg(&KEY, body, ty.clone());
        let mut attrs = self
            .module
            .op_as::<Func>(func)
            .expect("an argument is added to a function")
            .attrs()
            .clone();
        attrs.params.resize(attrs.args.len(), None);
        attrs.args.push(ty);
        attrs.params.push(param);
        attrs.params = any_items(attrs.params);
        self.module.set_op(&KEY, func, attrs.into());
        self.touched.push(func);
        arg
    }
}
