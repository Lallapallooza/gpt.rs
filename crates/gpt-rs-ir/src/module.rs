//! The IR container: instructions, SSA values and regions, with def-use chains and the facts of
//! [`Module::is_static`] and [`Module::reads_param`] kept current on every edit.
//!
//! A region is one block with arguments. Control flow is structured only. The module's body is a
//! region that holds the `func.func` ops.

use std::collections::HashSet;

use anyhow::{bail, ensure, Result};

use crate::dense::Dense;
use crate::op::{InOp, Matched, Values};
use crate::rewrite::bound_params;
use crate::traits::{IsolatedFromAbove, Terminator};
use crate::types::Type;
use crate::Op;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Inst(u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Value(u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Region(u32);

/// Where a value comes from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValueDef {
    /// Result `index` of an instruction.
    Result(Inst, usize),
    /// Block argument `index` of a region.
    Arg(Region, usize),
}

/// Operand `index` of `inst`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Use {
    pub inst: Inst,
    pub index: usize,
}

#[derive(Debug, Clone)]
struct InstData {
    op: Op,
    operands: Vec<Value>,
    operand_segments: Option<Vec<usize>>,
    results: Vec<Value>,
    result_segments: Option<Vec<usize>>,
    regions: Vec<Region>,
    parent: Option<Region>,
    prev: Option<Inst>,
    next: Option<Inst>,
    erased: bool,
}

#[derive(Debug, Clone)]
struct ValueData {
    ty: Type,
    def: ValueDef,
    uses: Vec<Use>,
    facts: Facts,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct Facts {
    is_static: bool,
    reads_param: bool,
}

#[derive(Debug, Clone)]
struct RegionData {
    args: Vec<Value>,
    first: Option<Inst>,
    last: Option<Inst>,
    parent: Option<Inst>,
}

/// An instruction's op and operands: equal keys compute equal values when the op is pure.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct OpKey {
    op: Op,
    operands: Vec<Value>,
    operand_segments: Option<Vec<usize>>,
}

/// Instruction contents for [`Module::create_inst`].
pub(crate) struct NewInst {
    pub op: Op,
    pub operands: Vec<Value>,
    pub operand_segments: Option<Vec<usize>>,
    pub result_types: Vec<Type>,
    pub result_segments: Option<Vec<usize>>,
    pub regions: Vec<Region>,
}

#[derive(Debug, Clone)]
pub struct Module {
    insts: Vec<InstData>,
    values: Vec<ValueData>,
    regions: Vec<RegionData>,
    body: Region,
}

impl Default for Module {
    fn default() -> Self {
        Self::new()
    }
}

fn idx(raw: u32) -> usize {
    raw as usize
}

impl Module {
    pub fn new() -> Self {
        let mut module = Self {
            insts: Vec::new(),
            values: Vec::new(),
            regions: Vec::new(),
            body: Region(0),
        };
        module.body = module.new_region(Vec::new());
        module
    }

    /// The region holding the top-level ops.
    pub fn body(&self) -> Region {
        self.body
    }

    fn inst(&self, inst: Inst) -> &InstData {
        &self.insts[idx(inst.0)]
    }

    fn inst_mut(&mut self, inst: Inst) -> &mut InstData {
        &mut self.insts[idx(inst.0)]
    }

    fn value(&self, value: Value) -> &ValueData {
        &self.values[idx(value.0)]
    }

    fn value_mut(&mut self, value: Value) -> &mut ValueData {
        &mut self.values[idx(value.0)]
    }

    fn region(&self, region: Region) -> &RegionData {
        &self.regions[idx(region.0)]
    }

    fn region_mut(&mut self, region: Region) -> &mut RegionData {
        &mut self.regions[idx(region.0)]
    }

    pub fn op(&self, inst: Inst) -> &Op {
        &self.inst(inst).op
    }

    /// Replaces the op of `inst` with one of the same shape. The caller keeps operands and results
    /// consistent with it.
    pub fn set_op(&mut self, inst: Inst, op: Op) {
        self.inst_mut(inst).op = op;
        self.refresh(vec![inst]);
        for region in self.inst(inst).regions.clone() {
            self.refresh_args(region);
        }
    }

    pub fn operands(&self, inst: Inst) -> &[Value] {
        &self.inst(inst).operands
    }

    pub fn operand_segments(&self, inst: Inst) -> Option<&[usize]> {
        self.inst(inst).operand_segments.as_deref()
    }

    pub fn results(&self, inst: Inst) -> &[Value] {
        &self.inst(inst).results
    }

    pub fn result_segments(&self, inst: Inst) -> Option<&[usize]> {
        self.inst(inst).result_segments.as_deref()
    }

    pub fn regions(&self, inst: Inst) -> &[Region] {
        &self.inst(inst).regions
    }

    /// The region that holds `inst`, or `None` while it is detached.
    pub fn parent_region(&self, inst: Inst) -> Option<Region> {
        self.inst(inst).parent
    }

    /// The op whose region this is, or `None` for the module body.
    pub fn parent_op(&self, region: Region) -> Option<Inst> {
        self.region(region).parent
    }

    pub fn args(&self, region: Region) -> &[Value] {
        &self.region(region).args
    }

    pub fn ty(&self, value: Value) -> &Type {
        &self.value(value).ty
    }

    pub fn def(&self, value: Value) -> ValueDef {
        self.value(value).def
    }

    pub fn uses(&self, value: Value) -> &[Use] {
        &self.value(value).uses
    }

    pub fn is_erased(&self, inst: Inst) -> bool {
        self.inst(inst).erased
    }

    /// The instruction that defines `value`, unless it is a block argument.
    pub fn def_inst(&self, value: Value) -> Option<Inst> {
        match self.def(value) {
            ValueDef::Result(inst, _) => Some(inst),
            ValueDef::Arg(..) => None,
        }
    }

    /// The instructions of `region`, in order.
    pub fn insts(&self, region: Region) -> InstIter<'_> {
        InstIter {
            module: self,
            next: self.region(region).first,
        }
    }

    pub fn last_inst(&self, region: Region) -> Option<Inst> {
        self.region(region).last
    }

    /// Every live instruction under `region`, outer before inner, in order.
    pub fn walk(&self, region: Region) -> Vec<Inst> {
        let mut out = Vec::new();
        self.walk_into(region, &mut out);
        out
    }

    fn walk_into(&self, region: Region, out: &mut Vec<Inst>) {
        for inst in self.insts(region) {
            out.push(inst);
            for &inner in self.regions(inst) {
                self.walk_into(inner, out);
            }
        }
    }

    /// The `func.func` ops at the top level.
    pub fn funcs(&self) -> Vec<Inst> {
        self.insts(self.body)
            .filter(|&inst| self.op(inst).as_func().is_some())
            .collect()
    }

    /// The `func.func` named `name`.
    pub fn func(&self, name: &str) -> Option<Inst> {
        self.insts(self.body)
            .find(|&inst| self.op(inst).as_func().is_some_and(|f| f.name == name))
    }

    /// Creates a detached region with block arguments of the given types.
    pub fn new_region(&mut self, arg_types: Vec<Type>) -> Region {
        let region = Region(self.regions.len() as u32);
        self.regions.push(RegionData {
            args: Vec::new(),
            first: None,
            last: None,
            parent: None,
        });
        let args = arg_types
            .into_iter()
            .enumerate()
            .map(|(index, ty)| self.new_value(ty, ValueDef::Arg(region, index)))
            .collect();
        self.region_mut(region).args = args;
        region
    }

    /// Adds a block argument of type `ty` after the arguments `region` has.
    pub fn append_arg(&mut self, region: Region, ty: Type) -> Value {
        let index = self.region(region).args.len();
        let value = self.new_value(ty, ValueDef::Arg(region, index));
        self.region_mut(region).args.push(value);
        self.value_mut(value).facts = self.arg_facts(region, index);
        value
    }

    /// Removes argument `index` of the function `func`, with the parameter it binds. The argument
    /// must have no uses.
    pub fn remove_arg(&mut self, func: Inst, index: usize) -> Result<()> {
        let Some(mut attrs) = self.op(func).as_func().cloned() else {
            bail!("{} is not a function", self.op(func).name());
        };
        let body = self.regions(func)[0];
        let args = self.args(body);
        let Some(&arg) = args.get(index) else {
            bail!(
                "@{} has {} arguments, so no argument {index}",
                attrs.name,
                args.len()
            );
        };
        if let Some(user) = self.uses(arg).first() {
            bail!(
                "argument {index} of @{} is still used by {}",
                attrs.name,
                self.op(user.inst).name()
            );
        }
        self.region_mut(body).args.remove(index);
        for (i, later) in self.args(body).to_vec().into_iter().enumerate().skip(index) {
            self.value_mut(later).def = ValueDef::Arg(body, i);
        }
        attrs.args.remove(index);
        if !attrs.params.is_empty() {
            attrs.params.remove(index);
        }
        attrs.params = bound_params(attrs.params);
        self.set_op(func, attrs.into());
        Ok(())
    }

    fn new_value(&mut self, ty: Type, def: ValueDef) -> Value {
        let value = Value(self.values.len() as u32);
        self.values.push(ValueData {
            ty,
            def,
            uses: Vec::new(),
            facts: Facts::default(),
        });
        value
    }

    /// Creates a detached instruction. Its regions must be detached too, and they become its own.
    pub(crate) fn create_inst(&mut self, new: NewInst) -> Inst {
        let inst = Inst(self.insts.len() as u32);
        for &region in &new.regions {
            debug_assert!(self.region(region).parent.is_none(), "region already owned");
            self.region_mut(region).parent = Some(inst);
        }
        for (index, &operand) in new.operands.iter().enumerate() {
            self.value_mut(operand).uses.push(Use { inst, index });
        }
        self.insts.push(InstData {
            op: new.op,
            operands: new.operands,
            operand_segments: new.operand_segments,
            results: Vec::new(),
            result_segments: new.result_segments,
            regions: new.regions,
            parent: None,
            prev: None,
            next: None,
            erased: false,
        });
        let results: Vec<Value> = new
            .result_types
            .into_iter()
            .enumerate()
            .map(|(index, ty)| self.new_value(ty, ValueDef::Result(inst, index)))
            .collect();
        let facts = self.result_facts(inst);
        for &result in &results {
            self.value_mut(result).facts = facts;
        }
        self.inst_mut(inst).results = results;
        for region in self.inst(inst).regions.clone() {
            self.refresh_args(region);
        }
        inst
    }

    /// Appends a detached instruction to the end of `region`.
    pub fn append(&mut self, region: Region, inst: Inst) {
        debug_assert!(
            self.inst(inst).parent.is_none(),
            "instruction already placed"
        );
        let last = self.region(region).last;
        {
            let data = self.inst_mut(inst);
            data.parent = Some(region);
            data.prev = last;
            data.next = None;
        }
        match last {
            Some(last) => self.inst_mut(last).next = Some(inst),
            None => self.region_mut(region).first = Some(inst),
        }
        self.region_mut(region).last = Some(inst);
    }

    /// Places a detached instruction right before `anchor`.
    pub fn insert_before(&mut self, anchor: Inst, inst: Inst) {
        debug_assert!(
            self.inst(inst).parent.is_none(),
            "instruction already placed"
        );
        let region = self.inst(anchor).parent.expect("anchor is placed");
        let prev = self.inst(anchor).prev;
        {
            let data = self.inst_mut(inst);
            data.parent = Some(region);
            data.prev = prev;
            data.next = Some(anchor);
        }
        self.inst_mut(anchor).prev = Some(inst);
        match prev {
            Some(prev) => self.inst_mut(prev).next = Some(inst),
            None => self.region_mut(region).first = Some(inst),
        }
    }

    fn unlink(&mut self, inst: Inst) {
        let Some(region) = self.inst(inst).parent else {
            return;
        };
        let (prev, next) = (self.inst(inst).prev, self.inst(inst).next);
        match prev {
            Some(prev) => self.inst_mut(prev).next = next,
            None => self.region_mut(region).first = next,
        }
        match next {
            Some(next) => self.inst_mut(next).prev = prev,
            None => self.region_mut(region).last = prev,
        }
        let data = self.inst_mut(inst);
        data.parent = None;
        data.prev = None;
        data.next = None;
    }

    pub fn set_operand(&mut self, inst: Inst, index: usize, value: Value) {
        let old = self.inst(inst).operands[index];
        self.value_mut(old)
            .uses
            .retain(|u| !(u.inst == inst && u.index == index));
        self.inst_mut(inst).operands[index] = value;
        self.value_mut(value).uses.push(Use { inst, index });
        self.refresh(vec![inst]);
    }

    /// Makes every use of `from` a use of `to`, terminators included.
    pub fn replace_all_uses(&mut self, from: Value, to: Value) {
        if from == to {
            return;
        }
        let uses = std::mem::take(&mut self.value_mut(from).uses);
        for u in &uses {
            self.inst_mut(u.inst).operands[u.index] = to;
        }
        let users = uses.iter().map(|u| u.inst).collect();
        self.value_mut(to).uses.extend(uses);
        self.refresh(users);
    }

    /// Removes `inst` and everything in its regions. Fails while a result is still used outside.
    pub fn erase(&mut self, inst: Inst) -> Result<()> {
        ensure!(!self.inst(inst).erased, "instruction is already erased");
        let nested: Vec<Inst> = self
            .regions(inst)
            .iter()
            .flat_map(|&region| self.walk(region))
            .collect();
        let inside_set: HashSet<Inst> = nested.iter().copied().collect();
        let inside = |u: &Use| u.inst == inst || inside_set.contains(&u.inst);
        for &result in self.results(inst) {
            if let Some(user) = self.uses(result).iter().find(|u| !inside(u)) {
                bail!(
                    "cannot erase {}: result {} is still used by {}",
                    self.op(inst).name(),
                    self.def_index(result),
                    self.op(user.inst).name()
                );
            }
        }
        for &inner in nested.iter().rev() {
            self.drop_inst(inner);
        }
        self.drop_inst(inst);
        Ok(())
    }

    fn def_index(&self, value: Value) -> usize {
        match self.def(value) {
            ValueDef::Result(_, i) | ValueDef::Arg(_, i) => i,
        }
    }

    fn drop_inst(&mut self, inst: Inst) {
        self.unlink(inst);
        let operands = std::mem::take(&mut self.inst_mut(inst).operands);
        for (index, operand) in operands.into_iter().enumerate() {
            self.value_mut(operand)
                .uses
                .retain(|u| !(u.inst == inst && u.index == index));
        }
        self.inst_mut(inst).erased = true;
    }

    pub fn operand_types(&self, inst: Inst) -> Vec<&Type> {
        self.operands(inst).iter().map(|&v| self.ty(v)).collect()
    }

    pub fn result_types(&self, inst: Inst) -> Vec<&Type> {
        self.results(inst).iter().map(|&v| self.ty(v)).collect()
    }

    /// The constant that defines `value`, if one does.
    pub fn constant(&self, value: Value) -> Option<&Dense> {
        self.op(self.def_inst(value)?).as_constant()
    }

    /// Whether `inst` can go: it is pure, no result is used, and it is neither a terminator nor an
    /// isolated op such as `func.func`, which no SSA value uses.
    pub fn is_trivially_dead(&self, inst: Inst) -> bool {
        let op = self.op(inst);
        op.effects().is_pure()
            && !op.has_trait::<Terminator>()
            && !op.has_trait::<IsolatedFromAbove>()
            && self.results(inst).iter().all(|&v| self.uses(v).is_empty())
    }

    /// Whether `value` is known before the run: a constant, a function argument bound to a
    /// parameter, or a result of a pure op without regions whose operands are all known.
    pub fn is_static(&self, value: Value) -> bool {
        self.value(value).facts.is_static
    }

    /// Whether a parameter goes into `value`: it is a function argument bound to one, or an op
    /// computes it from such a value.
    pub fn reads_param(&self, value: Value) -> bool {
        self.value(value).facts.reads_param
    }

    /// The facts of the results of `inst`, from its op and the facts of its operands.
    fn result_facts(&self, inst: Inst) -> Facts {
        let data = self.inst(inst);
        let operands = || data.operands.iter().map(|&v| self.value(v).facts);
        Facts {
            is_static: data.op.effects().is_pure()
                && data.regions.is_empty()
                && operands().all(|f| f.is_static),
            reads_param: operands().any(|f| f.reads_param),
        }
    }

    fn arg_facts(&self, region: Region, index: usize) -> Facts {
        let param = self.is_param(region, index);
        Facts {
            is_static: param,
            reads_param: param,
        }
    }

    /// Recomputes the facts of the results of `insts`, and of every value computed from a result
    /// whose facts changed.
    fn refresh(&mut self, mut insts: Vec<Inst>) {
        while let Some(inst) = insts.pop() {
            let facts = self.result_facts(inst);
            for index in 0..self.inst(inst).results.len() {
                let result = self.inst(inst).results[index];
                if std::mem::replace(&mut self.value_mut(result).facts, facts) != facts {
                    insts.extend(self.uses(result).iter().map(|u| u.inst));
                }
            }
        }
    }

    /// Recomputes the facts of the block arguments of `region`, and of everything computed from
    /// them.
    fn refresh_args(&mut self, region: Region) {
        for (index, arg) in self.region(region).args.clone().into_iter().enumerate() {
            let facts = self.arg_facts(region, index);
            if std::mem::replace(&mut self.value_mut(arg).facts, facts) != facts {
                let users = self.uses(arg).iter().map(|u| u.inst).collect();
                self.refresh(users);
            }
        }
    }

    /// Whether block argument `index` of `region` is a function argument bound to a parameter.
    fn is_param(&self, region: Region, index: usize) -> bool {
        self.parent_op(region)
            .and_then(|inst| self.op(inst).as_func())
            .is_some_and(|func| func.param(index).is_some())
    }

    /// What makes two instructions compute the same values, or `None` when `inst` may not be
    /// merged with another: it has effects or regions, or is a terminator.
    pub fn cse_key(&self, inst: Inst) -> Option<OpKey> {
        let op = self.op(inst);
        if !op.effects().is_pure() || op.has_trait::<Terminator>() || !self.regions(inst).is_empty()
        {
            return None;
        }
        Some(OpKey {
            op: op.clone(),
            operands: self.operands(inst).to_vec(),
            operand_segments: self.operand_segments(inst).map(<[usize]>::to_vec),
        })
    }

    /// The op of `inst` as `T`, with its named operand and result groups.
    pub fn op_as<T: InOp>(&self, inst: Inst) -> Option<Matched<T>> {
        let attrs = T::cast(self.op(inst))?.clone();
        let operands = T::split_operands::<Values>(
            self.operands(inst).iter().copied(),
            self.operand_segments(inst),
        )
        .ok()?;
        let results = T::split_results::<Values>(
            self.results(inst).iter().copied(),
            self.result_segments(inst),
        )
        .ok()?;
        Some(Matched {
            inst,
            attrs,
            operands,
            results,
        })
    }

    /// The op that defines `value`, if it is a `T`.
    pub fn def_as<T: InOp>(&self, value: Value) -> Option<Matched<T>> {
        self.op_as(self.def_inst(value)?)
    }

    pub fn is<T: InOp>(&self, inst: Inst) -> bool {
        T::cast(self.op(inst)).is_some()
    }
}

pub struct InstIter<'a> {
    module: &'a Module,
    next: Option<Inst>,
}

impl Iterator for InstIter<'_> {
    type Item = Inst;

    fn next(&mut self) -> Option<Inst> {
        let inst = self.next?;
        self.next = self.module.inst(inst).next;
        Some(inst)
    }
}
