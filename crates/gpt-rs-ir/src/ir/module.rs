//! The IR container: instructions, SSA values and regions, with def-use chains and the facts of
//! [`Module::known_before_run`] and [`Module::depends_on_parameter`] kept current on every edit.
//! [`ValueRef`] and [`RegionRef`] read a value or a region together with the module.
//!
//! A region is one block with arguments. Control flow is structured only. The module's body is a
//! region that holds the `func.func` ops.

use std::collections::{HashMap, HashSet};

use anyhow::{bail, ensure, Result};

use crate::dialects::func::{Func, FuncOp};
use crate::host::HostScalar;
use crate::ir::{Dense, Flat, InOp, Location, OpDyn, OpRef, RunningPass, TensorType, Type};
use crate::rewrite::EditKey;
use crate::traits::{IsolatedFromAbove, Terminator};
use crate::Op;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Inst(u32);

/// An instruction, or what holds one: a view of an op.
pub trait AsInst {
    fn inst(&self) -> Inst;
}

impl AsInst for Inst {
    fn inst(&self) -> Inst {
        *self
    }
}

impl<A: AsInst> AsInst for &A {
    fn inst(&self) -> Inst {
        (*self).inst()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Value(u32);

impl Value {
    /// The bit that marks a value a rewrite created and has not applied yet.
    const STAGED: u32 = 1 << 31;

    /// The value a rewrite created at `index` of its staged ops.
    pub(crate) fn staged(index: u32) -> Self {
        Self(index | Self::STAGED)
    }

    /// The index among a rewrite's staged ops, when a rewrite created this value.
    pub(crate) fn staged_index(self) -> Option<u32> {
        (self.0 & Self::STAGED != 0).then_some(self.0 & !Self::STAGED)
    }

    /// The value at `index` of a module.
    pub(crate) fn at(index: u32) -> Self {
        Self(index)
    }

    /// The index of the value in its module.
    pub(crate) fn index(self) -> u32 {
        self.0
    }
}

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
    /// Increases along the region, so two instructions of a region compare by it. Placing an
    /// instruction takes a key between its neighbors', renumbering the region when none is left.
    order: u64,
    location: Option<Location>,
    erased: bool,
}

/// The gap between the order keys of instructions that a region gets in a row, which leaves room
/// for the instructions placed between them later.
const ORDER_GAP: u64 = 1 << 16;

#[derive(Debug, Clone)]
struct ValueData {
    ty: Type,
    def: ValueDef,
    uses: Vec<Use>,
    facts: Facts,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct Facts {
    known_before_run: bool,
    depends_on_parameter: bool,
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
    pub location: Option<Location>,
}

#[derive(Debug, Clone)]
pub struct Module {
    insts: Vec<InstData>,
    values: Vec<ValueData>,
    regions: Vec<RegionData>,
    body: Region,
    /// The pass that runs on the module. Each op it makes records the pass on its trail.
    pub(super) running: RunningPass,
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
            running: RunningPass::default(),
        };
        module.body = module.new_region(Vec::new());
        module
    }

    /// The region holding the top-level ops.
    pub fn body(&self) -> Region {
        self.body
    }

    fn inst(&self, op: impl AsInst) -> &InstData {
        &self.insts[idx(op.inst().0)]
    }

    fn inst_mut(&mut self, op: impl AsInst) -> &mut InstData {
        &mut self.insts[idx(op.inst().0)]
    }

    fn value(&self, value: Value) -> &ValueData {
        assert!(
            value.staged_index().is_none(),
            "a value that a rewrite created is not in the module until the rewrite applies"
        );
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

    pub fn op(&self, op: impl AsInst) -> &Op {
        &self.inst(op).op
    }

    /// Replaces the op of `op` with `new`, an op of the same kind and shape. The caller keeps
    /// operands and results consistent with it.
    pub fn set_op(&mut self, _key: &EditKey, op: impl AsInst, new: Op) {
        let inst = op.inst();
        debug_assert_eq!(
            self.op(inst).name(),
            new.name(),
            "an op keeps its kind once made"
        );
        self.inst_mut(inst).op = new;
        self.refresh(vec![inst]);
        for region in self.inst(inst).regions.clone() {
            self.refresh_args(region);
        }
        if let Some(region) = self.inst(inst).parent {
            self.refresh_holders(region);
        }
    }

    pub fn operands(&self, op: impl AsInst) -> &[Value] {
        &self.inst(op).operands
    }

    pub fn operand_segments(&self, op: impl AsInst) -> Option<&[usize]> {
        self.inst(op).operand_segments.as_deref()
    }

    pub fn results(&self, op: impl AsInst) -> &[Value] {
        &self.inst(op).results
    }

    pub fn result_segments(&self, op: impl AsInst) -> Option<&[usize]> {
        self.inst(op).result_segments.as_deref()
    }

    pub fn regions(&self, op: impl AsInst) -> &[Region] {
        &self.inst(op).regions
    }

    /// The region that holds `op`, or `None` while it is detached.
    pub fn parent_region(&self, op: impl AsInst) -> Option<Region> {
        self.inst(op).parent
    }

    /// The op whose region this is, or `None` for the module body.
    pub fn parent_op(&self, region: Region) -> Option<Inst> {
        self.region(region).parent
    }

    /// The op of the module body that holds `op` in its regions, or `op` when it is one.
    pub fn top_level_op(&self, op: impl AsInst) -> Inst {
        let mut outer = op.inst();
        while let Some(parent) = self
            .parent_region(outer)
            .and_then(|region| self.parent_op(region))
        {
            outer = parent;
        }
        outer
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

    pub fn is_erased(&self, op: impl AsInst) -> bool {
        self.inst(op).erased
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

    /// A copy of `inst` of `source`, with its regions, unplaced: its operands are the values `map`
    /// gives for its own, and `map` gains its results and region arguments. Fails when it reads a
    /// value `map` does not give.
    pub fn copy_inst(
        &mut self,
        source: &Module,
        inst: Inst,
        map: &mut HashMap<Value, Value>,
    ) -> Result<Inst> {
        let mapped = |values: &[Value], copies: &HashMap<Value, Value>| -> Result<Vec<Value>> {
            values
                .iter()
                .map(|value| match copies.get(value) {
                    Some(&mapped) => Ok(mapped),
                    None => bail!(
                        "{} reads a value from outside what it copies",
                        source.op(inst).name()
                    ),
                })
                .collect()
        };
        let mut regions = Vec::new();
        for &region in source.regions(inst) {
            let arg_types: Vec<Type> = source
                .args(region)
                .iter()
                .map(|&arg| source.ty(arg).clone())
                .collect();
            let copy = self.new_region(arg_types);
            map.extend(
                source
                    .args(region)
                    .iter()
                    .copied()
                    .zip(self.args(copy).to_vec()),
            );
            for inner in source.insts(region) {
                let inner_copy = self.copy_inst(source, inner, map)?;
                self.link_last(copy, inner_copy);
            }
            regions.push(copy);
        }
        let copy = self.create_inst(NewInst {
            op: source.op(inst).clone(),
            operands: mapped(source.operands(inst), map)?,
            operand_segments: source.operand_segments(inst).map(<[usize]>::to_vec),
            result_types: source.result_types(inst).into_iter().cloned().collect(),
            result_segments: source.result_segments(inst).map(<[usize]>::to_vec),
            regions,
            location: source.location(inst).cloned(),
        });
        map.extend(
            source
                .results(inst)
                .iter()
                .copied()
                .zip(self.results(copy).to_vec()),
        );
        Ok(copy)
    }

    /// `inst` and every live instruction in its regions, outer before inner, in order.
    pub fn walk_inst(&self, op: impl AsInst) -> Vec<Inst> {
        let inst = op.inst();
        let mut out = vec![inst];
        for &inner in self.regions(inst) {
            self.walk_into(inner, &mut out);
        }
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
    pub fn funcs(&self) -> Vec<FuncOp<'_>> {
        let mut funcs = Vec::new();
        for inst in self.insts(self.body) {
            funcs.extend(self.op_as::<Func>(inst));
        }
        funcs
    }

    /// The `func.func` named `name`.
    pub fn func(&self, name: &str) -> Option<FuncOp<'_>> {
        for inst in self.insts(self.body) {
            if let Some(func) = self.op_as::<Func>(inst).filter(|func| func.name == name) {
                return Some(func);
            }
        }
        None
    }

    /// The top-level op that defines the symbol `name`: a `func.func` or a `tile.dispatch`.
    pub fn symbol(&self, name: &str) -> Option<Inst> {
        self.insts(self.body)
            .find(|&inst| self.op(inst).symbol_name() == Some(name))
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
    pub fn append_arg(&mut self, _key: &EditKey, region: Region, ty: Type) -> Value {
        let index = self.region(region).args.len();
        let value = self.new_value(ty, ValueDef::Arg(region, index));
        self.region_mut(region).args.push(value);
        self.value_mut(value).facts = self.arg_facts(region, index);
        value
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
            order: 0,
            location: new.location,
            erased: false,
        });
        let results: Vec<Value> = new
            .result_types
            .into_iter()
            .enumerate()
            .map(|(index, ty)| self.new_value(ty, ValueDef::Result(inst, index)))
            .collect();
        self.inst_mut(inst).results = results;
        self.refresh(vec![inst]);
        for region in self.inst(inst).regions.clone() {
            self.refresh_args(region);
        }
        inst
    }

    /// Appends a detached instruction to the end of `region`.
    pub fn append(&mut self, _key: &EditKey, region: Region, inst: Inst) {
        self.link_last(region, inst);
    }

    fn link_last(&mut self, region: Region, inst: Inst) {
        self.number_for_trails();
        debug_assert!(
            self.inst(inst).parent.is_none(),
            "instruction already placed"
        );
        let last = self.region(region).last;
        let order = last.map_or(0, |tail| self.inst(tail).order) + ORDER_GAP;
        {
            let data = self.inst_mut(inst);
            data.parent = Some(region);
            data.prev = last;
            data.next = None;
            data.order = order;
        }
        match last {
            Some(tail) => self.inst_mut(tail).next = Some(inst),
            None => self.region_mut(region).first = Some(inst),
        }
        self.region_mut(region).last = Some(inst);
        self.refresh_holders(region);
    }

    /// Places a detached instruction right before `anchor`.
    pub fn insert_before(&mut self, _key: &EditKey, anchor: Inst, inst: Inst) {
        self.link_before(anchor, inst);
    }

    fn link_before(&mut self, anchor: Inst, inst: Inst) {
        self.number_for_trails();
        debug_assert!(
            self.inst(inst).parent.is_none(),
            "instruction already placed"
        );
        let region = self.inst(anchor).parent.expect("anchor is placed");
        let prev = self.inst(anchor).prev;
        let gap = self.inst(anchor).order - prev.map_or(0, |before| self.inst(before).order);
        if gap < 2 {
            self.renumber(region);
        }
        let lower = prev.map_or(0, |before| self.inst(before).order);
        let order = lower + (self.inst(anchor).order - lower) / 2;
        {
            let data = self.inst_mut(inst);
            data.parent = Some(region);
            data.prev = prev;
            data.next = Some(anchor);
            data.order = order;
        }
        self.inst_mut(anchor).prev = Some(inst);
        match prev {
            Some(before) => self.inst_mut(before).next = Some(inst),
            None => self.region_mut(region).first = Some(inst),
        }
        self.refresh_holders(region);
    }

    /// Spreads the order keys of `region`'s instructions [`ORDER_GAP`] apart again.
    fn renumber(&mut self, region: Region) {
        let mut next = self.region(region).first;
        let mut order = 0;
        while let Some(inst) = next {
            order += ORDER_GAP;
            self.inst_mut(inst).order = order;
            next = self.inst(inst).next;
        }
    }

    /// Moves `inst`, which must be placed, right before `anchor`, which may be in another region.
    /// Its operands must be defined before `anchor`, and its results used after it.
    pub fn move_before(&mut self, _key: &EditKey, op: impl AsInst, anchor: impl AsInst) {
        let inst = op.inst();
        self.unlink(inst);
        self.link_before(anchor.inst(), inst);
    }

    /// Whether `first` comes before `second`, two instructions of one region.
    pub fn is_before(&self, first: Inst, second: Inst) -> bool {
        debug_assert_eq!(
            self.inst(first).parent,
            self.inst(second).parent,
            "the instructions are in different regions"
        );
        self.inst(first).order < self.inst(second).order
    }

    /// The instruction of `region` that holds `inst`: `inst` itself, or the op of `region` whose
    /// regions hold it, unless an op between them is isolated from above.
    fn ancestor_in(&self, inst: Inst, region: Region) -> Option<Inst> {
        let mut current = inst;
        loop {
            let parent = self.inst(current).parent?;
            if parent == region {
                return Some(current);
            }
            let op = self.region(parent).parent?;
            if self.op(op).has_trait::<IsolatedFromAbove>() {
                return None;
            }
            current = op;
        }
    }

    /// Whether `value` is defined where `user` reads it: an argument of a region that holds
    /// `user`, or the result of an instruction before `user` or before the op whose regions hold
    /// it. A region of an op isolated from above sees nothing defined outside it.
    pub fn dominates(&self, value: Value, user: Inst) -> bool {
        match self.def(value) {
            ValueDef::Arg(region, _) => self.ancestor_in(user, region).is_some(),
            ValueDef::Result(def, _) => {
                let Some(region) = self.inst(def).parent else {
                    return false;
                };
                self.ancestor_in(user, region)
                    .is_some_and(|holder| holder != def && self.is_before(def, holder))
            }
        }
    }

    /// Where the op of `inst` comes from, when that is known.
    pub fn location(&self, op: impl AsInst) -> Option<&Location> {
        self.inst(op).location.as_ref()
    }

    pub fn set_location(&mut self, _key: &EditKey, op: impl AsInst, location: Option<Location>) {
        self.inst_mut(op).location = location;
    }

    fn unlink(&mut self, inst: Inst) {
        self.number_for_trails();
        let Some(region) = self.inst(inst).parent else {
            return;
        };
        let (prev, next) = (self.inst(inst).prev, self.inst(inst).next);
        match prev {
            Some(before) => self.inst_mut(before).next = next,
            None => self.region_mut(region).first = next,
        }
        match next {
            Some(after) => self.inst_mut(after).prev = prev,
            None => self.region_mut(region).last = prev,
        }
        let data = self.inst_mut(inst);
        data.parent = None;
        data.prev = None;
        data.next = None;
        self.refresh_holders(region);
    }

    /// Removes block argument `index` of `region`, which nothing may use.
    pub fn remove_region_arg(
        &mut self,
        _key: &EditKey,
        region: Region,
        index: usize,
    ) -> Result<()> {
        let Some(&arg) = self.args(region).get(index) else {
            bail!("the region has no argument {index}");
        };
        if let Some(user) = self.uses(arg).first() {
            bail!(
                "region argument {index} is still used by {}",
                self.op(user.inst).name()
            );
        }
        self.region_mut(region).args.remove(index);
        for (position, later) in self
            .args(region)
            .to_vec()
            .into_iter()
            .enumerate()
            .skip(index)
        {
            self.value_mut(later).def = ValueDef::Arg(region, position);
        }
        Ok(())
    }

    /// Replaces every operand of `inst` with `operands`. The operand groups stay when the count
    /// does, and otherwise must follow from the op, which has at most one variable group.
    pub fn set_operands(&mut self, _key: &EditKey, op: impl AsInst, operands: Vec<Value>) {
        let inst = op.inst();
        for (index, &old) in self.inst(inst).operands.clone().iter().enumerate() {
            self.value_mut(old)
                .uses
                .retain(|use_site| !(use_site.inst == inst && use_site.index == index));
        }
        for (index, &value) in operands.iter().enumerate() {
            self.value_mut(value).uses.push(Use { inst, index });
        }
        let data = self.inst_mut(inst);
        if data.operands.len() != operands.len() {
            data.operand_segments = None;
        }
        data.operands = operands;
        self.refresh(vec![inst]);
    }

    pub fn set_operand(&mut self, _key: &EditKey, op: impl AsInst, index: usize, value: Value) {
        let inst = op.inst();
        let old = self.inst(inst).operands[index];
        self.value_mut(old)
            .uses
            .retain(|use_site| !(use_site.inst == inst && use_site.index == index));
        self.inst_mut(inst).operands[index] = value;
        self.value_mut(value).uses.push(Use { inst, index });
        self.refresh(vec![inst]);
    }

    /// Makes every use of `from` a use of `to`, terminators included.
    pub fn replace_all_uses(&mut self, _key: &EditKey, from: Value, to: Value) {
        if from == to {
            return;
        }
        let uses = std::mem::take(&mut self.value_mut(from).uses);
        for use_site in &uses {
            self.inst_mut(use_site.inst).operands[use_site.index] = to;
        }
        let users = uses.iter().map(|use_site| use_site.inst).collect();
        self.value_mut(to).uses.extend(uses);
        self.refresh(users);
    }

    /// Removes `inst` and everything in its regions. Fails while a result is still used outside.
    pub fn erase(&mut self, _key: &EditKey, op: impl AsInst) -> Result<()> {
        let inst = op.inst();
        ensure!(!self.inst(inst).erased, "instruction is already erased");
        let nested: Vec<Inst> = self
            .regions(inst)
            .iter()
            .flat_map(|&region| self.walk(region))
            .collect();
        let inside_set: HashSet<Inst> = nested.iter().copied().collect();
        let inside = |use_site: &Use| use_site.inst == inst || inside_set.contains(&use_site.inst);
        for &result in self.results(inst) {
            if let Some(user) = self.uses(result).iter().find(|use_site| !inside(use_site)) {
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
            ValueDef::Result(_, index) | ValueDef::Arg(_, index) => index,
        }
    }

    fn drop_inst(&mut self, inst: Inst) {
        self.unlink(inst);
        let operands = std::mem::take(&mut self.inst_mut(inst).operands);
        for (index, operand) in operands.into_iter().enumerate() {
            self.value_mut(operand)
                .uses
                .retain(|use_site| !(use_site.inst == inst && use_site.index == index));
        }
        self.inst_mut(inst).erased = true;
    }

    pub fn operand_types(&self, op: impl AsInst) -> Vec<&Type> {
        self.operands(op)
            .iter()
            .map(|&operand| self.ty(operand))
            .collect()
    }

    /// The result types and result groups that `op` gives on `operands`, in the groups
    /// `segments`, with `regions`: what its type rule says. `rewritten` is the module a rewrite
    /// reads, when this module holds the ops it stages.
    pub fn infer_results(
        &self,
        op: &Op,
        operands: &[Value],
        segments: Option<&[usize]>,
        regions: &[Region],
        rewritten: Option<&Module>,
    ) -> Result<(Flat<Type>, Option<Vec<usize>>)> {
        let types: Vec<&Type> = operands.iter().map(|&operand| self.ty(operand)).collect();
        op.dyn_result_types(&types, segments, regions, self, rewritten)
    }

    pub fn result_types(&self, op: impl AsInst) -> Vec<&Type> {
        self.results(op)
            .iter()
            .map(|&result| self.ty(result))
            .collect()
    }

    /// The constant that defines `value`, if one does.
    pub fn constant(&self, value: Value) -> Option<&Dense> {
        self.op(self.def_inst(value)?).as_constant()
    }

    /// Whether `inst` can go: it and every op its regions hold are pure, no result is used, and it
    /// is neither a terminator nor an isolated op without results such as `func.func`, which no
    /// SSA value uses.
    pub fn is_trivially_dead(&self, inst: Inst) -> bool {
        let op = self.op(inst);
        !op.has_trait::<Terminator>()
            && !(op.has_trait::<IsolatedFromAbove>() && self.results(inst).is_empty())
            && self
                .results(inst)
                .iter()
                .all(|&result| self.uses(result).is_empty())
            && self.is_pure_within(inst)
    }

    /// Whether neither `inst` nor any op its regions hold has an effect: a loop whose body stores
    /// has the store's.
    fn is_pure_within(&self, inst: Inst) -> bool {
        self.op(inst).effects().is_pure()
            && self
                .regions(inst)
                .iter()
                .all(|&region| self.insts(region).all(|inner| self.is_pure_within(inner)))
    }

    /// Whether `value` is known before the run: a constant, a function argument bound to a
    /// parameter, or a result of a pure op whose operands are all known, and whose regions, if it
    /// has any, read only their own arguments and hold only pure ops.
    pub fn known_before_run(&self, value: Value) -> bool {
        self.value(value).facts.known_before_run
    }

    /// Whether a parameter goes into `value`: it is a function argument bound to one, or an op
    /// computes it from such a value.
    pub fn depends_on_parameter(&self, value: Value) -> bool {
        self.value(value).facts.depends_on_parameter
    }

    /// The facts of the results of `inst`, from its op and the facts of its operands.
    fn result_facts(&self, inst: Inst) -> Facts {
        let data = self.inst(inst);
        if data.results.is_empty() {
            return Facts::default();
        }
        let operands = || {
            data.operands
                .iter()
                .map(|&operand| self.value(operand).facts)
        };
        let isolated = data.regions.is_empty() || data.op.has_trait::<IsolatedFromAbove>();
        Facts {
            known_before_run: isolated
                && operands().all(|facts| facts.known_before_run)
                && self.is_pure_within(inst),
            depends_on_parameter: operands().any(|facts| facts.depends_on_parameter),
        }
    }

    fn arg_facts(&self, region: Region, index: usize) -> Facts {
        let param = self.is_param(region, index);
        Facts {
            known_before_run: param,
            depends_on_parameter: param,
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
                    insts.extend(self.uses(result).iter().map(|use_site| use_site.inst));
                }
            }
        }
    }

    /// Recomputes the facts of the ops that hold `region`, which depend on what their regions hold,
    /// and of everything computed from them.
    fn refresh_holders(&mut self, region: Region) {
        let mut holders = Vec::new();
        let mut current = self.parent_op(region);
        while let Some(holder) = current {
            holders.push(holder);
            current = self
                .parent_region(holder)
                .and_then(|outer| self.parent_op(outer));
        }
        self.refresh(holders);
    }

    /// Recomputes the facts of the block arguments of `region`, and of everything computed from
    /// them.
    fn refresh_args(&mut self, region: Region) {
        for (index, arg) in self.region(region).args.clone().into_iter().enumerate() {
            let facts = self.arg_facts(region, index);
            if std::mem::replace(&mut self.value_mut(arg).facts, facts) != facts {
                let users = self
                    .uses(arg)
                    .iter()
                    .map(|use_site| use_site.inst)
                    .collect();
                self.refresh(users);
            }
        }
    }

    /// Whether block argument `index` of `region` is a function argument bound to a parameter.
    fn is_param(&self, region: Region, index: usize) -> bool {
        self.parent_op(region)
            .and_then(|inst| self.op(inst).get::<Func>())
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

    /// The op of `inst` as a `T`, with its named groups, when it is one.
    pub fn op_as<T: InOp>(&self, inst: Inst) -> Option<OpRef<'_, T>> {
        OpRef::new(self, inst)
    }

    /// `value` read together with this module.
    pub fn value_ref(&self, value: impl Into<Value>) -> ValueRef<'_> {
        ValueRef {
            module: self,
            value: value.into(),
        }
    }

    /// `region` read together with this module.
    pub fn region_ref(&self, region: Region) -> RegionRef<'_> {
        RegionRef {
            module: self,
            region,
        }
    }

    /// The values that the terminator of `region` gives: what the region yields or returns.
    pub fn yielded(&self, region: Region) -> &[Value] {
        self.last_inst(region)
            .map_or(&[], |terminator| self.operands(terminator))
    }
}

/// A value of a module, read together with the module: what defines it, who uses it, its type and
/// what is known about it. It is two references, so it is free to copy and pass around.
#[derive(Clone, Copy)]
pub struct ValueRef<'m> {
    module: &'m Module,
    value: Value,
}

impl<'m> ValueRef<'m> {
    pub fn id(self) -> Value {
        self.value
    }

    pub fn module(self) -> &'m Module {
        self.module
    }

    /// The op that defines the value, when it is a `T`.
    pub fn defining_op<T: InOp>(self) -> Option<OpRef<'m, T>> {
        self.module.op_as(self.defining_inst()?)
    }

    /// The instruction that defines the value, unless it is a block argument.
    pub fn defining_inst(self) -> Option<Inst> {
        self.module.def_inst(self.value)
    }

    pub fn uses(self) -> &'m [Use] {
        self.module.uses(self.value)
    }

    pub fn has_one_use(self) -> bool {
        self.uses().len() == 1
    }

    /// The type of the value.
    pub fn value_type(self) -> &'m Type {
        self.module.ty(self.value)
    }

    /// The type of the value, a tensor. Panics otherwise: a verified module gives a tensor to
    /// every operand and result that a tensor op declares as one.
    pub fn tensor_type(self) -> &'m TensorType {
        let ty = self.value_type();
        ty.as_tensor()
            .unwrap_or_else(|| panic!("{:?} is {ty}, not a tensor", self.value))
    }

    /// See [`Module::known_before_run`].
    pub fn known_before_run(self) -> bool {
        self.module.known_before_run(self.value)
    }

    /// See [`Module::depends_on_parameter`].
    pub fn depends_on_parameter(self) -> bool {
        self.module.depends_on_parameter(self.value)
    }

    /// The constant that defines the value, if one does.
    pub fn constant(self) -> Option<&'m Dense> {
        self.module.constant(self.value)
    }

    /// The element of the constant that defines the value, when every element equals it.
    pub fn splat(self) -> Option<HostScalar> {
        let dense = self.constant()?;
        let element = Dense::splat(dense.dtype, [0usize; 0], dense.splat_bytes()?);
        HostScalar::from_dense(&element).ok()
    }
}

impl From<ValueRef<'_>> for Value {
    fn from(value: ValueRef) -> Value {
        value.value
    }
}

impl PartialEq for ValueRef<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value
    }
}

impl Eq for ValueRef<'_> {}

impl std::fmt::Debug for ValueRef<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        self.value.fmt(f)
    }
}

/// A region of a module, read together with the module.
#[derive(Clone, Copy)]
pub struct RegionRef<'m> {
    module: &'m Module,
    region: Region,
}

impl<'m> RegionRef<'m> {
    pub fn id(self) -> Region {
        self.region
    }

    pub fn module(self) -> &'m Module {
        self.module
    }

    /// The block arguments.
    pub fn args(self) -> Vec<ValueRef<'m>> {
        let module = self.module;
        module
            .args(self.region)
            .iter()
            .map(|&value| module.value_ref(value))
            .collect()
    }

    /// The instructions, in order.
    pub fn insts(self) -> InstIter<'m> {
        self.module.insts(self.region)
    }

    /// The values its terminator gives.
    pub fn yielded(self) -> Vec<ValueRef<'m>> {
        let module = self.module;
        module
            .yielded(self.region)
            .iter()
            .map(|&value| module.value_ref(value))
            .collect()
    }
}

impl From<RegionRef<'_>> for Region {
    fn from(region: RegionRef) -> Region {
        region.region
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
