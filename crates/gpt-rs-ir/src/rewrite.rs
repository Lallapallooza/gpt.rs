//! Building and rewriting IR: [`Builder`], [`Rewriter`], and [`Pattern`]s rooted at an op type
//! or at every op with a trait.
//!
//! The builder takes result types from the op's `result_types`, never from its caller.

use std::marker::PhantomData;

use anyhow::{bail, ensure, Result};

use crate::dense::Dense;
use crate::dialects::constant_op;
use crate::dialects::func::{Func, ParamId, Return};
use crate::doc::{short_name, Doc};
use crate::fold::{fold_op, FoldResult};
use crate::module::{Inst, Module, NewInst, Region, Value};
use crate::op::{Folded, InOp, Matched, OpDef, OpDyn, Values};
use crate::traits::OpTrait;
use crate::types::Type;
use crate::Op;

pub use gpt_rs_ir_macros::pattern;

/// Where a builder places new instructions.
#[derive(Debug, Clone, Copy)]
enum InsertPoint {
    End(Region),
    Before(Inst),
}

pub struct Builder<'m> {
    pub module: &'m mut Module,
    point: InsertPoint,
    /// Instructions this builder created, in order.
    created: Vec<Inst>,
}

impl<'m> Builder<'m> {
    pub fn at_end(module: &'m mut Module, region: Region) -> Self {
        Self {
            module,
            point: InsertPoint::End(region),
            created: Vec::new(),
        }
    }

    pub fn before(module: &'m mut Module, inst: Inst) -> Self {
        Self {
            module,
            point: InsertPoint::Before(inst),
            created: Vec::new(),
        }
    }

    fn place(&mut self, inst: Inst) {
        match self.point {
            InsertPoint::End(region) => self.module.append(region, inst),
            InsertPoint::Before(anchor) => self.module.insert_before(anchor, inst),
        }
        self.created.push(inst);
    }

    /// Builds `op` from already created regions, taking result types from its `result_types`.
    pub fn build_op(
        &mut self,
        op: Op,
        operands: Vec<Value>,
        operand_segments: Option<Vec<usize>>,
        regions: Vec<Region>,
    ) -> Result<Inst> {
        let types: Vec<&Type> = operands.iter().map(|&v| self.module.ty(v)).collect();
        let (result_types, result_segments) = op
            .dyn_result_types(&types, operand_segments.as_deref(), &regions, self.module)
            .map_err(|err| anyhow::anyhow!("{}: {err:#}", op.name()))?;
        let inst = self.module.create_inst(NewInst {
            op,
            operands,
            operand_segments,
            result_types: result_types.into_vec(),
            result_segments,
            regions,
        });
        self.place(inst);
        Ok(inst)
    }

    /// Builds an op without regions and returns its named results.
    pub fn build<T: OpDef + InOp>(
        &mut self,
        op: T,
        ins: impl Into<T::Operands<Values>>,
    ) -> Result<T::Results<Values>> {
        ensure!(
            T::REGIONS.is_empty(),
            "{} has regions; use build_regions",
            op.name()
        );
        let (operands, segments) = T::flatten_operands::<Values>(ins.into());
        let inst = self.build_op(op.into(), operands.into_vec(), segments, Vec::new())?;
        T::split_results::<Values>(
            self.module.results(inst).iter().copied(),
            self.module.result_segments(inst),
        )
    }

    /// Builds an op with regions. `body(index, builder, args)` fills region `index`, whose block
    /// arguments are `args`, and must end it with its terminator.
    pub fn build_regions<T: OpDef + InOp>(
        &mut self,
        op: T,
        ins: impl Into<T::Operands<Values>>,
        mut body: impl FnMut(usize, &mut Builder<'_>, &[Value]) -> Result<()>,
    ) -> Result<T::Results<Values>> {
        let (operands, segments) = T::flatten_operands::<Values>(ins.into());
        let types: Vec<&Type> = operands.iter().map(|&v| self.module.ty(v)).collect();
        let op = op.into();
        let sigs = op.dyn_region_signatures(&types, segments.as_deref())?;
        let mut regions = Vec::new();
        for (index, sig) in sigs.into_iter().enumerate() {
            let region = self.module.new_region(sig.args);
            let args = self.module.args(region).to_vec();
            let mut inner = Builder::at_end(self.module, region);
            body(index, &mut inner, &args)?;
            regions.push(region);
        }
        let inst = self.build_op(op, operands.into_vec(), segments, regions)?;
        T::split_results::<Values>(
            self.module.results(inst).iter().copied(),
            self.module.result_segments(inst),
        )
    }

    /// Builds a constant with the value `dense`.
    pub fn constant(&mut self, dense: Dense) -> Result<Value> {
        let inst = self.build_op(constant_op(dense), Vec::new(), None, Vec::new())?;
        Ok(self.module.results(inst)[0])
    }

    /// Builds `op`, unless it folds: then returns what it folds to and builds at most constants.
    pub fn build_or_fold<T: OpDef + InOp>(
        &mut self,
        op: T,
        ins: impl Into<T::Operands<Values>>,
    ) -> Result<T::Results<Values>> {
        let (operands, segments) = T::flatten_operands::<Values>(ins.into());
        let mut operands = operands.into_vec();
        let op = op.into();
        let types: Vec<&Type> = operands.iter().map(|&v| self.module.ty(v)).collect();
        let (result_types, result_segments) = op
            .dyn_result_types(&types, segments.as_deref(), &[], self.module)
            .map_err(|err| anyhow::anyhow!("{}: {err:#}", op.name()))?;
        match fold_op(
            self.module,
            &op,
            &operands,
            segments.as_deref(),
            &result_types,
        ) {
            Some(FoldResult::Results(folded)) => {
                let values = self.materialize(folded)?;
                return T::split_results::<Values>(values.into_iter(), result_segments.as_deref());
            }
            Some(FoldResult::Operands(canonical)) => operands = canonical,
            None => {}
        }
        let inst = self.build_op(op, operands, segments, Vec::new())?;
        T::split_results::<Values>(
            self.module.results(inst).iter().copied(),
            self.module.result_segments(inst),
        )
    }

    /// The values of folded results, building a constant for each `Folded::Constant`.
    fn materialize(&mut self, folded: Vec<Folded>) -> Result<Vec<Value>> {
        folded
            .into_iter()
            .map(|f| match f {
                Folded::Value(value) => Ok(value),
                Folded::Constant(dense) => self.constant(dense),
            })
            .collect()
    }

    /// Ends the current function body.
    pub fn ret(&mut self, values: &[Value]) -> Result<()> {
        self.build(Return, values.to_vec())?;
        Ok(())
    }
}

impl Module {
    /// Adds `func.func @name`. `body(builder, args)` builds the body and returns the values to
    /// return. The function's result types are the types of those values.
    pub fn build_func(
        &mut self,
        name: &str,
        args: Vec<Type>,
        params: Vec<Option<ParamId>>,
        body: impl FnOnce(&mut Builder<'_>, &[Value]) -> Result<Vec<Value>>,
    ) -> Result<Inst> {
        let region = self.new_region(args.clone());
        let arg_values = self.args(region).to_vec();
        let mut builder = Builder::at_end(self, region);
        let returned = body(&mut builder, &arg_values)?;
        builder.ret(&returned)?;
        let results = returned.iter().map(|&v| self.ty(v).clone()).collect();
        let func = Func {
            name: name.to_string(),
            args,
            results,
            params: bound_params(params),
            result_params: Vec::new(),
        };
        let top = self.body();
        let mut builder = Builder::at_end(self, top);
        builder.build_op(func.into(), Vec::new(), None, vec![region])
    }
}

/// `params`, or an empty list when none of them is bound. A `Func` stores its `params` this way.
pub(crate) fn bound_params(params: Vec<Option<ParamId>>) -> Vec<Option<ParamId>> {
    if params.iter().any(Option::is_some) {
        params
    } else {
        Vec::new()
    }
}

/// Edits made by one pattern application.
pub struct Rewriter<'m> {
    pub module: &'m mut Module,
    /// Instructions the driver revisits: those this rewrite created or changed, and those that lost
    /// a use.
    pub touched: Vec<Inst>,
}

impl<'m> Rewriter<'m> {
    pub fn new(module: &'m mut Module) -> Self {
        Self {
            module,
            touched: Vec::new(),
        }
    }

    /// Builds `op` right before `at` from an op value, such as a copy of a matched op.
    pub fn insert_op_before(&mut self, at: Inst, op: Op, operands: Vec<Value>) -> Result<Inst> {
        let mut builder = Builder::before(self.module, at);
        let inst = builder.build_op(op, operands, None, Vec::new())?;
        self.touched.extend(builder.created);
        Ok(inst)
    }

    /// Builds a constant with the value `dense` right before `at`.
    pub fn constant_before(&mut self, at: Inst, dense: Dense) -> Result<Value> {
        let mut builder = Builder::before(self.module, at);
        let value = builder.constant(dense)?;
        self.touched.extend(builder.created);
        Ok(value)
    }

    /// Builds an op right before `at` and returns its named results.
    pub fn insert_before<T: OpDef + InOp>(
        &mut self,
        at: Inst,
        op: T,
        ins: impl Into<T::Operands<Values>>,
    ) -> Result<T::Results<Values>> {
        let mut builder = Builder::before(self.module, at);
        let results = builder.build(op, ins)?;
        self.touched.extend(builder.created);
        Ok(results)
    }

    /// Replaces every use of `inst`'s results, terminators included, then erases it.
    pub fn replace(&mut self, inst: Inst, with: &[Value]) -> Result<()> {
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
            self.touched
                .extend(self.module.uses(from).iter().map(|u| u.inst));
            self.module.replace_all_uses(from, to);
        }
        self.erase(inst)
    }

    /// Replaces `inst` by what it folded to: existing values, or constants built before it.
    fn apply_fold(&mut self, inst: Inst, folded: Vec<Folded>) -> Result<()> {
        let mut builder = Builder::before(self.module, inst);
        let values = builder.materialize(folded)?;
        self.touched.extend(builder.created);
        self.replace(inst, &values)
    }

    /// Replaces `inst` by `other`, which computes the same values.
    pub fn replace_with_inst(&mut self, inst: Inst, other: Inst) -> Result<()> {
        let values = self.module.results(other).to_vec();
        self.replace(inst, &values)
    }

    /// Gives `inst` new operands that compute the same results, such as its operands in canonical
    /// order.
    fn set_operands(&mut self, inst: Inst, operands: &[Value]) {
        debug_assert_eq!(operands.len(), self.module.operands(inst).len());
        for (index, &value) in operands.iter().enumerate() {
            self.module.set_operand(inst, index, value);
        }
        self.touched.push(inst);
    }

    /// Applies what `inst` folded to.
    pub fn apply_fold_result(&mut self, inst: Inst, fold: FoldResult) -> Result<()> {
        match fold {
            FoldResult::Results(folded) => self.apply_fold(inst, folded),
            FoldResult::Operands(operands) => {
                self.set_operands(inst, &operands);
                Ok(())
            }
        }
    }

    /// Replaces the op of `inst` by one that computes the same results, such as the same op with
    /// attributes in canonical form.
    pub fn set_op(&mut self, inst: Inst, op: impl Into<Op>) {
        self.module.set_op(inst, op.into());
        self.touched.push(inst);
    }

    /// Erases `inst`, and fails while its results are used. The driver revisits the ops that
    /// defined its operands, since they may have lost their last use.
    pub fn erase(&mut self, inst: Inst) -> Result<()> {
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

    pub fn def_as<T: InOp>(&self, value: Value) -> Option<Matched<T>> {
        self.module.def_as(value)
    }
}

/// What a pattern is rooted at: ops of one type, or every op that has a trait ([`Has`]).
pub trait Root {
    /// The root as the pattern receives it.
    type Matched;
    fn find(module: &Module, inst: Inst) -> Option<Self::Matched>;
    fn info() -> RootInfo;
}

/// What a pattern is rooted at, as the reference pages show it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RootInfo {
    /// The ops of the op struct of this name.
    Op(&'static str),
    /// Every op with the op trait of this name.
    Trait(&'static str),
}

impl<T: InOp> Root for T {
    type Matched = Matched<T>;
    fn find(module: &Module, inst: Inst) -> Option<Matched<T>> {
        module.op_as(inst)
    }
    fn info() -> RootInfo {
        RootInfo::Op(T::INFO.rust)
    }
}

/// Roots a pattern at every op that has the trait `T`.
pub struct Has<T>(PhantomData<T>);

/// An op found by its trait `T`.
pub struct WithTrait<T: 'static> {
    pub inst: Inst,
    pub op: Op,
    /// The op's `T`, with its parameters.
    pub fact: &'static T,
}

impl<T: OpTrait> Root for Has<T> {
    type Matched = WithTrait<T>;
    fn find(module: &Module, inst: Inst) -> Option<WithTrait<T>> {
        let op = module.op(inst);
        let fact = op.get_trait::<T>()?;
        Some(WithTrait {
            inst,
            op: op.clone(),
            fact,
        })
    }
    fn info() -> RootInfo {
        RootInfo::Trait(short_name::<T>())
    }
}

/// A local rewrite rooted at `Root`. `#[pattern]` on the struct implements [`Doc`] from its doc
/// comment, and `#[pattern(canonical)]` also registers it with `canonicalize`.
pub trait Pattern: Doc {
    type Root: Root;
    /// Rewrites `root` and returns `true`, or returns `false` and leaves the IR unchanged.
    fn rewrite(&self, root: <Self::Root as Root>::Matched, rw: &mut Rewriter<'_>) -> Result<bool>;
}

/// A [`Pattern`] whose root type is erased, so different patterns fit one list.
pub trait AnyPattern: Doc {
    fn try_rewrite(&self, inst: Inst, rw: &mut Rewriter<'_>) -> Result<bool>;
    fn root(&self) -> RootInfo;
}

impl<P: Pattern> AnyPattern for P {
    fn try_rewrite(&self, inst: Inst, rw: &mut Rewriter<'_>) -> Result<bool> {
        match P::Root::find(rw.module, inst) {
            Some(root) => self.rewrite(root, rw),
            None => Ok(false),
        }
    }
    fn root(&self) -> RootInfo {
        P::Root::info()
    }
}

/// The patterns a greedy rewrite tries on each op, in order.
#[derive(Default)]
pub struct PatternSet {
    patterns: Vec<Box<dyn AnyPattern>>,
}

impl PatternSet {
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds `pattern` at the end.
    pub fn add(&mut self, pattern: impl AnyPattern + 'static) -> &mut Self {
        self.patterns.push(Box::new(pattern));
        self
    }

    pub fn iter(&self) -> impl Iterator<Item = &dyn AnyPattern> {
        self.patterns.iter().map(Box::as_ref)
    }

    pub fn len(&self) -> usize {
        self.patterns.len()
    }

    pub fn is_empty(&self) -> bool {
        self.patterns.is_empty()
    }
}

/// The canonical patterns, each added by the function that `#[pattern(canonical)]` registers.
#[linkme::distributed_slice]
pub static CANONICAL_PATTERNS: [fn(&mut PatternSet)];

/// Every canonical pattern, in the order of their names: the patterns of `canonicalize`.
pub fn canonical_patterns() -> PatternSet {
    let mut set = PatternSet::new();
    for add in CANONICAL_PATTERNS {
        add(&mut set);
    }
    set.patterns.sort_by_key(|pattern| pattern.name());
    set
}
