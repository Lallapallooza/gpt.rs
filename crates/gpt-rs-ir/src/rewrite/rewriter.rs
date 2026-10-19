//! [`Rewriter`]: what a pattern changes, staged while the pattern reads the module.
//!
//! A pattern reads the module through shared references, [`OpRef`](crate::ir::OpRef) and
//! [`ValueRef`](crate::ir::ValueRef), for as long as it runs, so the rewriter cannot change the
//! module while it does. It stages the changes instead: it builds new ops into a scratch module and
//! records the edits of existing ops, in order. When the pattern succeeds the driver applies them,
//! and when it fails the driver drops them, so a pattern that does not apply changes nothing.
//!
//! A value a pattern creates is a staged value until the changes apply: it goes into other
//! creates and edits, and the module does not know it yet.

use std::collections::HashMap;

use crate::dialects::func::{FuncOp, ParamId};
use crate::ir::{
    AsInst, Created, FoldIn, Groups, InOp, Inst, Location, Module, OpDyn, OpKind, Operands, Region,
    TensorType, Type, Value, Values,
};
use crate::rewrite::build::materialize;
use crate::rewrite::builder::StagedRegions;
use crate::rewrite::{Build, Builder, IntoValues, KEY};
use crate::Op;

/// What a pattern changes, staged until the driver applies it.
pub struct Rewriter<'m> {
    module: &'m Module,
    /// The instruction new ops go before: the pattern's root, unless the pattern moves it.
    insertion_point: Inst,
    /// The op that the ops the pattern creates are made from: its root, unless the pattern sets
    /// another ([`Rewriter::set_origin`]).
    origin: Inst,
    /// The location of the ops the pattern creates, once the pattern creates the first: the
    /// location of their origin, with the running pass on its trail.
    location: Option<Option<Location>>,
    staging: Option<Box<Staging>>,
    edits: Vec<Edit>,
}

/// The ops a pattern created, in a scratch module.
pub(crate) struct Staging {
    pub(crate) scratch: Module,
    /// The scratch value that stands for each value of the module a staged op reads.
    pub(crate) imports: HashMap<Value, Value>,
    /// The region that holds the imported values as its arguments.
    imports_region: Region,
    /// The region that holds the staged ops, in the order the pattern created them.
    staged_region: Region,
    /// Each staged op, with the instruction of the module it goes before.
    pub(crate) anchors: Vec<(Inst, Inst)>,
    /// The region that holds a placeholder argument for each argument a function gains.
    added_args_region: Region,
    /// The ops of the module that move into the regions of staged ops, in order.
    pub(crate) moves: Vec<Move>,
}

/// An op of the module that moves into a region of a staged op: before the op at `position` among
/// the staged ops of the region, all of which the builder of the region creates.
pub(crate) struct Move {
    pub(crate) op: Inst,
    pub(crate) region: Region,
    pub(crate) position: usize,
}

/// An edit of the module's existing IR. Values in it may be staged.
pub(crate) enum Edit {
    ReplaceOp {
        inst: Inst,
        values: Vec<Value>,
    },
    ReplaceAllUses {
        from: Value,
        to: Value,
    },
    SetOp {
        inst: Inst,
        op: Box<Op>,
    },
    SetOperands {
        inst: Inst,
        operands: Vec<Value>,
    },
    SetOperand {
        inst: Inst,
        index: usize,
        value: Value,
    },
    Erase(Inst),
    /// The ops of `region`, all but its terminator, moved before `before`, each argument of the
    /// region replaced by the value at its position in `args`.
    Inline {
        region: Region,
        before: Inst,
        args: Vec<Value>,
    },
    RemoveArg {
        region: Region,
        index: usize,
    },
    /// An argument of type `ty` that the function `func` gains, bound to the parameter `param`,
    /// which the scratch argument `placeholder` stands for.
    AddArg {
        func: Inst,
        ty: Type,
        param: Option<ParamId>,
        placeholder: Value,
    },
}

/// The changes of a pattern that succeeded, for the driver to apply.
pub struct Staged {
    pub(crate) staging: Option<Box<Staging>>,
    pub(crate) edits: Vec<Edit>,
}

impl Staged {
    /// Whether the pattern staged no change.
    pub(crate) fn is_empty(&self) -> bool {
        self.edits.is_empty()
            && self
                .staging
                .as_ref()
                .is_none_or(|staging| staging.anchors.is_empty())
    }
}

impl<'m> Rewriter<'m> {
    /// A rewriter for a pattern rooted at `root`: new ops go before it.
    pub(crate) fn new(module: &'m Module, root: Inst) -> Self {
        Self {
            module,
            insertion_point: root,
            origin: root,
            location: None,
            staging: None,
            edits: Vec::new(),
        }
    }

    pub(crate) fn finish(self) -> Staged {
        Staged {
            staging: self.staging,
            edits: self.edits,
        }
    }

    /// The module the pattern reads.
    pub fn module(&self) -> &'m Module {
        self.module
    }

    /// The type of `value`, a tensor, whether the module holds it or this rewrite created it.
    pub fn tensor_type(&self, value: impl Into<Value>) -> &TensorType {
        Build::tensor_type(self, value.into())
    }

    /// The location of the ops the pattern creates.
    fn created_location(&mut self) -> Option<Location> {
        let (module, origin) = (self.module, self.origin);
        self.location
            .get_or_insert_with(|| module.made_from(origin))
            .clone()
    }

    /// New ops are made from `op`, another op the pattern reads, from here on, in place of its
    /// root: they take its location, with the running pass on its trail.
    pub fn set_origin(&mut self, op: impl AsInst) {
        self.origin = op.inst();
        self.location = None;
    }

    /// New ops go before `inst` from here on.
    pub fn set_insertion_point(&mut self, inst: impl AsInst) {
        self.insertion_point = inst.inst();
    }

    /// Creates `op`, which has no regions, on `operands`, and gives its results.
    #[track_caller]
    pub fn create<T: InOp>(
        &mut self,
        op: T,
        operands: impl Into<Operands<T, Values>>,
    ) -> Created<T> {
        let (flat_operands, segments) = T::Groups::flatten_operands::<Values>(operands.into());
        let inst = self.stage(op.into(), &flat_operands, segments);
        let staging = self.staging();
        staged_results::<T>(&staging.scratch, inst)
    }

    /// Creates `op` on `operands`. `body(index, builder, args)` fills region `index`, whose
    /// block arguments are `args`, and ends it with its terminator. The builder builds into the
    /// staged region, on values of that region only.
    #[track_caller]
    pub fn create_with_regions<T: InOp>(
        &mut self,
        op: T,
        operands: impl Into<Operands<T, Values>>,
        body: impl FnMut(usize, &mut Builder, &[Value]),
    ) -> Created<T> {
        let (flat_operands, segments) = T::Groups::flatten_operands::<Values>(operands.into());
        let imported = self.import_all(&flat_operands);
        let anchor = self.insertion_point;
        let location = self.created_location();
        let mut builder = self.staged_builder();
        builder.set_location(location);
        let inst = builder.create_op_with_regions(op.into(), imported, segments, body);
        let staging = self.staging();
        staging.anchors.push((inst, anchor));
        staged_results::<T>(&staging.scratch, inst)
    }

    /// Creates the op value `op`, such as a copy of a matched op, on `operands`, and gives its
    /// results.
    #[track_caller]
    pub fn create_op(&mut self, op: Op, operands: impl IntoValues) -> Vec<Value> {
        let inst = self.stage(op, &operands.into_values(), None);
        self.staged_values(inst).0
    }

    /// Replaces every use of `op`'s results by `values`, in order, and erases `op`.
    pub fn replace_op(&mut self, op: impl AsInst, values: impl IntoValues) {
        self.edits.push(Edit::ReplaceOp {
            inst: op.inst(),
            values: values.into_values(),
        });
    }

    /// Makes every use of `from` use `to`.
    pub fn replace_all_uses(&mut self, from: impl Into<Value>, to: impl Into<Value>) {
        self.edits.push(Edit::ReplaceAllUses {
            from: from.into(),
            to: to.into(),
        });
    }

    /// Gives `op` the op value `new`, of the same kind, which computes the same results: what
    /// [`OpRef::update`](crate::ir::OpRef::update) stages.
    pub(crate) fn set_op(&mut self, op: impl AsInst, new: impl Into<Op>) {
        self.edits.push(Edit::SetOp {
            inst: op.inst(),
            op: Box::new(new.into()),
        });
    }

    /// Gives `op` the operands `operands`: as many as it has, in its groups, or any number for an
    /// op of one operand group.
    pub fn set_operands(&mut self, op: impl AsInst, operands: impl IntoValues) {
        self.edits.push(Edit::SetOperands {
            inst: op.inst(),
            operands: operands.into_values(),
        });
    }

    /// Makes operand `index` of `op` use `value`.
    pub fn set_operand(&mut self, op: impl AsInst, index: usize, value: impl Into<Value>) {
        self.edits.push(Edit::SetOperand {
            inst: op.inst(),
            index,
            value: value.into(),
        });
    }

    /// Erases `op`, whose results nothing may use once the edits before it apply.
    pub fn erase(&mut self, op: impl AsInst) {
        self.edits.push(Edit::Erase(op.inst()));
    }

    /// Moves the ops of `region`, all but its terminator, before `before`, each use of an argument
    /// of the region using the value at its position in `args`. The terminator stays in the region,
    /// for the op that holds it to go with.
    pub fn inline_region(
        &mut self,
        region: impl Into<Region>,
        before: impl AsInst,
        args: impl IntoValues,
    ) {
        self.edits.push(Edit::Inline {
            region: region.into(),
            before: before.inst(),
            args: args.into_values(),
        });
    }

    /// Removes block argument `index` of `region`, which nothing may use once the edits before
    /// it apply. A function loses the argument from its signature too, with the parameter it
    /// binds.
    pub fn remove_arg(&mut self, region: impl Into<Region>, index: usize) {
        self.edits.push(Edit::RemoveArg {
            region: region.into(),
            index,
        });
    }

    /// Adds an argument of type `ty` after the arguments of the function `func`, bound to the
    /// parameter `param`, and gives it.
    pub fn add_arg(&mut self, func: &FuncOp, ty: Type, param: Option<ParamId>) -> Value {
        let staging = self.staging();
        let placeholder = staging
            .scratch
            .append_arg(&KEY, staging.added_args_region, ty.clone());
        self.edits.push(Edit::AddArg {
            func: func.inst(),
            ty,
            param,
            placeholder,
        });
        staged(placeholder)
    }

    /// Adds a copy of `symbol`, a top-level op of `source` such as a function or a dispatch,
    /// before the top-level op that holds the insertion point. A symbol whose text names it, such
    /// as a dispatch named by the hash of its body, is built in a module of its own first.
    pub fn add_symbol(&mut self, source: &Module, symbol: impl AsInst) {
        let anchor = self.module.top_level_op(self.insertion_point);
        let location = self.created_location();
        let staging = self.staging();
        // At the top of the scratch module, where the type rules of staged ops that name it find it.
        let top = staging.scratch.body();
        let mut builder = Builder::at_end(&mut staging.scratch, top);
        let copy = builder
            .create_copy(source, symbol, &mut HashMap::new())
            .expect("a symbol reads no value from outside it");
        staging.scratch.set_location(&KEY, copy, location);
        staging.anchors.push((copy, anchor));
    }

    /// Creates the op value `op`, which has no regions, on `operands` in the groups `segments`, in
    /// the scratch module, before the insertion point.
    #[track_caller]
    fn stage(&mut self, op: Op, operands: &[Value], segments: Option<Vec<usize>>) -> Inst {
        let imported = self.import_all(operands);
        let anchor = self.insertion_point;
        let location = self.created_location();
        let mut builder = self.staged_builder();
        builder.set_location(location);
        let inst = builder.create_op(op, imported, segments, Vec::new());
        self.staging().anchors.push((inst, anchor));
        inst
    }

    /// The builder of staged ops, at the end of the staged region: the type rules of the ops it
    /// creates also see the symbols of the module the rewrite reads, and the builders of their
    /// regions move ops of that module.
    fn staged_builder(&mut self) -> Builder<'_> {
        let source = self.module;
        let staging = self.staging();
        let staged = StagedRegions {
            source,
            moves: &mut staging.moves,
            imports: &mut staging.imports,
            imports_region: staging.imports_region,
        };
        Builder::staged(&mut staging.scratch, staging.staged_region, staged)
    }

    /// The results of the staged op `inst`, as staged values, and their groups.
    fn staged_values(&self, inst: Inst) -> (Vec<Value>, Option<Vec<usize>>) {
        let scratch = &self
            .staging
            .as_ref()
            .expect("a staged op comes from a create")
            .scratch;
        let values = scratch
            .results(inst)
            .iter()
            .map(|result| staged(*result))
            .collect();
        (values, scratch.result_segments(inst).map(<[usize]>::to_vec))
    }

    /// What the op value `op` on `operands` folds to by its own folder, which reads the operands
    /// as the module or this rewrite holds them, with the results' groups. The driver applies the
    /// folds of the op's traits once the changes apply.
    #[track_caller]
    fn fold_staged(
        &mut self,
        op: &Op,
        operands: &[Value],
        segments: Option<&[usize]>,
    ) -> Option<(Vec<Value>, Option<Vec<usize>>)> {
        let scratch = self.staging.as_deref().map(|staging| &staging.scratch);
        let mut ins = Vec::new();
        for &value in operands {
            let (holder, local) = match value.staged_index() {
                Some(index) => (
                    scratch.expect("a staged value comes from a create"),
                    Value::at(index),
                ),
                None => (self.module, value),
            };
            ins.push(FoldIn {
                value,
                ty: holder.ty(local),
                constant: holder.constant(local),
            });
        }
        let types: Vec<&Type> = ins.iter().map(|input| input.ty).collect();
        let (module, rewritten) = match scratch {
            Some(scratch_module) => (scratch_module, Some(self.module)),
            None => (self.module, None),
        };
        let (result_types, result_segments) = op
            .dyn_result_types(&types, segments, &[], module, rewritten)
            .unwrap_or_else(|err| panic!("building {}: {err:#}", op.name()));
        let folded = op.dyn_fold(&ins, segments, &result_types)?;
        let values = materialize(folded, &result_types, |constant| {
            self.create_op(constant, Vec::<Value>::new())[0]
        });
        Some((values, result_segments))
    }

    fn staging(&mut self) -> &mut Staging {
        self.staging.get_or_insert_with(|| {
            let mut scratch = Module::new();
            let imports_region = scratch.new_region(Vec::new());
            let staged_region = scratch.new_region(Vec::new());
            let added_args_region = scratch.new_region(Vec::new());
            Box::new(Staging {
                scratch,
                imports: HashMap::new(),
                imports_region,
                staged_region,
                anchors: Vec::new(),
                added_args_region,
                moves: Vec::new(),
            })
        })
    }

    /// The scratch values that stand for `values`: a staged value is its scratch value, and a
    /// value of the module is imported once, as an argument of its type.
    fn import_all(&mut self, values: &[Value]) -> Vec<Value> {
        let module = self.module;
        let staging = self.staging();
        values
            .iter()
            .map(|&value| match value.staged_index() {
                Some(index) => Value::at(index),
                None => *staging.imports.entry(value).or_insert_with(|| {
                    let ty = module.ty(value).clone();
                    staging.scratch.append_arg(&KEY, staging.imports_region, ty)
                }),
            })
            .collect()
    }
}

impl Build for Rewriter<'_> {
    fn create_with_regions<T: InOp>(
        &mut self,
        op: T,
        operands: impl Into<Operands<T, Values>>,
        body: impl FnMut(usize, &mut Builder, &[Value]),
    ) -> Created<T> {
        Rewriter::create_with_regions(self, op, operands, body)
    }

    #[track_caller]
    fn create_or_fold_op(
        &mut self,
        op: Op,
        operands: Vec<Value>,
        segments: Option<Vec<usize>>,
    ) -> (Vec<Value>, Option<Vec<usize>>) {
        if let Some(folded) = self.fold_staged(&op, &operands, segments.as_deref()) {
            return folded;
        }
        let inst = self.stage(op, &operands, segments);
        self.staged_values(inst)
    }

    #[track_caller]
    fn create<T: InOp>(&mut self, op: T, operands: impl Into<Operands<T, Values>>) -> Created<T> {
        Rewriter::create(self, op, operands)
    }

    fn value_type(&self, value: Value) -> &Type {
        match value.staged_index() {
            Some(index) => {
                let staging = self
                    .staging
                    .as_ref()
                    .expect("a staged value comes from a create");
                staging.scratch.ty(Value::at(index))
            }
            None => self.module.ty(value),
        }
    }
}

/// The staged value that stands for the scratch value `value`.
fn staged(value: Value) -> Value {
    Value::staged(value.index())
}

/// The results of the staged op `inst` as `create` gives them, as staged values.
fn staged_results<T: OpKind>(scratch: &Module, inst: Inst) -> Created<T> {
    let results = T::Groups::split_results::<Values>(
        scratch.results(inst).iter().map(|&result| staged(result)),
        scratch.result_segments(inst),
    )
    .expect("the type rule gave the result groups");
    T::Groups::created(results)
}
