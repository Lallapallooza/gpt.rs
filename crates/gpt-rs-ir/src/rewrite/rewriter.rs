//! [`Rewriter`]: what a pattern changes, staged while the pattern reads the module.
//!
//! A pattern reads the module through shared references, [`Matched`](crate::op::Matched) and
//! [`ValueRef`](crate::module::ValueRef), for as long as it runs, so the rewriter cannot change the
//! module while it does. It stages the changes instead: it builds new ops into a scratch module and
//! records the edits of existing ops, in order. When the pattern succeeds the driver applies them,
//! and when it fails the driver drops them, so a pattern that does not apply changes nothing.
//!
//! A value a pattern creates is a staged value until the changes apply: it goes into other
//! creates and edits, and the module does not know it yet.

use std::collections::HashMap;

use crate::location::Location;
use crate::module::{Inst, Module, Region, Value};
use crate::op::{Created, FoldIn, Groups, InOp, OpDyn, OpKind, Operands, Values};
use crate::rewrite::build::materialize;
use crate::rewrite::{AsInst, Build, Builder, IntoValues};
use crate::types::{TensorType, Type};
use crate::Op;

/// What a pattern changes, staged until the driver applies it.
pub struct Rewriter<'m> {
    module: &'m Module,
    /// The instruction new ops go before: the pattern's root, unless the pattern moves it.
    insertion_point: Inst,
    /// The location of the ops the pattern creates: its root's.
    location: Option<Location>,
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
}

/// An edit of the module's existing IR. Values in it may be staged.
pub(crate) enum Edit {
    ReplaceOp { inst: Inst, values: Vec<Value> },
    ReplaceAllUses { from: Value, to: Value },
    SetOp { inst: Inst, op: Box<Op> },
    SetOperands { inst: Inst, operands: Vec<Value> },
    Erase(Inst),
    RemoveArg { region: Region, index: usize },
}

/// The changes of a pattern that succeeded, for the driver to apply.
pub struct Staged {
    pub(crate) staging: Option<Box<Staging>>,
    pub(crate) edits: Vec<Edit>,
}

impl Staged {
    /// Whether the pattern staged no change.
    pub(crate) fn is_empty(&self) -> bool {
        self.edits.is_empty() && self.staging.as_ref().is_none_or(|s| s.anchors.is_empty())
    }
}

impl<'m> Rewriter<'m> {
    /// A rewriter for a pattern rooted at `root`: new ops go before it.
    pub(crate) fn new(module: &'m Module, root: Inst) -> Self {
        Self {
            module,
            insertion_point: root,
            location: module.location(root).cloned(),
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
        let value = value.into();
        let ty = match value.staged_index() {
            Some(index) => {
                let staging = self
                    .staging
                    .as_ref()
                    .expect("a staged value comes from a create");
                staging.scratch.ty(Value::at(index))
            }
            None => self.module.ty(value),
        };
        ty.as_tensor()
            .unwrap_or_else(|| panic!("{value:?} is {ty}, not a tensor"))
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
        let (operands, segments) = T::Groups::flatten_operands::<Values>(operands.into());
        let inst = self.stage(op.into(), &operands, segments);
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
        body: impl FnMut(usize, &mut Builder<'_>, &[Value]),
    ) -> Created<T> {
        let (operands, segments) = T::Groups::flatten_operands::<Values>(operands.into());
        let operands = self.import_all(&operands);
        let anchor = self.insertion_point;
        let location = self.location.clone();
        let staging = self.staging();
        let mut builder = Builder::at_end(&mut staging.scratch, staging.staged_region);
        builder.set_location(location);
        let inst = builder.create_op_with_regions(op.into(), operands, segments, body);
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

    /// Gives `op` the op value `new`, which computes the same results, such as the same op with
    /// its attributes in canonical form.
    pub fn set_op(&mut self, op: impl AsInst, new: impl Into<Op>) {
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

    /// Erases `op`, whose results nothing may use once the edits before it apply.
    pub fn erase(&mut self, op: impl AsInst) {
        self.edits.push(Edit::Erase(op.inst()));
    }

    /// Removes block argument `index` of `region`, which nothing may use once the edits before
    /// it apply.
    pub fn remove_arg(&mut self, region: impl Into<Region>, index: usize) {
        self.edits.push(Edit::RemoveArg {
            region: region.into(),
            index,
        });
    }

    /// Creates the op value `op`, which has no regions, on `operands` in the groups `segments`, in
    /// the scratch module, before the insertion point.
    #[track_caller]
    fn stage(&mut self, op: Op, operands: &[Value], segments: Option<Vec<usize>>) -> Inst {
        let operands = self.import_all(operands);
        let anchor = self.insertion_point;
        let location = self.location.clone();
        let staging = self.staging();
        let mut builder = Builder::at_end(&mut staging.scratch, staging.staged_region);
        builder.set_location(location);
        let inst = builder.create_op(op, operands, segments, Vec::new());
        staging.anchors.push((inst, anchor));
        inst
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
        let (result_types, result_segments) = op
            .dyn_result_types(&types, segments, &[], self.module)
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
            Box::new(Staging {
                scratch,
                imports: HashMap::new(),
                imports_region,
                staged_region,
                anchors: Vec::new(),
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
                    staging.scratch.append_arg(staging.imports_region, ty)
                }),
            })
            .collect()
    }
}

impl Build for Rewriter<'_> {
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

    fn tensor_type(&self, value: Value) -> &TensorType {
        Rewriter::tensor_type(self, value)
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
