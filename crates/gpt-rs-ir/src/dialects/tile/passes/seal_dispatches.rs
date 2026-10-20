//! Outlining groups of blocks into dispatches.

use std::collections::{BTreeSet, HashMap, HashSet};
use std::marker::PhantomData;

use anyhow::Result;

use crate::dialects::func::Return;
use crate::dialects::host;
use crate::dialects::tile::utils::in_function_outside_groups;
use crate::dialects::tile::{Block, BlockOp, Dispatch, Group, GroupOp};
use crate::ir::{Dim, InOp, Inst, Module, OpRef, Sym, SymbolRef, Type, Value};
use crate::passes::{pass, Pass};
use crate::require;
use crate::rewrite::{
    pattern, Builder, Greedy, NoMatch, Pattern, PatternSet, Rewriter, Rewrites, Rewriting,
};
use crate::traits::IsolatedFromAbove;

/// Outlines each group of blocks into a `tile.dispatch` that the function calls with
/// `host.dispatch`.
///
/// - Each `tile.group` becomes a dispatch, and so does each `tile.block` outside a group, as a
///   group of its own.
/// - The dispatch takes the values its blocks read from outside the group, in the order they are
///   first read, and gives what the group yields. When a block uses a dimension symbol that none
///   of those values binds, the dispatch also takes the nearest region argument that does: one of
///   the function, or of a region of control flow around the group.
/// - Its name is a hash of its text, so equal groups share one dispatch: layers that compute the
///   same blocks call one.
#[pass(
    name = "seal-dispatches",
    stats(sealed = "groups outlined into dispatches")
)]
pub struct SealDispatches;

fn populate_seal_dispatches_patterns(set: &mut PatternSet) {
    set.add(Seal::<Group>(PhantomData));
    set.add(Seal::<Block>(PhantomData));
}

impl Pass for SealDispatches {
    fn rewrites(&self) -> Rewrites {
        Rewrites::once(populate_seal_dispatches_patterns)
    }

    fn run(&self, module: &mut Rewriting) -> Result<bool> {
        Ok(Greedy::new(self.rewrites()).run(module)?.changed())
    }
}

/// What `seal-dispatches` outlines into a dispatch: a `tile.group`, or a `tile.block` outside
/// one, as a group of its own.
trait Sealed: InOp {
    /// The ops that the dispatch for `op` holds, and the values of those ops it gives.
    fn members(op: &OpRef<Self>) -> (Vec<Inst>, Vec<Value>);
}

impl Sealed for Group {
    fn members(group: &GroupOp) -> (Vec<Inst>, Vec<Value>) {
        let body = group.body();
        let mut ops: Vec<Inst> = body.insts().collect();
        ops.pop();
        let given = body.yielded().iter().map(|value| value.id()).collect();
        (ops, given)
    }
}

impl Sealed for Block {
    fn members(block: &BlockOp) -> (Vec<Inst>, Vec<Value>) {
        let given = block.outs().iter().map(|out| out.id()).collect();
        (vec![block.inst()], given)
    }
}

/// A group of a function, or a block of one outside a group, becomes a `tile.dispatch` that the
/// function calls with `host.dispatch` in its place. Equal groups share one dispatch.
#[pattern]
struct Seal<T>(PhantomData<T>);

impl<T: Sealed> Pattern for Seal<T> {
    type Root = T;

    fn rewrite(&self, root: OpRef<T>, rewriter: &mut Rewriter) -> Result<(), NoMatch> {
        let module = root.module();
        require!(
            in_function_outside_groups(module, root.inst()),
            "the op is in a group, or outside a function"
        );
        let (ops, given) = T::members(&root);
        let mut args = captured(module, &ops);
        add_symbol_args(module, root.inst(), &ops, &mut args);
        let outlined = Outlined {
            module,
            ops: &ops,
            args: &args,
            given: &given,
        };
        let name =
            Dispatch::add_named_by_text(rewriter, |scratch, name| outlined.build(scratch, name));
        let call = rewriter.create(
            host::Dispatch {
                callee: SymbolRef(name),
            },
            args,
        );
        rewriter.replace_op(&root, call);
        SealDispatches::SEALED.add(1);
        Ok(())
    }
}

/// The values the ops of `ops` read from outside them, in the order they are first read.
fn captured(module: &Module, ops: &[Inst]) -> Vec<Value> {
    let mut defined = HashSet::new();
    let mut captured = Vec::new();
    for &inst in ops {
        for &operand in module.operands(inst) {
            if !defined.contains(&operand) && !captured.contains(&operand) {
                captured.push(operand);
            }
        }
        defined.extend(module.results(inst).iter().copied());
    }
    captured
}

/// The dimension symbols that `types` name.
fn syms_of<'t>(types: impl IntoIterator<Item = &'t Type>) -> BTreeSet<Sym> {
    let mut syms = BTreeSet::new();
    for ty in types {
        let Some(tensor) = ty.as_tensor() else {
            continue;
        };
        for dim in &tensor.dims {
            if let Dim::Dynamic(sym) = dim {
                syms.insert(sym.clone());
            }
        }
    }
    syms
}

/// The region arguments that `root` sees, innermost first: those of each region around it, up to
/// and including the first region of an op isolated from above.
fn visible_args(module: &Module, root: Inst) -> Vec<Value> {
    let mut args = Vec::new();
    let mut region = module.parent_region(root);
    while let Some(current) = region {
        args.extend(module.args(current).iter().copied());
        let Some(parent) = module.parent_op(current) else {
            break;
        };
        if module.op(parent).has_trait::<IsolatedFromAbove>() {
            break;
        }
        region = module.parent_region(parent);
    }
    args
}

/// Adds to `captured` a region argument that `root` sees for each symbol that `ops` use and no
/// captured value binds.
fn add_symbol_args(module: &Module, root: Inst, ops: &[Inst], captured: &mut Vec<Value>) {
    let mut used = BTreeSet::new();
    for &inst in ops {
        for inner in module.walk_inst(inst) {
            if let Some(block) = module.op(inner).get::<Block>() {
                used.extend(block.syms(module, module.regions(inner)[0]));
            }
        }
        used.extend(syms_of(module.result_types(inst)));
    }
    let args = visible_args(module, root);
    loop {
        let bound = syms_of(captured.iter().map(|&value| module.ty(value)));
        let Some(sym) = used.iter().find(|sym| !bound.contains(*sym)) else {
            return;
        };
        let binding = args
            .iter()
            .copied()
            .find(|&arg| syms_of([module.ty(arg)]).contains(sym))
            .unwrap_or_else(|| panic!("no argument in reach binds {{{}}}", sym.as_str()));
        captured.push(binding);
    }
}

/// The dispatch that `seal-dispatches` outlines: it copies `ops` of `module`, takes `args` and gives
/// `given`.
struct Outlined<'a> {
    module: &'a Module,
    ops: &'a [Inst],
    args: &'a [Value],
    given: &'a [Value],
}

impl Outlined<'_> {
    /// Builds the dispatch into `scratch`, named `name`.
    fn build(&self, scratch: &mut Module, name: &str) -> Inst {
        let module = self.module;
        let arg_types: Vec<Type> = self
            .args
            .iter()
            .map(|&arg| module.ty(arg).clone())
            .collect();
        let results: Vec<Type> = self
            .given
            .iter()
            .map(|&value| module.ty(value).clone())
            .collect();
        let body = scratch.new_region(arg_types.clone());
        let mut map: HashMap<Value, Value> = self
            .args
            .iter()
            .copied()
            .zip(scratch.args(body).to_vec())
            .collect();
        let mut body_builder = Builder::at_end(scratch, body);
        for &inst in self.ops {
            body_builder
                .create_copy(module, inst, &mut map)
                .expect("a dispatch takes every value its ops read");
        }
        let returned: Vec<Value> = self.given.iter().map(|value| map[value]).collect();
        body_builder.create(Return, returned);
        let dispatch = Dispatch {
            name: name.to_string(),
            args: arg_types,
            results,
            facts: Default::default(),
            target: None,
            numerics: Default::default(),
        };
        let top = scratch.body();
        Builder::at_end(scratch, top).create_op(dispatch.into(), Vec::new(), None, vec![body])
    }
}
