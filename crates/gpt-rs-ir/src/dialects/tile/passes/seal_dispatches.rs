//! Outlining groups of blocks into dispatches.

use std::collections::{BTreeSet, HashMap, HashSet};

use anyhow::{anyhow, Result};

use crate::attr::SymbolRef;
use crate::dialects::func::Return;
use crate::dialects::host;
use crate::dialects::tile::{Block, Dispatch, Group};
use crate::index::IterList;
use crate::module::{Inst, Module, Region, Value};
use crate::passes::{pass, Pass};
use crate::rewrite::Builder;
use crate::traits::IsolatedFromAbove;
use crate::types::{Dim, Sym, Type};

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

impl Pass for SealDispatches {
    fn run(&self, module: &mut Module) -> Result<bool> {
        let mut sealed = 0;
        for func in module.funcs() {
            for root in groups(module, func) {
                seal(module, func, root)?;
                sealed += 1;
            }
        }
        SealDispatches::SEALED.add(sealed);
        Ok(sealed > 0)
    }
}

/// The `tile.group`s of `func` and its `tile.block`s outside a group, in order.
fn groups(module: &Module, func: Inst) -> Vec<Inst> {
    let mut roots = Vec::new();
    for &region in module.regions(func) {
        collect_groups(module, region, &mut roots);
    }
    roots
}

/// Adds to `roots` the groups of `region` and of the regions of its other ops, in order.
fn collect_groups(module: &Module, region: Region, roots: &mut Vec<Inst>) {
    for inst in module.insts(region) {
        let op = module.op(inst);
        if op.get::<Group>().is_some() || op.get::<Block>().is_some() {
            roots.push(inst);
            continue;
        }
        for &inner in module.regions(inst) {
            collect_groups(module, inner, roots);
        }
    }
}

/// The ops that a dispatch for `root` holds, and the values of those ops it gives.
fn members(module: &Module, root: Inst) -> (Vec<Inst>, Vec<Value>) {
    if module.op(root).get::<Group>().is_none() {
        return (vec![root], module.results(root).to_vec());
    }
    let body = module.regions(root)[0];
    let mut ops: Vec<Inst> = module.insts(body).collect();
    ops.pop();
    (ops, module.yielded(body).to_vec())
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
fn add_symbol_args(
    module: &Module,
    root: Inst,
    ops: &[Inst],
    captured: &mut Vec<Value>,
) -> Result<()> {
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
            return Ok(());
        };
        let binding = args
            .iter()
            .copied()
            .find(|&arg| syms_of([module.ty(arg)]).contains(sym))
            .ok_or_else(|| anyhow!("no argument in reach binds {{{}}}", sym.as_str()))?;
        captured.push(binding);
    }
}

/// Outlines `root` of `func` into a dispatch, which the function calls in its place.
fn seal(module: &mut Module, func: Inst, root: Inst) -> Result<()> {
    let (ops, given) = members(module, root);
    let mut args = captured(module, &ops);
    add_symbol_args(module, root, &ops, &mut args)?;
    let grid = module
        .op(root)
        .get::<Group>()
        .map_or_else(IterList::default, |group| group.grid);

    // The dispatch in a module of its own, whose text names it.
    let mut scratch = Module::new();
    let dispatch = outline(&mut scratch, module, &ops, &args, &given, grid)?;
    let hash = blake3::hash(scratch.to_text().as_bytes());
    let bytes: [u8; 16] = hash.as_bytes()[..16].try_into().expect("16 bytes");
    let name = format!("dispatch.{:032x}", u128::from_le_bytes(bytes));
    if module.symbol(&name).is_none() {
        let mut op = scratch
            .op(dispatch)
            .get::<Dispatch>()
            .expect("a dispatch")
            .clone();
        op.name = name.clone();
        scratch.set_op(dispatch, op.into());
        let mut map = HashMap::new();
        let mut builder = Builder::before(module, func);
        builder.create_copy(&scratch, dispatch, &mut map)?;
    }

    let mut builder = Builder::before(module, root);
    let call = builder.create(
        host::Dispatch {
            callee: SymbolRef(name),
        },
        args,
    );
    for (from, to) in module.results(root).to_vec().into_iter().zip(call) {
        module.replace_all_uses(from, to);
    }
    module.erase(root)
}

/// Builds into `scratch` the dispatch that copies `ops` of `module`, taking `args` and giving
/// `given`.
fn outline(
    scratch: &mut Module,
    module: &Module,
    ops: &[Inst],
    args: &[Value],
    given: &[Value],
    grid: IterList,
) -> Result<Inst> {
    let arg_types: Vec<Type> = args.iter().map(|&arg| module.ty(arg).clone()).collect();
    let results: Vec<Type> = given
        .iter()
        .map(|&value| module.ty(value).clone())
        .collect();
    let body = scratch.new_region(arg_types.clone());
    let mut map: HashMap<Value, Value> = args
        .iter()
        .copied()
        .zip(scratch.args(body).to_vec())
        .collect();
    let mut body_builder = Builder::at_end(scratch, body);
    for &inst in ops {
        body_builder.create_copy(module, inst, &mut map)?;
    }
    let returned: Vec<Value> = given.iter().map(|value| map[value]).collect();
    body_builder.create(Return, returned);
    let dispatch = Dispatch {
        name: "dispatch".to_string(),
        args: arg_types,
        results,
        grid,
        facts: Default::default(),
        schedule: Default::default(),
    };
    let top = scratch.body();
    let mut top_builder = Builder::at_end(scratch, top);
    Ok(top_builder.create_op(dispatch.into(), Vec::new(), None, vec![body]))
}
