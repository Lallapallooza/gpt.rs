//! Moving computation over parameters into load-time functions.

use std::collections::{HashMap, HashSet};

use anyhow::Result;

use super::{pass, Pass};
use crate::dialects::func::{Func, ParamId};
use crate::module::{Inst, Module, Region, Value, ValueDef};
use crate::traits::{Movement, Terminator};
use crate::types::Type;
use crate::Op;

/// Moves each static value that a run-time op reads into a load-time function, which the runtime
/// runs once when it loads the parameters. The function that reads the value gets a new argument
/// bound to the parameter the load-time function computes.
///
/// - What moves is the value before an op that only repeats its elements, such as a
///   `broadcast_to`, so the runtime keeps the smaller tensor.
/// - A static value computed from no parameter stays: as a parameter, it would be stored once for
///   each shape a program is compiled for.
/// - The values stored from a parameter take at most the parameter's bytes in total. The pass
///   allocates those bytes in the order the values are defined. So the values a function stores
///   never take more bytes than the parameters they read. A value that does not fit, such as a
///   weight cast to a wider dtype, stays, and the static values it reads move instead. A value or
///   parameter of unknown size never fits.
/// - The parameter's id is a hash of the load-time function's text, so equal computations share
///   one parameter across functions and programs. A load-time function can read the result of
///   one defined before it.
#[pass(
    name = "outline-static",
    stats(outlined = "values moved into load-time functions")
)]
pub struct OutlineStatic;

impl Pass for OutlineStatic {
    fn run(&self, module: &mut Module) -> Result<bool> {
        let mut outlined = 0;
        for func in module.funcs() {
            outlined += outline(module, func)?;
        }
        OutlineStatic::OUTLINED.add(outlined);
        Ok(outlined > 0)
    }
}

/// A static value to move, and the operands that read it at run time.
struct Root {
    value: Value,
    uses: Vec<(Inst, usize)>,
}

/// The number of values of `func` it moves into load-time functions.
fn outline(module: &mut Module, func: Inst) -> Result<usize> {
    if module
        .op(func)
        .as_func()
        .expect("a function")
        .is_load_time()
    {
        return Ok(0);
    }
    let body = module.regions(func)[0];
    let mut outlined = 0;
    for root in roots(module, body) {
        let Some(id) = load_time_function(module, root.value)? else {
            continue;
        };
        let arg = add_param_arg(module, func, module.ty(root.value).clone(), id);
        for (inst, index) in root.uses {
            module.set_operand(inst, index, arg);
        }
        outlined += 1;
    }
    Ok(outlined)
}

/// Whether `inst` runs at run time: it is the terminator, or one of its results is not static.
fn runs_at_run_time(module: &Module, statics: &HashSet<Value>, inst: Inst) -> bool {
    module.op(inst).has_trait::<Terminator>()
        || module.results(inst).iter().any(|r| !statics.contains(r))
}

/// The input of `inst` when it only repeats the elements of its one operand into more of them.
fn expansion_input(module: &Module, inst: Inst) -> Option<Value> {
    let &[input] = module.operands(inst) else {
        return None;
    };
    let &[output] = module.results(inst) else {
        return None;
    };
    let count = |v: Value| module.ty(v).as_tensor()?.element_count();
    let expands = count(output)? > count(input)?;
    (module.op(inst).has_trait::<Movement>() && expands).then_some(input)
}

/// The static values of `body` read at run time, in the order they are defined, each moved back
/// through the expansion that produces it. A value that does not fit its parameters' bytes stays
/// at run time with everything computed from it, until every root fits.
fn roots(module: &Module, body: Region) -> Vec<Root> {
    let mut statics: HashSet<Value> = module
        .insts(body)
        .flat_map(|inst| module.results(inst).iter().copied())
        .filter(|&v| module.is_static(v))
        .collect();
    loop {
        let roots = roots_among(module, body, &statics);
        let unfit = unfit(module, &roots);
        if unfit.is_empty() {
            return roots;
        }
        drop_dependents(module, body, &mut statics, unfit);
    }
}

/// The bytes a value of type `ty` takes, when its size is known.
fn bytes(ty: &Type) -> Option<usize> {
    let tensor = ty.as_tensor()?;
    tensor
        .element_count()?
        .checked_mul(tensor.dtype.size_in_bytes()?)
}

/// The roots that do not fit the bytes their parameters have left, taking the roots in order.
/// A root that fits takes its bytes from its parameters in id order.
fn unfit(module: &Module, roots: &[Root]) -> HashSet<Value> {
    let mut left: HashMap<ParamId, Option<usize>> = HashMap::new();
    let mut unfit = HashSet::new();
    for root in roots {
        let mut params = subgraph(module, root.value).params;
        params.dedup_by_key(|&mut (_, id, _)| id);
        for (_, id, ty) in &params {
            left.entry(*id).or_insert_with(|| bytes(ty));
        }
        let available: Option<usize> = params.iter().map(|(_, id, _)| left[id]).sum();
        let fits = |need: &usize| available.is_some_and(|available| *need <= available);
        let Some(mut need) = bytes(module.ty(root.value)).filter(fits) else {
            unfit.insert(root.value);
            continue;
        };
        for (_, id, _) in &params {
            let have = left
                .get_mut(id)
                .and_then(Option::as_mut)
                .expect("a known size");
            let taken = need.min(*have);
            *have -= taken;
            need -= taken;
        }
    }
    unfit
}

/// Removes `dropped` from `statics`, with every value of `body` computed from them.
fn drop_dependents(
    module: &Module,
    body: Region,
    statics: &mut HashSet<Value>,
    mut dropped: HashSet<Value>,
) {
    for inst in module.insts(body) {
        if module.operands(inst).iter().any(|v| dropped.contains(v)) {
            dropped.extend(module.results(inst).iter().copied());
        }
    }
    statics.retain(|v| !dropped.contains(v));
}

/// [`roots`] when `statics` are the static values.
fn roots_among(module: &Module, body: Region, statics: &HashSet<Value>) -> Vec<Root> {
    let mut roots: Vec<Root> = Vec::new();
    for inst in module.insts(body) {
        for &value in module.results(inst) {
            if !statics.contains(&value) {
                continue;
            }
            let uses: Vec<(Inst, usize)> = module
                .uses(value)
                .iter()
                .filter(|u| runs_at_run_time(module, statics, u.inst))
                .map(|u| (u.inst, u.index))
                .collect();
            if uses.is_empty() {
                continue;
            }
            // Through an expansion, the run-time op reads the expansion's input instead.
            let (value, uses) = match expansion_input(module, inst) {
                Some(input) => (input, vec![(inst, 0)]),
                None => (value, uses),
            };
            match roots.iter_mut().find(|r| r.value == value) {
                Some(root) => root.uses.extend(uses),
                None => roots.push(Root { value, uses }),
            }
        }
    }
    roots.retain(|r| matches!(module.def(r.value), ValueDef::Result(..)));
    roots
}

/// What computes a static value: its parameters, ordered by id, and its ops in order.
struct Subgraph {
    params: Vec<(Value, ParamId, Type)>,
    ops: Vec<Inst>,
    output: Value,
}

fn subgraph(module: &Module, output: Value) -> Subgraph {
    let mut seen: HashSet<Value> = HashSet::new();
    let mut ops: HashSet<Inst> = HashSet::new();
    let mut params = Vec::new();
    let mut stack = vec![output];
    while let Some(value) = stack.pop() {
        if !seen.insert(value) {
            continue;
        }
        match module.def(value) {
            ValueDef::Arg(region, index) => {
                let func = module.parent_op(region).expect("a function argument");
                let id = module.op(func).as_func().and_then(|f| f.param(index));
                let id = id.expect("a static argument is a parameter");
                params.push((value, id, module.ty(value).clone()));
            }
            ValueDef::Result(inst, _) => {
                ops.insert(inst);
                stack.extend(module.operands(inst));
            }
        }
    }
    params.sort_by_key(|&(_, id, _)| id);
    let region = match module.def(output) {
        ValueDef::Result(inst, _) => module.parent_region(inst).expect("in a region"),
        ValueDef::Arg(region, _) => region,
    };
    let ops = module.insts(region).filter(|i| ops.contains(i)).collect();
    Subgraph {
        params,
        ops,
        output,
    }
}

/// One op of a subgraph, copied out of its module.
struct CopiedOp {
    op: Op,
    operands: Vec<Value>,
    segments: Option<Vec<usize>>,
    results: Vec<Value>,
}

fn copy_ops(module: &Module, graph: &Subgraph) -> Vec<CopiedOp> {
    graph
        .ops
        .iter()
        .map(|&inst| CopiedOp {
            op: module.op(inst).clone(),
            operands: module.operands(inst).to_vec(),
            segments: module.operand_segments(inst).map(<[usize]>::to_vec),
            results: module.results(inst).to_vec(),
        })
        .collect()
}

/// Adds `graph`, whose ops are `ops`, to `dst` as the load-time function `name` that computes
/// `id`.
fn emit(
    dst: &mut Module,
    graph: &Subgraph,
    ops: &[CopiedOp],
    name: &str,
    id: ParamId,
) -> Result<()> {
    let types = graph.params.iter().map(|(_, _, ty)| ty.clone()).collect();
    let ids = graph.params.iter().map(|&(_, id, _)| Some(id)).collect();
    let func = dst.build_func(name, types, ids, |b, args| {
        let mut map: HashMap<Value, Value> = graph
            .params
            .iter()
            .zip(args)
            .map(|(&(value, _, _), &arg)| (value, arg))
            .collect();
        for copied in ops {
            let operands = copied.operands.iter().map(|v| map[v]).collect();
            let inst = b.build_op(
                copied.op.clone(),
                operands,
                copied.segments.clone(),
                Vec::new(),
            )?;
            map.extend(
                copied
                    .results
                    .iter()
                    .copied()
                    .zip(b.module.results(inst).to_vec()),
            );
        }
        Ok(vec![map[&graph.output]])
    })?;
    let mut attrs: Func = dst.op(func).as_func().expect("a function").clone();
    attrs.result_params = vec![id];
    dst.set_op(func, attrs.into());
    Ok(())
}

/// The parameter that a load-time function computes as `value`. Creates the function unless the
/// module has it. `None` when no parameter goes into `value`.
fn load_time_function(module: &mut Module, value: Value) -> Result<Option<ParamId>> {
    let graph = subgraph(module, value);
    if graph.params.is_empty() {
        return Ok(None);
    }
    let ops = copy_ops(module, &graph);
    let mut scratch = Module::new();
    emit(&mut scratch, &graph, &ops, "static", ParamId(0))?;
    let hash = blake3::hash(scratch.to_text().as_bytes());
    let bytes: [u8; 16] = hash.as_bytes()[..16].try_into().expect("16 bytes");
    let id = ParamId(u128::from_le_bytes(bytes));
    let name = format!("static.{:032x}", id.0);
    if module.func(&name).is_none() {
        emit(module, &graph, &ops, &name, id)?;
    }
    Ok(Some(id))
}

/// Adds an argument of type `ty` bound to parameter `id` to the function `func`.
fn add_param_arg(module: &mut Module, func: Inst, ty: Type, id: ParamId) -> Value {
    let body = module.regions(func)[0];
    let arg = module.append_arg(body, ty.clone());
    let mut attrs: Func = module.op(func).as_func().expect("a function").clone();
    attrs.params.resize(attrs.args.len(), None);
    attrs.args.push(ty);
    attrs.params.push(Some(id));
    module.set_op(func, attrs.into());
    arg
}
