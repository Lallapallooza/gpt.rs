//! Moving computation over parameters into load-time functions.

use std::collections::{HashMap, HashSet};

use anyhow::Result;

use super::{pass, Pass};
use crate::analysis::tensor_bytes;
use crate::dialects::func::{Func, FuncOp, ParamId};
use crate::ir::{Inst, Module, Region, Type, Value, ValueDef};
use crate::require;
use crate::rewrite::{
    pattern, Greedy, NoMatch, Pattern, PatternSet, Rewriter, Rewrites, Rewriting,
};
use crate::traits::{Movement, Terminator};

/// Moves the static values that the run-time ops of a function read into one load-time function,
/// which the runtime runs once when it loads the parameters: it takes the parameters they read
/// and gives each value as a parameter of its own. The function that reads a value gets a new
/// argument bound to that parameter.
///
/// - What moves is the value before an op that only repeats its elements, such as a
///   `broadcast_to`, so the runtime keeps the smaller tensor.
/// - A static value computed from no parameter stays: as a parameter, it would be stored once for
///   each shape a program is compiled for.
/// - The values stored from a parameter take at most the parameter's bytes in total. The pass
///   allocates those bytes in the order the values are defined. So the values a function stores
///   never take more bytes than the parameters they read, with one exception. A value of their
///   dtype may take up to one slice of its outermost axis more, less one byte, as a layout in
///   panels does when the parameter does not fill the last panel. A value that does not fit, such
///   as a weight cast to a wider dtype, stays, and the static values it reads move instead. A value
///   or parameter of unknown size never fits.
/// - The id of a value's parameter is a hash of the text of a function that computes it alone from
///   the parameters it reads, so equal computations share one parameter across functions and
///   programs, and the runtime skips a load-time function whose results it holds. Ops that
///   several values share are computed once.
#[pass(
    name = "outline-static",
    stats(outlined = "values moved into load-time functions")
)]
pub struct OutlineStatic;

impl OutlineStatic {
    /// Whether the pass would store a static value of type `ty` computed from the static values
    /// `sources`, by the rule of its budget: it takes no more bytes than the parameters that
    /// they read (`fits`). A pass that moves work onto parameters asks it before it writes work
    /// that pays only once it is stored.
    pub fn would_store(module: &Module, ty: &Type, sources: &[Value]) -> bool {
        let mut params: Vec<(Value, ParamId, Type)> = Vec::new();
        for &source in sources {
            for param in subgraph(module, &[source]).params {
                if !params.iter().any(|(_, id, _)| *id == param.1) {
                    params.push(param);
                }
            }
        }
        let mut param_types = Vec::with_capacity(params.len());
        for (_, _, param) in &params {
            param_types.push(param);
        }
        Self::would_store_over(ty, &param_types)
    }

    /// Whether the pass would store a value of type `ty` computed from parameters of the types
    /// `params`, by the rule of its budget (`fits`): for a value whose parameters are known by
    /// their types alone, such as the operands of a block known before the run.
    pub fn would_store_over(ty: &Type, params: &[&Type]) -> bool {
        let mut available = Some(0);
        for param in params {
            available = available.zip(bytes(param)).map(|(sum, size)| sum + size);
        }
        fits(ty, available, params)
    }
}

fn populate_outline_static_patterns(set: &mut PatternSet) {
    set.add(OutlineStaticValues);
}

impl Pass for OutlineStatic {
    fn rewrites(&self) -> Rewrites {
        Rewrites::once(populate_outline_static_patterns)
    }

    fn run(&self, module: &mut Rewriting) -> Result<bool> {
        Ok(Greedy::new(self.rewrites()).run(module)?.changed())
    }
}

/// A run-time function's static values that run-time ops read, and that parameters go into, move
/// into one load-time function: the function gains an argument bound to the parameter each
/// computes, which the ops read in place of the value.
#[pattern]
struct OutlineStaticValues;

impl Pattern for OutlineStaticValues {
    type Root = Func;

    fn rewrite(&self, func: FuncOp, rewriter: &mut Rewriter) -> Result<(), NoMatch> {
        require!(!func.is_load_time(), "the function runs at load time");
        let module = func.module();
        let mut moved: Vec<(Root, ParamId)> = Vec::new();
        for root in roots(module, func.body().id()) {
            let graph = subgraph(module, &[root.value]);
            if graph.params.is_empty() {
                continue;
            }
            let id = param_of(module, &graph);
            moved.push((root, id));
        }
        require!(
            !moved.is_empty(),
            "no static value read at run time has a parameter in it"
        );
        let mut outputs = Vec::with_capacity(moved.len());
        let mut ids = Vec::with_capacity(moved.len());
        for (root, id) in &moved {
            outputs.push(root.value);
            ids.push(*id);
        }
        let graph = subgraph(module, &outputs);
        // Built under a name of its own first, whose text then names it.
        let mut unnamed = Module::new();
        let draft = emit(&mut unnamed, module, &graph, "static", ids);
        let name = format!("static.{:032x}", param_of_text(&unnamed.inst_text(draft)).0);
        if module.func(&name).is_none() {
            let Some(built) = unnamed.op_as::<Func>(draft) else {
                panic!("`emit` builds a function");
            };
            let named = Func {
                name,
                ..built.attrs().clone()
            };
            rewriter.add_symbol_as(&unnamed, draft, named.into());
        }
        OutlineStatic::OUTLINED.add(moved.len());
        for (root, id) in moved {
            let arg = rewriter.add_arg(&func, module.ty(root.value).clone(), Some(id));
            for (inst, index) in root.uses {
                rewriter.set_operand(inst, index, arg);
            }
        }
        Ok(())
    }
}

/// The parameter that a load-time function computing `graph` of `module` alone gives: a hash of
/// its text.
fn param_of(module: &Module, graph: &Subgraph) -> ParamId {
    let mut unnamed = Module::new();
    let draft = emit(&mut unnamed, module, graph, "static", Vec::new());
    param_of_text(&unnamed.inst_text(draft))
}

/// A hash of the text `text` of a load-time function, as the id of a parameter or a name.
fn param_of_text(text: &str) -> ParamId {
    let hash = blake3::hash(text.as_bytes());
    let bytes: [u8; 16] = hash.as_bytes()[..16].try_into().expect("16 bytes");
    ParamId(u128::from_le_bytes(bytes))
}

/// A static value to move, and the operands that read it at run time.
struct Root {
    value: Value,
    uses: Vec<(Inst, usize)>,
}

/// Whether `inst` runs at run time: it is the terminator, or one of its results is not static.
fn runs_at_run_time(module: &Module, statics: &HashSet<Value>, inst: Inst) -> bool {
    module.op(inst).has_trait::<Terminator>()
        || module
            .results(inst)
            .iter()
            .any(|result| !statics.contains(result))
}

/// The input of `inst` when it only repeats the elements of its one operand into more of them.
fn expansion_input(module: &Module, inst: Inst) -> Option<Value> {
    let &[input] = module.operands(inst) else {
        return None;
    };
    let &[output] = module.results(inst) else {
        return None;
    };
    let count = |value: Value| module.ty(value).as_tensor()?.element_count();
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
        .filter(|&value| module.known_before_run(value))
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

/// The bytes a tensor of type `ty` takes, when its size is known.
fn bytes(ty: &Type) -> Option<usize> {
    usize::try_from(tensor_bytes(ty.as_tensor()?)?).ok()
}

/// The roots that do not fit the bytes their parameters have left, taking the roots in order.
/// A root that fits takes its bytes from its parameters in id order.
fn unfit(module: &Module, roots: &[Root]) -> HashSet<Value> {
    let mut left: HashMap<ParamId, Option<usize>> = HashMap::new();
    let mut unfit = HashSet::new();
    for root in roots {
        let mut params = subgraph(module, &[root.value]).params;
        params.dedup_by_key(|&mut (_, id, _)| id);
        for (_, id, ty) in &params {
            left.entry(*id).or_insert_with(|| bytes(ty));
        }
        let available: Option<usize> = params.iter().map(|(_, id, _)| left[id]).sum();
        let mut param_types = Vec::with_capacity(params.len());
        for (_, _, param) in &params {
            param_types.push(param);
        }
        let ty = module.ty(root.value);
        if !fits(ty, available, &param_types) {
            unfit.insert(root.value);
            continue;
        }
        let mut need = bytes(ty).expect("a value that fits has a size");
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

/// Whether the bytes of a value of type `ty` fit the `available` bytes of the parameters `params`
/// that it reads, plus [`panel_padding`]. A value of unknown size, or one of parameters of unknown
/// size, never fits.
fn fits(ty: &Type, available: Option<usize>, params: &[&Type]) -> bool {
    match (bytes(ty), available) {
        (Some(need), Some(limit)) => need <= limit + panel_padding(ty, limit, params),
        _ => false,
    }
}

/// The bytes that a value of type `ty` may take past the `available` bytes of the parameters
/// `params` that it reads: less than one slice of its outermost axis, as a layout in panels pads
/// its last panel. This holds for a value of the parameters' dtype whose slices each take no more
/// bytes than the parameters. Any other value, such as one slice that holds them all, may take
/// none.
fn panel_padding(ty: &Type, available: usize, params: &[&Type]) -> usize {
    let Some(tensor) = ty.as_tensor() else {
        return 0;
    };
    let same_dtype = params.iter().all(|param| {
        param
            .as_tensor()
            .is_some_and(|read| read.dtype == tensor.dtype)
    });
    let outermost = tensor.dims.first().and_then(|dim| dim.as_static());
    let slice = bytes(ty)
        .zip(outermost.filter(|&count| count > 0))
        .map(|(total, count)| total / count);
    match (same_dtype, slice) {
        (true, Some(panel)) if panel <= available => panel.saturating_sub(1),
        _ => 0,
    }
}

/// Removes `dropped` from `statics`, with every value of `body` computed from them.
fn drop_dependents(
    module: &Module,
    body: Region,
    statics: &mut HashSet<Value>,
    mut dropped: HashSet<Value>,
) {
    for inst in module.insts(body) {
        if module
            .operands(inst)
            .iter()
            .any(|operand| dropped.contains(operand))
        {
            dropped.extend(module.results(inst).iter().copied());
        }
    }
    statics.retain(|value| !dropped.contains(value));
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
                .filter(|use_site| runs_at_run_time(module, statics, use_site.inst))
                .map(|use_site| (use_site.inst, use_site.index))
                .collect();
            if uses.is_empty() {
                continue;
            }
            // Through an expansion, the run-time op reads the expansion's input instead.
            let (root_value, root_uses) = match expansion_input(module, inst) {
                Some(input) => (input, vec![(inst, 0)]),
                None => (value, uses),
            };
            match roots.iter_mut().find(|root| root.value == root_value) {
                Some(root) => root.uses.extend(root_uses),
                None => roots.push(Root {
                    value: root_value,
                    uses: root_uses,
                }),
            }
        }
    }
    roots.retain(|root| matches!(module.def(root.value), ValueDef::Result(..)));
    roots
}

/// What computes static values: their parameters, ordered by id, their ops in order, and the
/// values.
struct Subgraph {
    params: Vec<(Value, ParamId, Type)>,
    ops: Vec<Inst>,
    outputs: Vec<Value>,
}

/// What computes `outputs`, values of one region, from the parameters.
fn subgraph(module: &Module, outputs: &[Value]) -> Subgraph {
    let mut seen: HashSet<Value> = HashSet::new();
    let mut subgraph_insts: HashSet<Inst> = HashSet::new();
    let mut params = Vec::new();
    let mut stack = outputs.to_vec();
    while let Some(value) = stack.pop() {
        if !seen.insert(value) {
            continue;
        }
        match module.def(value) {
            ValueDef::Arg(region, index) => {
                let func = module.parent_op(region).expect("a function argument");
                let id = module
                    .op_as::<Func>(func)
                    .and_then(|func_op| func_op.param(index))
                    .expect("a static argument is a parameter");
                params.push((value, id, module.ty(value).clone()));
            }
            ValueDef::Result(inst, _) => {
                subgraph_insts.insert(inst);
                stack.extend_from_slice(module.operands(inst));
            }
        }
    }
    params.sort_by_key(|&(_, id, _)| id);
    let region = match module.def(outputs[0]) {
        ValueDef::Result(inst, _) => module.parent_region(inst).expect("in a region"),
        ValueDef::Arg(region, _) => region,
    };
    let ops = module
        .insts(region)
        .filter(|inst| subgraph_insts.contains(inst))
        .collect();
    Subgraph {
        params,
        ops,
        outputs: outputs.to_vec(),
    }
}

/// Adds a copy of `graph` of `source` to `dst` as the function `name` whose results are the
/// parameters `result_params`, and gives it. Each op keeps its result types, since the symbols
/// that its calls name are not in `dst`.
fn emit(
    dst: &mut Module,
    source: &Module,
    graph: &Subgraph,
    name: &str,
    result_params: Vec<ParamId>,
) -> Inst {
    let types = graph.params.iter().map(|(_, _, ty)| ty.clone()).collect();
    let ids = graph.params.iter().map(|&(_, id, _)| Some(id)).collect();
    dst.build_func(name, types, ids, result_params, |builder, args| {
        let mut map: HashMap<Value, Value> = graph
            .params
            .iter()
            .zip(args)
            .map(|(&(value, _, _), &arg)| (value, arg))
            .collect();
        for &inst in &graph.ops {
            builder.create_copy(source, inst, &mut map)?;
        }
        let mut results = Vec::with_capacity(graph.outputs.len());
        for output in &graph.outputs {
            results.push(map[output]);
        }
        Ok(results)
    })
    .expect("a subgraph reads only its parameters and its own values")
}
