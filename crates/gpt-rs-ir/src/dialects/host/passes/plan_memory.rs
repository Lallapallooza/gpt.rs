//! Planning the memory of a function: the buffer form of the `host` dialect.

use std::collections::HashMap;

use anyhow::{anyhow, bail, ensure, Result};

use crate::dialects::func::{Func, FuncOp};
use crate::dialects::host::{
    Alloc, Arena, Copy, Dispatch, Launch, LaunchOperands, Slice, Take, ToBuffer, ToTensor,
};
use crate::dialects::loops::{self, Access};
use crate::ir::{Inst, Module, Region, SymbolRef, TensorType, Value};
use crate::passes::{pass, Pass, Registered};
use crate::require;
use crate::rewrite::{
    pattern, Greedy, NoMatch, Pattern, PatternSet, Rewriter, Rewrites, Rewriting,
};
use crate::targets::{required, Target};

/// Gives every value that a function's dispatches compute a buffer, and calls each `loop.func`
/// with `host.launch` on buffers: the buffer form of the `host` dialect.
///
/// - Values live in one arena, `host.arena`. A greedy best fit places them, the largest first. Each
///   goes at the lowest offset of the smallest gap between the placed values live at the same time,
///   or after all of them when no gap fits. So values that are never live at once share bytes. The
///   arena, each value in it and the memory of each allocation start at a multiple of the alignment
///   that the target states for its buffers.
/// - An `inout` parameter updates the buffer of its argument when that argument is the result of
///   an earlier launch, or an argument of the function that only launches use, and dies at this
///   launch: the launch takes it once and nothing uses it later. It updates a copy otherwise.
///   An argument of the function that an `inout` parameter updates last is taken (`host.take`),
///   so a runtime updates the caller's memory when the caller holds the tensor no more.
/// - An `out<over = k>` parameter writes over the buffer of the argument of `in` parameter `k`
///   when that argument is a result of an earlier launch, has the type of the parameter's result
///   and dies at this launch: the launch takes it once and nothing uses it later. Of the results
///   of one launch, only the first writes over a given argument.
/// - A value that the function returns, or that an op other than a launch reads, lives in memory
///   of its own (`host.alloc`), or in the taken memory of the argument it updates.
/// - The function keeps its tensor arguments and results. `host.to_buffer` gives launches the
///   elements of an argument or of the value of another op, and `host.to_tensor` gives other ops
///   and the return the elements of a buffer.
///
/// It plans the dispatches of a function's body. A dispatch in a region of another op keeps the
/// value form. Every top-level `host.dispatch` must call a `loop.func`, and every result of one
/// must have static extents.
#[pass(
    name = "plan-memory",
    stats(
        planned = "functions given a memory plan",
        in_place = "values written over a value that dies"
    )
)]
#[derive(Default)]
pub struct PlanMemory {
    /// The target that states the alignment of buffers: the arena and each allocation start at a
    /// multiple of it.
    pub target: Option<Target>,
}

fn populate_plan_memory_patterns(set: &mut PatternSet, alignment: Option<u64>) {
    set.add(PlanFunction { alignment });
}

impl Pass for PlanMemory {
    fn rewrites(&self) -> Rewrites {
        let alignment = self
            .target
            .as_ref()
            .and_then(|target| target.buffer_alignment().ok());
        let mut once = PatternSet::new();
        populate_plan_memory_patterns(&mut once, alignment);
        Rewrites {
            once,
            ..Rewrites::default()
        }
    }

    fn run(&self, module: &mut Rewriting) -> Result<bool> {
        required(self.target.as_ref(), self.name())?.buffer_alignment()?;
        // What a plan can fail on fails before any change: a call of a dispatch that is not a
        // `loop.func`, and a result without static extents.
        for func in module.funcs() {
            let body = module.regions(func)[0];
            for (position, inst) in module.insts(body).enumerate() {
                let Some(call) = call_at(module, inst, position)? else {
                    continue;
                };
                for &result in call.gives.iter().flatten() {
                    static_tensor(module, result)?;
                }
            }
        }
        Ok(Greedy::new(self.rewrites()).run(module)?.changed())
    }
}

/// A function that calls `loop.func`s takes the buffer form under its memory plan.
#[pattern]
struct PlanFunction {
    alignment: Option<u64>,
}

impl Pattern for PlanFunction {
    type Root = Func;

    fn rewrite(&self, func: FuncOp, rewriter: &mut Rewriter) -> Result<(), NoMatch> {
        require!(let Some(alignment) = self.alignment, "the target states no alignment of buffers");
        let planned = Plan::of(func.module(), func.body().id(), alignment)
            .expect("the pass checks each call and the extents of its results");
        require!(let Some(plan) = planned, "the function calls no loop.func");
        PlanMemory::IN_PLACE.add(plan.in_place);
        plan.apply(&func, rewriter);
        PlanMemory::PLANNED.add(1);
        Ok(())
    }
}

/// A `host.dispatch` of a `loop.func`: its position among the ops of the function, and the
/// argument and the result of each parameter of the callee.
struct Call {
    inst: Inst,
    position: usize,
    callee: String,
    access: Vec<Access>,
    /// For each parameter, the argument it takes, if any.
    takes: Vec<Option<Value>>,
    /// For each parameter, the result it gives, if any.
    gives: Vec<Option<Value>>,
}

/// Where the values of a chain live.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Storage {
    /// In the arena.
    Arena,
    /// In memory of its own, for a value that the function returns or that an op other than a
    /// launch reads.
    Alloc,
    /// In the memory of the chain's first value, an argument of the function.
    Take,
}

/// Values that share one buffer, one after another: each is written over the one before, which
/// dies where it is written.
struct Chain {
    /// Its last value, which the ops after it read.
    last: Value,
    ty: TensorType,
    storage: Storage,
    /// The positions of the ops where its first value is written and its last one dies.
    start: usize,
    end: usize,
    /// Its offset in the arena.
    offset: u64,
}

/// How a function's ops use a value.
#[derive(Default)]
struct Uses {
    /// The positions of the launches that take it, once per operand.
    calls: Vec<usize>,
    /// Whether another op uses it, or the return.
    other: bool,
    /// The position of its last use.
    last: usize,
}

/// The memory plan of one function.
struct Plan {
    /// The ops of the function's body, in order.
    insts: Vec<Inst>,
    calls: Vec<Call>,
    chains: Vec<Chain>,
    /// The chain of each value that has a buffer of a chain.
    chain_of: HashMap<Value, usize>,
    /// The values to copy into the buffer of a result before a launch, by the launch's position.
    copies: HashMap<usize, Vec<(Value, Value)>>,
    /// How the function uses each value that a launch takes or gives.
    uses: HashMap<Value, Uses>,
    arena_bytes: u64,
    /// The bytes that every buffer's offset is a multiple of.
    alignment: u64,
    in_place: usize,
}

impl Plan {
    /// The plan of the function whose body is `body`, or `None` when it calls no `loop.func`.
    fn of(module: &Module, body: Region, alignment: u64) -> Result<Option<Plan>> {
        let insts: Vec<Inst> = module.insts(body).collect();
        let position: HashMap<Inst, usize> = insts
            .iter()
            .enumerate()
            .map(|(index, &inst)| (inst, index))
            .collect();
        let mut calls = Vec::new();
        for (index, &inst) in insts.iter().enumerate() {
            if let Some(call) = call_at(module, inst, index)? {
                calls.push(call);
            }
        }
        if calls.is_empty() {
            return Ok(None);
        }
        let uses = uses_of(module, body, &insts, &position, &calls);
        let mut plan = Plan {
            insts,
            calls,
            chains: Vec::new(),
            chain_of: HashMap::new(),
            copies: HashMap::new(),
            uses,
            arena_bytes: 0,
            alignment,
            in_place: 0,
        };
        plan.chain(module)?;
        plan.place();
        Ok(Some(plan))
    }

    /// Puts each result of a launch into a chain: that of the argument it updates or writes
    /// over, when the argument dies at the launch and no result before it joined the argument's
    /// chain, or a chain of its own.
    fn chain(&mut self, module: &Module) -> Result<()> {
        for index in 0..self.calls.len() {
            let call = &self.calls[index];
            let position = call.position;
            let mut joins = Vec::new();
            for (param, access) in call.access.iter().enumerate() {
                let Some(result) = call.gives[param] else {
                    continue;
                };
                let over = match *access {
                    Access::Inout => call.takes[param],
                    Access::Out { over: Some(shared) } => {
                        call.takes[shared].filter(|&arg| module.ty(arg) == module.ty(result))
                    }
                    _ => None,
                };
                joins.push((result, over, *access == Access::Inout));
            }
            let mut joined_args = Vec::new();
            for (result, over, updates) in joins {
                let joined = over.filter(|&arg| {
                    !joined_args.contains(&arg) && self.dies_at(module, arg, position, updates)
                });
                match joined {
                    Some(arg) => {
                        joined_args.push(arg);
                        let chain = match self.chain_of.get(&arg) {
                            Some(&chain) => chain,
                            None => self.new_chain(module, arg, Storage::Take, position)?,
                        };
                        self.chains[chain].last = result;
                        self.chain_of.insert(result, chain);
                        self.in_place += 1;
                    }
                    None => {
                        self.new_chain(module, result, Storage::Arena, position)?;
                        if let (true, Some(arg)) = (updates, over) {
                            self.copies.entry(position).or_default().push((arg, result));
                        }
                    }
                }
            }
        }
        for index in 0..self.chains.len() {
            let last = self.chains[index].last;
            let escapes = self.escapes(last);
            let chain = &mut self.chains[index];
            chain.end = self.uses.get(&last).map_or(chain.start, |uses| uses.last);
            if chain.storage == Storage::Arena && escapes {
                chain.storage = Storage::Alloc;
            }
        }
        Ok(())
    }

    /// Whether `arg`, which the launch at `position` takes, dies there, so a result may be written
    /// over its buffer: the launch takes it once and nothing uses it later. An argument without
    /// a chain must also be an argument of the function that no op but launches uses, and is
    /// taken only for an update: the memory of another op's value, such as a constant's, is not
    /// the function's to write.
    fn dies_at(&self, module: &Module, arg: Value, position: usize, updates: bool) -> bool {
        let Some(uses) = self.uses.get(&arg) else {
            return false;
        };
        let taken_once = uses.calls.iter().filter(|&&at| at == position).count() == 1;
        if !taken_once || uses.last != position || uses.other {
            return false;
        }
        let function_arg = module.def_inst(arg).is_none();
        self.chain_of.contains_key(&arg) || (updates && function_arg)
    }

    /// Whether the return or another op takes `value` as a tensor.
    fn escapes(&self, value: Value) -> bool {
        self.uses.get(&value).is_some_and(|uses| uses.other)
    }

    /// Starts a chain whose first value is `value`, written at `position`.
    fn new_chain(
        &mut self,
        module: &Module,
        value: Value,
        storage: Storage,
        position: usize,
    ) -> Result<usize> {
        let ty = static_tensor(module, value)?;
        let chain = self.chains.len();
        self.chains.push(Chain {
            last: value,
            ty,
            storage,
            start: position,
            end: position,
            offset: 0,
        });
        self.chain_of.insert(value, chain);
        Ok(chain)
    }

    /// Places the chains of the arena: the largest first, each at the lowest offset of the
    /// smallest gap between the chains already placed that are live at the same time, or after
    /// them all.
    fn place(&mut self) {
        let mut order: Vec<usize> = (0..self.chains.len())
            .filter(|&chain| self.chains[chain].storage == Storage::Arena)
            .collect();
        order.sort_by_key(|&index| {
            let chain = &self.chains[index];
            (
                std::cmp::Reverse(aligned_bytes(&chain.ty, self.alignment)),
                chain.start,
            )
        });
        let mut placed: Vec<usize> = Vec::new();
        for chain in order {
            let size = aligned_bytes(&self.chains[chain].ty, self.alignment);
            let (start, end) = (self.chains[chain].start, self.chains[chain].end);
            let mut taken: Vec<(u64, u64)> = Vec::new();
            for &placed_chain in &placed {
                let other = &self.chains[placed_chain];
                if other.start <= end && start <= other.end {
                    taken.push((
                        other.offset,
                        other.offset + aligned_bytes(&other.ty, self.alignment),
                    ));
                }
            }
            taken.sort_unstable();
            let mut best: Option<(u64, u64)> = None;
            let mut free = 0;
            for &(begin, finish) in &taken {
                let gap = begin.saturating_sub(free);
                if gap >= size && best.is_none_or(|(_, best_gap)| gap < best_gap) {
                    best = Some((free, gap));
                }
                free = free.max(finish);
            }
            let offset = best.map_or(free, |(offset, _)| offset);
            self.chains[chain].offset = offset;
            self.arena_bytes = self.arena_bytes.max(offset + size);
            placed.push(chain);
        }
    }

    /// Rewrites `func`, whose plan this is, into the buffer form.
    fn apply(&self, func: &FuncOp, rewriter: &mut Rewriter) {
        let module = func.module();
        let first = self.insts[0];
        let mut buffers = Buffers {
            chains: vec![None; self.chains.len()],
            values: HashMap::new(),
        };
        rewriter.set_insertion_point(first);
        let in_arena = self
            .chains
            .iter()
            .any(|chain| chain.storage == Storage::Arena);
        let arena = in_arena.then(|| {
            rewriter.create(
                Arena {
                    bytes: self.arena_bytes,
                    alignment: self.alignment,
                },
                (),
            )
        });
        for (index, chain) in self.chains.iter().enumerate() {
            let shape = chain.ty.static_dims().expect("chains have static extents");
            let dtype = chain.ty.dtype;
            buffers.chains[index] = match chain.storage {
                Storage::Arena => Some(rewriter.create(
                    Slice {
                        offset: chain.offset,
                        shape,
                        dtype,
                    },
                    arena.expect("a chain in the arena makes one"),
                )),
                Storage::Alloc => Some(rewriter.create(
                    Alloc {
                        shape,
                        dtype,
                        alignment: self.alignment,
                    },
                    (),
                )),
                Storage::Take => None,
            };
        }
        let args: Vec<Value> = func.body().args().iter().map(|arg| arg.id()).collect();
        self.give_buffers(rewriter, &args, first, &mut buffers);
        for (index, &inst) in self.insts.iter().enumerate() {
            let Some(&next) = self.insts.get(index + 1) else {
                break;
            };
            let Some(call) = self.calls.iter().find(|call| call.position == index) else {
                let results = module.results(inst).to_vec();
                rewriter.set_origin(inst);
                self.give_buffers(rewriter, &results, next, &mut buffers);
                continue;
            };
            self.launch(rewriter, call, &buffers);
            rewriter.set_insertion_point(next);
            let mut tensors = Vec::new();
            for &result in call.gives.iter().flatten() {
                if self.escapes(result) {
                    let tensor = rewriter.create(ToTensor, self.buffer_of(result, &buffers));
                    tensors.push((result, tensor));
                }
            }
            for (result, tensor) in tensors {
                rewriter.replace_all_uses(result, tensor);
            }
        }
        for call in self.calls.iter().rev() {
            rewriter.erase(call.inst);
        }
    }

    /// The buffer that holds `value`.
    fn buffer_of(&self, value: Value, buffers: &Buffers) -> Value {
        match self.chain_of.get(&value) {
            Some(&chain) => buffers.chains[chain].expect("a chain's buffer comes first"),
            None => buffers.values[&value],
        }
    }

    /// Gives a buffer, before `next`, to each of `values` that launches take: an argument of the
    /// function that starts a chain is taken, and the others get `host.to_buffer`.
    fn give_buffers(
        &self,
        rewriter: &mut Rewriter,
        values: &[Value],
        next: Inst,
        buffers: &mut Buffers,
    ) {
        rewriter.set_insertion_point(next);
        for &value in values {
            if self
                .uses
                .get(&value)
                .is_none_or(|uses| uses.calls.is_empty())
            {
                continue;
            }
            match self.chain_of.get(&value) {
                Some(&chain) => buffers.chains[chain] = Some(rewriter.create(Take, value)),
                None => {
                    let buffer = rewriter.create(ToBuffer, value);
                    buffers.values.insert(value, buffer);
                }
            }
        }
    }

    /// Adds, before the dispatch of `call`, the copies it needs and a launch on the buffers.
    fn launch(&self, rewriter: &mut Rewriter, call: &Call, buffers: &Buffers) {
        rewriter.set_insertion_point(call.inst);
        rewriter.set_origin(call.inst);
        for &(from, to) in self.copies.get(&call.position).into_iter().flatten() {
            let operands = (self.buffer_of(from, buffers), self.buffer_of(to, buffers));
            rewriter.create(Copy, operands);
        }
        // The buffer of a parameter that gives a result is the result's, and of any other the
        // argument's.
        let mut params = Vec::with_capacity(call.access.len());
        for (given, taken) in call.gives.iter().zip(&call.takes) {
            let value = given
                .or(*taken)
                .expect("a call has a value for each parameter");
            params.push(self.buffer_of(value, buffers));
        }
        let operands = LaunchOperands::of_params(&call.access, params);
        let callee = SymbolRef(call.callee.clone());
        rewriter.create(Launch { callee }, operands);
    }
}

/// The buffers of a plan, once made: of each chain, and of each value outside the chains.
struct Buffers {
    chains: Vec<Option<Value>>,
    values: HashMap<Value, Value>,
}

/// The call that `inst`, at `position`, makes, when it is a `host.dispatch` of a `loop.func`.
fn call_at(module: &Module, inst: Inst, position: usize) -> Result<Option<Call>> {
    let Some(dispatch) = module.op(inst).get::<Dispatch>() else {
        return Ok(None);
    };
    let name = &dispatch.callee.0;
    let func = module
        .symbol(name)
        .and_then(|symbol| module.op(symbol).get::<loops::Func>())
        .ok_or_else(|| anyhow!("plan-memory launches `loop.func`s, and @{name} is not one"))?;
    let (mut args, mut results) = (
        module.operands(inst).iter().copied(),
        module.results(inst).iter().copied(),
    );
    let mut takes = Vec::with_capacity(func.access.len());
    let mut gives = Vec::with_capacity(func.access.len());
    for access in &func.access {
        let arg = match access.takes_argument() {
            true => args.next(),
            false => None,
        };
        let result = match access.gives_result() {
            true => results.next(),
            false => None,
        };
        takes.push(arg);
        gives.push(result);
    }
    ensure!(
        args.next().is_none() && results.next().is_none(),
        "the call of @{name} does not match its parameters"
    );
    Ok(Some(Call {
        inst,
        position,
        callee: name.clone(),
        access: func.access.clone(),
        takes,
        gives,
    }))
}

/// How the ops of the body use each value that a launch takes or gives.
fn uses_of(
    module: &Module,
    body: Region,
    insts: &[Inst],
    position: &HashMap<Inst, usize>,
    calls: &[Call],
) -> HashMap<Value, Uses> {
    let mut values = Vec::new();
    for call in calls {
        values.extend(call.takes.iter().flatten());
        values.extend(call.gives.iter().flatten());
    }
    let call_at: HashMap<usize, &Call> = calls.iter().map(|call| (call.position, call)).collect();
    let mut uses: HashMap<Value, Uses> = HashMap::new();
    for value in values {
        if uses.contains_key(&value) {
            continue;
        }
        let defined = module
            .def_inst(value)
            .and_then(|inst| position.get(&inst).copied())
            .unwrap_or(0);
        let mut found = Uses {
            last: defined,
            ..Uses::default()
        };
        for usage in module.uses(value) {
            let at = top_position(module, body, usage.inst, position);
            found.last = found.last.max(at);
            let is_call = call_at.contains_key(&at) && usage.inst == insts[at];
            match is_call {
                true => found.calls.push(at),
                false => found.other = true,
            }
        }
        uses.insert(value, found);
    }
    uses
}

/// The position of the op of `body` that holds `inst`.
fn top_position(
    module: &Module,
    body: Region,
    inst: Inst,
    position: &HashMap<Inst, usize>,
) -> usize {
    let mut current = inst;
    loop {
        let region = module
            .parent_region(current)
            .expect("an op of a function is in a region");
        if region == body {
            return position[&current];
        }
        current = module
            .parent_op(region)
            .expect("a region inside a function belongs to an op");
    }
}

/// The type of `value`, a tensor with static extents.
fn static_tensor(module: &Module, value: Value) -> Result<TensorType> {
    let ty = module.ty(value);
    let tensor = ty
        .as_tensor()
        .ok_or_else(|| anyhow!("a launch takes and gives tensors, not {ty}"))?;
    if tensor.static_dims().is_none() {
        bail!("plan-memory plans values of static extents, not {tensor}");
    }
    Ok(tensor.clone())
}

/// The bytes the elements of `ty` take, rounded up to a multiple of `alignment`.
fn aligned_bytes(ty: &TensorType, alignment: u64) -> u64 {
    let count = ty.element_count().expect("chains have static extents");
    let bytes = ty.dtype.bytes_for(count).expect("a value's bytes fit") as u64;
    bytes.next_multiple_of(alignment)
}
