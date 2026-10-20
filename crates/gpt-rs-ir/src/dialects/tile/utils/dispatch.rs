//! The name a `tile.dispatch`'s text gives it, its callers, the rounding changes it records, and
//! which results of its body may take the buffer of an argument.

use std::collections::BTreeSet;

use crate::dialects::host;
use crate::dialects::tile::ir::{Block, Coords, Dispatch, IterKind, Numerics};
use crate::ir::{Inst, Module, OperandRef, Region, Space, SymbolRef, Type, Value};
use crate::rewrite::Rewriter;
use crate::targets::TargetDesc;

impl Dispatch {
    /// The buffer space of a result of a block that the dispatch does not give and that no block
    /// placed after it takes as it is computed: private scratch, the scratch of the worker that
    /// runs the dispatch's function, which holds the result whole, and whose extents the verifier
    /// keeps static.
    pub const SCRATCH_SPACE: Space = Space::Private;

    /// The `tile.dispatch` named by `callee`, when the module has it.
    pub fn named(module: &Module, callee: &SymbolRef) -> Option<Inst> {
        let inst = module.symbol(&callee.0)?;
        module.op(inst).get::<Dispatch>().map(|_| inst)
    }

    /// The `host.dispatch` ops of `module` that call `dispatch`, in order.
    pub fn callers(module: &Module, dispatch: Inst) -> Vec<Inst> {
        let name = &module
            .op(dispatch)
            .get::<Dispatch>()
            .expect("a dispatch")
            .name;
        let mut calls = Vec::new();
        for inst in module.walk(module.body()) {
            let calls_it = module
                .op(inst)
                .get::<host::Dispatch>()
                .is_some_and(|call| call.callee.0 == *name);
            if calls_it {
                calls.push(inst);
            }
        }
        calls
    }

    /// The blocks of the dispatch body `body`, in order.
    pub fn blocks(module: &Module, body: Region) -> Vec<(Inst, &Block)> {
        let mut blocks = Vec::new();
        for inst in module.insts(body) {
            if let Some(block) = module.op(inst).get::<Block>() {
                blocks.push((inst, block));
            }
        }
        blocks
    }

    /// The lanes, `(iterator, lanes)`, that the first of `blocks`, the blocks of a dispatch in
    /// order, computes in on `target` ([`Block::vector_lanes`]), when each block after it is the
    /// epilogue of the block before it ([`Block::takes_epilogue`]) and computes in the same lanes.
    pub fn vector_lanes(
        module: &Module,
        blocks: &[(Inst, &Block)],
        target: &TargetDesc,
    ) -> Option<(usize, usize)> {
        let (&(first, _), epilogues) = blocks.split_first()?;
        let lanes = Block::vector_lanes(module, first, target)?;
        for pair in blocks.windows(2) {
            let [(before, earlier), (after, later)] = pair else {
                continue;
            };
            let input = Block::epilogue_input(module, *before, *after)?;
            if !earlier.takes_epilogue(later, input) {
                return None;
            }
        }
        for &(epilogue, _) in epilogues {
            if Block::vector_lanes(module, epilogue, target)? != lanes {
                return None;
            }
        }
        Some(lanes)
    }

    /// The name of the dispatch that `scratch` holds alone: `dispatch.` and a hash of the
    /// module's text, so that dispatches that compute the same share one name.
    pub fn name_from_text(scratch: &Module) -> String {
        let hash = blake3::hash(scratch.to_text().as_bytes());
        let bytes: [u8; 16] = hash.as_bytes()[..16].try_into().expect("16 bytes");
        format!("dispatch.{:032x}", u128::from_le_bytes(bytes))
    }

    /// Adds through `rewriter` the dispatch that `build` builds alone into an empty module under
    /// the name it is given, and gives its name: the one its text gives it under the name
    /// `dispatch`. A dispatch of that name in the module is the same one, so it is not added
    /// again.
    pub(crate) fn add_named_by_text(
        rewriter: &mut Rewriter,
        build: impl Fn(&mut Module, &str) -> Inst,
    ) -> String {
        let mut unnamed = Module::new();
        build(&mut unnamed, "dispatch");
        let name = Self::name_from_text(&unnamed);
        if rewriter.module().symbol(&name).is_none() {
            let mut named = Module::new();
            let dispatch = build(&mut named, &name);
            rewriter.add_symbol(&named, dispatch);
        }
        name
    }

    /// For each result of the dispatch body `body`, the argument that it updates in place: a block
    /// writes the result as an updated copy of the argument, which has no other use in the body
    /// and none of whose elements the block reads. The parameters of the dispatch's function take
    /// the arguments and give the results in order, so an argument that comes before the one an
    /// earlier result updates is updated in a copy instead.
    pub fn updated_args(module: &Module, body: Region) -> Vec<Option<usize>> {
        let args = module.args(body);
        let returned = module.yielded(body);
        let mut updated = vec![None; returned.len()];
        let mut next_arg = 0;
        for (position, &value) in returned.iter().enumerate() {
            if returned[..position].contains(&value) {
                continue;
            }
            let Some((inst, block, result)) = block_of(module, value) else {
                continue;
            };
            let Some(write) = block.writes().nth(result) else {
                continue;
            };
            let Some(OperandRef(operand)) = write.into else {
                continue;
            };
            let base = module.operands(inst)[operand];
            let Some(arg) = args.iter().position(|&arg| arg == base) else {
                continue;
            };
            let alone = module.uses(base).len() == 1 && !block.read_operands().contains(&operand);
            if alone && arg >= next_arg {
                updated[position] = Some(arg);
                next_arg = arg + 1;
            }
        }
        updated
    }

    /// For each result of the dispatch body `body`, an argument whose buffer it may share: a map
    /// writes the result at the position of each point, and reads the argument, which nothing else
    /// in the body uses, only at that position. Each argument goes to one result at most.
    pub fn shareable_args(module: &Module, body: Region) -> Vec<Option<usize>> {
        let args = module.args(body);
        let returned = module.yielded(body);
        let mut shared = vec![None; returned.len()];
        let mut taken = Vec::new();
        for (position, &value) in returned.iter().enumerate() {
            if returned[..position].contains(&value) {
                continue;
            }
            let Some((inst, block, _)) = block_of(module, value) else {
                continue;
            };
            if !writes_each_point(block) {
                continue;
            }
            let write = &block.folds[0].writes[0];
            let operands = module.operands(inst);
            let found = args.iter().enumerate().find(|&(arg, &arg_value)| {
                let uses = module.uses(arg_value);
                !taken.contains(&arg)
                    && !uses.is_empty()
                    && uses.iter().all(|usage| usage.inst == inst)
                    && same_layout(module.ty(arg_value), module.ty(value))
                    && reads_only_at(block, operands, arg_value, &write.coords)
            });
            if let Some((arg, _)) = found {
                shared[position] = Some(arg);
                taken.push(arg);
            }
        }
        shared
    }
}

impl Numerics {
    /// The rounding changes that a dispatch compiled under these may hold on a target: every one
    /// of these, and `contract` only where the target computes a product and a sum rounded once,
    /// as `has_fma` says.
    pub fn on_target(self, has_fma: bool) -> Numerics {
        Numerics {
            contract: self.contract && has_fma,
            ..self
        }
    }
}

/// The block that defines `value`, its op and which of its results `value` is.
fn block_of(module: &Module, value: Value) -> Option<(Inst, &Block, usize)> {
    let inst = module.def_inst(value)?;
    let block = module.op(inst).get::<Block>()?;
    let result = module
        .results(inst)
        .iter()
        .position(|&result_value| result_value == value)?;
    Some((inst, block, result))
}

/// Whether `block` is a map over parallel iterators that writes each point at an element of its
/// own ([`crate::dialects::tile::Write::translates`]).
fn writes_each_point(block: &Block) -> bool {
    block.is_map()
        && block.iters.iter().all(|iter| iter.kind == IterKind::Par)
        && block.folds[0].writes[0].translates(&block.iters)
}

/// Whether two tensor types put elements of one size at the same positions.
fn same_layout(first: &Type, second: &Type) -> bool {
    let (Some(first_tensor), Some(second_tensor)) = (first.as_tensor(), second.as_tensor()) else {
        return false;
    };
    let size = first_tensor.dtype.size_in_bytes();
    first_tensor.dims == second_tensor.dims
        && size.is_some()
        && size == second_tensor.dtype.size_in_bytes()
}

/// Whether every read that `block`, with the operands `operands`, makes of `value` is at
/// `coords`, without a pad, and no index expression reads it.
fn reads_only_at(block: &Block, operands: &[Value], value: Value, coords: &Coords) -> bool {
    let mut indexed = BTreeSet::new();
    for expr in block.exprs() {
        expr.collect_operands(&mut indexed);
    }
    let holds = |operand: usize| operands[operand] == value;
    if indexed.into_iter().any(holds) {
        return false;
    }
    block.reads.iter().all(|read| {
        !read.source.operands().any(holds) || (read.pad.is_none() && read.coords == *coords)
    })
}
