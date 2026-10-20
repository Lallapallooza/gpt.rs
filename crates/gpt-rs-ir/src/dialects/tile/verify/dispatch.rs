//! The type rules of `tile.group` and `tile.dispatch`: the types each op gives, and what it
//! rejects.

use std::collections::BTreeSet;

use anyhow::{anyhow, ensure, Result};

use crate::dialects::tile::ir::*;
use crate::dialects::tile::verify::check_lanes;
use crate::ir::{
    Dim, Inst, Module, OpDef, OwnedTypes, Region, RegionOpDef, RegionSig, Type, TypeCx, Types,
};
use crate::targets::Target;
use crate::traits::Terminator;

impl RegionOpDef for Group {
    fn region_signatures(&self, _ins: GroupOperands<Types>) -> Result<GroupRegions<RegionSig>> {
        Ok(GroupRegions {
            body: RegionSig::args_only([]),
        })
    }
}

impl OpDef for Group {
    fn result_types(
        &self,
        _ins: GroupOperands<Types>,
        cx: &TypeCx<GroupRegions<Region>>,
    ) -> Result<GroupResults<OwnedTypes>> {
        let terminator = cx
            .module
            .last_inst(cx.regions.body)
            .ok_or_else(|| anyhow!("the body is empty"))?;
        let outs = cx
            .module
            .operand_types(terminator)
            .into_iter()
            .map(|ty| {
                ty.as_tensor()
                    .cloned()
                    .ok_or_else(|| anyhow!("a group yields tensors, got {ty}"))
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(GroupResults { outs })
    }
}

impl RegionOpDef for Dispatch {
    fn region_signatures(
        &self,
        _ins: DispatchOperands<Types>,
    ) -> Result<DispatchRegions<RegionSig>> {
        Ok(DispatchRegions {
            body: RegionSig::new(self.args.clone(), self.results.clone()),
        })
    }
}

impl OpDef for Dispatch {
    fn result_types(
        &self,
        _ins: DispatchOperands<Types>,
        cx: &TypeCx<DispatchRegions<Region>>,
    ) -> Result<DispatchResults<OwnedTypes>> {
        for types in [&self.args, &self.results] {
            for ty in types {
                ensure!(
                    ty.as_tensor().is_some(),
                    "a dispatch takes and gives tensors, got {ty}"
                );
            }
        }
        let mut bound = BTreeSet::new();
        for tensor in self.args.iter().filter_map(Type::as_tensor) {
            for dim in &tensor.dims {
                if let Dim::Dynamic(sym) = dim {
                    bound.insert(sym.as_str().to_string());
                }
            }
        }
        for (name, fact) in &self.facts {
            ensure!(
                bound.contains(name),
                "facts name {{{name}}}, which no argument binds"
            );
            ensure!(
                fact.min <= fact.max,
                "facts give {{{name}}} the empty range [{}, {}]",
                fact.min,
                fact.max
            );
            ensure!(fact.divisor > 0, "facts give {{{name}}} the divisor 0");
        }
        // A dispatch runs on its arguments alone, so they bind every symbol its blocks use.
        let module = cx.module;
        for inst in module.walk(cx.regions.body) {
            let Some(block) = module.op(inst).get::<Block>() else {
                continue;
            };
            for sym in block.syms(module, module.regions(inst)[0]) {
                ensure!(
                    bound.contains(sym.as_str()),
                    "a block uses {{{}}}, which no argument binds",
                    sym.as_str()
                );
            }
        }
        check_body(module, cx.regions.body)?;
        check_placements(module, &Dispatch::blocks(module, cx.regions.body))?;
        if let Some(reference) = &self.target {
            check_on_target(module, cx.regions.body, &reference.target()?)?;
        }
        Ok(DispatchResults::default())
    }
}

/// Fails unless every op of the dispatch body `body` but its terminator is a `tile.block`, and
/// every result of a block that the dispatch does not give, unless the block after it is placed in
/// its loops and takes it as it is computed, has static extents: it lives in scratch of its own.
fn check_body(module: &Module, body: Region) -> Result<()> {
    let insts: Vec<Inst> = module.insts(body).collect();
    let given = module.yielded(body);
    for (index, &inst) in insts.iter().enumerate() {
        let op = module.op(inst);
        if op.has_trait::<Terminator>() {
            continue;
        }
        ensure!(
            op.get::<Block>().is_some(),
            "op {index} is {}, and a dispatch holds blocks",
            op.name()
        );
        let next = insts
            .get(index + 1)
            .and_then(|&after| module.op(after).get::<Block>());
        if next.is_some_and(|block| block.reads.iter().any(|read| read.at.is_some())) {
            continue;
        }
        for &result in module.results(inst) {
            let ty = module.ty(result);
            let dynamic = ty
                .as_tensor()
                .is_some_and(|tensor| tensor.static_dims().is_none());
            ensure!(
                given.contains(&result) || !dynamic,
                "block {index} gives a result of {ty} that the dispatch does not give, which lives \
                 in scratch of its own, whose extents must be static"
            );
        }
    }
    Ok(())
}

/// Fails with the first fact of `target`, the target the dispatch names, that a block of the
/// dispatch body `body` needs and the target lacks: the execution level of each mode of workers,
/// the memory of each holding, a cache or scratch with a buffer space for a copy or a hint for a
/// prefetch, and the vector forms of the work of each register tile, of its block and of each
/// block placed at each of its points.
fn check_on_target(module: &Module, body: Region, target: &Target) -> Result<()> {
    let blocks = Dispatch::blocks(module, body);
    let mut run_lanes = None;
    for (position, &(inst, block)) in blocks.iter().enumerate() {
        for (index, mode) in block.loops.iter().enumerate() {
            if let Some(memory) = &mode.memory {
                ensure!(
                    target.holding_memory(memory).is_some(),
                    "block {position}: mode {index} walks the pieces of {memory}, which is not a \
                     cache or scratch of {}",
                    target.name
                );
            }
            let Some(ModeKind::Workers(workers)) = &mode.kind else {
                continue;
            };
            ensure!(
                target.units(&workers.level).is_some(),
                "block {position}: mode {index} runs on the workers of {}, which {} does not have",
                workers.level,
                target.name
            );
        }
        for read in &block.reads {
            let Some(Holding::Copy(held) | Holding::Prefetch(held)) = &read.held else {
                continue;
            };
            let memory = target.holding_memory(&held.level).ok_or_else(|| {
                anyhow!(
                    "block {position}: the read of ${} is held for {}, a memory {} does not have",
                    read.source.first(),
                    held.level,
                    target.name
                )
            })?;
            let (kept, what) = match read.held {
                Some(Holding::Prefetch(_)) => (
                    memory.prefetch.is_some(),
                    "no prefetch that asks it for lines",
                ),
                _ => (memory.space.is_some(), "no buffer space that holds a copy"),
            };
            ensure!(
                kept,
                "block {position}: the read of ${} is held for {}, for which {} has {what}",
                read.source.first(),
                held.level,
                target.name
            );
        }
        let placed = block.reads.iter().any(|read| read.at.is_some());
        run_lanes = match (placed, position.checked_sub(1)) {
            (true, Some(before)) => {
                let (before_inst, earlier) = blocks[before];
                let input = Block::epilogue_input(module, before_inst, inst);
                let at_point = input.is_some_and(|operand| earlier.read_at_point(block, operand));
                run_lanes.filter(|_| at_point)
            }
            _ => block
                .walk()
                .atom
                .map(|atom| (atom.iter.position(), atom.lanes)),
        };
        let Some((lane, lanes)) = run_lanes else {
            continue;
        };
        let in_vectors = Block::vector_work(module, inst, lane)
            .and_then(|work| target.computes_vectors(&work.ops, &work.dtypes, lanes));
        in_vectors.map_err(|err| anyhow!("block {position}: {err:#}"))?;
    }
    Ok(())
}

/// Fails with the rule that the placements of `blocks`, the blocks of a dispatch body in order,
/// break. A pass that places blocks checks the blocks it would write with it.
///
/// - A placed block is placed by one read, of the result of the block before it, which has no
///   other use, and has no modes of its own; the first block of the body is placed by none.
/// - It is placed at the register tile of the first block of its run or at a mode of that block's
///   loops other than those of its register tile, every placed block of a run at the same piece.
/// - It runs on the elements that the block before it finishes in the piece there
///   ([`Block::placeable`]); when it reads them across the piece, rather than at its point, the
///   piece's extent along each parallel iterator of the block before it is static.
pub(crate) fn check_placements(module: &Module, blocks: &[(Inst, &Block)]) -> Result<()> {
    let mut first = 0;
    let mut run_placement = None;
    let mut in_register_tile = true;
    for (position, &(after, later)) in blocks.iter().enumerate() {
        let Some((read_index, placement)) = placing_read(position, later)? else {
            first = position;
            run_placement = None;
            in_register_tile = true;
            continue;
        };
        ensure!(
            position > 0,
            "block 0 is placed, and no block comes before it"
        );
        let (before, earlier) = blocks[position - 1];
        let input_operand = Block::epilogue_input(module, before, after);
        let reads_input = matches!(
            (&later.reads[read_index].source, input_operand),
            (Source::Operand(operand), Some(read_operand)) if *operand == read_operand
        );
        ensure!(
            reads_input,
            "block {position} is placed by read {read_index}, which does not read the result of \
             the block before it"
        );
        let input = input_operand.expect("a placed block reads the result of the block before it");
        ensure!(
            later.loops.is_empty(),
            "block {position} runs in the loops of the block before it, and has modes of its own"
        );
        ensure!(
            later.reads.iter().all(|read| read.held.is_none()),
            "block {position} runs in the loops of the block before it, and holds a read of its own"
        );
        ensure!(
            module.uses(module.results(before)[0]).len() == 1,
            "block {position} runs in the loops of the block before it, whose result has another \
             use"
        );
        let run_at = *run_placement.get_or_insert(placement);
        ensure!(
            placement == run_at,
            "block {position} is placed at {placement}, and the block before it at {run_at}: the \
             blocks of a run are placed at one piece"
        );
        let head = blocks[first].1;
        check_piece(position, head, (earlier, later, input), placement)?;
        // A block that runs at each point of the register tile, as the blocks before it in its
        // run do, computes in its vectors.
        in_register_tile &= earlier.read_at_point(later, input);
        if let (true, Some(atom)) = (in_register_tile, head.walk().atom) {
            check_lanes(later, atom.iter.position(), &[input], None)
                .map_err(|err| anyhow!("block {position}: {err}"))?;
        }
    }
    Ok(())
}

/// The read of block `position`, `block`, that places it, and where, if one does; fails when two
/// do.
fn placing_read(position: usize, block: &Block) -> Result<Option<(usize, Placement)>> {
    let mut placing = None;
    for (read_index, read) in block.reads.iter().enumerate() {
        let Some(placement) = read.at else {
            continue;
        };
        ensure!(
            placing.is_none(),
            "block {position} is placed by two of its reads, where it runs in one block's loops"
        );
        placing = Some((read_index, placement));
    }
    Ok(placing)
}

/// Fails with the rule that block `position`, `later`, placed at `placement` in the loops of
/// `head`, the first block of its run, after `earlier`, whose result is its operand `input`,
/// breaks about the piece it runs in.
fn check_piece(
    position: usize,
    head: &Block,
    (earlier, later, input): (&Block, &Block, usize),
    placement: Placement,
) -> Result<()> {
    let walk = head.walk();
    if let Placement::Mode(mode) = placement {
        let modes = head.loops.len();
        ensure!(
            mode < modes,
            "block {position} is placed at mode {mode} of the first block of its run, of {modes} \
             modes"
        );
        ensure!(
            walk.nest_of_mode[mode].is_some(),
            "block {position} is placed at mode {mode}, a mode of the register tile, where a block \
             is placed at `reg`"
        );
    }
    let (_, tiles) = walk.piece(head, placement);
    let whole =
        |iter: usize| tiles[iter].as_static().is_some() && tiles[iter] == head.iters[iter].extent;
    earlier.placeable(later, input, &whole).map_err(|err| {
        anyhow!(
            "block {position} does not run on the elements that the block before it finishes \
             where it is placed: {err}"
        )
    })?;
    if earlier.read_at_point(later, input) {
        return Ok(());
    }
    for (iter, kind) in earlier.iters.iter().map(|iter| iter.kind).enumerate() {
        ensure!(
            kind != IterKind::Par || tiles[iter].as_static().is_some(),
            "block {position} reads the result of the block before it across the piece where it \
             is placed, whose extent along d{iter} is not static"
        );
    }
    Ok(())
}
