//! A `tile.block` in a `loop.func`: each of its folds as a loop nest that runs the block's reads,
//! constraints and body at each point, with the blocks placed in its loops.
//!
//! Each fold is planned first (`plan`): a list of plain values that follow from the block's walk
//! before any `loop` op is built, its loops, the state of a fold into new results, the piece that
//! placed blocks read across, and what the body does at each point. The emitters then build each
//! node: the loops (`nest`), which carry a [`Scope`] down, where the code stands and what it knows
//! there; the state (`fold`); the body at each point (`body`), which reads from the operands in
//! place or from the copies that holdings make (`stage`); the placed blocks (`placed`); and the
//! clips, which solve where coordinates stay in range (`bounds`). The emitters choose nothing the
//! plan does not say.

mod body;
mod bounds;
mod fold;
mod nest;
mod placed;
mod plan;
mod scope;
mod stage;

use std::ops::Range;
use std::rc::Rc;

use anyhow::Result;

use self::body::Input;
use self::fold::FoldState;
use self::plan::Leaf;
use self::scope::Scope;
use super::combine::{Emit, State};
use super::utils::{each_point, extent, load, store};
use crate::dialects::tile::{Block, CombineKind, Fold, Step, Walk};
use crate::ir::{DType, Dim, Inst, Module, OperandRef, Region, Space, Value};
use crate::rewrite::Builder;
use crate::targets::TargetDesc;

/// The buffers of the `loop.func` that a block reads and writes: the buffer of each operand with
/// its extents, and the buffer of each result.
pub(super) struct Buffers {
    pub(super) operands: Vec<OperandBuffer>,
    pub(super) results: Vec<Value>,
}

/// An operand of a block in the `loop.func`: the buffer it is in, none for the operand of a
/// placed block that the block before it gives, element by element or across a piece; and its
/// extents.
pub(super) struct OperandBuffer {
    pub(super) buffer: Option<Value>,
    pub(super) dims: Vec<Dim>,
}

impl OperandBuffer {
    /// The buffer that the block reads the operand from in place.
    pub(super) fn in_place(&self) -> Value {
        self.buffer
            .expect("a block reads in place no operand that the block before it gives")
    }
}

/// What the blocks of a run lower by, besides themselves: the walk of its first block, whether a
/// loop may skip a point where a padded read gives the pad 0 to products summed into a float, and
/// the dispatch's target, whose memories give the buffer space of each copy.
#[derive(Clone, Copy)]
pub(super) struct RunFacts<'m> {
    pub(super) walk: &'m Walk,
    pub(super) skip_pad: bool,
    pub(super) target: Option<&'m TargetDesc>,
}

/// One block of a dispatch, with the buffers of the `loop.func` it reads and writes, and the walk
/// of the run of blocks it lowers in: a block, and the blocks placed in its loops after it.
pub(super) struct BlockLowering<'m> {
    module: &'m Module,
    inst: Inst,
    block: &'m Block,
    body: Region,
    /// The buffer of each operand, and its extents.
    operands: Vec<OperandBuffer>,
    /// The buffer of each result: none for a block whose result the next block of its run takes.
    results: Vec<Value>,
    /// The walk of the first block of the run.
    walk: &'m Walk,
    /// Whether a loop may skip a point where a padded read gives the pad 0 to products summed
    /// into a float.
    skip_pad: bool,
    /// The dispatch's target.
    target: Option<&'m TargetDesc>,
    /// For a placed block, its operand that the result of the block before it is.
    input: Option<usize>,
    /// For a placed block, whether it runs at the point where the block before it finishes the
    /// element it reads, rather than once the piece is finished.
    at_point: bool,
    /// For the first block of a run, the blocks placed after it, in order.
    placed: Vec<BlockLowering<'m>>,
}

/// Where a block lowers in its run: first, with the blocks placed after it; or placed, reading
/// its operand `input`, the result of the block before it, at the point where that block finishes
/// each element, or once the piece is finished.
pub(super) enum Place<'m> {
    First { placed: Vec<BlockLowering<'m>> },
    Placed { input: usize, at_point: bool },
}

/// One fold of a block: its kind, the yielded values it takes, and the results it gives.
struct Part<'f> {
    fold: &'f Fold,
    yields: Range<usize>,
    results: Range<usize>,
}

impl<'m> BlockLowering<'m> {
    /// The lowering of the block `block` of `inst`, with its buffers, in a run of blocks that
    /// lowers by `facts`, at `place` in the run.
    pub(super) fn new(
        module: &'m Module,
        (inst, block): (Inst, &'m Block),
        buffers: Buffers,
        facts: RunFacts<'m>,
        place: Place<'m>,
    ) -> Self {
        let (placed, input, at_point) = match place {
            Place::First { placed } => (placed, None, false),
            Place::Placed { input, at_point } => (Vec::new(), Some(input), at_point),
        };
        BlockLowering {
            module,
            inst,
            block,
            body: module.regions(inst)[0],
            operands: buffers.operands,
            results: buffers.results,
            walk: facts.walk,
            skip_pad: facts.skip_pad,
            target: facts.target,
            input,
            at_point,
            placed,
        }
    }
}

impl BlockLowering<'_> {
    /// Lowers the folds of the block, in order, one walk of the points for each group of folds
    /// that share one ([`Block::fold_walks`]), each with the loops of the block's walk and their
    /// clips ([`crate::dialects::tile::Walk::clips`]).
    pub(super) fn lower(&self, builder: &mut Builder) -> Result<()> {
        let steps = self.walk.steps();
        let clips = self
            .walk
            .clips(self.module, self.inst, &steps, self.skip_pad);
        let before = Scope::outside(
            self.walk.nest.len(),
            vec![None; self.operands.len()],
            Rc::new(clips),
        );
        let outside = self.stage(builder, None, &before)?;
        let dtypes = Block::yield_types(self.module, self.body)?;
        let ranges = self.block.fold_ranges();
        for walk in self.block.fold_walks() {
            let mut parts = Vec::with_capacity(walk.len());
            for index in walk {
                let (yields, results) = ranges[index].clone();
                parts.push(Part {
                    fold: &self.block.folds[index],
                    yields,
                    results,
                });
            }
            let first = &parts[0];
            if first.fold.maps_new() {
                let nodes = self.plan_maps(&parts, &steps);
                self.emit(builder, &nodes, &outside)?;
                continue;
            }
            match first.fold.writes[0].into {
                Some(OperandRef(operand)) => {
                    self.lower_into(builder, (first, operand), (&dtypes, &steps), &outside)?
                }
                None => {
                    let nodes = self.plan_fold(first, &dtypes, &steps);
                    self.emit(builder, &nodes, &outside)?
                }
            }
        }
        Ok(())
    }

    /// The fold `part` into a copy of its operand `operand`, whose body yields values of `dtypes`,
    /// with the loops `steps`: the copy first, unless the result is the operand's buffer, then
    /// every point in order, which writes its value, or folds it into the element it names in
    /// the state of one element, which lives in private scratch.
    fn lower_into(
        &self,
        builder: &mut Builder,
        (part, operand): (&Part, usize),
        (dtypes, steps): (&[DType], &[Step]),
        outside: &Scope,
    ) -> Result<()> {
        let result = self.results[part.results.start];
        let source = &self.operands[operand];
        if source.in_place() != result {
            copy(builder, source.in_place(), result, &source.dims)?;
        }
        let mut inside = outside.clone();
        if part.fold.combine != CombineKind::None {
            let yield_dtypes = &dtypes[part.yields.clone()];
            let folding = part.fold.combine.fold();
            let buffers = fold::state_buffers(builder, folding, yield_dtypes, &[], Space::Private);
            inside.state = Some(Rc::new(FoldState::one_element(buffers)));
        }
        let nodes = self.plan_into(part, dtypes, steps);
        self.emit(builder, &nodes, &inside)
    }

    /// The body at the point of `scope`, and what `leaf` does with what it yields.
    fn emit_leaf(&self, builder: &mut Builder, leaf: &Leaf, scope: &Scope) -> Result<()> {
        match *leaf {
            Leaf::Maps(parts) => self.write_maps(builder, parts, scope),
            Leaf::Into { part, dtypes } => self.write_into(builder, part, dtypes, scope),
            Leaf::Push(fold) => self.push(builder, fold, scope),
        }
    }

    /// Writes the element of each of the `none` folds `parts` at the point of `scope`, where the
    /// block's constraints hold, or gives it to the blocks placed after this one.
    fn write_maps(&self, builder: &mut Builder, parts: &[Part], scope: &Scope) -> Result<()> {
        self.guarded(builder, scope, &mut |guarded_builder| {
            let yielded = self.point(guarded_builder, scope, Input::None)?;
            for part in parts {
                let value = yielded[part.yields.start];
                if !self.placed.is_empty() {
                    self.give(guarded_builder, scope, 0, value)?;
                    continue;
                }
                let coords =
                    self.coords(guarded_builder, &part.fold.writes[0].coords, &scope.point);
                store(
                    guarded_builder,
                    value,
                    self.results[part.results.start],
                    coords,
                );
            }
            Ok(())
        })
    }

    /// Writes the value of the fold `part` into a copy of an operand at the point of `scope`, or
    /// folds it into the element it names there, where the block's constraints hold: the state
    /// that the scope holds starts from the element, takes the value, and gives what the copy
    /// keeps.
    fn write_into(
        &self,
        builder: &mut Builder,
        part: &Part,
        dtypes: &[DType],
        scope: &Scope,
    ) -> Result<()> {
        let result = self.results[part.results.start];
        self.guarded(builder, scope, &mut |guarded_builder| {
            let yielded = self.point(guarded_builder, scope, Input::None)?;
            let values = &yielded[part.yields.clone()];
            let coords = self.coords(guarded_builder, &part.fold.writes[0].coords, &scope.point);
            let Some(state) = &scope.state else {
                store(guarded_builder, values[0], result, coords);
                return Ok(());
            };
            let current = load(guarded_builder, result, coords.clone(), None);
            let folding = part.fold.combine.fold();
            let mut emit = Emit {
                builder: guarded_builder,
                state: State {
                    buffers: &state.buffers,
                    at: &[],
                },
            };
            folding.start(&mut emit, &dtypes[part.yields.clone()], &[current])?;
            folding.push(&mut emit, 0, values)?;
            let folded = folding.finish(&mut emit)?;
            store(emit.builder, folded[0][0], result, coords);
            Ok(())
        })
    }
}

/// Copies every element of `from` into `to`, two buffers of the extents `dims`.
pub(super) fn copy(builder: &mut Builder, from: Value, to: Value, dims: &[Dim]) -> Result<()> {
    let mut counts = Vec::with_capacity(dims.len());
    for dim in dims {
        counts.push(extent(builder, dim));
    }
    copy_box(builder, (from, &[]), (to, &[]), &counts)
}

/// Copies a box of the counts `counts` of elements from `from` to `to`: each element whose
/// position in `from` is `from_at` then its offsets in the box, to the position `to_at` then the
/// same offsets in `to`.
fn copy_box(
    builder: &mut Builder,
    (from, from_at): (Value, &[Value]),
    (to, to_at): (Value, &[Value]),
    counts: &[Value],
) -> Result<()> {
    each_point(builder, counts, None, &mut |point_builder, offsets| {
        let element = load(point_builder, from, [from_at, offsets].concat(), None);
        store(point_builder, element, to, [to_at, offsets].concat());
        Ok(())
    })
}
