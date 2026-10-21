//! A fold into new results. The fold starts at one point of the loop nest. Its state holds one
//! element for each point of the tiles that the loops inside that point walk. Each element starts
//! there, folds every point in one or two sweeps, and then finishes into the results or goes to the
//! blocks placed after it. Without a holding, the state lives in registers, or in one element of
//! private scratch for a block without a register tile. The state of a holding lives in private
//! scratch. Where the loops inside a register tile walk only reduced iterators, that tile folds
//! into a copy of its part in registers.

use std::rc::Rc;

use anyhow::Result;
use tinyvec::ArrayVec;

use super::body::Input;
use super::nest::clip_points;
use super::plan::{Node, StateFold, StatePlan};
use super::scope::{Point, Scope};
use super::{copy_box, BlockLowering};
use crate::conversion::tile_to_loop::combine::{Emit, Folding, State};
use crate::conversion::tile_to_loop::utils::{
    alloc, constant, each_point, for_range, index, load, offset_from, store,
};
use crate::ir::{DType, IndexExpr, OperandRef, Space, Value, MAX_ITERS};
use crate::rewrite::Builder;

/// What a loop over the elements of a state builds at each: from the builder, the scope there and
/// the element's position in the state.
type ElementBody<'a> = dyn FnMut(&mut Builder<'_>, &Scope, &[Value]) -> Result<()> + 'a;

/// The tiles of parallel iterators that a fold's state covers: the iterator at a position and the
/// extent of its tile, for each.
pub(super) type StateTiles = ArrayVec<[(usize, usize); MAX_ITERS]>;

/// The state that the points of a fold fold into while its loops are built.
pub(super) struct FoldState {
    /// The buffer of each component of the state.
    pub(super) buffers: Vec<Value>,
    /// The tiles that the state covers: the buffers hold one element for each of their points.
    pub(super) tiles: StateTiles,
    /// Where the tile of each iterator that the state covers starts, by position.
    pub(super) origins: Point,
}

impl FoldState {
    /// The state of no tile in `buffers`: one element.
    pub(super) fn one_element(buffers: Vec<Value>) -> FoldState {
        FoldState {
            buffers,
            tiles: StateTiles::new(),
            origins: [None; MAX_ITERS],
        }
    }

    /// The position in the state of the element at the point of `scope`.
    fn position(&self, builder: &mut Builder, scope: &Scope) -> Vec<Value> {
        let mut at = Vec::with_capacity(self.tiles.len());
        for &(position, _) in &self.tiles {
            let value = scope.point[position].expect("an element of the state's tile");
            let origin = self.origins[position].expect("the state's tile starts");
            at.push(offset_from(builder, value, origin));
        }
        at
    }

    /// The state of the element at position `at`.
    fn element<'s>(&'s self, at: &'s [Value]) -> State<'s> {
        State {
            buffers: &self.buffers,
            at,
        }
    }
}

/// Which way [`BlockLowering::copy_registers`] copies the part of a state that a register tile
/// covers.
#[derive(Clone, Copy)]
enum Copying {
    IntoRegisters,
    BackToState,
}

/// Copies the state components of one element, of the extents `shapes`, from position `from_at`
/// of the buffers `from` to position `to_at` of the buffers `to`.
fn copy_state(
    builder: &mut Builder,
    shapes: &[(DType, Vec<usize>)],
    (from, from_at): (&[Value], &[Value]),
    (to, to_at): (&[Value], &[Value]),
) -> Result<()> {
    for ((&source, &target), (_, shape)) in from.iter().zip(to).zip(shapes) {
        let mut counts = Vec::with_capacity(shape.len());
        for &extent in shape {
            counts.push(constant(builder, extent as i64));
        }
        copy_box(builder, (source, from_at), (target, to_at), &counts)?;
    }
    Ok(())
}

/// The buffers of the state components of `folding` for yielded values of `dtypes`, in `space`,
/// each with `extents` of elements in front of its own.
pub(super) fn state_buffers(
    builder: &mut Builder,
    folding: Folding,
    dtypes: &[DType],
    extents: &[usize],
    space: Space,
) -> Vec<Value> {
    let mut buffers = Vec::new();
    for (dtype, shape) in folding.components(dtypes) {
        buffers.push(alloc(
            builder,
            &[extents, &shape[..]].concat(),
            dtype,
            space,
        ));
    }
    buffers
}

impl BlockLowering<'_> {
    /// The state of `plan` over the tiles of the scope: each element started, folded by the nodes
    /// `inner` in each sweep, and finished.
    pub(super) fn emit_state(
        &self,
        builder: &mut Builder,
        plan: &StatePlan,
        inner: &[Node],
        scope: &Scope,
    ) -> Result<()> {
        let fold = plan.fold;
        let mut extents: ArrayVec<[usize; MAX_ITERS]> = ArrayVec::new();
        for &(_, extent) in &plan.tiles {
            extents.push(extent);
        }
        let buffers = state_buffers(builder, fold.folding, fold.dtypes, &extents, plan.space);
        let mut origins = [None; MAX_ITERS];
        for &(position, _) in &plan.tiles {
            origins[position] = Some(self.range(builder, scope, position).start);
        }
        let state = Rc::new(FoldState {
            buffers,
            tiles: plan.tiles,
            origins,
        });
        self.each_element(
            builder,
            &state,
            scope,
            &mut |element_builder, element_scope, at| {
                self.start_element(element_builder, fold, state.element(at), element_scope)
            },
        )?;
        for sweep in 0..fold.folding.sweeps() {
            if sweep > 0 {
                self.each_element(builder, &state, scope, &mut |element_builder, _, at| {
                    let mut emit = Emit {
                        builder: element_builder,
                        state: state.element(at),
                    };
                    fold.folding.next_sweep(&mut emit, sweep)
                })?;
            }
            let sweeping = Scope {
                state: Some(Rc::clone(&state)),
                sweep,
                ..scope.clone()
            };
            self.emit(builder, inner, &sweeping)?;
        }
        self.each_element(
            builder,
            &state,
            scope,
            &mut |element_builder, element_scope, at| {
                self.finish_element(element_builder, fold, state.element(at), element_scope)
            },
        )
    }

    /// The prefix of the scope's element, its points of the prefix iterator at `position` up to
    /// and including the element's, each building the nodes `inner`.
    pub(super) fn emit_prefix(
        &self,
        builder: &mut Builder,
        position: usize,
        inner: &[Node],
        scope: &Scope,
    ) -> Result<()> {
        let at = scope.point[position].expect("the prefix iterator is outside");
        let start = constant(builder, 0);
        let end = index(builder, IndexExpr::iter(0) + IndexExpr::constant(1), &[at]);
        for_range(builder, start, end, |loop_builder, step| {
            self.emit(loop_builder, inner, &scope.at(position, step))
        })
    }

    /// Starts `element`, the element of the state of `fold` at the scope's point, from the fold's
    /// initial values.
    fn start_element(
        &self,
        builder: &mut Builder,
        fold: StateFold,
        element: State,
        scope: &Scope,
    ) -> Result<()> {
        let part = fold.part;
        let mut init = Vec::with_capacity(part.fold.init.len());
        for (component, &OperandRef(operand)) in part.fold.init.iter().enumerate() {
            let write = &part.fold.writes[component];
            let coords = self.coords(builder, &write.coords, &scope.point);
            init.push(load(
                builder,
                self.operands[operand].in_place(),
                coords,
                None,
            ));
        }
        let mut emit = Emit {
            builder,
            state: element,
        };
        fold.folding.start(&mut emit, fold.dtypes, &init)
    }

    /// Writes the results of `element`, the element of the state of `fold` at the scope's point,
    /// or gives the one result to the blocks placed after this one.
    fn finish_element(
        &self,
        builder: &mut Builder,
        fold: StateFold,
        element: State,
        scope: &Scope,
    ) -> Result<()> {
        let part = fold.part;
        let rule = part.fold.combine.rule();
        let mut emit = Emit {
            builder,
            state: element,
        };
        let finished = fold.folding.finish(&mut emit)?;
        if !self.placed.is_empty() {
            return self.give(builder, scope, 0, finished[0][0]);
        }
        for (component, values) in finished.into_iter().enumerate() {
            let write = &part.fold.writes[component];
            let result = self.results[part.results.start + component];
            let coords = self.coords(builder, &write.coords, &scope.point);
            if rule.extra_dims(component).is_empty() {
                store(builder, values[0], result, coords);
                continue;
            }
            // Along the dimension the kind appends.
            for (rank, value) in values.into_iter().enumerate() {
                let rank_index = constant(builder, rank as i64);
                store(
                    builder,
                    value,
                    result,
                    [coords.clone(), vec![rank_index]].concat(),
                );
            }
        }
        Ok(())
    }

    /// Folds the values that the scope's point yields, where the block's constraints hold, into
    /// the element at the point of the state of `fold` that the scope holds.
    pub(super) fn push(&self, builder: &mut Builder, fold: StateFold, scope: &Scope) -> Result<()> {
        let state = scope
            .state
            .as_ref()
            .expect("a fold's points fold into its state");
        self.guarded(builder, scope, &mut |guarded_builder| {
            let yielded = self.point(guarded_builder, scope, Input::None)?;
            let at = state.position(guarded_builder, scope);
            let mut emit = Emit {
                builder: guarded_builder,
                state: state.element(&at),
            };
            let values = &yielded[fold.part.yields.clone()];
            fold.folding.push(&mut emit, scope.sweep, values)
        })
    }

    /// The nodes `inner`, loops over one register tile, folding into a copy in registers of the
    /// part of the state of `fold`, which the scope holds, that the tile covers: copied in before
    /// them, and back after.
    pub(super) fn emit_copies(
        &self,
        builder: &mut Builder,
        fold: StateFold,
        inner: &[Node],
        scope: &Scope,
    ) -> Result<()> {
        let state = scope
            .state
            .as_ref()
            .expect("a register tile copies a state");
        let mut extents: ArrayVec<[usize; MAX_ITERS]> = ArrayVec::new();
        let mut tiles = StateTiles::new();
        let mut origins = [None; MAX_ITERS];
        for dim in &self.walk.reg {
            let position = dim.iter.position();
            extents.push(dim.extent);
            tiles.push((position, dim.extent));
            origins[position] = Some(self.range(builder, scope, position).start);
        }
        let buffers = state_buffers(builder, fold.folding, fold.dtypes, &extents, Space::Reg);
        let registers = Rc::new(FoldState {
            buffers,
            tiles,
            origins,
        });
        let in_registers = Scope {
            state: Some(Rc::clone(&registers)),
            ..scope.clone()
        };
        let copied = (state.as_ref(), registers.as_ref());
        self.copy_registers(builder, fold, copied, Copying::IntoRegisters, scope)?;
        self.emit(builder, inner, &in_registers)?;
        self.copy_registers(builder, fold, copied, Copying::BackToState, scope)
    }

    /// Copies each element of `registers`, the register tile's part of `state`, the way `copying`
    /// says.
    fn copy_registers(
        &self,
        builder: &mut Builder,
        fold: StateFold,
        (state, registers): (&FoldState, &FoldState),
        copying: Copying,
        scope: &Scope,
    ) -> Result<()> {
        let shapes = fold.folding.components(fold.dtypes);
        self.each_element(
            builder,
            registers,
            scope,
            &mut |element_builder, element_scope, at| {
                let outer = state.position(element_builder, element_scope);
                let in_state = (&state.buffers[..], &outer[..]);
                let in_registers = (&registers.buffers[..], at);
                match copying {
                    Copying::IntoRegisters => {
                        copy_state(element_builder, &shapes, in_state, in_registers)
                    }
                    Copying::BackToState => {
                        copy_state(element_builder, &shapes, in_registers, in_state)
                    }
                }
            },
        )
    }

    /// Loops over the elements of the tiles of `state`, each clipped to its iterator's range, and
    /// builds `inner` at each, with the scope there and the element's position in the state. The
    /// last tiles that the register tile computes at once
    /// ([`crate::dialects::tile::Walk::state_atom_tiles`]) are one `loop.atom`.
    fn each_element(
        &self,
        builder: &mut Builder,
        state: &FoldState,
        scope: &Scope,
        inner: &mut ElementBody,
    ) -> Result<()> {
        let tiles = &state.tiles;
        let atom_from = tiles.len() - self.walk.state_atom_tiles(tiles);
        let mut loops = Vec::with_capacity(atom_from);
        for &(position, extent) in &tiles[..atom_from] {
            let origin = state.origins[position].expect("the state's tile starts");
            let span = self.range(builder, scope, position);
            loops.push(clip_points(builder, extent, origin, span.end));
        }
        let mut held = Vec::with_capacity(tiles.len() - atom_from);
        for &(position, extent) in &tiles[atom_from..] {
            let span = self.range(builder, scope, position);
            held.push((position, span, extent));
        }
        let atom = match held.is_empty() {
            true => None,
            false => Some(self.atom_box(builder, &held)),
        };
        each_point(
            builder,
            &loops,
            atom.as_ref(),
            &mut |point_builder, offsets| {
                let mut at_point = scope.clone();
                for (&(position, _), &offset) in tiles.iter().zip(offsets) {
                    let origin = state.origins[position].expect("the state's tile starts");
                    let value = index(
                        point_builder,
                        IndexExpr::iter(0) + IndexExpr::iter(1),
                        &[origin, offset],
                    );
                    at_point.point[position] = Some(value);
                }
                inner(point_builder, &at_point, offsets)
            },
        )
    }
}
