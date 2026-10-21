//! The combine kinds of `tile.block` as `loop` IR: each kind's [`CombineFold`] on ops that the
//! builder builds, with the state of an element in private scratch or registers. Each gives the
//! bits that the interpreter's fold of the kind ([`Acc`](crate::dialects::tile::Acc)) gives, which
//! the filetests' differential checks.

use anyhow::Result;

use super::utils::{cell, for_range, get, if_then, index, lit, load, predicate, set, store};
use crate::dialects::arith;
use crate::dialects::tile::{CombineFold, Scalars, SlotBody, WhenBody};
use crate::host::HostScalar;
use crate::ir::{Constraint, DType, IndexExpr, Value};
use crate::rewrite::{Build, Builder};
use crate::Op;

/// The state of a fold for one element: a buffer per state component, which holds the state of
/// every element of a tile, and the element's position in them.
#[derive(Clone, Copy)]
pub(super) struct State<'s> {
    pub(super) buffers: &'s [Value],
    pub(super) at: &'s [Value],
}

impl State<'_> {
    /// The position of the value of a component at `slot`, when it holds several.
    fn position(&self, slot: Option<Value>) -> Vec<Value> {
        [self.at, slot.as_slice()].concat()
    }
}

/// The fold of a combine kind as `tile-to-loop` builds it ([`CombineKind::fold`]): each of its
/// calls builds its ops on the state of an element through an [`Emit`].
///
/// [`CombineKind::fold`]: crate::dialects::tile::CombineKind::fold
pub(super) type Folding<'k> = &'k dyn CombineFold<Value, Value, Value>;

/// The scalar operations of a fold as `loop` IR: ops that `builder` builds, on the state of one
/// element, `state`.
pub(super) struct Emit<'b, 'm> {
    pub(super) builder: &'b mut Builder<'m>,
    pub(super) state: State<'b>,
}

impl Scalars for Emit<'_, '_> {
    type Value = Value;
    type Slot = Value;
    type Cell = Value;

    fn apply(&mut self, op: Op, operands: &[Value]) -> Result<Value> {
        let inst = self
            .builder
            .create_op(op, operands.to_vec(), None, Vec::new());
        Ok(self.builder.module.results(inst)[0])
    }

    fn constant(&mut self, value: HostScalar) -> Value {
        lit(self.builder, value)
    }

    fn dtype(&self, value: Value) -> DType {
        self.builder
            .value_type(value)
            .as_scalar()
            .expect("a fold computes on scalars")
    }

    fn get(&mut self, component: usize, slot: Option<Value>) -> Value {
        let at = self.state.position(slot);
        load(self.builder, self.state.buffers[component], at, None)
    }

    fn set(&mut self, component: usize, slot: Option<Value>, value: Value) {
        let at = self.state.position(slot);
        store(self.builder, value, self.state.buffers[component], at);
    }

    fn cell(&mut self, dtype: DType) -> Value {
        cell(self.builder, dtype)
    }

    fn load(&mut self, cell: Value) -> Value {
        get(self.builder, cell)
    }

    fn store(&mut self, cell: Value, value: Value) {
        set(self.builder, cell, value);
    }

    fn slot_of(&mut self, value: Value) -> Value {
        self.builder.create(arith::ToIndex, value)
    }

    fn slot(&mut self, expr: IndexExpr, operands: &[Value]) -> Value {
        index(self.builder, expr, operands)
    }

    fn holds(&mut self, constraints: &[Constraint], operands: &[Value]) -> Value {
        predicate(self.builder, constraints, operands)
    }

    fn each_slot(
        &mut self,
        start: Value,
        end: Value,
        body: &mut SlotBody<Value, Value, Value>,
    ) -> Result<()> {
        let state = self.state;
        for_range(self.builder, start, end, |builder, slot| {
            body(&mut Emit { builder, state }, slot)
        })
    }

    fn when(&mut self, condition: Value, body: &mut WhenBody<Value, Value, Value>) -> Result<()> {
        let state = self.state;
        if_then(self.builder, condition, |builder| {
            body(&mut Emit { builder, state })
        })
    }
}
