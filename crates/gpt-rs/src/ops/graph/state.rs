//! Internal graph bookkeeping structures shared by the arena and builder.

use std::any::Any;
use std::cell::RefCell;
use std::collections::HashMap;
use std::collections::HashSet;
use std::sync::Arc;

use anyhow::{anyhow, Result};
use foldhash::fast::FixedState;
use smallvec::SmallVec;

use gpt_rs_ir::types::{DType, TensorType, Type};
use gpt_rs_ir::Op;

use crate::backend::PortableBackend;
use crate::ops::graph::ValueId;
use crate::tensor::{InputRole, Materialized};

/// Hash map keyed by arena-internal ids. Keys are never attacker-controlled, so a fast fixed-seed
/// hasher is safe.
pub(super) type IdMap<K, V> = HashMap<K, V, FixedState>;

pub(super) type ValueSet = HashSet<ValueId, FixedState>;

/// A scalar operand broadcast to some extents: its dtype, its little-endian bits and the extents.
pub(super) type ScalarKey = (DType, u64, SmallVec<[usize; 4]>);

/// Mutable graph storage protected by a mutex inside [`GraphArena`](super::arena::GraphArena).
/// A value's id is its index in `values`.
pub(super) struct GraphInner<B: PortableBackend + 'static> {
    /// Every value of the graph, by id.
    pub(super) values: Vec<ValueRecord<B>>,
    /// The recorded ops, in record order.
    pub(super) ops: Vec<OpRecord>,
    /// The operands of every op. Each op holds the range of its own.
    pub(super) operands: Vec<ValueId>,
    /// The value of each captured tensor, by its role and id.
    pub(super) inputs: IdMap<(InputRole, u128), ValueId>,
    /// The constant of each scalar operand, by its dtype, little-endian bits and extents.
    pub(super) scalars: IdMap<ScalarKey, ValueId>,
    /// The values marked for export, which every program of the graph computes.
    pub(super) exports: ValueSet,
    /// Bumped when the graph gains a value, so a plan built for an older version is stale.
    pub(super) version: u64,
}

thread_local! {
    /// The emptied storage of the last graph that this thread dropped. The next graph that the
    /// thread creates fills it, so recording a step reuses the memory of an earlier one.
    static SPARE: RefCell<Option<Box<dyn Any>>> = const { RefCell::new(None) };
}

/// The vectors and maps of a [`GraphInner`], kept empty for reuse.
struct Storage<B: PortableBackend + 'static> {
    values: Vec<ValueRecord<B>>,
    ops: Vec<OpRecord>,
    operands: Vec<ValueId>,
    inputs: IdMap<(InputRole, u128), ValueId>,
    scalars: IdMap<ScalarKey, ValueId>,
    exports: ValueSet,
}

impl<B: PortableBackend + 'static> Default for Storage<B> {
    fn default() -> Self {
        Storage {
            values: Vec::new(),
            ops: Vec::new(),
            operands: Vec::new(),
            inputs: IdMap::default(),
            scalars: IdMap::default(),
            exports: ValueSet::default(),
        }
    }
}

impl<B: PortableBackend + 'static> Storage<B> {
    fn clear(&mut self) {
        self.values.clear();
        self.ops.clear();
        self.operands.clear();
        self.inputs.clear();
        self.scalars.clear();
        self.exports.clear();
    }
}

impl<B: PortableBackend + 'static> GraphInner<B> {
    /// An empty graph, in the spare storage of this thread when it has some.
    pub(super) fn new() -> Self {
        let spare = SPARE
            .try_with(|spare| spare.borrow_mut().take())
            .ok()
            .flatten()
            .and_then(|spare| spare.downcast::<Storage<B>>().ok());
        let Storage {
            values,
            ops,
            operands,
            inputs,
            scalars,
            exports,
        } = spare.map(|spare| *spare).unwrap_or_default();
        GraphInner {
            values,
            ops,
            operands,
            inputs,
            scalars,
            exports,
            version: 0,
        }
    }

    pub(super) fn bump_version(&mut self) {
        self.version = self.version.wrapping_add(1);
    }

    pub(super) fn value(&self, value: ValueId) -> Result<&ValueRecord<B>> {
        self.values
            .get(value.0 as usize)
            .ok_or_else(|| anyhow!("{value:?} is not captured in this graph"))
    }

    /// The op of a recorded value and which of its results the value is, or `None` for a captured
    /// tensor.
    pub(super) fn node(&self, value: ValueId) -> Option<(&OpRecord, usize)> {
        match self.values.get(value.0 as usize)?.def {
            ValueDef::Node { op, result, .. } => Some((&self.ops[op as usize], result as usize)),
            ValueDef::Input(_) => None,
        }
    }

    pub(super) fn operands(&self, op: &OpRecord) -> &[ValueId] {
        &self.operands[op.operands.0 as usize..op.operands.1 as usize]
    }

    pub(super) fn push_value(&mut self, ty: TensorType, def: ValueDef<B>) -> ValueId {
        let value = ValueId(self.values.len() as u32);
        self.values.push(ValueRecord {
            ty: Type::Tensor(ty),
            def,
        });
        value
    }
}

impl<B: PortableBackend + 'static> Drop for GraphInner<B> {
    fn drop(&mut self) {
        let mut storage = Storage {
            values: std::mem::take(&mut self.values),
            ops: std::mem::take(&mut self.ops),
            operands: std::mem::take(&mut self.operands),
            inputs: std::mem::take(&mut self.inputs),
            scalars: std::mem::take(&mut self.scalars),
            exports: std::mem::take(&mut self.exports),
        };
        storage.clear();
        let _ = SPARE.try_with(|spare| *spare.borrow_mut() = Some(Box::new(storage)));
    }
}

/// A value of the graph: its type, and what defines it.
pub(super) struct ValueRecord<B: PortableBackend + 'static> {
    /// Always a tensor type. The type rules read operand types as [`Type`]s.
    pub(super) ty: Type,
    pub(super) def: ValueDef<B>,
}

impl<B: PortableBackend + 'static> ValueRecord<B> {
    pub(super) fn tensor_type(&self) -> &TensorType {
        self.ty.as_tensor().expect("graph values are tensors")
    }
}

pub(super) enum ValueDef<B: PortableBackend + 'static> {
    /// Result `result` of the op `op` (an index into [`GraphInner::ops`]).
    Node {
        op: u32,
        result: u32,
        state: NodeState<B>,
    },
    /// A tensor the backend holds, captured as an input of the graph.
    Input(Arc<Materialized<B>>),
}

/// A recorded op: the op, the range of its operands in [`GraphInner::operands`], and the value
/// of its result 0. Its other results follow that value.
pub(super) struct OpRecord {
    pub(super) op: Op,
    pub(super) operands: (u32, u32),
    pub(super) first: ValueId,
}

/// Current materialization state of a recorded node.
/// Nodes start pending and transition to ready once the backend produces a handle.
pub(super) enum NodeState<B: PortableBackend + 'static> {
    /// Node still needs to run during program capture.
    Pending,
    /// Node already materialized on the backend and holds a tensor handle.
    Ready(B::TensorHandle),
}
