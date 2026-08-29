//! Internal graph bookkeeping structures shared by the arena and builder.

use std::collections::HashMap;
use std::collections::HashSet;
use std::sync::Arc;

use foldhash::fast::FixedState;

use crate::backend::spec::{Operand, Operation, PortableBackend, TensorSpec, ValueId};
use crate::params::{BaseParamId, ParamSource};
use crate::tensor::InputRole;

/// Hash map keyed by arena-internal ids. Keys are never attacker-controlled, so a fast fixed-seed
/// hasher is safe.
pub(super) type IdMap<K, V> = HashMap<K, V, FixedState>;

pub(super) type ValueMap<V> = IdMap<ValueId, V>;
pub(super) type ValueSet = HashSet<ValueId, FixedState>;

/// Mutable graph storage protected by a mutex inside [`GraphArena`](super::arena::GraphArena).
/// It tracks recorded nodes, insertion order, and imported parameters for the current arena.
pub(super) struct GraphInner<B: PortableBackend + 'static> {
    pub(super) next_value: u32,
    pub(super) nodes: ValueMap<NodeRecord<B>>,
    pub(super) order: Vec<ValueId>,
    pub(super) parameters: Vec<ParameterRecord<B>>,
    pub(super) parameter_by_value: ValueMap<usize>,
    pub(super) parameter_lookup: IdMap<(InputRole, u128), ValueId>,
    pub(super) param_sources: IdMap<u128, ParamSourceRecord<B>>,
    pub(super) exports: ValueSet,
    pub(super) version: u64,
}

impl<B: PortableBackend + 'static> GraphInner<B> {
    /// Constructs an empty graph state ready for incremental population.
    pub(super) fn new() -> Self {
        GraphInner {
            next_value: 0,
            nodes: ValueMap::default(),
            order: Vec::new(),
            parameters: Vec::new(),
            parameter_by_value: ValueMap::default(),
            parameter_lookup: IdMap::default(),
            param_sources: IdMap::default(),
            exports: ValueSet::default(),
            version: 0,
        }
    }

    pub(super) fn bump_version(&mut self) {
        self.version = self.version.wrapping_add(1);
    }

    pub(super) fn parameter(&self, value: ValueId) -> Option<&ParameterRecord<B>> {
        self.parameter_by_value
            .get(&value)
            .copied()
            .and_then(|idx| self.parameters.get(idx))
    }

    pub(super) fn parameter_mut(&mut self, value: ValueId) -> Option<&mut ParameterRecord<B>> {
        let idx = self.parameter_by_value.get(&value).copied()?;
        self.parameters.get_mut(idx)
    }

    pub(super) fn push_parameter(&mut self, record: ParameterRecord<B>) {
        let idx = self.parameters.len();
        self.parameter_by_value.insert(record.value, idx);
        self.parameters.push(record);
    }
}

/// Recorded node metadata kept while the graph remains in a lazy state.
/// Each record stores the operation, operands, static specification, and current state.
pub(super) struct NodeRecord<B: PortableBackend + 'static> {
    pub(super) op: Operation,
    pub(super) operands: Vec<Operand>,
    pub(super) spec: TensorSpec,
    pub(super) state: NodeState<B>,
}

/// Current materialisation state of a recorded node.
/// Nodes start pending and transition to ready once the backend produces a handle.
pub(super) enum NodeState<B: PortableBackend + 'static> {
    /// Node still needs to run during program capture.
    Pending,
    /// Node already materialised on the backend and holds a tensor handle.
    Ready(B::TensorHandle),
}

/// Descriptor for graph inputs imported from host tensors or other arenas.
/// Parameters behave like captured constants and are reused across program executions.
pub(super) struct ParameterRecord<B: PortableBackend + 'static> {
    pub(super) value: ValueId,
    pub(super) spec: TensorSpec,
    pub(super) handle: Option<B::TensorHandle>,
    pub(super) role: InputRole,
    pub(super) stable_id: Option<u128>,
}

/// Runtime source binding used to lazily load parameter handles on resolver misses.
pub(super) struct ParamSourceRecord<B: PortableBackend + 'static> {
    pub(super) base_id: BaseParamId,
    pub(super) source: Arc<dyn ParamSource<B>>,
}
