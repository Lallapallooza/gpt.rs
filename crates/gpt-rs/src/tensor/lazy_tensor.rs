//! Lazy tensor handles that delay backend materialization until required.

use crate::backend::PortableBackend;
use crate::ops::graph::GraphArena;
use crate::ops::graph::ValueId;
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum InputRole {
    Arg,
    Param,
}

static ARG_ID_COUNTER: AtomicU64 = AtomicU64::new(1);

/// A tensor the backend holds. A graph captures it as an input under its role and id.
pub(crate) struct Materialized<B: PortableBackend + 'static> {
    pub(crate) id: u128,
    pub(crate) role: InputRole,
    pub(crate) tensor: B::TensorHandle,
}

impl<B: PortableBackend + 'static> Materialized<B> {
    /// A run-time argument under a fresh id.
    pub(crate) fn arg(tensor: B::TensorHandle) -> Arc<Self> {
        Arc::new(Materialized {
            id: u128::from(ARG_ID_COUNTER.fetch_add(1, Ordering::Relaxed)),
            role: InputRole::Arg,
            tensor,
        })
    }
}

/// Captures either a materialized backend tensor or a lazy graph node. Both clone without
/// allocating.
pub(crate) enum LazyHandle<B: PortableBackend + 'static> {
    Input(Arc<Materialized<B>>),
    /// Lazily-evaluated graph node that will be flushed into a concrete handle on demand.
    Node {
        graph: Arc<GraphArena<B>>,
        value: ValueId,
    },
}

impl<B: PortableBackend + 'static> Clone for LazyHandle<B> {
    fn clone(&self) -> Self {
        match self {
            LazyHandle::Input(input) => LazyHandle::Input(Arc::clone(input)),
            LazyHandle::Node { graph, value } => LazyHandle::Node {
                graph: Arc::clone(graph),
                value: *value,
            },
        }
    }
}

impl<B: PortableBackend + 'static> LazyHandle<B> {
    /// Returns the graph arena when the handle points to a deferred node.
    pub fn graph(&self) -> Option<Arc<GraphArena<B>>> {
        match self {
            LazyHandle::Input(_) => None,
            LazyHandle::Node { graph, .. } => Some(Arc::clone(graph)),
        }
    }
}
