//! Lazy tensor handles that delay backend materialization until required.

use crate::backend::spec::{PortableBackend, ValueId};
use crate::ops::graph::GraphArena;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum InputRole {
    Arg,
    Param,
}

/// Captures either a materialized backend tensor or a lazy graph node.
pub(crate) enum LazyHandle<B: PortableBackend + 'static> {
    /// Direct tensor handle that can be reused without additional graph work.
    Input {
        id: u64,
        role: InputRole,
        tensor: B::TensorHandle,
    },
    /// Lazily-evaluated graph node that will be flushed into a concrete handle on demand.
    Node {
        graph: Arc<GraphArena<B>>,
        value: ValueId,
    },
}

impl<B: PortableBackend + 'static> LazyHandle<B> {
    pub(crate) fn id(&self) -> Option<u64> {
        match self {
            LazyHandle::Input { id, .. } => Some(*id),
            LazyHandle::Node { .. } => None,
        }
    }

    pub(crate) fn role(&self) -> Option<InputRole> {
        match self {
            LazyHandle::Input { role, .. } => Some(*role),
            LazyHandle::Node { .. } => None,
        }
    }

    /// Returns the graph arena when the handle points to a deferred node.
    pub fn graph(&self) -> Option<Arc<GraphArena<B>>> {
        match self {
            LazyHandle::Input { .. } => None,
            LazyHandle::Node { graph, .. } => Some(Arc::clone(graph)),
        }
    }
}
