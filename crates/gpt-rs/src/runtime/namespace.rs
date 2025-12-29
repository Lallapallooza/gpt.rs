use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};

use crate::params::ModelNamespaceId;

static NEXT_NAMESPACE: AtomicU64 = AtomicU64::new(1);

pub fn next_namespace() -> ModelNamespaceId {
    ModelNamespaceId(u128::from(
        NEXT_NAMESPACE.fetch_add(1, AtomicOrdering::Relaxed),
    ))
}
