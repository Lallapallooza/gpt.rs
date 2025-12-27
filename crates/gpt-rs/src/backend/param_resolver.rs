//! Backend-provided store for parameter handles keyed by stable ids.
//!
//! This is used by graph compilation/optimization to memoize derived parameter representations
//! (for example, packed weights) and by the runtime to resolve parameter inputs when executing
//! cached plans.

use std::collections::HashMap;
use std::sync::Mutex;

/// Thread-safe resolver keyed by stable parameter id.
pub trait ParamResolver: Send + Sync {
    type Handle: Clone + Send + Sync + 'static;

    fn get(&self, stable_id: u128) -> Option<Self::Handle>;

    fn set(&self, stable_id: u128, handle: Self::Handle);
}

/// Default in-memory implementation used when backends do not provide a resolver.
#[derive(Debug, Default)]
pub struct InMemoryParamResolver<H: Clone + Send + Sync + 'static> {
    map: Mutex<HashMap<u128, H>>,
}

impl<H: Clone + Send + Sync + 'static> InMemoryParamResolver<H> {
    pub fn new() -> Self {
        Self {
            map: Mutex::new(HashMap::new()),
        }
    }
}

impl<H: Clone + Send + Sync + 'static> ParamResolver for InMemoryParamResolver<H> {
    type Handle = H;

    fn get(&self, stable_id: u128) -> Option<Self::Handle> {
        self.map
            .lock()
            .expect("param resolver poisoned")
            .get(&stable_id)
            .cloned()
    }

    fn set(&self, stable_id: u128, handle: Self::Handle) {
        self.map
            .lock()
            .expect("param resolver poisoned")
            .insert(stable_id, handle);
    }
}
