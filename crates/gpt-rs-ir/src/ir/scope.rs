//! Name lookup across nested regions.

use std::borrow::Borrow;
use std::collections::HashMap;
use std::hash::Hash;

/// A map whose lookups also see the enclosing scopes, up to the nearest isolated one.
pub(crate) struct ScopedMap<K, V> {
    /// The scopes, innermost last, each marked when it is isolated.
    scopes: Vec<(HashMap<K, V>, bool)>,
}

impl<K: Eq + Hash, V: Copy> ScopedMap<K, V> {
    /// A map with one isolated scope, which stays.
    pub(crate) fn new() -> Self {
        Self {
            scopes: vec![(HashMap::new(), true)],
        }
    }

    pub(crate) fn push(&mut self, isolated: bool) {
        self.scopes.push((HashMap::new(), isolated));
    }

    pub(crate) fn pop(&mut self) {
        self.scopes.pop();
    }

    pub(crate) fn get<Q: Eq + Hash + ?Sized>(&self, key: &Q) -> Option<V>
    where
        K: Borrow<Q>,
    {
        for (scope, isolated) in self.scopes.iter().rev() {
            if let Some(&value) = scope.get(key) {
                return Some(value);
            }
            if *isolated {
                return None;
            }
        }
        None
    }

    /// Adds `key` to the innermost scope, and returns the value it had there.
    pub(crate) fn insert(&mut self, key: K, value: V) -> Option<V> {
        self.scopes
            .last_mut()
            .expect("the first scope stays")
            .0
            .insert(key, value)
    }
}
