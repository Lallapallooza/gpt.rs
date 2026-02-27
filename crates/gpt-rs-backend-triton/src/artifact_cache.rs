use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::hash::Hash;
use std::sync::{Arc, Mutex};

pub(crate) struct DecodedArtifactCache<K, V> {
    entries: Mutex<HashMap<K, Arc<V>>>,
}

impl<K, V> DecodedArtifactCache<K, V>
where
    K: Eq + Hash,
{
    pub(crate) fn new() -> Self {
        Self {
            entries: Mutex::new(HashMap::new()),
        }
    }

    pub(crate) fn get_or_try_insert_with<E, F>(&self, key: K, build: F) -> Result<(Arc<V>, bool), E>
    where
        F: FnOnce() -> Result<V, E>,
    {
        if let Some(existing) = self
            .entries
            .lock()
            .expect("decoded artifact cache mutex poisoned")
            .get(&key)
            .cloned()
        {
            return Ok((existing, true));
        }

        let built = Arc::new(build()?);
        let mut entries = self
            .entries
            .lock()
            .expect("decoded artifact cache mutex poisoned");
        match entries.entry(key) {
            Entry::Occupied(entry) => Ok((Arc::clone(entry.get()), true)),
            Entry::Vacant(entry) => {
                entry.insert(Arc::clone(&built));
                Ok((built, false))
            }
        }
    }
}

impl<K, V> Default for DecodedArtifactCache<K, V>
where
    K: Eq + Hash,
{
    fn default() -> Self {
        Self::new()
    }
}
