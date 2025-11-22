//! Runtime helpers for managing functional registries and constructing cache keys.
//!
//! The utilities expose a thread-local stack for temporary registry overrides and provide
//! deterministic hashing helpers used by benchmark and memoization layers.

use std::any::{Any, TypeId};
use std::cell::RefCell;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use crate::backend::spec::PortableBackend;
use crate::tensor::DeviceTensor;

use super::registry::{FunctionalRegistry, FunctionalRegistryHandle};

thread_local! {
    static REGISTRY_STACK: RefCell<Vec<RegistryEntry>> = const { RefCell::new(Vec::new()) };
}

struct RegistryEntry {
    type_id: TypeId,
    registry: Arc<dyn Any + Send + Sync>,
}

/// RAII guard that keeps a registry active on the thread-local stack.
pub struct FunctionalRegistryGuard {
    type_id: TypeId,
}

impl FunctionalRegistryGuard {
    /// Pushes a registry for backend `B` onto the stack and returns the guard.
    pub fn push<B: PortableBackend + 'static>(registry: FunctionalRegistryHandle<B>) -> Self {
        let type_id = TypeId::of::<B>();
        let registry_any: Arc<dyn Any + Send + Sync> = registry;
        REGISTRY_STACK.with(|stack| {
            stack.borrow_mut().push(RegistryEntry {
                type_id,
                registry: registry_any,
            });
        });
        FunctionalRegistryGuard { type_id }
    }
}

impl Drop for FunctionalRegistryGuard {
    fn drop(&mut self) {
        REGISTRY_STACK.with(|stack| {
            let mut stack = stack.borrow_mut();
            let entry = stack
                .pop()
                .expect("functional registry stack underflow: unmatched pop");
            assert_eq!(
                entry.type_id, self.type_id,
                "functional registry stack pop type mismatch"
            );
        });
    }
}

/// Convenience helper for pushing a registry and returning its guard.
pub fn push_registry<B: PortableBackend + 'static>(
    registry: FunctionalRegistryHandle<B>,
) -> FunctionalRegistryGuard {
    FunctionalRegistryGuard::push(registry)
}

/// Runs `f` with `registry` temporarily installed on the stack.
pub fn with_registry<B, F, R>(registry: FunctionalRegistryHandle<B>, f: F) -> R
where
    B: PortableBackend + 'static,
    F: FnOnce() -> R,
{
    let guard = push_registry(registry);
    let result = f();
    drop(guard);
    result
}

/// Returns the top-most registry for backend `B`, if any.
pub fn current_registry<B: PortableBackend + 'static>() -> Option<FunctionalRegistryHandle<B>> {
    let type_id = TypeId::of::<B>();
    REGISTRY_STACK.with(|stack| {
        for entry in stack.borrow().iter().rev() {
            if entry.type_id == type_id {
                let arc = entry.registry.clone();
                return arc.downcast::<FunctionalRegistry<B>>().ok();
            }
        }
        None
    })
}

/// Opaque identifier used to memoize benchmark results for functional invocations.
/// Keys can be generated via [`CacheKeyBuilder`] and stored alongside implementation names.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FunctionalCacheKey(u64);

/// Helper for deterministically hashing a collection of cache key arguments.
pub struct CacheKeyBuilder {
    hasher: DefaultHasher,
    empty: bool,
}

impl CacheKeyBuilder {
    /// Creates a builder with an empty key.
    pub fn new() -> Self {
        CacheKeyBuilder {
            hasher: DefaultHasher::new(),
            empty: true,
        }
    }

    /// Mixes an additional value into the hash state.
    pub fn combine_hash<T: Hash + ?Sized>(&mut self, value: &T) {
        self.empty = false;
        value.hash(&mut self.hasher);
    }

    /// Finalizes the hash and returns a cache key unless no values were added.
    pub fn finish(self) -> Option<FunctionalCacheKey> {
        if self.empty {
            None
        } else {
            Some(FunctionalCacheKey(self.hasher.finish()))
        }
    }
}

impl Default for CacheKeyBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Trait for types that can contribute to a functional cache key.
pub trait CacheKeyArg {
    fn add_to_cache_key(&self, builder: &mut CacheKeyBuilder);
}

/// Adds `value` into the cache key using the [`CacheKeyArg`] trait.
pub fn accumulate_cache_key<T: CacheKeyArg + ?Sized>(builder: &mut CacheKeyBuilder, value: &T) {
    value.add_to_cache_key(builder);
}

impl<B: PortableBackend + 'static> CacheKeyArg for DeviceTensor<B> {
    fn add_to_cache_key(&self, builder: &mut CacheKeyBuilder) {
        builder.combine_hash(self.shape().dims());
        builder.combine_hash(&self.dtype());
        builder.combine_hash(&self.requires_grad_flag());
    }
}

impl<T: CacheKeyArg + ?Sized> CacheKeyArg for &T {
    fn add_to_cache_key(&self, builder: &mut CacheKeyBuilder) {
        (*self).add_to_cache_key(builder);
    }
}

impl<T: CacheKeyArg> CacheKeyArg for Option<T> {
    fn add_to_cache_key(&self, builder: &mut CacheKeyBuilder) {
        match self {
            Some(value) => {
                builder.combine_hash(&1u8);
                value.add_to_cache_key(builder);
            }
            None => builder.combine_hash(&0u8),
        }
    }
}

impl CacheKeyArg for [usize] {
    fn add_to_cache_key(&self, builder: &mut CacheKeyBuilder) {
        builder.combine_hash(self);
    }
}

impl CacheKeyArg for Vec<usize> {
    fn add_to_cache_key(&self, builder: &mut CacheKeyBuilder) {
        builder.combine_hash(self);
    }
}

impl<const N: usize> CacheKeyArg for [usize; N] {
    fn add_to_cache_key(&self, builder: &mut CacheKeyBuilder) {
        builder.combine_hash(self);
    }
}

impl CacheKeyArg for usize {
    fn add_to_cache_key(&self, builder: &mut CacheKeyBuilder) {
        builder.combine_hash(self);
    }
}

impl CacheKeyArg for u64 {
    fn add_to_cache_key(&self, builder: &mut CacheKeyBuilder) {
        builder.combine_hash(self);
    }
}

impl CacheKeyArg for i64 {
    fn add_to_cache_key(&self, builder: &mut CacheKeyBuilder) {
        builder.combine_hash(self);
    }
}

impl CacheKeyArg for bool {
    fn add_to_cache_key(&self, builder: &mut CacheKeyBuilder) {
        builder.combine_hash(&(*self as u8));
    }
}

impl CacheKeyArg for f32 {
    fn add_to_cache_key(&self, builder: &mut CacheKeyBuilder) {
        builder.combine_hash(&self.to_bits());
    }
}

impl CacheKeyArg for () {
    fn add_to_cache_key(&self, _builder: &mut CacheKeyBuilder) {}
}
