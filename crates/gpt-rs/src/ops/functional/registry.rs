//! Runtime registry for selecting and benchmarking functional implementations.
//!
//! The registry enables swapping kernels at runtime, forcing specific implementations, and
//! caching benchmark results so the fastest variant can be reused without repeating timing runs.

use std::any::Any;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;
use std::num::NonZeroUsize;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};

use crate::backend::spec::PortableBackend;
use lru::LruCache;

use super::attention::{AttentionEntry, AttentionImplementation};
use super::runtime::FunctionalCacheKey;

/// Shared pointer to a functional registry tied to a specific backend.
pub type FunctionalRegistryHandle<B> = Arc<FunctionalRegistry<B>>;

/// Symbolic key used to index functional implementations (e.g., "attention").
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct FunctionalKey(&'static str);

impl FunctionalKey {
    /// Creates a new key from a static identifier.
    pub const fn new(name: &'static str) -> Self {
        FunctionalKey(name)
    }

    /// Returns the string representation of the key.
    /// This is used as the lookup key inside override maps and log messages.
    pub fn as_str(self) -> &'static str {
        self.0
    }
}

impl Hash for FunctionalKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.hash(state);
    }
}

const DEFAULT_BENCHMARK_CACHE_SIZE: usize = 128;

/// User-configurable overrides that steer how functionals pick implementations.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FunctionalOverrides {
    #[serde(flatten)]
    overrides: HashMap<String, FunctionalPolicySetting>,
}

impl FunctionalOverrides {
    /// Returns the policy configured for the provided key.
    /// Defaults to [`FunctionalPolicy::Default`] when no explicit override exists.
    pub fn policy(&self, key: FunctionalKey) -> FunctionalPolicy {
        self.overrides
            .get(key.as_str())
            .cloned()
            .map(FunctionalPolicy::from_setting)
            .unwrap_or(FunctionalPolicy::Default)
    }

    /// Whether any overrides are currently recorded.
    /// Helpful when deciding if the registry needs to consult override logic at runtime.
    pub fn is_empty(&self) -> bool {
        self.overrides.is_empty()
    }
}

/// Effective policy applied to a functional key.
/// Policies decide whether to execute the default implementation, force a specific variant, or benchmark candidates.
#[derive(Debug, Clone)]
pub enum FunctionalPolicy {
    Default,
    Force { implementation: String },
    Benchmark { cache_size: usize },
}

impl FunctionalPolicy {
    fn from_setting(setting: FunctionalPolicySetting) -> Self {
        match setting {
            FunctionalPolicySetting::Force(name) if name.is_empty() => FunctionalPolicy::Default,
            FunctionalPolicySetting::Force(name) => FunctionalPolicy::Force {
                implementation: name,
            },
            FunctionalPolicySetting::Benchmark { cache_size } => {
                FunctionalPolicy::Benchmark { cache_size }
            }
        }
    }
}

/// Raw settings parsed from configuration files before being resolved into a policy.
/// Strings like `force=my_impl` or `benchmark(cache=64)` land here before normalisation.
#[derive(Debug, Clone)]
pub enum FunctionalPolicySetting {
    Force(String),
    Benchmark { cache_size: usize },
}

impl<'de> Deserialize<'de> for FunctionalPolicySetting {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let raw = String::deserialize(deserializer)?;
        parse_policy_string(&raw).map_err(serde::de::Error::custom)
    }
}

impl Serialize for FunctionalPolicySetting {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        match self {
            FunctionalPolicySetting::Force(name) => {
                if name.contains('=') {
                    serializer.serialize_str(&format!("force={}", name))
                } else {
                    serializer.serialize_str(name)
                }
            }
            FunctionalPolicySetting::Benchmark { cache_size } => {
                if *cache_size == DEFAULT_BENCHMARK_CACHE_SIZE {
                    serializer.serialize_str("benchmark")
                } else {
                    serializer.serialize_str(&format!("benchmark(cache={})", cache_size))
                }
            }
        }
    }
}

fn parse_policy_string(raw: &str) -> Result<FunctionalPolicySetting> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Ok(FunctionalPolicySetting::Force(String::new()));
    }

    if let Some(rest) = trimmed.strip_prefix("force=") {
        return Ok(FunctionalPolicySetting::Force(rest.trim().to_string()));
    }

    if trimmed.eq_ignore_ascii_case("benchmark") {
        return Ok(FunctionalPolicySetting::Benchmark {
            cache_size: DEFAULT_BENCHMARK_CACHE_SIZE,
        });
    }

    if let Some(rest) = trimmed
        .strip_prefix("benchmark(")
        .and_then(|inner| inner.strip_suffix(')'))
    {
        let mut cache_size = DEFAULT_BENCHMARK_CACHE_SIZE;
        for part in rest.split(',') {
            let part = part.trim();
            if part.is_empty() {
                continue;
            }
            if let Some(value) = part.strip_prefix("cache=") {
                cache_size = value
                    .parse()
                    .map_err(|_| anyhow!("invalid benchmark cache size: {}", value))?;
                if cache_size == 0 {
                    return Err(anyhow!("benchmark cache size must be greater than zero"));
                }
            } else {
                return Err(anyhow!("unknown benchmark override option: {}", part));
            }
        }
        return Ok(FunctionalPolicySetting::Benchmark { cache_size });
    }

    // Legacy compatibility: treat bare string as force override.
    Ok(FunctionalPolicySetting::Force(trimmed.to_string()))
}

struct OpRegistry<I: ?Sized> {
    implementations: Mutex<Vec<Arc<I>>>,
    policy: FunctionalPolicy,
    benchmark_cache: Mutex<Option<LruCache<FunctionalCacheKey, &'static str>>>,
}

impl<I: ?Sized> OpRegistry<I> {
    fn new(policy: FunctionalPolicy) -> Self {
        let benchmark_cache = match policy {
            FunctionalPolicy::Benchmark { cache_size } => {
                let capacity = cache_size.max(1);
                let capacity = NonZeroUsize::new(capacity).unwrap();
                Some(LruCache::new(capacity))
            }
            _ => None,
        };

        Self {
            implementations: Mutex::new(Vec::new()),
            policy,
            benchmark_cache: Mutex::new(benchmark_cache),
        }
    }

    fn register(&self, implementation: Arc<I>) {
        self.implementations.lock().unwrap().push(implementation);
    }

    fn implementations(&self) -> Vec<Arc<I>> {
        self.implementations.lock().unwrap().clone()
    }

    fn select_by_name<B, E>(&self, name: &str) -> Option<Arc<I>>
    where
        B: PortableBackend + 'static,
        E: FunctionalRegistryEntry<B, Impl = I>,
    {
        self.implementations()
            .into_iter()
            .find(|candidate| E::name(candidate.as_ref()) == name)
    }

    fn call_forward<B, E>(&self, ctx: E::ForwardCtx<'_>) -> Result<(E::ForwardOutput, &'static str)>
    where
        B: PortableBackend + 'static,
        E: FunctionalRegistryEntry<B, Impl = I>,
    {
        match &self.policy {
            FunctionalPolicy::Default => self.call_forward_default::<B, E>(ctx),
            FunctionalPolicy::Force { implementation } => {
                self.call_forward_force::<B, E>(implementation, ctx)
            }
            FunctionalPolicy::Benchmark { .. } => self.call_forward_benchmark::<B, E>(ctx),
        }
    }

    fn call_forward_default<B, E>(
        &self,
        ctx: E::ForwardCtx<'_>,
    ) -> Result<(E::ForwardOutput, &'static str)>
    where
        B: PortableBackend + 'static,
        E: FunctionalRegistryEntry<B, Impl = I>,
    {
        let key = E::key();
        let implementation = self
            .implementations()
            .into_iter()
            .find(|candidate| E::supports(candidate.as_ref(), &ctx))
            .ok_or_else(|| {
                anyhow!(
                    "no implementation registered for functional op {}",
                    key.as_str()
                )
            })?;

        let name = E::name(implementation.as_ref());
        let _prof_guard = crate::profiling::functional_scope(key.as_str(), name);
        let output = E::forward(implementation.as_ref(), ctx)?;
        Ok((output, name))
    }

    fn call_forward_force<B, E>(
        &self,
        forced: &str,
        ctx: E::ForwardCtx<'_>,
    ) -> Result<(E::ForwardOutput, &'static str)>
    where
        B: PortableBackend + 'static,
        E: FunctionalRegistryEntry<B, Impl = I>,
    {
        if let Some(candidate) = self.select_by_name::<B, E>(forced) {
            if E::supports(candidate.as_ref(), &ctx) {
                let key = E::key();
                let name = E::name(candidate.as_ref());
                let _prof_guard = crate::profiling::functional_scope(key.as_str(), name);
                let output = E::forward(candidate.as_ref(), ctx)?;
                return Ok((output, name));
            }
        }

        self.call_forward_default::<B, E>(ctx)
    }

    fn call_forward_benchmark<B, E>(
        &self,
        ctx: E::ForwardCtx<'_>,
    ) -> Result<(E::ForwardOutput, &'static str)>
    where
        B: PortableBackend + 'static,
        E: FunctionalRegistryEntry<B, Impl = I>,
    {
        let key = E::key();
        let cache_key = match E::cache_key(&ctx) {
            Some(key) => key,
            None => return self.call_forward_default::<B, E>(ctx),
        };

        if let Some(cached_name) = self.lookup_benchmark_cache(cache_key) {
            if let Some(candidate) = self.select_by_name::<B, E>(cached_name) {
                if E::supports(candidate.as_ref(), &ctx) {
                    let _prof_guard = crate::profiling::functional_scope(key.as_str(), cached_name);
                    let output = E::forward(candidate.as_ref(), ctx)?;
                    return Ok((output, cached_name));
                }
            }
            self.remove_benchmark_entry(cache_key);
        }

        let mut best_name: Option<&'static str> = None;
        let mut best_output: Option<E::ForwardOutput> = None;
        let mut best_time = None;

        for candidate in self.implementations() {
            if !E::supports(candidate.as_ref(), &ctx) {
                continue;
            }

            let name = E::name(candidate.as_ref());
            let ctx_clone = ctx.clone();
            let start = Instant::now();
            let _prof_guard = crate::profiling::functional_scope(key.as_str(), name);
            let output = E::forward(candidate.as_ref(), ctx_clone)?;
            let elapsed = start.elapsed();

            if best_time.is_none_or(|best| elapsed < best) {
                best_time = Some(elapsed);
                best_output = Some(output);
                best_name = Some(name);
            }
        }

        let output = best_output.ok_or_else(|| {
            anyhow!(
                "no implementation registered for functional op {}",
                key.as_str()
            )
        })?;
        let name = best_name.expect("best implementation missing name");

        self.update_benchmark_cache(cache_key, name);

        Ok((output, name))
    }

    fn lookup_benchmark_cache(&self, key: FunctionalCacheKey) -> Option<&'static str> {
        let mut guard = self.benchmark_cache.lock().unwrap();
        guard.as_mut().and_then(|cache| cache.get(&key).copied())
    }

    fn remove_benchmark_entry(&self, key: FunctionalCacheKey) {
        if let Some(cache) = self.benchmark_cache.lock().unwrap().as_mut() {
            cache.pop(&key);
        }
    }

    fn update_benchmark_cache(&self, key: FunctionalCacheKey, name: &'static str) {
        if let Some(cache) = self.benchmark_cache.lock().unwrap().as_mut() {
            cache.put(key, name);
        }
    }
}

/// Trait implemented by functional families that can be stored in the registry.
pub trait FunctionalRegistryEntry<B: PortableBackend + 'static>: Send + Sync + 'static {
    type Impl: ?Sized + Send + Sync + 'static;
    type ForwardCtx<'a>: Clone;
    type ForwardOutput;

    /// Returns the registry key associated with this entry type.
    fn key() -> FunctionalKey;
    /// Human-readable name for the implementation, used in logs and overrides.
    fn name(implementation: &Self::Impl) -> &'static str;
    /// Reports whether the implementation can run with the provided forward context.
    fn supports<'a>(implementation: &Self::Impl, ctx: &Self::ForwardCtx<'a>) -> bool;
    /// Executes the forward pass for the functional.
    fn forward<'a>(
        implementation: &Self::Impl,
        ctx: Self::ForwardCtx<'a>,
    ) -> Result<Self::ForwardOutput>;

    /// Optional hook for generating a cache key used by benchmark mode.
    fn cache_key(_ctx: &Self::ForwardCtx<'_>) -> Option<FunctionalCacheKey> {
        None
    }
}

/// Registry responsible for storing and selecting functional implementations for a backend.
pub struct FunctionalRegistry<B: PortableBackend + 'static> {
    overrides: FunctionalOverrides,
    registries: Mutex<HashMap<FunctionalKey, Arc<dyn Any + Send + Sync>>>,
    backend: PhantomData<B>,
}

impl<B: PortableBackend + 'static> FunctionalRegistry<B> {
    /// Creates a new registry and pre-registers portable baseline kernels.
    pub fn new(overrides: FunctionalOverrides) -> Self {
        let registry = FunctionalRegistry {
            overrides,
            registries: Mutex::new(HashMap::new()),
            backend: PhantomData,
        };
        registry.register::<AttentionEntry<B>, _>(AttentionImplementation::<B>::portable());
        registry
    }

    /// Registers a new implementation for the functional entry `E`.
    /// Implementations can be plain structs or builder closures thanks to [`IntoImplementation`].
    pub fn register<E, I>(&self, implementation: I)
    where
        E: FunctionalRegistryEntry<B>,
        I: IntoImplementation<E, B>,
    {
        let registry = self.ensure_registry::<E>();
        registry.register(implementation.into_impl());
    }

    /// Registers a lazily constructed implementation under the provided `name`.
    /// This is typically used to install default kernels the first time they are referenced.
    pub fn register_default<E, F, I>(&self, name: &'static str, builder: F)
    where
        E: FunctionalRegistryEntry<B>,
        F: FnOnce() -> I,
        I: IntoImplementation<E, B>,
    {
        let registry = self.ensure_registry::<E>();
        {
            let implementations = registry.implementations.lock().unwrap();
            if implementations
                .iter()
                .any(|candidate| E::name(candidate.as_ref()) == name)
            {
                return;
            }
        }

        registry.register(builder().into_impl());
    }

    pub fn register_attention<I>(&self, implementation: I)
    where
        I: IntoImplementation<AttentionEntry<B>, B>,
    {
        self.register::<AttentionEntry<B>, _>(implementation);
    }

    pub fn call_forward<'a, 'ctx, E>(
        &'a self,
        ctx: E::ForwardCtx<'ctx>,
    ) -> Result<(E::ForwardOutput, &'static str)>
    where
        E: FunctionalRegistryEntry<B>,
        'a: 'ctx,
    {
        let registry = self.ensure_registry::<E>();
        registry.call_forward::<B, E>(ctx)
    }

    fn ensure_registry<E>(&self) -> Arc<OpRegistry<E::Impl>>
    where
        E: FunctionalRegistryEntry<B>,
    {
        let key = E::key();
        let mut guard = self.registries.lock().unwrap();
        let entry = guard.entry(key).or_insert_with(|| {
            let policy = self.overrides.policy(key);
            let registry: Arc<OpRegistry<E::Impl>> = Arc::new(OpRegistry::new(policy));
            registry as Arc<dyn Any + Send + Sync>
        });
        let arc = entry.clone();
        drop(guard);

        arc.downcast::<OpRegistry<E::Impl>>()
            .expect("functional registry entry stored with wrong type")
    }
}

/// Constructs a registry using overrides and pre-baked implementations.
pub fn build_registry<B: PortableBackend + 'static>(
    overrides: &FunctionalOverrides,
) -> FunctionalRegistryHandle<B> {
    Arc::new(FunctionalRegistry::new(overrides.clone()))
}

/// Helper trait for converting builder types into registry implementations.
pub trait IntoImplementation<E, B>
where
    B: PortableBackend + 'static,
    E: FunctionalRegistryEntry<B>,
{
    fn into_impl(self) -> Arc<E::Impl>;
}

impl<B: PortableBackend> Default for FunctionalRegistry<B> {
    fn default() -> Self {
        Self::new(FunctionalOverrides::default())
    }
}
