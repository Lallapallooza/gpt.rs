use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex};

use anyhow::{anyhow, ensure, Context, Result};

use crate::backend::spec::{PortableBackend, TensorInit};
use crate::checkpoint::{CheckpointReader, CheckpointTensorEntry};
use crate::model::config::WeightStreamingConfig;
use crate::model::registry as model_registry;
use crate::model::ModelConfig;
use crate::ops::functional::{build_registry, FunctionalOverrides};
use crate::params::{param_key, BaseParamId, ModelNamespaceId, ParamSource};
use crate::tensor::DeviceTensor;

use super::handle::{LoadedModel, ModelHandle};
use super::namespace::next_namespace;

struct CheckpointParamSource<B: PortableBackend + 'static> {
    backend: Arc<B>,
    state: Mutex<CheckpointParamState<B>>,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct SourceCacheConfig {
    enabled: bool,
    budget_bytes: Option<u64>,
    small_param_persist_threshold: Option<u64>,
}

#[derive(Clone)]
struct CachedParamHandle<H: Clone> {
    handle: H,
    bytes: u64,
    pinned: bool,
    last_used: u64,
}

struct ParamHandleCache<H: Clone> {
    cfg: SourceCacheConfig,
    entries: HashMap<BaseParamId, CachedParamHandle<H>>,
    cached_bytes: u64,
    usage_tick: u64,
}

impl<H: Clone> ParamHandleCache<H> {
    fn new(cfg: SourceCacheConfig) -> Self {
        Self {
            cfg,
            entries: HashMap::new(),
            cached_bytes: 0,
            usage_tick: 0,
        }
    }

    fn get(&mut self, base_id: BaseParamId) -> Option<H> {
        if !self.cfg.enabled {
            return None;
        }
        let tick = self.next_tick();
        let entry = self.entries.get_mut(&base_id)?;
        entry.last_used = tick;
        Some(entry.handle.clone())
    }

    fn insert(&mut self, base_id: BaseParamId, bytes: u64, handle: H) {
        if !self.cfg.enabled || !self.should_cache(bytes) {
            return;
        }

        let pinned = self
            .cfg
            .small_param_persist_threshold
            .is_some_and(|threshold| bytes <= threshold);

        let touch_tick = self.next_tick();
        if let Some(existing) = self.entries.get_mut(&base_id) {
            existing.handle = handle;
            existing.last_used = touch_tick;
            return;
        }

        if let Some(budget) = self.cfg.budget_bytes {
            if bytes > budget {
                return;
            }

            let pinned_bytes = self
                .entries
                .values()
                .filter(|entry| entry.pinned)
                .fold(0u64, |acc, entry| acc.saturating_add(entry.bytes));
            if pinned_bytes.saturating_add(bytes) > budget {
                return;
            }

            while self.cached_bytes.saturating_add(bytes) > budget {
                if !self.evict_one_unpinned() {
                    return;
                }
            }
        }

        self.cached_bytes = self.cached_bytes.saturating_add(bytes);
        let tick = self.next_tick();
        self.entries.insert(
            base_id,
            CachedParamHandle {
                handle,
                bytes,
                pinned,
                last_used: tick,
            },
        );
    }

    fn should_cache(&self, bytes: u64) -> bool {
        match (
            self.cfg.small_param_persist_threshold,
            self.cfg.budget_bytes,
        ) {
            (Some(threshold), None) => bytes <= threshold,
            _ => true,
        }
    }

    fn evict_one_unpinned(&mut self) -> bool {
        let victim = self
            .entries
            .iter()
            .filter(|(_, entry)| !entry.pinned)
            .min_by(|(id_a, entry_a), (id_b, entry_b)| {
                entry_a
                    .last_used
                    .cmp(&entry_b.last_used)
                    .then_with(|| id_a.0.cmp(&id_b.0))
            })
            .map(|(id, _)| *id);
        let Some(victim) = victim else {
            return false;
        };

        if let Some(evicted) = self.entries.remove(&victim) {
            self.cached_bytes = self.cached_bytes.saturating_sub(evicted.bytes);
            return true;
        }
        false
    }

    fn next_tick(&mut self) -> u64 {
        self.usage_tick = self.usage_tick.wrapping_add(1);
        self.usage_tick
    }
}

struct CheckpointParamState<B: PortableBackend + 'static> {
    reader: CheckpointReader,
    entry_bytes: HashMap<BaseParamId, u64>,
    cache: ParamHandleCache<B::TensorHandle>,
}

impl<B: PortableBackend + 'static> CheckpointParamSource<B> {
    fn new(backend: Arc<B>, reader: CheckpointReader, cache_cfg: SourceCacheConfig) -> Self {
        let mut entry_bytes: HashMap<BaseParamId, u64> =
            HashMap::with_capacity(reader.entries().len());
        for entry in reader.entries() {
            entry_bytes.insert(entry.base_id, entry.len);
        }
        Self {
            backend,
            state: Mutex::new(CheckpointParamState {
                reader,
                entry_bytes,
                cache: ParamHandleCache::new(cache_cfg),
            }),
        }
    }
}

impl<B: PortableBackend + 'static> ParamSource<B> for CheckpointParamSource<B> {
    fn load(&self, base_id: BaseParamId) -> Result<B::TensorHandle> {
        let mut state = self.state.lock().expect("checkpoint source poisoned");
        if let Some(handle) = state.cache.get(base_id) {
            return Ok(handle);
        }

        let tensor = state.reader.get_by_base_id(base_id)?;
        let bytes = state.entry_bytes.get(&base_id).copied().unwrap_or(0);
        let literal = tensor.to_literal();
        let handle = self.backend.materialize(TensorInit::Literal(literal))?;
        state.cache.insert(base_id, bytes, handle.clone());
        Ok(handle)
    }
}

pub fn load_model<B: PortableBackend + 'static>(
    backend: Arc<B>,
    path: impl AsRef<Path>,
) -> Result<Box<dyn LoadedModel<B>>> {
    load_model_with_namespace(backend, path, next_namespace())
}

pub fn load_model_with_namespace<B: PortableBackend + 'static>(
    backend: Arc<B>,
    path: impl AsRef<Path>,
    namespace: ModelNamespaceId,
) -> Result<Box<dyn LoadedModel<B>>> {
    let reader = CheckpointReader::open(&path)
        .with_context(|| format!("failed to open checkpoint {}", path.as_ref().display()))?;
    let config = reader.config().clone();
    let weight_streaming = &config.runtime.weight_streaming;
    if let Some(percent) = weight_streaming.device_weights_percent {
        ensure!(
            (0.0..=1.0).contains(&percent),
            "runtime.weight_streaming.device_weights_percent must be in [0.0, 1.0], got {}",
            percent
        );
    }
    let streaming_enabled = weight_streaming_enabled(weight_streaming);
    let source_cache_cfg = source_cache_config(reader.entries(), weight_streaming);
    let registry = build_registry::<B>(&functional_overrides_from_config(&config)?);

    let mut specs: HashMap<String, CheckpointTensorEntry> =
        HashMap::with_capacity(reader.entries().len());
    for entry in reader.entries().iter() {
        specs.insert(entry.name.clone(), entry.clone());
    }

    let source: Arc<dyn ParamSource<B>> = Arc::new(CheckpointParamSource::new(
        Arc::clone(&backend),
        reader,
        source_cache_cfg,
    ));

    let backend_for_params = Arc::clone(&backend);
    let source_for_params = Arc::clone(&source);
    let mut get = move |name: &str| -> Result<DeviceTensor<B>> {
        let spec = specs
            .get(name)
            .ok_or_else(|| anyhow!("missing tensor '{}' in checkpoint", name))?;
        let key = param_key(namespace, spec.base_id);
        let cache_enabled = !streaming_enabled;
        Ok(DeviceTensor::lazy_param(
            Arc::clone(&backend_for_params),
            crate::tensor::Shape::new(spec.dims.clone()),
            spec.dtype,
            key.0,
            spec.base_id,
            Arc::clone(&source_for_params),
            cache_enabled,
        ))
    };

    let kind = config.kind.as_str();
    let factory = model_registry::model_factory::<B>(kind)
        .ok_or_else(|| anyhow!("unsupported model kind '{kind}'"))?;
    let model = (factory)(backend, &config, &mut get)?;
    Ok(Box::new(ModelHandle::new(model, registry)))
}

fn functional_overrides_from_config(cfg: &ModelConfig) -> Result<FunctionalOverrides> {
    if !cfg.runtime.functional_overrides.is_empty() {
        return Ok(cfg.runtime.functional_overrides.clone());
    }

    Ok(FunctionalOverrides::default())
}

fn weight_streaming_enabled(cfg: &WeightStreamingConfig) -> bool {
    cfg.enabled
        || cfg.device_budget_bytes.is_some()
        || cfg.device_weights_percent.is_some()
        || cfg.host_budget_bytes.is_some()
        || cfg.prefetch_layers.is_some()
        || cfg.small_param_persist_threshold.is_some()
}

fn effective_device_budget_bytes(
    entries: &[CheckpointTensorEntry],
    cfg: &WeightStreamingConfig,
) -> Option<u64> {
    let mut budget = cfg.device_budget_bytes;
    if let Some(percent) = cfg.device_weights_percent {
        let total_bytes = entries
            .iter()
            .fold(0u64, |acc, entry| acc.saturating_add(entry.len));
        let percent_budget = if percent >= 1.0 {
            total_bytes
        } else if percent <= 0.0 {
            0
        } else {
            ((total_bytes as f64) * f64::from(percent)).floor() as u64
        };
        budget = Some(match budget {
            Some(current) => current.min(percent_budget),
            None => percent_budget,
        });
    }
    budget
}

fn source_cache_config(
    entries: &[CheckpointTensorEntry],
    cfg: &WeightStreamingConfig,
) -> SourceCacheConfig {
    if !weight_streaming_enabled(cfg) {
        return SourceCacheConfig::default();
    }

    let budget_bytes = effective_device_budget_bytes(entries, cfg);
    let small_param_persist_threshold = cfg.small_param_persist_threshold;
    let enabled = budget_bytes.is_some() || small_param_persist_threshold.is_some();
    SourceCacheConfig {
        enabled,
        budget_bytes,
        small_param_persist_threshold,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(name: &str, id: u128, len: u64) -> CheckpointTensorEntry {
        CheckpointTensorEntry {
            name: name.to_string(),
            base_id: BaseParamId(id),
            dims: vec![1],
            dtype: crate::tensor::DType::F32,
            offset: 0,
            len,
        }
    }

    fn weight_cfg(
        enabled: bool,
        device_budget_bytes: Option<u64>,
        device_weights_percent: Option<f32>,
        small_param_persist_threshold: Option<u64>,
    ) -> WeightStreamingConfig {
        WeightStreamingConfig {
            enabled,
            device_budget_bytes,
            device_weights_percent,
            host_budget_bytes: None,
            prefetch_layers: None,
            small_param_persist_threshold,
        }
    }

    #[test]
    fn source_cache_disabled_when_streaming_disabled() {
        let entries = vec![entry("a", 1, 32), entry("b", 2, 48)];
        let cfg = source_cache_config(&entries, &WeightStreamingConfig::default());
        assert_eq!(cfg, SourceCacheConfig::default());
    }

    #[test]
    fn source_cache_disabled_when_streaming_enabled_without_limits() {
        let entries = vec![entry("a", 1, 32), entry("b", 2, 48)];
        let cfg = weight_cfg(true, None, None, None);
        let source_cfg = source_cache_config(&entries, &cfg);
        assert!(!source_cfg.enabled);
        assert_eq!(source_cfg.budget_bytes, None);
        assert_eq!(source_cfg.small_param_persist_threshold, None);
    }

    #[test]
    fn source_cache_enabled_for_threshold_only_streaming() {
        let entries = vec![entry("a", 1, 8), entry("b", 2, 16), entry("c", 3, 24)];
        let cfg = weight_cfg(true, None, None, Some(16));
        let source_cfg = source_cache_config(&entries, &cfg);
        assert!(source_cfg.enabled);
        assert_eq!(source_cfg.budget_bytes, None);
        assert_eq!(source_cfg.small_param_persist_threshold, Some(16));
    }

    #[test]
    fn source_cache_budget_uses_percent_when_explicit_is_absent() {
        let entries = vec![entry("a", 1, 30), entry("b", 2, 20), entry("c", 3, 10)];
        let cfg = weight_cfg(true, None, Some(0.5), None);
        let source_cfg = source_cache_config(&entries, &cfg);
        assert!(source_cfg.enabled);
        assert_eq!(source_cfg.budget_bytes, Some(30));
    }

    #[test]
    fn source_cache_budget_uses_tighter_of_explicit_and_percent_limits() {
        let entries = vec![entry("a", 1, 30), entry("b", 2, 20), entry("c", 3, 10)];
        let cfg = weight_cfg(true, Some(40), Some(0.8), None);
        let source_cfg = source_cache_config(&entries, &cfg);
        assert!(source_cfg.enabled);
        assert_eq!(source_cfg.budget_bytes, Some(40));

        let cfg_tighter_percent = weight_cfg(true, Some(40), Some(0.5), None);
        let source_cfg_tighter_percent = source_cache_config(&entries, &cfg_tighter_percent);
        assert_eq!(source_cfg_tighter_percent.budget_bytes, Some(30));
    }

    #[test]
    fn handle_cache_disabled_ignores_insertions() {
        let mut cache = ParamHandleCache::new(SourceCacheConfig::default());
        let a = BaseParamId(1);
        cache.insert(a, 8, 11u32);
        assert!(cache.get(a).is_none());
        assert!(cache.entries.is_empty());
    }

    #[test]
    fn threshold_only_without_budget_caches_only_small_entries() {
        let mut cache = ParamHandleCache::new(SourceCacheConfig {
            enabled: true,
            budget_bytes: None,
            small_param_persist_threshold: Some(8),
        });
        let small = BaseParamId(1);
        let large = BaseParamId(2);

        cache.insert(small, 4, 11u32);
        cache.insert(large, 16, 22u32);

        assert!(cache.entries.contains_key(&small));
        assert!(!cache.entries.contains_key(&large));
    }

    #[test]
    fn budgeted_cache_evicts_lru_non_pinned_entry() {
        let mut cache = ParamHandleCache::new(SourceCacheConfig {
            enabled: true,
            budget_bytes: Some(10),
            small_param_persist_threshold: None,
        });
        let a = BaseParamId(1);
        let b = BaseParamId(2);
        let c = BaseParamId(3);

        cache.insert(a, 4, 11u32);
        cache.insert(b, 4, 22u32);
        let _ = cache.get(a);
        cache.insert(c, 4, 33u32);

        assert!(cache.entries.contains_key(&a));
        assert!(cache.entries.contains_key(&c));
        assert!(!cache.entries.contains_key(&b));
    }

    #[test]
    fn pinned_entries_are_preserved_when_budget_is_tight() {
        let mut cache = ParamHandleCache::new(SourceCacheConfig {
            enabled: true,
            budget_bytes: Some(12),
            small_param_persist_threshold: Some(4),
        });
        let pinned = BaseParamId(1);
        let non_pinned_a = BaseParamId(2);
        let non_pinned_b = BaseParamId(3);
        let oversized = BaseParamId(4);

        cache.insert(pinned, 4, 11u32);
        cache.insert(non_pinned_a, 6, 22u32);
        cache.insert(non_pinned_b, 6, 33u32);

        assert!(cache.entries.contains_key(&pinned));
        assert!(!cache.entries.contains_key(&non_pinned_a));
        assert!(cache.entries.contains_key(&non_pinned_b));

        cache.insert(oversized, 10, 44u32);
        assert!(!cache.entries.contains_key(&oversized));
        assert!(cache.entries.contains_key(&pinned));
        assert!(cache.entries.contains_key(&non_pinned_b));
    }
}
