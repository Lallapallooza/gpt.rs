use std::collections::{HashMap, HashSet};
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

#[derive(Clone, Debug, Default, Eq, PartialEq)]
struct SourceCacheConfig {
    enabled: bool,
    budget_bytes: Option<u64>,
    small_param_persist_threshold: Option<u64>,
    prefetch_base_ids: HashSet<BaseParamId>,
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
        if !self.cfg.enabled || !self.should_cache(base_id, bytes) {
            return;
        }

        let pinned = self.is_pinned(base_id, bytes);

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

    fn should_cache(&self, base_id: BaseParamId, bytes: u64) -> bool {
        if self.cfg.prefetch_base_ids.contains(&base_id) {
            return true;
        }
        if self.cfg.budget_bytes.is_some() {
            return true;
        }
        self.cfg
            .small_param_persist_threshold
            .is_some_and(|threshold| bytes <= threshold)
    }

    fn is_pinned(&self, base_id: BaseParamId, bytes: u64) -> bool {
        self.cfg.prefetch_base_ids.contains(&base_id)
            || self
                .cfg
                .small_param_persist_threshold
                .is_some_and(|threshold| bytes <= threshold)
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
    fn new(
        backend: Arc<B>,
        reader: CheckpointReader,
        cache_cfg: SourceCacheConfig,
    ) -> Result<Self> {
        let mut entry_bytes: HashMap<BaseParamId, u64> =
            HashMap::with_capacity(reader.entries().len());
        for entry in reader.entries() {
            entry_bytes.insert(entry.base_id, entry.len);
        }
        let mut state = CheckpointParamState {
            reader,
            entry_bytes,
            cache: ParamHandleCache::new(cache_cfg),
        };
        prefetch_source_cache(&backend, &mut state)?;
        Ok(Self {
            backend,
            state: Mutex::new(state),
        })
    }
}

fn prefetch_source_cache<B: PortableBackend + 'static>(
    backend: &Arc<B>,
    state: &mut CheckpointParamState<B>,
) -> Result<()> {
    if !state.cache.cfg.enabled || state.cache.cfg.prefetch_base_ids.is_empty() {
        return Ok(());
    }

    let prefetched_ids: Vec<BaseParamId> = state
        .reader
        .entries()
        .iter()
        .filter_map(|entry| {
            state
                .cache
                .cfg
                .prefetch_base_ids
                .contains(&entry.base_id)
                .then_some(entry.base_id)
        })
        .collect();
    for base_id in prefetched_ids {
        if state.cache.get(base_id).is_some() {
            continue;
        }
        let tensor = state.reader.get_by_base_id(base_id)?;
        let bytes = state.entry_bytes.get(&base_id).copied().unwrap_or(0);
        let handle = backend.materialize(TensorInit::Literal(tensor.to_literal()))?;
        state.cache.insert(base_id, bytes, handle);
    }
    Ok(())
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
    validate_weight_streaming_config(weight_streaming)?;
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
    )?);

    let backend_for_params = Arc::clone(&backend);
    let source_for_params = Arc::clone(&source);
    let mut get = move |name: &str| -> Result<DeviceTensor<B>> {
        let spec = specs
            .get(name)
            .ok_or_else(|| anyhow!("missing tensor '{}' in checkpoint", name))?;
        let key = param_key(namespace, spec.base_id);
        let shape = crate::tensor::Shape::new(spec.dims.clone());
        if !streaming_enabled {
            let handle = source_for_params.load(spec.base_id)?;
            return DeviceTensor::from_handle(
                Arc::clone(&backend_for_params),
                shape,
                spec.dtype,
                handle,
            )
            .as_param_with_id(key.0);
        }
        Ok(DeviceTensor::lazy_param(
            Arc::clone(&backend_for_params),
            shape,
            spec.dtype,
            key.0,
            spec.base_id,
            Arc::clone(&source_for_params),
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
}

fn validate_weight_streaming_config(cfg: &WeightStreamingConfig) -> Result<()> {
    let has_streaming_knob = cfg.device_budget_bytes.is_some()
        || cfg.device_weights_percent.is_some()
        || cfg.cache_budget_cap_bytes.is_some()
        || cfg.prefetch_layers.is_some()
        || cfg.small_param_persist_threshold.is_some();
    ensure!(
        cfg.enabled || !has_streaming_knob,
        "runtime.weight_streaming.enabled=false but streaming knobs are set; set runtime.weight_streaming.enabled=true or clear streaming knobs"
    );
    if let Some(percent) = cfg.device_weights_percent {
        ensure!(
            (0.0..=1.0).contains(&percent),
            "runtime.weight_streaming.device_weights_percent must be in [0.0, 1.0], got {}",
            percent
        );
    }
    Ok(())
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

    let budget_bytes = effective_cache_budget_bytes(entries, cfg);
    let small_param_persist_threshold = cfg.small_param_persist_threshold;
    let prefetch_base_ids = prefetch_block_base_ids(entries, cfg.prefetch_layers);
    let enabled = budget_bytes.is_some()
        || small_param_persist_threshold.is_some()
        || !prefetch_base_ids.is_empty();
    SourceCacheConfig {
        enabled,
        budget_bytes,
        small_param_persist_threshold,
        prefetch_base_ids,
    }
}

fn prefetch_block_base_ids(
    entries: &[CheckpointTensorEntry],
    prefetch_layers: Option<usize>,
) -> HashSet<BaseParamId> {
    let Some(layer_limit) = prefetch_layers else {
        return HashSet::new();
    };
    if layer_limit == 0 {
        return HashSet::new();
    }
    entries
        .iter()
        .filter_map(|entry| {
            block_layer_index(&entry.name)
                .filter(|layer| *layer < layer_limit)
                .map(|_| entry.base_id)
        })
        .collect()
}

fn block_layer_index(name: &str) -> Option<usize> {
    let rest = name.strip_prefix("blocks.")?;
    let (layer_index, _) = rest.split_once('.')?;
    layer_index.parse::<usize>().ok()
}

fn effective_cache_budget_bytes(
    entries: &[CheckpointTensorEntry],
    cfg: &WeightStreamingConfig,
) -> Option<u64> {
    let device_budget = effective_device_budget_bytes(entries, cfg);
    match (device_budget, cfg.cache_budget_cap_bytes) {
        (Some(device), Some(host)) => Some(device.min(host)),
        (Some(device), None) => Some(device),
        (None, Some(host)) => Some(host),
        (None, None) => None,
    }
}
