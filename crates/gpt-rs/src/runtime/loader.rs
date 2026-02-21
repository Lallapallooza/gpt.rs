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
    reader: Mutex<CheckpointReader>,
}

impl<B: PortableBackend + 'static> ParamSource<B> for CheckpointParamSource<B> {
    fn load(&self, base_id: BaseParamId) -> Result<B::TensorHandle> {
        let tensor = {
            let mut reader = self.reader.lock().expect("checkpoint reader poisoned");
            reader.get_by_base_id(base_id)?
        };
        let literal = tensor.to_literal();
        Ok(self.backend.materialize(TensorInit::Literal(literal))?)
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
    let cached_param_ids = select_cached_param_ids(reader.entries(), weight_streaming);
    let registry = build_registry::<B>(&functional_overrides_from_config(&config)?);

    let mut specs: HashMap<String, CheckpointTensorEntry> =
        HashMap::with_capacity(reader.entries().len());
    for entry in reader.entries().iter() {
        specs.insert(entry.name.clone(), entry.clone());
    }

    let source: Arc<dyn ParamSource<B>> = Arc::new(CheckpointParamSource {
        backend: Arc::clone(&backend),
        reader: Mutex::new(reader),
    });

    let backend_for_params = Arc::clone(&backend);
    let source_for_params = Arc::clone(&source);
    let cached_param_ids_for_params = Arc::new(cached_param_ids);
    let mut get = move |name: &str| -> Result<DeviceTensor<B>> {
        let spec = specs
            .get(name)
            .ok_or_else(|| anyhow!("missing tensor '{}' in checkpoint", name))?;
        let key = param_key(namespace, spec.base_id);
        let cache_enabled = cached_param_ids_for_params.contains(&spec.base_id);
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

fn sorted_entries_for_selection(entries: &[CheckpointTensorEntry]) -> Vec<&CheckpointTensorEntry> {
    let mut sorted: Vec<&CheckpointTensorEntry> = entries.iter().collect();
    sorted.sort_by(|a, b| a.len.cmp(&b.len).then_with(|| a.name.cmp(&b.name)));
    sorted
}

fn select_cached_param_ids(
    entries: &[CheckpointTensorEntry],
    cfg: &WeightStreamingConfig,
) -> HashSet<BaseParamId> {
    if !weight_streaming_enabled(cfg) {
        return entries.iter().map(|entry| entry.base_id).collect();
    }

    let mut selected: HashSet<BaseParamId> = HashSet::new();
    let mut remaining_budget = effective_device_budget_bytes(entries, cfg);

    if let Some(threshold) = cfg.small_param_persist_threshold {
        let mut small_entries: Vec<&CheckpointTensorEntry> = entries
            .iter()
            .filter(|entry| entry.len <= threshold)
            .collect();
        small_entries.sort_by(|a, b| a.len.cmp(&b.len).then_with(|| a.name.cmp(&b.name)));

        if let Some(remaining) = remaining_budget.as_mut() {
            for entry in small_entries {
                if entry.len <= *remaining {
                    selected.insert(entry.base_id);
                    *remaining -= entry.len;
                }
            }
        } else {
            selected.extend(small_entries.iter().map(|entry| entry.base_id));
        }
    }

    if let Some(remaining) = remaining_budget.as_mut() {
        for entry in sorted_entries_for_selection(entries) {
            if selected.contains(&entry.base_id) {
                continue;
            }
            if entry.len <= *remaining {
                selected.insert(entry.base_id);
                *remaining -= entry.len;
            }
        }
    }

    selected
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
    fn default_config_caches_all_parameters() {
        let entries = vec![entry("a", 1, 32), entry("b", 2, 48)];
        let selected = select_cached_param_ids(&entries, &WeightStreamingConfig::default());
        assert_eq!(selected.len(), 2);
        assert!(selected.contains(&BaseParamId(1)));
        assert!(selected.contains(&BaseParamId(2)));
    }

    #[test]
    fn enabled_streaming_without_limits_streams_all_parameters() {
        let entries = vec![entry("a", 1, 32), entry("b", 2, 48)];
        let cfg = weight_cfg(true, None, None, None);
        let selected = select_cached_param_ids(&entries, &cfg);
        assert!(selected.is_empty());
    }

    #[test]
    fn threshold_only_caches_small_parameters() {
        let entries = vec![entry("a", 1, 8), entry("b", 2, 16), entry("c", 3, 24)];
        let cfg = weight_cfg(true, None, None, Some(16));
        let selected = select_cached_param_ids(&entries, &cfg);
        assert_eq!(selected.len(), 2);
        assert!(selected.contains(&BaseParamId(1)));
        assert!(selected.contains(&BaseParamId(2)));
        assert!(!selected.contains(&BaseParamId(3)));
    }

    #[test]
    fn budget_only_caches_smallest_parameters_until_full() {
        let entries = vec![entry("a", 1, 20), entry("b", 2, 8), entry("c", 3, 10)];
        let cfg = weight_cfg(true, Some(24), None, None);
        let selected = select_cached_param_ids(&entries, &cfg);
        assert_eq!(selected.len(), 2);
        assert!(selected.contains(&BaseParamId(2)));
        assert!(selected.contains(&BaseParamId(3)));
        assert!(!selected.contains(&BaseParamId(1)));
    }

    #[test]
    fn threshold_priority_still_respects_budget() {
        let entries = vec![entry("a", 1, 4), entry("b", 2, 6), entry("c", 3, 9)];
        let cfg = weight_cfg(true, Some(10), None, Some(4));
        let selected = select_cached_param_ids(&entries, &cfg);
        assert_eq!(selected.len(), 2);
        assert!(selected.contains(&BaseParamId(1)));
        assert!(selected.contains(&BaseParamId(2)));
        assert!(!selected.contains(&BaseParamId(3)));
    }

    #[test]
    fn explicit_and_percent_budgets_use_the_tighter_limit() {
        let entries = vec![
            entry("a", 1, 30),
            entry("b", 2, 25),
            entry("c", 3, 20),
            entry("d", 4, 25),
        ];
        let cfg = weight_cfg(true, Some(50), Some(0.6), None);
        let selected = select_cached_param_ids(&entries, &cfg);
        assert_eq!(selected.len(), 2);
        assert!(selected.contains(&BaseParamId(3)));
        assert!(selected.contains(&BaseParamId(2)));
        assert!(!selected.contains(&BaseParamId(1)));
        assert!(!selected.contains(&BaseParamId(4)));
    }
}
