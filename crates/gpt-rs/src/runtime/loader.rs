use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex};

use anyhow::{anyhow, ensure, Context, Result};

use crate::backend::spec::{PortableBackend, TensorInit};
use crate::checkpoint::{CheckpointReader, CheckpointTensorEntry};
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
    let weight_streaming_enabled = weight_streaming.enabled
        || weight_streaming.device_budget_bytes.is_some()
        || weight_streaming.device_weights_percent.is_some()
        || weight_streaming.host_budget_bytes.is_some()
        || weight_streaming.prefetch_layers.is_some()
        || weight_streaming.small_param_persist_threshold.is_some();
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
    let mut get = move |name: &str| -> Result<DeviceTensor<B>> {
        let spec = specs
            .get(name)
            .ok_or_else(|| anyhow!("missing tensor '{}' in checkpoint", name))?;
        let key = param_key(namespace, spec.base_id);
        Ok(DeviceTensor::lazy_param(
            Arc::clone(&backend_for_params),
            crate::tensor::Shape::new(spec.dims.clone()),
            spec.dtype,
            key.0,
            spec.base_id,
            Arc::clone(&source_for_params),
            !weight_streaming_enabled,
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
