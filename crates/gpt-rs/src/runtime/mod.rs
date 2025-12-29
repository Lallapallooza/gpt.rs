use std::collections::HashMap;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};
use std::sync::{Arc, Mutex};

use anyhow::{anyhow, Context, Result};

use crate::backend::spec::{PortableBackend, TensorInit};
use crate::checkpoint::{CheckpointReader, CheckpointTensorEntry};
use crate::inference::CausalLanguageModel;
use crate::model::registry as model_registry;
use crate::model::ModelConfig;
use crate::ops::functional::{build_registry, with_registry, FunctionalOverrides};
use crate::params::{param_key, BaseParamId, ModelNamespaceId, ParamSource};
use crate::tensor::{DeviceTensor, Tensor};

static NEXT_NAMESPACE: AtomicU64 = AtomicU64::new(1);

pub fn next_namespace() -> ModelNamespaceId {
    ModelNamespaceId(u128::from(
        NEXT_NAMESPACE.fetch_add(1, AtomicOrdering::Relaxed),
    ))
}

pub enum ModelInput<B: PortableBackend + 'static> {
    Tokens(Vec<usize>),
    Vision(DeviceTensor<B>),
}

pub enum ModelOutput {
    Tensor(Tensor),
}

pub trait LoadedModel<B: PortableBackend + 'static>: Send {
    fn kind(&self) -> &str;

    fn forward(&mut self, input: ModelInput<B>) -> Result<ModelOutput>;

    fn as_causal_lm(&self) -> Option<&dyn CausalLanguageModel<B>> {
        None
    }
}

pub struct ModelHandle<B: PortableBackend + 'static> {
    inner: Box<dyn LoadedModel<B>>,
    registry: crate::ops::functional::FunctionalRegistryHandle<B>,
}

impl<B: PortableBackend + 'static> ModelHandle<B> {
    pub fn new(
        inner: Box<dyn LoadedModel<B>>,
        registry: crate::ops::functional::FunctionalRegistryHandle<B>,
    ) -> Self {
        Self { inner, registry }
    }
}

impl<B: PortableBackend + 'static> LoadedModel<B> for ModelHandle<B> {
    fn kind(&self) -> &str {
        self.inner.kind()
    }

    fn forward(&mut self, input: ModelInput<B>) -> Result<ModelOutput> {
        with_registry(Arc::clone(&self.registry), || self.inner.forward(input))
    }

    fn as_causal_lm(&self) -> Option<&dyn CausalLanguageModel<B>> {
        self.inner.as_causal_lm().is_some().then_some(self)
    }
}

impl<B: PortableBackend + 'static> CausalLanguageModel<B> for ModelHandle<B> {
    fn context_length(&self) -> usize {
        with_registry(Arc::clone(&self.registry), || {
            self.inner
                .as_causal_lm()
                .expect("ModelHandle::context_length called on a non-causal model")
                .context_length()
        })
    }

    fn num_layers(&self) -> usize {
        with_registry(Arc::clone(&self.registry), || {
            self.inner
                .as_causal_lm()
                .expect("ModelHandle::num_layers called on a non-causal model")
                .num_layers()
        })
    }

    fn forward(&self, tokens: &[usize]) -> Result<Tensor> {
        with_registry(Arc::clone(&self.registry), || {
            self.inner
                .as_causal_lm()
                .expect("ModelHandle::forward called on a non-causal model")
                .forward(tokens)
        })
    }

    fn forward_with_decode_cache(
        &self,
        tokens: &[usize],
        position_offset: usize,
        caches: &mut [Option<crate::ops::functional::DecodeKvCache<B>>],
    ) -> Result<Tensor> {
        with_registry(Arc::clone(&self.registry), || {
            self.inner
                .as_causal_lm()
                .expect("ModelHandle::forward_with_decode_cache called on a non-causal model")
                .forward_with_decode_cache(tokens, position_offset, caches)
        })
    }

    fn forward_with_decode_cache_with_capacity(
        &self,
        tokens: &[usize],
        position_offset: usize,
        caches: &mut [Option<crate::ops::functional::DecodeKvCache<B>>],
        capacity: usize,
    ) -> Result<Tensor> {
        with_registry(Arc::clone(&self.registry), || {
            self.inner
                .as_causal_lm()
                .expect(
                    "ModelHandle::forward_with_decode_cache_with_capacity called on a non-causal model",
                )
                .forward_with_decode_cache_with_capacity(tokens, position_offset, caches, capacity)
        })
    }
}

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
            spec.requires_grad,
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
