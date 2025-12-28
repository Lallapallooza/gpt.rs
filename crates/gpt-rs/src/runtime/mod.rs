use std::collections::HashMap;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};
use std::sync::{Arc, Mutex};

use anyhow::{anyhow, bail, Context, Result};

use crate::backend::spec::{PortableBackend, TensorInit};
use crate::checkpoint::{CheckpointReader, CheckpointTensorEntry};
use crate::inference::CausalLanguageModel;
use crate::model::ModelConfig;
use crate::model::{Gpt, GptConfig, MobileNetV2, ResNet34};
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

pub type VisionTrace<B> = Vec<(String, DeviceTensor<B>)>;

pub trait LoadedModel<B: PortableBackend + 'static>: Send {
    fn kind(&self) -> &str;

    fn forward(&mut self, input: ModelInput<B>) -> Result<ModelOutput>;

    fn as_causal_lm(&self) -> Option<&dyn CausalLanguageModel<B>> {
        None
    }

    fn trace_vision(&self, _input_nchw: &DeviceTensor<B>) -> Result<Option<VisionTrace<B>>> {
        Ok(None)
    }
}

fn other_kind<B: PortableBackend + 'static>(input: &ModelInput<B>) -> &'static str {
    match input {
        ModelInput::Tokens(_) => "tokens",
        ModelInput::Vision(_) => "vision",
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
    let factory =
        model_factory::<B>(kind).ok_or_else(|| anyhow!("unsupported model kind '{kind}'"))?;
    factory(backend, &config, &mut get)
}

type BuildFn<B> = fn(
    Arc<B>,
    &ModelConfig,
    &mut dyn FnMut(&str) -> Result<DeviceTensor<B>>,
) -> Result<Box<dyn LoadedModel<B>>>;

fn build_gpt<B: PortableBackend + 'static>(
    backend: Arc<B>,
    cfg: &ModelConfig,
    get: &mut dyn FnMut(&str) -> Result<DeviceTensor<B>>,
) -> Result<Box<dyn LoadedModel<B>>> {
    let model_cfg: GptConfig =
        serde_json::from_value(cfg.config.clone()).with_context(|| "invalid gpt config")?;
    Ok(Box::new(Gpt::build_from_params(model_cfg, backend, get)?))
}

fn build_resnet34<B: PortableBackend + 'static>(
    backend: Arc<B>,
    _cfg: &ModelConfig,
    get: &mut dyn FnMut(&str) -> Result<DeviceTensor<B>>,
) -> Result<Box<dyn LoadedModel<B>>> {
    Ok(Box::new(ResNet34::build_from_params(backend, get)?))
}

fn build_mobilenet_v2<B: PortableBackend + 'static>(
    backend: Arc<B>,
    _cfg: &ModelConfig,
    get: &mut dyn FnMut(&str) -> Result<DeviceTensor<B>>,
) -> Result<Box<dyn LoadedModel<B>>> {
    Ok(Box::new(MobileNetV2::build_from_params(backend, get)?))
}

fn model_factory<B: PortableBackend + 'static>(kind: &str) -> Option<BuildFn<B>> {
    for (k, build) in model_factories::<B>() {
        if *k == kind {
            return Some(*build);
        }
    }
    None
}

fn model_factories<B: PortableBackend + 'static>() -> &'static [(&'static str, BuildFn<B>)] {
    &[
        ("gpt", build_gpt::<B>),
        ("resnet34", build_resnet34::<B>),
        ("mobilenet_v2", build_mobilenet_v2::<B>),
        ("mobilenet-v2", build_mobilenet_v2::<B>),
    ]
}

impl<B: PortableBackend + 'static> LoadedModel<B> for Gpt<B> {
    fn kind(&self) -> &str {
        "gpt"
    }

    fn forward(&mut self, input: ModelInput<B>) -> Result<ModelOutput> {
        match input {
            ModelInput::Tokens(tokens) => Ok(ModelOutput::Tensor(Gpt::forward(self, &tokens)?)),
            other => bail!("GPT does not accept input {:?}", other_kind(&other)),
        }
    }

    fn as_causal_lm(&self) -> Option<&dyn CausalLanguageModel<B>> {
        Some(self)
    }
}

impl<B: PortableBackend + 'static> LoadedModel<B> for ResNet34<B> {
    fn kind(&self) -> &str {
        "resnet34"
    }

    fn forward(&mut self, input: ModelInput<B>) -> Result<ModelOutput> {
        match input {
            ModelInput::Vision(input) => Ok(ModelOutput::Tensor(
                ResNet34::forward(self, &input)?.to_host()?,
            )),
            other => bail!("ResNet34 does not accept input {:?}", other_kind(&other)),
        }
    }

    fn trace_vision(
        &self,
        input_nchw: &DeviceTensor<B>,
    ) -> Result<Option<Vec<(String, DeviceTensor<B>)>>> {
        Ok(Some(self.forward_trace(input_nchw)?))
    }
}

impl<B: PortableBackend + 'static> LoadedModel<B> for MobileNetV2<B> {
    fn kind(&self) -> &str {
        "mobilenet_v2"
    }

    fn forward(&mut self, input: ModelInput<B>) -> Result<ModelOutput> {
        match input {
            ModelInput::Vision(input) => Ok(ModelOutput::Tensor(
                MobileNetV2::forward(self, &input)?.to_host()?,
            )),
            other => bail!("MobileNetV2 does not accept input {:?}", other_kind(&other)),
        }
    }

    fn trace_vision(
        &self,
        input_nchw: &DeviceTensor<B>,
    ) -> Result<Option<Vec<(String, DeviceTensor<B>)>>> {
        Ok(Some(self.forward_trace(input_nchw)?))
    }
}
