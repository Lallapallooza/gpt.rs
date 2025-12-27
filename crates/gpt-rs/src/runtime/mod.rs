use std::collections::HashMap;
use std::io::Read;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};
use std::sync::{Arc, Mutex};

use anyhow::{anyhow, bail, Context, Result};

use crate::backend::spec::{PortableBackend, TensorInit};
use crate::checkpoint::CheckpointReader;
use crate::io::tensor_archive::{TensorArchiveEntry, TensorArchiveReader};
use crate::model::conv::Conv2d;
use crate::model::{Gpt, MobileNetV2, ResNet34};
use crate::nn::layers::Linear;
use crate::ops::functional::{transpose, Padding2d};
use crate::params::{base_param_id, param_key, BaseParamId, ModelNamespaceId, ParamSource};
use crate::tensor::{DeviceTensor, Tensor};

static NEXT_NAMESPACE: AtomicU64 = AtomicU64::new(1);

pub fn next_namespace() -> ModelNamespaceId {
    ModelNamespaceId(u128::from(
        NEXT_NAMESPACE.fetch_add(1, AtomicOrdering::Relaxed),
    ))
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ModelKind {
    Gpt,
    ResNet34,
    MobileNetV2,
}

pub enum ModelInput<B: PortableBackend + 'static> {
    Tokens(Vec<usize>),
    Vision(DeviceTensor<B>),
}

pub enum ModelOutput {
    Tensor(Tensor),
}

pub enum LoadedModel<B: PortableBackend + 'static> {
    Gpt(Gpt<B>),
    ResNet34(ResNet34<B>),
    MobileNetV2(MobileNetV2<B>),
}

impl<B: PortableBackend + 'static> LoadedModel<B> {
    pub fn kind(&self) -> ModelKind {
        match self {
            LoadedModel::Gpt(_) => ModelKind::Gpt,
            LoadedModel::ResNet34(_) => ModelKind::ResNet34,
            LoadedModel::MobileNetV2(_) => ModelKind::MobileNetV2,
        }
    }

    pub fn forward(&mut self, input: ModelInput<B>) -> Result<ModelOutput> {
        match (self, input) {
            (LoadedModel::Gpt(model), ModelInput::Tokens(tokens)) => {
                Ok(ModelOutput::Tensor(model.forward(&tokens)?))
            }
            (LoadedModel::ResNet34(model), ModelInput::Vision(input)) => {
                Ok(ModelOutput::Tensor(model.forward(&input)?.to_host()?))
            }
            (LoadedModel::MobileNetV2(model), ModelInput::Vision(input)) => {
                Ok(ModelOutput::Tensor(model.forward(&input)?.to_host()?))
            }
            (LoadedModel::Gpt(_), other) => {
                bail!("GPT does not accept input {:?}", other_kind(&other))
            }
            (LoadedModel::ResNet34(_), other) => {
                bail!("ResNet34 does not accept input {:?}", other_kind(&other))
            }
            (LoadedModel::MobileNetV2(_), other) => {
                bail!("MobileNetV2 does not accept input {:?}", other_kind(&other))
            }
        }
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

struct TensorArchiveParamSource<B: PortableBackend + 'static> {
    backend: Arc<B>,
    reader: Mutex<TensorArchiveReader>,
    by_base_id: HashMap<BaseParamId, String>,
}

impl<B: PortableBackend + 'static> ParamSource<B> for TensorArchiveParamSource<B> {
    fn load(&self, base_id: BaseParamId) -> Result<B::TensorHandle> {
        let name = self
            .by_base_id
            .get(&base_id)
            .ok_or_else(|| anyhow!("tensor id {:?} not found in archive", base_id))?
            .clone();
        let tensor = {
            let mut reader = self.reader.lock().expect("tensor archive reader poisoned");
            reader.get(&name)?
        };
        let literal = tensor.to_literal();
        Ok(self.backend.materialize(TensorInit::Literal(literal))?)
    }
}

pub fn load_model<B: PortableBackend + 'static>(
    backend: Arc<B>,
    path: impl AsRef<Path>,
) -> Result<LoadedModel<B>> {
    let path = path.as_ref();
    let mut magic = [0u8; 8];
    std::fs::File::open(path)?.read_exact(&mut magic)?;

    match &magic {
        b"GPTRSCHK" => load_gpt_checkpoint(backend, path),
        b"GPTRSTEN" => bail!(
            "tensor archives are not self-describing yet; use load_resnet34_weights/load_mobilenet_v2_weights"
        ),
        _ => bail!("unrecognized model file magic header"),
    }
}

pub fn load_gpt_checkpoint<B: PortableBackend + 'static>(
    backend: Arc<B>,
    path: impl AsRef<Path>,
) -> Result<LoadedModel<B>> {
    load_gpt_checkpoint_with_namespace(backend, path, next_namespace()).map(LoadedModel::Gpt)
}

pub fn load_gpt_checkpoint_with_namespace<B: PortableBackend + 'static>(
    backend: Arc<B>,
    path: impl AsRef<Path>,
    namespace: ModelNamespaceId,
) -> Result<Gpt<B>> {
    let reader = CheckpointReader::open(&path)
        .with_context(|| format!("failed to open checkpoint {}", path.as_ref().display()))?;
    let config = reader.config().clone();

    let mut specs: HashMap<String, crate::checkpoint::CheckpointTensorEntry> =
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

    Gpt::build_from_params(config, backend, &mut get)
}

pub fn load_resnet34_weights<B: PortableBackend + 'static>(
    backend: Arc<B>,
    path: impl AsRef<Path>,
) -> Result<LoadedModel<B>> {
    load_resnet34_weights_with_namespace(backend, path, next_namespace()).map(LoadedModel::ResNet34)
}

pub fn load_resnet34_weights_with_namespace<B: PortableBackend + 'static>(
    backend: Arc<B>,
    path: impl AsRef<Path>,
    namespace: ModelNamespaceId,
) -> Result<ResNet34<B>> {
    let reader = TensorArchiveReader::open(&path)
        .with_context(|| format!("failed to open archive {}", path.as_ref().display()))?;

    let mut specs: HashMap<String, TensorArchiveEntry> =
        HashMap::with_capacity(reader.entries().len());
    let mut by_base_id: HashMap<BaseParamId, String> =
        HashMap::with_capacity(reader.entries().len());
    for entry in reader.entries().iter() {
        specs.insert(entry.name.clone(), entry.clone());
        let base_id = base_param_id(&entry.name)?;
        by_base_id.insert(base_id, entry.name.clone());
    }

    let source: Arc<dyn ParamSource<B>> = Arc::new(TensorArchiveParamSource {
        backend: Arc::clone(&backend),
        reader: Mutex::new(reader),
        by_base_id,
    });

    let backend_for_params = Arc::clone(&backend);
    let source_for_params = Arc::clone(&source);
    let mut get = move |name: &str| -> Result<DeviceTensor<B>> {
        let spec = specs
            .get(name)
            .ok_or_else(|| anyhow!("missing tensor '{}' in archive", name))?;
        let base_id = base_param_id(name)?;
        let key = param_key(namespace, base_id);
        Ok(DeviceTensor::lazy_param(
            Arc::clone(&backend_for_params),
            crate::tensor::Shape::new(spec.dims.clone()),
            spec.dtype,
            key.0,
            base_id,
            Arc::clone(&source_for_params),
            spec.requires_grad,
        ))
    };

    build_resnet34(backend, &mut get)
}

pub fn load_mobilenet_v2_weights<B: PortableBackend + 'static>(
    backend: Arc<B>,
    path: impl AsRef<Path>,
) -> Result<LoadedModel<B>> {
    load_mobilenet_v2_weights_with_namespace(backend, path, next_namespace())
        .map(LoadedModel::MobileNetV2)
}

pub fn load_mobilenet_v2_weights_with_namespace<B: PortableBackend + 'static>(
    backend: Arc<B>,
    path: impl AsRef<Path>,
    namespace: ModelNamespaceId,
) -> Result<MobileNetV2<B>> {
    let reader = TensorArchiveReader::open(&path)
        .with_context(|| format!("failed to open archive {}", path.as_ref().display()))?;

    let mut specs: HashMap<String, TensorArchiveEntry> =
        HashMap::with_capacity(reader.entries().len());
    let mut by_base_id: HashMap<BaseParamId, String> =
        HashMap::with_capacity(reader.entries().len());
    for entry in reader.entries().iter() {
        specs.insert(entry.name.clone(), entry.clone());
        let base_id = base_param_id(&entry.name)?;
        by_base_id.insert(base_id, entry.name.clone());
    }

    let source: Arc<dyn ParamSource<B>> = Arc::new(TensorArchiveParamSource {
        backend: Arc::clone(&backend),
        reader: Mutex::new(reader),
        by_base_id,
    });

    let backend_for_params = Arc::clone(&backend);
    let source_for_params = Arc::clone(&source);
    let mut get = move |name: &str| -> Result<DeviceTensor<B>> {
        let spec = specs
            .get(name)
            .ok_or_else(|| anyhow!("missing tensor '{}' in archive", name))?;
        let base_id = base_param_id(name)?;
        let key = param_key(namespace, base_id);
        Ok(DeviceTensor::lazy_param(
            Arc::clone(&backend_for_params),
            crate::tensor::Shape::new(spec.dims.clone()),
            spec.dtype,
            key.0,
            base_id,
            Arc::clone(&source_for_params),
            spec.requires_grad,
        ))
    };

    build_mobilenet_v2(backend, &mut get)
}

fn build_resnet34<B: PortableBackend + 'static>(
    backend: Arc<B>,
    get: &mut impl FnMut(&str) -> Result<DeviceTensor<B>>,
) -> Result<ResNet34<B>> {
    let conv1_weight = get("conv1.weight")?;
    let conv1_bias = get("conv1.bias")?;
    let conv1 = Conv2d::new(
        Arc::clone(&backend),
        conv1_weight,
        Some(conv1_bias),
        [7, 7],
        [2, 2],
        [1, 1],
        Padding2d {
            top: 3,
            bottom: 3,
            left: 3,
            right: 3,
        },
    )?;

    const STAGES: [(usize, usize); 4] = [(64, 3), (128, 4), (256, 6), (512, 3)];

    let mut in_channels = 64usize;
    let mut layers: [Vec<crate::model::resnet::BasicBlock<B>>; 4] =
        [Vec::new(), Vec::new(), Vec::new(), Vec::new()];

    for (stage_idx, (out_channels, blocks)) in STAGES.iter().copied().enumerate() {
        let stage_num = stage_idx + 1;
        for block_idx in 0..blocks {
            let stride = if stage_idx == 0 {
                1usize
            } else if block_idx == 0 {
                2usize
            } else {
                1usize
            };

            let conv1_key = format!("layer{stage_num}.{block_idx}.conv1");
            let conv2_key = format!("layer{stage_num}.{block_idx}.conv2");

            let block_conv1_weight = get(&format!("{conv1_key}.weight"))?;
            let block_conv1_bias = get(&format!("{conv1_key}.bias"))?;
            let conv1 = Conv2d::new(
                Arc::clone(&backend),
                block_conv1_weight,
                Some(block_conv1_bias),
                [3, 3],
                [stride, stride],
                [1, 1],
                Padding2d {
                    top: 1,
                    bottom: 1,
                    left: 1,
                    right: 1,
                },
            )?;

            let block_conv2_weight = get(&format!("{conv2_key}.weight"))?;
            let block_conv2_bias = get(&format!("{conv2_key}.bias"))?;
            let conv2 = Conv2d::new(
                Arc::clone(&backend),
                block_conv2_weight,
                Some(block_conv2_bias),
                [3, 3],
                [1, 1],
                [1, 1],
                Padding2d {
                    top: 1,
                    bottom: 1,
                    left: 1,
                    right: 1,
                },
            )?;

            let downsample = if block_idx == 0 && (stride != 1 || in_channels != out_channels) {
                let ds_key = format!("layer{stage_num}.{block_idx}.downsample");
                let ds_weight = get(&format!("{ds_key}.weight"))?;
                let ds_bias = get(&format!("{ds_key}.bias"))?;
                Some(Conv2d::new(
                    Arc::clone(&backend),
                    ds_weight,
                    Some(ds_bias),
                    [1, 1],
                    [stride, stride],
                    [1, 1],
                    Padding2d::zero(),
                )?)
            } else {
                None
            };

            layers[stage_idx].push(crate::model::resnet::BasicBlock::new(
                Arc::clone(&backend),
                conv1,
                conv2,
                downsample,
            ));

            in_channels = out_channels;
        }
    }

    let mut fc_weight = get("fc.weight")?;
    let fc_bias = get("fc.bias")?;
    let fc_bias_len = fc_bias.shape().dims()[0];
    let fc_weight_dims = fc_weight.shape().dims();
    if fc_weight_dims.len() == 2 && fc_weight_dims[0] == fc_bias_len {
        let stable_id = fc_weight
            .lazy_handle()
            .id()
            .ok_or_else(|| anyhow!("fc.weight missing stable id"))?;
        fc_weight = transpose(backend.as_ref(), &fc_weight, &[1, 0])?
            .freeze()?
            .as_param_with_id(stable_id)?;
    }
    let fc = Linear::new(Arc::clone(&backend), fc_weight, Some(fc_bias))?;

    let [layer1, layer2, layer3, layer4] = layers;
    Ok(ResNet34::new(
        backend, conv1, layer1, layer2, layer3, layer4, fc,
    ))
}

fn build_mobilenet_v2<B: PortableBackend + 'static>(
    backend: Arc<B>,
    get: &mut impl FnMut(&str) -> Result<DeviceTensor<B>>,
) -> Result<MobileNetV2<B>> {
    let stem_weight = get("stem.weight")?;
    let stem_bias = get("stem.bias")?;
    let stem = Conv2d::new(
        Arc::clone(&backend),
        stem_weight,
        Some(stem_bias),
        [3, 3],
        [2, 2],
        [1, 1],
        Padding2d {
            top: 1,
            bottom: 1,
            left: 1,
            right: 1,
        },
    )?;

    const SETTINGS: [(usize, usize, usize, usize); 7] = [
        (1, 16, 1, 1),
        (6, 24, 2, 2),
        (6, 32, 3, 2),
        (6, 64, 4, 2),
        (6, 96, 3, 1),
        (6, 160, 3, 2),
        (6, 320, 1, 1),
    ];

    let mut blocks = Vec::new();
    let mut input_channels = 32usize;
    let mut block_idx = 0usize;

    for (expand_ratio, out_channels, repeats, stage_stride) in SETTINGS {
        for repeat_idx in 0..repeats {
            let stride = if repeat_idx == 0 { stage_stride } else { 1 };
            let _expanded = input_channels
                .checked_mul(expand_ratio)
                .ok_or_else(|| anyhow!("mobilenet expanded channel overflow"))?;

            let expand = if expand_ratio != 1 {
                let w = get(&format!("blocks.{block_idx}.expand.weight"))?;
                let b = get(&format!("blocks.{block_idx}.expand.bias"))?;
                Some(Conv2d::new(
                    Arc::clone(&backend),
                    w,
                    Some(b),
                    [1, 1],
                    [1, 1],
                    [1, 1],
                    Padding2d::zero(),
                )?)
            } else {
                None
            };

            let dw_weight = get(&format!("blocks.{block_idx}.depthwise.weight"))?;
            let dw_bias = get(&format!("blocks.{block_idx}.depthwise.bias"))?;
            let groups = dw_weight.shape().dims()[0];
            let depthwise = Conv2d::new_grouped(
                Arc::clone(&backend),
                dw_weight,
                Some(dw_bias),
                [3, 3],
                [stride, stride],
                [1, 1],
                Padding2d {
                    top: 1,
                    bottom: 1,
                    left: 1,
                    right: 1,
                },
                groups,
            )?;

            let proj_weight = get(&format!("blocks.{block_idx}.project.weight"))?;
            let proj_bias = get(&format!("blocks.{block_idx}.project.bias"))?;
            let project = Conv2d::new(
                Arc::clone(&backend),
                proj_weight,
                Some(proj_bias),
                [1, 1],
                [1, 1],
                [1, 1],
                Padding2d::zero(),
            )?;

            let use_res_connect = stride == 1 && input_channels == out_channels;
            blocks.push(crate::model::mobilenet_v2::InvertedResidual::new(
                Arc::clone(&backend),
                expand,
                depthwise,
                project,
                use_res_connect,
            ));

            input_channels = out_channels;
            block_idx += 1;
        }
    }

    let head_weight = get("head.weight")?;
    let head_bias = get("head.bias")?;
    let head = Conv2d::new(
        Arc::clone(&backend),
        head_weight,
        Some(head_bias),
        [1, 1],
        [1, 1],
        [1, 1],
        Padding2d::zero(),
    )?;

    let mut classifier_weight = get("classifier.weight")?;
    let classifier_bias = get("classifier.bias")?;
    let bias_len = classifier_bias.shape().dims()[0];
    let weight_dims = classifier_weight.shape().dims();
    if weight_dims.len() == 2 && weight_dims[0] == bias_len {
        let stable_id = classifier_weight
            .lazy_handle()
            .id()
            .ok_or_else(|| anyhow!("classifier.weight missing stable id"))?;
        classifier_weight = transpose(backend.as_ref(), &classifier_weight, &[1, 0])?
            .freeze()?
            .as_param_with_id(stable_id)?;
    }
    let classifier = Linear::new(
        Arc::clone(&backend),
        classifier_weight,
        Some(classifier_bias),
    )?;

    Ok(MobileNetV2::new(backend, stem, blocks, head, classifier))
}
