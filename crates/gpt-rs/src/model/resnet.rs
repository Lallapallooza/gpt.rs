//! ResNet image classifier (`transformers` `ResNetForImageClassification`).
//!
//! NCHW pixels -> NHWC -> `embedder` \[7x7/2 conv layer -> 3x3/2 max pool\] -> `encoder.stages` x
//! `layers` x \[`layer` conv layers + `shortcut` -> `activation`\] -> global average pool ->
//! `classifier`.
//!
//! A conv layer ([`ResNetConvLayer`]) is a bias-free convolution, a [`BatchNorm2d`] and an
//! optional activation. A basic layer runs two 3x3 conv layers. A bottleneck layer runs 1x1, 3x3
//! and 1x1 conv layers. The shortcut, when the checkpoint has one, is a 1x1 conv layer without
//! activation, and the input otherwise. The first layer of each stage after the first halves the
//! resolution. With `downsample_in_first_stage`, the first stage does too.

use std::sync::Arc;

use anyhow::{anyhow, bail, ensure, Result};

use crate::backend::spec::PortableBackend;
use crate::module::Layer;
use crate::nn::layers::{Activation, ActivationFunction, BatchNorm2d, Conv2d, ConvPadding, Linear};
use crate::nn::{self, Params};
use crate::ops::functional::{global_avg_pool2d, max_pool2d, transpose, Padding2d};
use crate::tensor::{DeviceTensor, DeviceTensorOps};

pub const KIND: &str = "resnet";

/// `transformers` keeps the `nn.BatchNorm2d` default.
const BATCH_NORM_EPS: f32 = 1e-5;

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ResNetLayerType {
    Basic,
    Bottleneck,
}

/// The keys of a Hugging Face `ResNetConfig` that the model reads, with its defaults. Channel
/// counts come from the checkpoint.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct ResNetConfig {
    pub depths: Vec<usize>,
    pub layer_type: ResNetLayerType,
    pub hidden_act: ActivationFunction,
    pub downsample_in_first_stage: bool,
    pub downsample_in_bottleneck: bool,
}

impl Default for ResNetConfig {
    fn default() -> Self {
        Self {
            depths: vec![3, 4, 6, 3],
            layer_type: ResNetLayerType::Bottleneck,
            hidden_act: ActivationFunction::Relu,
            downsample_in_first_stage: false,
            downsample_in_bottleneck: false,
        }
    }
}

pub(crate) fn build_from_model_config<B: PortableBackend + 'static>(
    backend: Arc<B>,
    cfg: &super::ModelConfig,
    get: &mut dyn FnMut(&str) -> Result<Option<DeviceTensor<B>>>,
) -> Result<Box<dyn crate::runtime::LoadedModel<B>>> {
    let config: ResNetConfig = serde_json::from_value(cfg.config.clone())
        .map_err(|err| anyhow!("invalid {KIND} config: {err}"))?;
    let mut params =
        Params::new(backend, get).with_linear_input_dtype(cfg.runtime.matmul_input_dtype);
    Ok(Box::new(ResNet::new(&mut params, &config)?))
}

/// Bias-free convolution with centered padding, batch norm, then an optional activation.
#[nn::module]
pub struct ResNetConvLayer {
    convolution: Conv2d,
    normalization: BatchNorm2d,
    activation: Option<Activation>,
}

#[nn::module]
impl ResNetConvLayer {
    fn new(
        p: &mut Params<B>,
        stride: usize,
        activation: Option<ActivationFunction>,
    ) -> Result<Self> {
        Ok(Self {
            convolution: Conv2d::new(p, stride, 1, ConvPadding::Centered)?,
            normalization: BatchNorm2d::new(p, BATCH_NORM_EPS)?,
            activation: activation.map(ActivationFunction::layer),
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.convolution(x)?;
        let x = self.normalization(&x)?;
        Ok(self.activation(&x)?.unwrap_or(x))
    }
}

/// Residual layer: `ResNetBasicLayer` or `ResNetBottleNeckLayer`, which differ only in their conv
/// layers.
#[nn::module]
pub struct ResNetLayer {
    shortcut: Option<ResNetConvLayer>,
    layer: Vec<ResNetConvLayer>,
    activation: Activation,
}

#[nn::module]
impl ResNetLayer {
    fn new(p: &mut Params<B>, config: &ResNetConfig, stride: usize) -> Result<Self> {
        // `transformers` builds the inner conv layers with its default `relu`, not `hidden_act`.
        let relu = Some(ActivationFunction::Relu);
        let convs = match config.layer_type {
            ResNetLayerType::Basic => vec![(stride, relu), (1, None)],
            ResNetLayerType::Bottleneck if config.downsample_in_bottleneck => {
                vec![(stride, relu), (1, relu), (1, None)]
            }
            ResNetLayerType::Bottleneck => vec![(1, relu), (stride, relu), (1, None)],
        };
        Ok(Self {
            shortcut: p.optional(|p| ResNetConvLayer::new(p, stride, None))?,
            layer: p.each(convs, |p, (stride, activation)| {
                ResNetConvLayer::new(p, stride, activation)
            })?,
            activation: config.hidden_act.layer(),
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let residual = self.shortcut(x)?.unwrap_or_else(|| x.clone());
        let mut hidden = x.clone();
        for conv in &self.layer {
            hidden = conv.call(&hidden)?;
        }
        self.activation(&hidden.add(&residual)?)
    }
}

/// Layers at one resolution. The first one may downsample.
#[nn::module]
pub struct ResNetStage {
    layers: Vec<ResNetLayer>,
}

#[nn::module]
impl ResNetStage {
    fn new(p: &mut Params<B>, config: &ResNetConfig, stride: usize, depth: usize) -> Result<Self> {
        let strides = std::iter::once(stride)
            .chain(std::iter::repeat(1))
            .take(depth);
        Ok(Self {
            layers: p.each(strides, |p, stride| ResNetLayer::new(p, config, stride))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut x = x.clone();
        for layer in &self.layers {
            x = layer.call(&x)?;
        }
        Ok(x)
    }
}

/// The stages. Each stage after the first halves the resolution, and with
/// `downsample_in_first_stage` the first one does too.
#[nn::module]
pub struct ResNetEncoder {
    stages: Vec<ResNetStage>,
}

#[nn::module]
impl ResNetEncoder {
    fn new(p: &mut Params<B>, config: &ResNetConfig) -> Result<Self> {
        ensure!(!config.depths.is_empty(), "{KIND} config has no stages");
        let first_stride = if config.downsample_in_first_stage {
            2
        } else {
            1
        };
        let strides = std::iter::once(first_stride).chain(std::iter::repeat(2));
        Ok(Self {
            stages: p.each(strides.zip(&config.depths), |p, (stride, &depth)| {
                ResNetStage::new(p, config, stride, depth)
            })?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut x = x.clone();
        for stage in &self.stages {
            x = stage.call(&x)?;
        }
        Ok(x)
    }
}

/// The stem: a 7x7 stride-2 conv layer, then a 3x3 stride-2 max pool.
#[nn::module]
pub struct ResNetEmbeddings {
    embedder: ResNetConvLayer,
}

#[nn::module]
impl ResNetEmbeddings {
    fn new(p: &mut Params<B>, config: &ResNetConfig) -> Result<Self> {
        Ok(Self {
            embedder: ResNetConvLayer::new(p, 2, Some(config.hidden_act))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.embedder(x)?;
        let padding = Padding2d {
            top: 1,
            bottom: 1,
            left: 1,
            right: 1,
        };
        max_pool2d(&x, [3, 3], [2, 2], padding)
    }
}

/// ResNet without its classifier.
#[nn::module]
pub struct ResNetModel {
    embedder: ResNetEmbeddings,
    encoder: ResNetEncoder,
}

#[nn::module]
impl ResNetModel {
    fn new(p: &mut Params<B>, config: &ResNetConfig) -> Result<Self> {
        Ok(Self {
            embedder: ResNetEmbeddings::new(p, config)?,
            encoder: ResNetEncoder::new(p, config)?,
        })
    }

    /// Pooled features `[N, channels]` of NCHW images.
    fn forward(&self, pixel_values: &Tensor) -> Result<Tensor> {
        let x = transpose(pixel_values, &[0, 2, 3, 1])?;
        let x = self.embedder(&x)?;
        let x = self.encoder(&x)?;
        global_avg_pool2d(&x)
    }
}

/// The classification head, `nn.Sequential(nn.Flatten(), nn.Linear(..))` in `transformers`. The
/// pooled features are already flat.
#[nn::module]
pub struct ResNetClassifier {
    #[module(rename = "1")]
    linear: Linear,
}

#[nn::module]
impl ResNetClassifier {
    fn new(p: &mut Params<B>) -> Result<Self> {
        Ok(Self {
            linear: Linear::new(p)?,
        })
    }

    fn forward(&self, pooled: &Tensor) -> Result<Tensor> {
        self.linear(pooled)
    }
}

/// ResNet image classifier.
#[nn::module]
pub struct ResNet {
    resnet: ResNetModel,
    classifier: Option<ResNetClassifier>,
}

#[nn::module]
impl ResNet {
    pub fn new(p: &mut Params<B>, config: &ResNetConfig) -> Result<Self> {
        Ok(Self {
            resnet: ResNetModel::new(p, config)?,
            classifier: p.optional(ResNetClassifier::new)?,
        })
    }

    /// Class logits `[N, num_labels]` of NCHW images, or the pooled features when the checkpoint
    /// has no classifier.
    fn forward(&self, pixel_values: &Tensor) -> Result<Tensor> {
        let pooled = self.resnet(pixel_values)?;
        Ok(self.classifier(&pooled)?.unwrap_or(pooled))
    }
}

impl<B: PortableBackend + 'static> crate::runtime::LoadedModel<B> for ResNet<B> {
    fn kind(&self) -> &str {
        KIND
    }

    fn forward(
        &mut self,
        input: crate::runtime::ModelInput<B>,
    ) -> Result<crate::runtime::ModelOutput> {
        match input {
            crate::runtime::ModelInput::Vision(input) => Ok(crate::runtime::ModelOutput::Tensor(
                self.call(&input)?.to_host()?,
            )),
            crate::runtime::ModelInput::Tokens(_) => {
                bail!("model '{KIND}' expects vision input, got token input")
            }
        }
    }
}
