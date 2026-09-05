use std::sync::Arc;

use anyhow::{anyhow, bail, Result};

use crate::backend::spec::PortableBackend;
use crate::module::Layer;
use crate::nn::layers::{Conv2d, Linear};
use crate::nn::{self, LayerLoader};
use crate::ops::functional::{
    global_avg_pool2d, max_pool2d, relu, transpose, Conv2dParams2d, Padding2d,
};
use crate::tensor::{DeviceTensor, DeviceTensorOps};

pub const KIND: &str = "resnet34";

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ResNet34Config {
    pub num_classes: usize,
}

impl Default for ResNet34Config {
    fn default() -> Self {
        Self { num_classes: 1000 }
    }
}

pub(crate) fn build_from_model_config<B: PortableBackend + 'static>(
    backend: Arc<B>,
    cfg: &super::ModelConfig,
    get: &mut dyn FnMut(&str) -> Result<DeviceTensor<B>>,
) -> Result<Box<dyn crate::runtime::LoadedModel<B>>> {
    let config: ResNet34Config = serde_json::from_value(cfg.config.clone())
        .map_err(|err| anyhow!("invalid {KIND} config: {err}"))?;
    let mut params =
        LayerLoader::new(backend, get).with_linear_input_dtype(cfg.runtime.matmul_input_dtype);
    Ok(Box::new(ResNet34::load(&mut params, &config)?))
}

/// Residual block of two 3x3 convolutions: torchvision `BasicBlock` with its batch norms folded
/// into the convolutions.
#[nn::module]
pub struct BasicBlock {
    conv1: Conv2d,
    conv2: Conv2d,
    downsample: Option<Conv2d>,
}

#[nn::module]
impl BasicBlock {
    pub fn load(
        params: &mut LayerLoader<'_, B>,
        prefix: &str,
        in_channels: usize,
        out_channels: usize,
        stride: usize,
    ) -> Result<Self> {
        let name = |conv: &str| format!("{prefix}.{conv}");
        let conv1 = Conv2dParams2d::square(3, stride, 1);
        let conv2 = Conv2dParams2d::square(3, 1, 1);
        Ok(Self {
            conv1: params.conv2d(&name("conv1"), in_channels, out_channels, conv1)?,
            conv2: params.conv2d(&name("conv2"), out_channels, out_channels, conv2)?,
            downsample: (stride != 1 || in_channels != out_channels)
                .then(|| {
                    let downsample = Conv2dParams2d::square(1, stride, 0);
                    params.conv2d(&name("downsample"), in_channels, out_channels, downsample)
                })
                .transpose()?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let identity = self.downsample(x)?.unwrap_or_else(|| x.clone());
        let out = relu(&self.conv1(x)?)?;
        let out = self.conv2(&out)?.add(&identity)?;
        relu(&out)
    }
}

/// torchvision ResNet-34 with its batch norms folded into the convolutions.
#[nn::module]
pub struct ResNet34 {
    conv1: Conv2d,
    layer1: Vec<BasicBlock>,
    layer2: Vec<BasicBlock>,
    layer3: Vec<BasicBlock>,
    layer4: Vec<BasicBlock>,
    fc: Linear,
}

#[nn::module]
impl ResNet34 {
    pub fn load(params: &mut LayerLoader<'_, B>, config: &ResNet34Config) -> Result<Self> {
        let conv1 = params.conv2d("conv1", 3, 64, Conv2dParams2d::square(7, 2, 3))?;
        let mut in_channels = 64;
        let mut stage = |stage: usize, out_channels: usize, blocks: usize| {
            (0..blocks)
                .map(|block| {
                    let stride = if stage > 1 && block == 0 { 2 } else { 1 };
                    let prefix = format!("layer{stage}.{block}");
                    let block =
                        BasicBlock::load(params, &prefix, in_channels, out_channels, stride);
                    in_channels = out_channels;
                    block
                })
                .collect::<Result<Vec<_>>>()
        };
        Ok(Self {
            conv1,
            layer1: stage(1, 64, 3)?,
            layer2: stage(2, 128, 4)?,
            layer3: stage(3, 256, 6)?,
            layer4: stage(4, 512, 3)?,
            fc: params.linear("fc", 512, config.num_classes, true)?,
        })
    }

    /// Class logits `[N, num_classes]` of NCHW images.
    fn forward(&self, input_nchw: &Tensor) -> Result<Tensor> {
        let x = transpose(input_nchw, &[0, 2, 3, 1])?;
        let x = relu(&self.conv1(&x)?)?;

        // maxpool: 3x3 stride 2 padding 1 (NHWC).
        let padding = Padding2d {
            top: 1,
            bottom: 1,
            left: 1,
            right: 1,
        };
        let mut x = max_pool2d(&x, [3, 3], [2, 2], padding)?;
        let stages = [&self.layer1, &self.layer2, &self.layer3, &self.layer4];
        for block in stages.into_iter().flatten() {
            x = block.call(&x)?;
        }

        let x = global_avg_pool2d(&x)?;
        self.fc(&x)
    }
}

impl<B: PortableBackend + 'static> crate::runtime::LoadedModel<B> for ResNet34<B> {
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
