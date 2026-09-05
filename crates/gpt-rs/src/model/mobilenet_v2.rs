use std::sync::Arc;

use anyhow::{anyhow, bail, Result};

use crate::backend::spec::PortableBackend;
use crate::module::Layer;
use crate::nn::layers::{Conv2d, Linear};
use crate::nn::{self, LayerLoader};
use crate::ops::functional::{global_avg_pool2d, relu6, transpose, Conv2dParams2d};
use crate::tensor::{DeviceTensor, DeviceTensorOps};

pub const KIND: &str = "mobilenet_v2";

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MobileNetV2Config {
    pub num_classes: usize,
}

impl Default for MobileNetV2Config {
    fn default() -> Self {
        Self { num_classes: 1000 }
    }
}

pub(crate) fn build_from_model_config<B: PortableBackend + 'static>(
    backend: Arc<B>,
    cfg: &super::ModelConfig,
    get: &mut dyn FnMut(&str) -> Result<DeviceTensor<B>>,
) -> Result<Box<dyn crate::runtime::LoadedModel<B>>> {
    let config: MobileNetV2Config = serde_json::from_value(cfg.config.clone())
        .map_err(|err| anyhow!("invalid {KIND} config: {err}"))?;
    let mut params =
        LayerLoader::new(backend, get).with_linear_input_dtype(cfg.runtime.matmul_input_dtype);
    Ok(Box::new(MobileNetV2::load(&mut params, &config)?))
}

/// Inverted residual block: torchvision `InvertedResidual` with its batch norms folded into the
/// convolutions.
#[nn::module]
pub struct InvertedResidual {
    expand: Option<Conv2d>,
    depthwise: Conv2d,
    project: Conv2d,
    #[module(config)]
    use_res_connect: bool,
}

#[nn::module]
impl InvertedResidual {
    pub fn load(
        params: &mut LayerLoader<'_, B>,
        prefix: &str,
        in_channels: usize,
        out_channels: usize,
        expand_ratio: usize,
        stride: usize,
    ) -> Result<Self> {
        let name = |conv: &str| format!("{prefix}.{conv}");
        let hidden = in_channels * expand_ratio;
        let pointwise = Conv2dParams2d::square(1, 1, 0);
        let depthwise = Conv2dParams2d {
            groups: hidden,
            ..Conv2dParams2d::square(3, stride, 1)
        };
        Ok(Self {
            expand: (expand_ratio != 1)
                .then(|| params.conv2d(&name("expand"), in_channels, hidden, pointwise))
                .transpose()?,
            depthwise: params.conv2d(&name("depthwise"), hidden, hidden, depthwise)?,
            project: params.conv2d(&name("project"), hidden, out_channels, pointwise)?,
            use_res_connect: stride == 1 && in_channels == out_channels,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let hidden = match self.expand(x)? {
            Some(expanded) => relu6(&expanded)?,
            None => x.clone(),
        };
        let hidden = relu6(&self.depthwise(&hidden)?)?;
        let out = self.project(&hidden)?;
        if self.use_res_connect {
            out.add(x)
        } else {
            Ok(out)
        }
    }
}

/// torchvision MobileNetV2 with its batch norms folded into the convolutions.
#[nn::module]
pub struct MobileNetV2 {
    stem: Conv2d,
    blocks: Vec<InvertedResidual>,
    head: Conv2d,
    classifier: Linear,
}

#[nn::module]
impl MobileNetV2 {
    pub fn load(params: &mut LayerLoader<'_, B>, config: &MobileNetV2Config) -> Result<Self> {
        /// `(expand_ratio, out_channels, repeats, stride)` of each stage.
        const SETTINGS: [(usize, usize, usize, usize); 7] = [
            (1, 16, 1, 1),
            (6, 24, 2, 2),
            (6, 32, 3, 2),
            (6, 64, 4, 2),
            (6, 96, 3, 1),
            (6, 160, 3, 2),
            (6, 320, 1, 1),
        ];

        let stem = params.conv2d("stem", 3, 32, Conv2dParams2d::square(3, 2, 1))?;
        let mut blocks = Vec::new();
        let mut in_channels = 32;
        for (expand_ratio, out_channels, repeats, stage_stride) in SETTINGS {
            for repeat in 0..repeats {
                let stride = if repeat == 0 { stage_stride } else { 1 };
                let prefix = format!("blocks.{}", blocks.len());
                blocks.push(InvertedResidual::load(
                    params,
                    &prefix,
                    in_channels,
                    out_channels,
                    expand_ratio,
                    stride,
                )?);
                in_channels = out_channels;
            }
        }
        Ok(Self {
            stem,
            blocks,
            head: params.conv2d("head", in_channels, 1280, Conv2dParams2d::square(1, 1, 0))?,
            classifier: params.linear("classifier", 1280, config.num_classes, true)?,
        })
    }

    /// Class logits `[N, num_classes]` of NCHW images.
    fn forward(&self, input_nchw: &Tensor) -> Result<Tensor> {
        let x = transpose(input_nchw, &[0, 2, 3, 1])?;
        let mut x = relu6(&self.stem(&x)?)?;
        for block in &self.blocks {
            x = block.call(&x)?;
        }
        let x = relu6(&self.head(&x)?)?;

        let x = global_avg_pool2d(&x)?;
        self.classifier(&x)
    }
}

impl<B: PortableBackend + 'static> crate::runtime::LoadedModel<B> for MobileNetV2<B> {
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
