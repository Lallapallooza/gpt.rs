//! MobileNetV2 image classifier (`transformers` `MobileNetV2ForImageClassification`).
//!
//! NCHW pixels -> NHWC -> `conv_stem` \[3x3/2 `first_conv` -> optional `expand_1x1` -> depthwise
//! `conv_3x3` -> `reduce_1x1`\] -> 16 x `layer` \[`expand_1x1` -> depthwise `conv_3x3` ->
//! `reduce_1x1` (+ input when the shape is kept)\] -> `conv_1x1` -> global average pool ->
//! `classifier`.
//!
//! A conv layer ([`ConvNormActivation`]) is a bias-free convolution, a batch norm and the
//! `hidden_act` activation, except the `reduce_1x1` projections, which have no activation. With
//! `tf_padding` the convolutions pad like TensorFlow `SAME`, asymmetrically when a strided
//! convolution meets an even size. Once the resolution has shrunk by `output_stride`, depthwise
//! convolutions dilate instead of striding. The checkpoint shapes give the channel counts, so
//! the model does not read `depth_multiplier`.

use anyhow::Result;

use crate::backend::PortableBackend;
use crate::inference::vision::VisionConfig;
use crate::nn::layers::{ActivationFunction, ConvNormActivation, ConvPadding, Linear};
use crate::nn::{self, Params};
use crate::ops::functional::global_avg_pool2d;

pub const KIND: &str = "mobilenet_v2";

/// Stride of each inverted residual layer.
const STRIDES: [usize; 16] = [2, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1];

/// The keys of a Hugging Face `MobileNetV2Config` that the model reads, with its defaults.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct MobileNetV2Config {
    pub output_stride: usize,
    pub hidden_act: ActivationFunction,
    pub tf_padding: bool,
    pub layer_norm_eps: f32,
}

impl Default for MobileNetV2Config {
    fn default() -> Self {
        Self {
            output_stride: 32,
            hidden_act: ActivationFunction::Relu6,
            tf_padding: true,
            layer_norm_eps: 0.001,
        }
    }
}

impl MobileNetV2Config {
    fn padding(&self) -> ConvPadding {
        if self.tf_padding {
            ConvPadding::TfSame
        } else {
            ConvPadding::Centered
        }
    }

    /// Stride and dilation of each inverted residual layer. Once the resolution has shrunk by
    /// `output_stride`, each layer keeps stride 1 and dilates by the product of the strides that
    /// earlier layers skipped.
    fn layer_strides(&self) -> Vec<(usize, usize)> {
        let (mut current_stride, mut dilation) = (2, 1);
        STRIDES
            .iter()
            .map(|&stride| {
                if current_stride == self.output_stride {
                    let layer_dilation = dilation;
                    dilation *= stride;
                    (1, layer_dilation)
                } else {
                    current_stride *= stride;
                    (stride, 1)
                }
            })
            .collect()
    }
}

impl VisionConfig for MobileNetV2Config {
    const KIND: &'static str = KIND;

    type Model<B: PortableBackend + 'static> = MobileNetV2<B>;

    fn build<B: PortableBackend + 'static>(&self, p: &mut Params<B>) -> Result<MobileNetV2<B>> {
        MobileNetV2::new(p, self)
    }
}

/// A conv layer with the `hidden_act` activation.
fn conv_layer<B: PortableBackend + 'static>(
    p: &mut Params<B>,
    config: &MobileNetV2Config,
    stride: usize,
    dilation: usize,
) -> Result<ConvNormActivation<B>> {
    let eps = config.layer_norm_eps;
    let activation = Some(config.hidden_act);
    ConvNormActivation::new(p, stride, dilation, config.padding(), eps, activation)
}

/// A conv layer without activation, for the `reduce_1x1` projections.
fn linear_conv_layer<B: PortableBackend + 'static>(
    p: &mut Params<B>,
    config: &MobileNetV2Config,
) -> Result<ConvNormActivation<B>> {
    ConvNormActivation::new(p, 1, 1, config.padding(), config.layer_norm_eps, None)
}

/// Inverted residual: a 1x1 expansion, a depthwise 3x3 convolution and a linear 1x1 projection,
/// plus the input when the layer keeps the shape.
#[nn::module]
pub struct MobileNetV2InvertedResidual {
    expand_1x1: ConvNormActivation,
    conv_3x3: ConvNormActivation,
    reduce_1x1: ConvNormActivation,
}

#[nn::module]
impl MobileNetV2InvertedResidual {
    fn new(
        p: &mut Params<B>,
        config: &MobileNetV2Config,
        stride: usize,
        dilation: usize,
    ) -> Result<Self> {
        Ok(Self {
            expand_1x1: conv_layer(p, config, 1, 1)?,
            conv_3x3: conv_layer(p, config, stride, dilation)?,
            reduce_1x1: linear_conv_layer(p, config)?,
        })
    }

    fn forward(&self, x: &Tensor) -> Tensor {
        let features = self.expand_1x1(x);
        let features = self.conv_3x3(&features);
        let features = self.reduce_1x1(&features);
        // `transformers` adds the input when the stride is 1 and the channel count stays, so the
        // shape is unchanged.
        if features.shape() == x.shape() {
            x + features
        } else {
            features
        }
    }
}

/// The stem: a 3x3 stride-2 convolution, then an inverted residual without the residual add. When
/// the checkpoint has no `expand_1x1`, the first convolution is the expansion.
#[nn::module]
pub struct MobileNetV2Stem {
    first_conv: ConvNormActivation,
    expand_1x1: Option<ConvNormActivation>,
    conv_3x3: ConvNormActivation,
    reduce_1x1: ConvNormActivation,
}

#[nn::module]
impl MobileNetV2Stem {
    fn new(p: &mut Params<B>, config: &MobileNetV2Config) -> Result<Self> {
        Ok(Self {
            first_conv: conv_layer(p, config, 2, 1)?,
            expand_1x1: p.optional(|p| conv_layer(p, config, 1, 1))?,
            conv_3x3: conv_layer(p, config, 1, 1)?,
            reduce_1x1: linear_conv_layer(p, config)?,
        })
    }

    fn forward(&self, x: &Tensor) -> Tensor {
        let features = self.first_conv(x);
        let features = self.expand_1x1(&features).unwrap_or(features);
        let features = self.conv_3x3(&features);
        self.reduce_1x1(&features)
    }
}

/// MobileNetV2 without its classifier.
#[nn::module]
pub struct MobileNetV2Model {
    conv_stem: MobileNetV2Stem,
    layer: Vec<MobileNetV2InvertedResidual>,
    conv_1x1: ConvNormActivation,
}

#[nn::module]
impl MobileNetV2Model {
    fn new(p: &mut Params<B>, config: &MobileNetV2Config) -> Result<Self> {
        Ok(Self {
            conv_stem: MobileNetV2Stem::new(p, config)?,
            layer: p.each(config.layer_strides(), |p, (stride, dilation)| {
                MobileNetV2InvertedResidual::new(p, config, stride, dilation)
            })?,
            conv_1x1: conv_layer(p, config, 1, 1)?,
        })
    }

    /// Pooled features `[N, channels]` of NCHW images.
    fn forward(&self, pixel_values: &Tensor) -> Tensor {
        let x = pixel_values.transpose([0, 2, 3, 1]);
        let mut x = self.conv_stem(&x);
        for layer in &self.layer {
            x = layer.call(&x);
        }
        let x = self.conv_1x1(&x);
        global_avg_pool2d(&x)
    }
}

/// MobileNetV2 image classifier.
#[nn::module]
pub struct MobileNetV2 {
    mobilenet_v2: MobileNetV2Model,
    classifier: Option<Linear>,
}

#[nn::module]
impl MobileNetV2 {
    pub fn new(p: &mut Params<B>, config: &MobileNetV2Config) -> Result<Self> {
        Ok(Self {
            mobilenet_v2: MobileNetV2Model::new(p, config)?,
            classifier: p.optional(Linear::new)?,
        })
    }

    /// Class logits `[N, num_labels]` of NCHW images, or the pooled features when the checkpoint
    /// has no classifier.
    fn forward(&self, pixel_values: &Tensor) -> Tensor {
        let pooled = self.mobilenet_v2(pixel_values);
        self.classifier(&pooled).unwrap_or(pooled)
    }
}
