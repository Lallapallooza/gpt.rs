//! MobileNetV3-Large image classifier (timm `mobilenetv3_large_100`).
//!
//! NCHW pixels -> NHWC -> `conv_stem` (3x3/2) -> `bn1` -> hard swish -> `blocks`, 7 stages of
//! blocks -> global average pool -> `conv_head` (a 1x1 convolution of the pooled pixel, with bias)
//! -> hard swish -> `classifier`.
//!
//! A block is one of three kinds. A depthwise separable block runs `conv_dw` -> `bn1` -> act ->
//! optional `se` -> `conv_pw` -> `bn2`. An inverted residual runs `conv_pw` -> `bn1` -> act ->
//! `conv_dw` -> `bn2` -> act -> optional `se` -> `conv_pwl` -> `bn3`. Both add their input when
//! they keep its shape. The last stage's block is `conv` -> `bn1` -> act. The squeeze-excitation
//! `se` scales the channels by `hard_sigmoid(conv_expand(relu(conv_reduce(mean))))`, the mean taken
//! over the image.
//!
//! The convolutions pad by `(kernel - 1) / 2` on each side. timm's architecture fixes each block's
//! kind, stride, and activation: ReLU in the first three stages, hard swish after. The checkpoint
//! shapes give the channels, the kernel sizes, and which blocks have an `se`.

use anyhow::Result;

use crate::backend::PortableBackend;
use crate::inference::vision::VisionConfig;
use crate::nn::layers::{Activation, ActivationFunction, BatchNorm2d, Conv2d, ConvPadding, Linear};
use crate::nn::{self, Params};
use crate::ops::functional::{global_avg_pool2d, hard_sigmoid, relu};

pub const KIND: &str = "mobilenetv3_large_100";

/// The epsilon of timm's batch norms, Torch's default.
const BATCH_NORM_EPS: f32 = 1e-5;

/// How a block computes.
#[derive(Debug, Clone, Copy)]
enum BlockKind {
    DepthwiseSeparable,
    InvertedResidual,
    ConvBnAct,
}

/// Each block of each stage of timm's `mobilenetv3_large_100`: its kind, its stride and its
/// activation.
const STAGES: [&[(BlockKind, usize, ActivationFunction)]; 7] = {
    use ActivationFunction::{HardSwish, Relu};
    use BlockKind::{ConvBnAct, DepthwiseSeparable, InvertedResidual};
    [
        &[(DepthwiseSeparable, 1, Relu)],
        &[(InvertedResidual, 2, Relu), (InvertedResidual, 1, Relu)],
        &[
            (InvertedResidual, 2, Relu),
            (InvertedResidual, 1, Relu),
            (InvertedResidual, 1, Relu),
        ],
        &[
            (InvertedResidual, 2, HardSwish),
            (InvertedResidual, 1, HardSwish),
            (InvertedResidual, 1, HardSwish),
            (InvertedResidual, 1, HardSwish),
        ],
        &[
            (InvertedResidual, 1, HardSwish),
            (InvertedResidual, 1, HardSwish),
        ],
        &[
            (InvertedResidual, 2, HardSwish),
            (InvertedResidual, 1, HardSwish),
            (InvertedResidual, 1, HardSwish),
        ],
        &[(ConvBnAct, 1, HardSwish)],
    ]
};

/// The keys of a timm MobileNetV3 config that the model reads: none, because timm's architecture
/// fixes what the checkpoint does not hold.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct MobileNetV3Config {}

impl VisionConfig for MobileNetV3Config {
    const KIND: &'static str = KIND;

    type Model<B: PortableBackend + 'static> = MobileNetV3<B>;

    fn build<B: PortableBackend + 'static>(&self, p: &mut Params<B>) -> Result<MobileNetV3<B>> {
        MobileNetV3::new(p)
    }
}

/// Squeeze-excitation: the channels scaled by a gate computed from their mean over the image. Its
/// 1x1 convolutions of the pooled pixel are projections ([`Linear::pointwise`]).
#[nn::module]
pub struct SqueezeExcite {
    conv_reduce: Linear,
    conv_expand: Linear,
}

#[nn::module]
impl SqueezeExcite {
    fn new(p: &mut Params<B>) -> Result<Self> {
        Ok(Self {
            conv_reduce: Linear::pointwise(p)?,
            conv_expand: Linear::pointwise(p)?,
        })
    }

    fn forward(&self, x: &Tensor) -> Tensor {
        let pooled = global_avg_pool2d(x);
        let reduced = relu(&self.conv_reduce(&pooled));
        let gate = hard_sigmoid(&self.conv_expand(&reduced));
        let [batch, channels] = gate.shape().dims() else {
            panic!("the gate is [N, C]");
        };
        x * gate.reshape([*batch, 1, 1, *channels])
    }
}

/// A depthwise separable block: a depthwise convolution, an optional squeeze-excitation and a
/// linear 1x1 projection.
#[nn::module]
pub struct DepthwiseSeparable {
    conv_dw: Conv2d,
    bn1: BatchNorm2d,
    se: Option<SqueezeExcite>,
    conv_pw: Conv2d,
    bn2: BatchNorm2d,
    activation: Activation,
}

#[nn::module]
impl DepthwiseSeparable {
    fn new(p: &mut Params<B>, stride: usize, activation: ActivationFunction) -> Result<Self> {
        Ok(Self {
            conv_dw: Conv2d::new(p, stride, 1, ConvPadding::Centered)?,
            bn1: BatchNorm2d::new(p, BATCH_NORM_EPS)?,
            se: p.optional(SqueezeExcite::new)?,
            conv_pw: Conv2d::new(p, 1, 1, ConvPadding::Centered)?,
            bn2: BatchNorm2d::new(p, BATCH_NORM_EPS)?,
            activation: activation.layer(),
        })
    }

    fn forward(&self, x: &Tensor) -> Tensor {
        let depthwise = self.activation(&self.bn1(&self.conv_dw(x)));
        let excited = self.se(&depthwise).unwrap_or(depthwise);
        let features = self.bn2(&self.conv_pw(&excited));
        if features.shape() == x.shape() {
            x + features
        } else {
            features
        }
    }
}

/// An inverted residual: a 1x1 expansion, a depthwise convolution, an optional
/// squeeze-excitation and a linear 1x1 projection.
#[nn::module]
pub struct InvertedResidual {
    conv_pw: Conv2d,
    bn1: BatchNorm2d,
    conv_dw: Conv2d,
    bn2: BatchNorm2d,
    se: Option<SqueezeExcite>,
    conv_pwl: Conv2d,
    bn3: BatchNorm2d,
    activation: Activation,
}

#[nn::module]
impl InvertedResidual {
    fn new(p: &mut Params<B>, stride: usize, activation: ActivationFunction) -> Result<Self> {
        Ok(Self {
            conv_pw: Conv2d::new(p, 1, 1, ConvPadding::Centered)?,
            bn1: BatchNorm2d::new(p, BATCH_NORM_EPS)?,
            conv_dw: Conv2d::new(p, stride, 1, ConvPadding::Centered)?,
            bn2: BatchNorm2d::new(p, BATCH_NORM_EPS)?,
            se: p.optional(SqueezeExcite::new)?,
            conv_pwl: Conv2d::new(p, 1, 1, ConvPadding::Centered)?,
            bn3: BatchNorm2d::new(p, BATCH_NORM_EPS)?,
            activation: activation.layer(),
        })
    }

    fn forward(&self, x: &Tensor) -> Tensor {
        let expanded = self.activation(&self.bn1(&self.conv_pw(x)));
        let depthwise = self.activation(&self.bn2(&self.conv_dw(&expanded)));
        let excited = self.se(&depthwise).unwrap_or(depthwise);
        let features = self.bn3(&self.conv_pwl(&excited));
        if features.shape() == x.shape() {
            x + features
        } else {
            features
        }
    }
}

/// A convolution, a batch norm and an activation.
#[nn::module]
pub struct ConvBnAct {
    conv: Conv2d,
    bn1: BatchNorm2d,
    activation: Activation,
}

#[nn::module]
impl ConvBnAct {
    fn new(p: &mut Params<B>, stride: usize, activation: ActivationFunction) -> Result<Self> {
        Ok(Self {
            conv: Conv2d::new(p, stride, 1, ConvPadding::Centered)?,
            bn1: BatchNorm2d::new(p, BATCH_NORM_EPS)?,
            activation: activation.layer(),
        })
    }

    fn forward(&self, x: &Tensor) -> Tensor {
        self.activation(&self.bn1(&self.conv(x)))
    }
}

/// A block of one of the three kinds.
#[nn::module]
// A model holds 16 blocks, built once, so the sizes of the variants cost nothing.
#[allow(clippy::large_enum_variant)]
pub enum MobileNetV3Block {
    DepthwiseSeparable(DepthwiseSeparable),
    InvertedResidual(InvertedResidual),
    ConvBnAct(ConvBnAct),
}

#[nn::module]
impl MobileNetV3Block {
    fn new(
        p: &mut Params<B>,
        (kind, stride, activation): (BlockKind, usize, ActivationFunction),
    ) -> Result<Self> {
        Ok(match kind {
            BlockKind::DepthwiseSeparable => {
                Self::DepthwiseSeparable(DepthwiseSeparable::new(p, stride, activation)?)
            }
            BlockKind::InvertedResidual => {
                Self::InvertedResidual(InvertedResidual::new(p, stride, activation)?)
            }
            BlockKind::ConvBnAct => Self::ConvBnAct(ConvBnAct::new(p, stride, activation)?),
        })
    }

    fn forward(&self, x: &Tensor) -> Tensor {
        match self {
            Self::DepthwiseSeparable(block) => block.call(x),
            Self::InvertedResidual(block) => block.call(x),
            Self::ConvBnAct(block) => block.call(x),
        }
    }
}

/// The blocks of one stage, under `blocks.{stage}.{block}`.
#[nn::module]
pub struct MobileNetV3Stage {
    #[module(flatten)]
    blocks: Vec<MobileNetV3Block>,
}

#[nn::module]
impl MobileNetV3Stage {
    fn new(p: &mut Params<B>, blocks: &[(BlockKind, usize, ActivationFunction)]) -> Result<Self> {
        Ok(Self {
            blocks: p.each(blocks.iter().copied(), MobileNetV3Block::new)?,
        })
    }

    fn forward(&self, x: &Tensor) -> Tensor {
        let mut hidden = x.clone();
        for block in &self.blocks {
            hidden = block.call(&hidden);
        }
        hidden
    }
}

/// MobileNetV3-Large image classifier.
#[nn::module]
pub struct MobileNetV3 {
    conv_stem: Conv2d,
    bn1: BatchNorm2d,
    blocks: Vec<MobileNetV3Stage>,
    conv_head: Linear,
    classifier: Linear,
    activation: Activation,
}

#[nn::module]
impl MobileNetV3 {
    pub fn new(p: &mut Params<B>) -> Result<Self> {
        Ok(Self {
            conv_stem: Conv2d::new(p, 2, 1, ConvPadding::Centered)?,
            bn1: BatchNorm2d::new(p, BATCH_NORM_EPS)?,
            blocks: p.each(STAGES, MobileNetV3Stage::new)?,
            conv_head: Linear::pointwise(p)?,
            classifier: Linear::new(p)?,
            activation: ActivationFunction::HardSwish.layer(),
        })
    }

    /// Class logits `[N, num_classes]` of NCHW images.
    fn forward(&self, pixel_values: &Tensor) -> Tensor {
        let nhwc = pixel_values.transpose([0, 2, 3, 1]);
        let mut hidden = self.activation(&self.bn1(&self.conv_stem(&nhwc)));
        for stage in &self.blocks {
            hidden = stage.call(&hidden);
        }
        let pooled = global_avg_pool2d(&hidden);
        let features = self.activation(&self.conv_head(&pooled));
        self.classifier(&features)
    }
}
