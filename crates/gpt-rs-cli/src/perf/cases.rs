//! The cases: the ops, layers and blocks of ResNet-18 and MobileNetV2 at 224x224 and of GPT-2
//! small, at the shapes those models run them, and a few kernels on their own. Weights are
//! parameters, as a loaded model's are, so the passes that fold and outline them run as they do
//! for the model.

use std::sync::Arc;

use anyhow::{ensure, Result};
use gpt_rs::backend::PortableBackend;
use gpt_rs::inference::LayerCache;
use gpt_rs::model::gpt::GptConfig;
use gpt_rs::module::Layer;
use gpt_rs::nn::{
    ActivationFunction, AttentionConfig, CausalSelfAttention, ConvNormActivation, ConvPadding,
    DecoderBlock, FeedForward, LayerNorm, Linear, MixerConfig, MlpConfig, NormConfig,
};
use gpt_rs::ops::functional::{self, AttentionPositions, DecodeKvCache, Padding2d, WeightLayout};
use gpt_rs::tensor::{DeviceTensor, HostTensor};
use gpt_rs_backend_tests::layers::{
    load_layer, random_vec, tensor_from_vec, HostBatchNorm, HostConv1D, HostLinear,
};
use rand::rngs::StdRng;
use rand::Rng;

/// One case: a name that says what it computes, the model it comes from, and the computation.
pub struct Case {
    pub name: String,
    pub model: &'static str,
    pub kind: Kind,
}

/// A convolution layer of the vision models: `ConvNormActivation` on an NHWC input.
#[derive(Debug, Clone, Copy)]
pub struct Conv {
    /// The height and the width of the input.
    pub size: usize,
    pub c_in: usize,
    pub c_out: usize,
    pub kernel: usize,
    pub stride: usize,
    /// One filter per channel, as MobileNetV2's depthwise 3x3 convolutions.
    pub depthwise: bool,
    pub activation: Option<ActivationFunction>,
    pub padding: ConvPadding,
}

/// GPT-2 on `tokens` new tokens after `past` ones in the cache.
#[derive(Debug, Clone, Copy)]
pub struct Step {
    pub tokens: usize,
    pub past: usize,
}

/// What a case computes.
#[derive(Debug, Clone)]
#[expect(
    clippy::min_ident_chars,
    reason = "`perf list` prints `Matmul { m, k, n }`, the extents of `[m, k]` by `[k, n]`"
)]
pub enum Kind {
    ConvLayer(Conv),
    /// ResNet's basic block: two 3x3 conv layers, a 1x1 shortcut when it downsamples, the sum,
    /// then ReLU.
    ResNetBlock {
        size: usize,
        c_in: usize,
        c_out: usize,
        stride: usize,
    },
    /// MobileNetV2's inverted residual: a 1x1 expansion, a depthwise 3x3, a linear 1x1
    /// projection, and the input added when the shape stays.
    InvertedResidual {
        size: usize,
        c_in: usize,
        c_out: usize,
        stride: usize,
        expansion: usize,
    },
    /// `nn::Linear` on `[rows, input]`.
    Linear {
        rows: usize,
        input: usize,
        output: usize,
        layout: WeightLayout,
        bias: bool,
    },
    LayerNorm {
        rows: usize,
        dim: usize,
    },
    /// GPT-2's attention layer: fused QKV projection, causal attention over the cache, output
    /// projection.
    Gpt2Attention(Step),
    /// GPT-2's MLP: `c_fc`, tanh GELU, `c_proj`.
    Gpt2Mlp {
        tokens: usize,
    },
    /// A GPT-2 decoder layer.
    Gpt2Block(Step),
    /// GPT-2's token and position embeddings.
    Gpt2Embedding {
        tokens: usize,
    },
    /// The final norm and the tied LM head.
    Gpt2Head {
        tokens: usize,
    },
    /// `x + y`, then the activation.
    Residual {
        dims: Vec<usize>,
        activation: Option<ActivationFunction>,
    },
    Activation {
        dims: Vec<usize>,
        activation: ActivationFunction,
    },
    MaxPool {
        size: usize,
        channels: usize,
    },
    GlobalAvgPool {
        size: usize,
        channels: usize,
    },
    /// Softmax along the last axis.
    Softmax {
        dims: Vec<usize>,
    },
    Transpose {
        dims: Vec<usize>,
        perm: Vec<usize>,
    },
    Matmul {
        m: usize,
        k: usize,
        n: usize,
    },
}

/// A case ready to run: each call records its computation once more and gives the tensors to
/// materialize.
pub type Workload<B> = Box<dyn Fn() -> Vec<DeviceTensor<B>>>;

const GPT2_DIM: usize = 768;
const GPT2_HEADS: usize = 12;
const GPT2_MLP: usize = 3072;
const GPT2_VOCAB: usize = 50257;
const GPT2_POSITIONS: usize = 1024;
const GPT2_EPS: f32 = 1e-5;
const BN_EPS: f32 = 1e-5;
const MOBILENET_EPS: f32 = 1e-3;

/// Every case, model by model.
pub fn catalog() -> Vec<Case> {
    let mut cases = resnet18();
    cases.extend(mobilenet_v2());
    cases.extend(gpt2());
    cases.extend(kernels());
    cases
}

fn case(model: &'static str, name: impl Into<String>, kind: Kind) -> Case {
    Case {
        name: format!("{model}/{}", name.into()),
        model,
        kind,
    }
}

fn conv(size: usize, c_in: usize, c_out: usize, kernel: usize, stride: usize) -> Conv {
    Conv {
        size,
        c_in,
        c_out,
        kernel,
        stride,
        depthwise: false,
        activation: Some(ActivationFunction::Relu),
        padding: ConvPadding::Centered,
    }
}

/// ResNet-18 at 224x224, batch 1: its stem, the conv layers and blocks of each stage, and its
/// head.
fn resnet18() -> Vec<Case> {
    let model = "resnet18";
    let mut cases = vec![
        case(
            model,
            "input.nchw_to_nhwc 3x224x224",
            Kind::Transpose {
                dims: vec![1, 3, 224, 224],
                perm: vec![0, 2, 3, 1],
            },
        ),
        case(
            model,
            "stem.conv 7x7/2 3>64 @224",
            Kind::ConvLayer(conv(224, 3, 64, 7, 2)),
        ),
        case(
            model,
            "stem.maxpool 3x3/2 64 @112",
            Kind::MaxPool {
                size: 112,
                channels: 64,
            },
        ),
    ];
    // Each stage: its input size and channels, and its output channels.
    let stages = [(56, 64, 64), (56, 64, 128), (28, 128, 256), (14, 256, 512)];
    for (index, (size, c_in, c_out)) in stages.into_iter().enumerate() {
        cases.extend(resnet18_stage(index + 1, size, c_in, c_out));
    }
    cases.push(case(
        model,
        "pool 512 @7",
        Kind::GlobalAvgPool {
            size: 7,
            channels: 512,
        },
    ));
    cases.push(case(
        model,
        "fc 512>1000",
        Kind::Linear {
            rows: 1,
            input: 512,
            output: 1000,
            layout: WeightLayout::OutIn,
            bias: true,
        },
    ));
    cases
}

/// Stage `stage` of ResNet-18 on a `size`x`size` input of `c_in` channels: its conv layers, its
/// residual sum and its blocks.
fn resnet18_stage(stage: usize, size: usize, c_in: usize, c_out: usize) -> Vec<Case> {
    let model = "resnet18";
    let relu = Some(ActivationFunction::Relu);
    let stride = if c_in == c_out { 1 } else { 2 };
    let out = size / stride;
    let mut cases = vec![
        case(
            model,
            format!("layer{stage}.conv1 3x3/{stride} {c_in}>{c_out} @{size}"),
            Kind::ConvLayer(conv(size, c_in, c_out, 3, stride)),
        ),
        case(
            model,
            format!("layer{stage}.conv2 3x3 {c_out}>{c_out} @{out}"),
            Kind::ConvLayer(Conv {
                activation: None,
                ..conv(out, c_out, c_out, 3, 1)
            }),
        ),
    ];
    if stride == 2 {
        cases.push(case(
            model,
            format!("layer{stage}.shortcut 1x1/2 {c_in}>{c_out} @{size}"),
            Kind::ConvLayer(Conv {
                activation: None,
                ..conv(size, c_in, c_out, 1, 2)
            }),
        ));
    }
    cases.push(case(
        model,
        format!("layer{stage}.residual add+relu {c_out} @{out}"),
        Kind::Residual {
            dims: vec![1, out, out, c_out],
            activation: relu,
        },
    ));
    cases.push(case(
        model,
        format!("layer{stage}.block {c_in}>{c_out}/{stride} @{size}"),
        Kind::ResNetBlock {
            size,
            c_in,
            c_out,
            stride,
        },
    ));
    if stride == 2 {
        cases.push(case(
            model,
            format!("layer{stage}.block {c_out}>{c_out} @{out}"),
            Kind::ResNetBlock {
                size: out,
                c_in: c_out,
                c_out,
                stride: 1,
            },
        ));
    }
    cases
}

/// A conv layer of MobileNetV2, with TensorFlow `SAME` padding.
fn mobilenet_conv(
    size: usize,
    c_in: usize,
    c_out: usize,
    kernel: usize,
    stride: usize,
    depthwise: bool,
    activation: Option<ActivationFunction>,
) -> Conv {
    Conv {
        size,
        c_in,
        c_out,
        kernel,
        stride,
        depthwise,
        activation,
        padding: ConvPadding::TfSame,
    }
}

/// MobileNetV2 1.0 at 224x224, batch 1, with TensorFlow `SAME` padding as the published
/// checkpoints have: its stem, the 1x1 and depthwise layers and blocks of each stage, and its
/// head.
fn mobilenet_v2() -> Vec<Case> {
    let model = "mobilenet_v2";
    let relu6 = Some(ActivationFunction::Relu6);
    let mut cases = vec![
        case(
            model,
            "stem.conv 3x3/2 3>32 @224",
            Kind::ConvLayer(mobilenet_conv(224, 3, 32, 3, 2, false, relu6)),
        ),
        case(
            model,
            "stem.dw 3x3 32 @112",
            Kind::ConvLayer(mobilenet_conv(112, 32, 32, 3, 1, true, relu6)),
        ),
        case(
            model,
            "stem.project 1x1 32>16 @112",
            Kind::ConvLayer(mobilenet_conv(112, 32, 16, 1, 1, false, None)),
        ),
    ];
    // Each stage after the stem: input size, input and output channels, stride.
    let stages = [
        (112, 16, 24, 2),
        (56, 24, 32, 2),
        (28, 32, 64, 2),
        (14, 64, 96, 1),
        (14, 96, 160, 2),
        (7, 160, 320, 1),
    ];
    for (index, (size, c_in, c_out, stride)) in stages.into_iter().enumerate() {
        cases.extend(mobilenet_v2_stage(index + 2, size, c_in, c_out, stride));
    }
    cases.push(case(
        model,
        "head.conv 1x1 320>1280 @7",
        Kind::ConvLayer(mobilenet_conv(7, 320, 1280, 1, 1, false, relu6)),
    ));
    cases.push(case(
        model,
        "head.pool 1280 @7",
        Kind::GlobalAvgPool {
            size: 7,
            channels: 1280,
        },
    ));
    cases.push(case(
        model,
        "classifier 1280>1000",
        Kind::Linear {
            rows: 1,
            input: 1280,
            output: 1000,
            layout: WeightLayout::OutIn,
            bias: true,
        },
    ));
    cases.push(case(
        model,
        "relu6 96 @112",
        Kind::Activation {
            dims: vec![1, 112, 112, 96],
            activation: ActivationFunction::Relu6,
        },
    ));
    cases
}

/// Stage `stage` of MobileNetV2 on a `size`x`size` input of `c_in` channels: the layers of its
/// first block and the block, then the depthwise layer of its repeated block and that block, which
/// the last stage does not have.
fn mobilenet_v2_stage(
    stage: usize,
    size: usize,
    c_in: usize,
    c_out: usize,
    stride: usize,
) -> Vec<Case> {
    let model = "mobilenet_v2";
    let relu6 = Some(ActivationFunction::Relu6);
    let out = size / stride;
    let hidden = c_in * 6;
    let mut cases = vec![
        case(
            model,
            format!("stage{stage}.expand 1x1 {c_in}>{hidden} @{size}"),
            Kind::ConvLayer(mobilenet_conv(size, c_in, hidden, 1, 1, false, relu6)),
        ),
        case(
            model,
            format!("stage{stage}.dw 3x3/{stride} {hidden} @{size}"),
            Kind::ConvLayer(mobilenet_conv(size, hidden, hidden, 3, stride, true, relu6)),
        ),
        case(
            model,
            format!("stage{stage}.project 1x1 {hidden}>{c_out} @{out}"),
            Kind::ConvLayer(mobilenet_conv(out, hidden, c_out, 1, 1, false, None)),
        ),
        case(
            model,
            format!("stage{stage}.block {c_in}>{c_out}/{stride} @{size}"),
            Kind::InvertedResidual {
                size,
                c_in,
                c_out,
                stride,
                expansion: 6,
            },
        ),
    ];
    if c_out != 320 {
        let repeat_hidden = c_out * 6;
        cases.push(case(
            model,
            format!("stage{stage}.dw 3x3 {repeat_hidden} @{out}"),
            Kind::ConvLayer(mobilenet_conv(
                out,
                repeat_hidden,
                repeat_hidden,
                3,
                1,
                true,
                relu6,
            )),
        ));
        cases.push(case(
            model,
            format!("stage{stage}.block {c_out}>{c_out} @{out}"),
            Kind::InvertedResidual {
                size: out,
                c_in: c_out,
                c_out,
                stride: 1,
                expansion: 6,
            },
        ));
    }
    cases
}

/// GPT-2 small: decode, one token after 128 and 512, and prefill of 128 and 512 tokens.
fn gpt2() -> Vec<Case> {
    let model = "gpt2";
    let mut cases = Vec::new();
    let steps = [
        (
            "decode@128",
            Step {
                tokens: 1,
                past: 128,
            },
        ),
        (
            "decode@512",
            Step {
                tokens: 1,
                past: 512,
            },
        ),
        (
            "prefill128",
            Step {
                tokens: 128,
                past: 0,
            },
        ),
        (
            "prefill512",
            Step {
                tokens: 512,
                past: 0,
            },
        ),
    ];
    for (label, step) in steps {
        cases.push(case(
            model,
            format!("{label}.attn"),
            Kind::Gpt2Attention(step),
        ));
        cases.push(case(model, format!("{label}.block"), Kind::Gpt2Block(step)));
    }
    for tokens in [1, 128, 512] {
        cases.extend(gpt2_layers(tokens));
    }
    cases.push(case(
        model,
        "decode.head ln_f+lm_head",
        Kind::Gpt2Head { tokens: 1 },
    ));
    cases.push(case(
        model,
        "prefill128.softmax 12x128x128",
        Kind::Softmax {
            dims: vec![GPT2_HEADS, 128, 128],
        },
    ));
    cases
}

/// GPT-2's norm, projections, activation, MLP, residual sum and embeddings on `tokens` tokens: a
/// decode step for one, a prefill for more.
fn gpt2_layers(tokens: usize) -> Vec<Case> {
    let model = "gpt2";
    let label = if tokens == 1 {
        "decode".to_string()
    } else {
        format!("prefill{tokens}")
    };
    let rows = tokens;
    let linear = |input, output| Kind::Linear {
        rows,
        input,
        output,
        layout: WeightLayout::InOut,
        bias: true,
    };
    vec![
        case(
            model,
            format!("{label}.ln {GPT2_DIM}"),
            Kind::LayerNorm {
                rows,
                dim: GPT2_DIM,
            },
        ),
        case(
            model,
            format!("{label}.c_attn {GPT2_DIM}>{}", 3 * GPT2_DIM),
            linear(GPT2_DIM, 3 * GPT2_DIM),
        ),
        case(
            model,
            format!("{label}.attn.c_proj {GPT2_DIM}>{GPT2_DIM}"),
            linear(GPT2_DIM, GPT2_DIM),
        ),
        case(
            model,
            format!("{label}.c_fc {GPT2_DIM}>{GPT2_MLP}"),
            linear(GPT2_DIM, GPT2_MLP),
        ),
        case(
            model,
            format!("{label}.mlp.c_proj {GPT2_MLP}>{GPT2_DIM}"),
            linear(GPT2_MLP, GPT2_DIM),
        ),
        case(
            model,
            format!("{label}.gelu_tanh {GPT2_MLP}"),
            Kind::Activation {
                dims: vec![rows, GPT2_MLP],
                activation: ActivationFunction::GeluTanh,
            },
        ),
        case(model, format!("{label}.mlp"), Kind::Gpt2Mlp { tokens }),
        case(
            model,
            format!("{label}.residual add {GPT2_DIM}"),
            Kind::Residual {
                dims: vec![rows, GPT2_DIM],
                activation: None,
            },
        ),
        case(
            model,
            format!("{label}.embed"),
            Kind::Gpt2Embedding { tokens },
        ),
    ]
}

/// Kernels on their own, at shapes that no model case uses.
fn kernels() -> Vec<Case> {
    let model = "kernel";
    let mut cases = Vec::new();
    for size in [256, 512, 1024] {
        cases.push(case(
            model,
            format!("matmul {size}x{size}x{size}"),
            Kind::Matmul {
                m: size,
                k: size,
                n: size,
            },
        ));
    }
    cases.extend([
        case(
            model,
            "gemv 1x4096x4096",
            Kind::Matmul {
                m: 1,
                k: 4096,
                n: 4096,
            },
        ),
        case(
            model,
            "softmax 512x1024",
            Kind::Softmax {
                dims: vec![512, 1024],
            },
        ),
        case(
            model,
            "layernorm 512x4096",
            Kind::LayerNorm {
                rows: 512,
                dim: 4096,
            },
        ),
        case(
            model,
            "transpose 2048x2048",
            Kind::Transpose {
                dims: vec![2048, 2048],
                perm: vec![1, 0],
            },
        ),
        case(
            model,
            "relu 1M",
            Kind::Activation {
                dims: vec![1024, 1024],
                activation: ActivationFunction::Relu,
            },
        ),
        case(
            model,
            "gelu_tanh 1M",
            Kind::Activation {
                dims: vec![1024, 1024],
                activation: ActivationFunction::GeluTanh,
            },
        ),
        case(
            model,
            "add 1M",
            Kind::Residual {
                dims: vec![1024, 1024],
                activation: None,
            },
        ),
    ]);
    cases
}

/// A tensor of the extents `dims` on `backend`, of values from [-1, 1).
fn upload<B: PortableBackend + 'static>(
    backend: &Arc<B>,
    rng: &mut StdRng,
    dims: &[usize],
) -> Result<DeviceTensor<B>> {
    let count = dims.iter().product();
    DeviceTensor::from_host(
        Arc::clone(backend),
        tensor_from_vec(dims, random_vec(rng, count)),
    )
}

/// The tensors of a `ConvNormActivation` under `prefix`.
fn conv_tensors(rng: &mut StdRng, prefix: &str, conv: &Conv) -> Vec<(String, HostTensor)> {
    let group_channels = if conv.depthwise { 1 } else { conv.c_in };
    let dims = [conv.c_out, group_channels, conv.kernel, conv.kernel];
    let weight = tensor_from_vec(&dims, random_vec(rng, dims.iter().product()));
    let norm = HostBatchNorm::random(rng, conv.c_out);
    let mut tensors = vec![(format!("{prefix}convolution.weight"), weight)];
    tensors.extend(norm.tensors(&format!("{prefix}normalization")));
    tensors
}

fn conv_layer<B: PortableBackend + 'static>(
    backend: &Arc<B>,
    rng: &mut StdRng,
    conv: &Conv,
    eps: f32,
) -> Result<ConvNormActivation<B>> {
    load_layer(backend, conv_tensors(rng, "", conv), |p| {
        ConvNormActivation::new(p, conv.stride, 1, conv.padding, eps, conv.activation)
    })
}

/// The tensors of an `nn::Linear` named `name`, its weight in `layout`: a Hugging Face `Linear`,
/// or for `in, out` a GPT-2 `Conv1D`, which always has a bias.
fn linear_tensors(
    rng: &mut StdRng,
    name: &str,
    input: usize,
    output: usize,
    layout: WeightLayout,
    bias: bool,
) -> Result<Vec<(String, HostTensor)>> {
    match layout {
        WeightLayout::OutIn => Ok(HostLinear::random(rng, input, output, bias).tensors(name)),
        WeightLayout::InOut => {
            ensure!(bias, "a Conv1D has a bias");
            Ok(HostConv1D::random(rng, input, output).tensors(name))
        }
    }
}

/// The weight and the bias of a `LayerNorm` named `name`.
fn norm_tensors(rng: &mut StdRng, name: &str, dim: usize) -> Vec<(String, HostTensor)> {
    vec![
        (
            format!("{name}.weight"),
            tensor_from_vec(&[dim], random_vec(rng, dim)),
        ),
        (
            format!("{name}.bias"),
            tensor_from_vec(&[dim], random_vec(rng, dim)),
        ),
    ]
}

fn gpt2_attention_tensors(rng: &mut StdRng, prefix: &str) -> Result<Vec<(String, HostTensor)>> {
    Ok([
        linear_tensors(
            rng,
            &format!("{prefix}c_attn"),
            GPT2_DIM,
            3 * GPT2_DIM,
            WeightLayout::InOut,
            true,
        )?,
        linear_tensors(
            rng,
            &format!("{prefix}c_proj"),
            GPT2_DIM,
            GPT2_DIM,
            WeightLayout::InOut,
            true,
        )?,
    ]
    .concat())
}

fn gpt2_mlp_tensors(rng: &mut StdRng, prefix: &str) -> Result<Vec<(String, HostTensor)>> {
    Ok([
        linear_tensors(
            rng,
            &format!("{prefix}c_fc"),
            GPT2_DIM,
            GPT2_MLP,
            WeightLayout::InOut,
            true,
        )?,
        linear_tensors(
            rng,
            &format!("{prefix}c_proj"),
            GPT2_MLP,
            GPT2_DIM,
            WeightLayout::InOut,
            true,
        )?,
    ]
    .concat())
}

fn gpt2_attention_config() -> Result<AttentionConfig> {
    Ok(AttentionConfig::with_equal_heads(GPT2_DIM, GPT2_HEADS)?
        .with_fused_qkv()
        .with_weight_layout(WeightLayout::InOut))
}

fn gpt2_mlp_config() -> MlpConfig {
    MlpConfig {
        gated: false,
        act: ActivationFunction::GeluTanh,
        weight_layout: WeightLayout::InOut,
    }
}

/// A key and value cache of GPT-2 with `past` positions filled and room for `tokens` more, and
/// the positions of those tokens.
fn gpt2_cache<B: PortableBackend + 'static>(
    backend: &Arc<B>,
    rng: &mut StdRng,
    step: Step,
) -> Result<(DecodeKvCache<B>, AttentionPositions<B>)> {
    let head_dim = GPT2_DIM / GPT2_HEADS;
    let capacity = step.past + step.tokens;
    let dims = [GPT2_HEADS, capacity, head_dim];
    let cache = DecodeKvCache::new(
        upload(backend, rng, &dims)?,
        upload(backend, rng, &dims)?,
        step.past,
    );
    let position_ids = (step.past..capacity)
        .map(|position| position as i32)
        .collect();
    let host_positions = HostTensor::si32([step.tokens], position_ids)?;
    let positions = DeviceTensor::from_host(Arc::clone(backend), host_positions)?;
    Ok((cache, AttentionPositions::new(positions, None)))
}

impl Kind {
    /// The case's layers and inputs on `backend`, from `rng`, and the computation on them.
    pub fn prepare<B: PortableBackend + 'static>(
        &self,
        backend: &Arc<B>,
        rng: &mut StdRng,
    ) -> Result<Workload<B>> {
        match self.clone() {
            Kind::ConvLayer(conv) => conv_workload(backend, rng, &conv),
            Kind::ResNetBlock {
                size,
                c_in,
                c_out,
                stride,
            } => resnet_block_workload(backend, rng, size, c_in, c_out, stride),
            Kind::InvertedResidual {
                size,
                c_in,
                c_out,
                stride,
                expansion,
            } => inverted_residual_workload(backend, rng, size, c_in, c_out, stride, expansion),
            Kind::Linear {
                rows,
                input,
                output,
                layout,
                bias,
            } => linear_workload(backend, rng, rows, input, output, layout, bias),
            Kind::LayerNorm { rows, dim } => layer_norm_workload(backend, rng, rows, dim),
            Kind::Gpt2Attention(step) => gpt2_attention_workload(backend, rng, step),
            Kind::Gpt2Mlp { tokens } => gpt2_mlp_workload(backend, rng, tokens),
            Kind::Gpt2Block(step) => gpt2_block_workload(backend, rng, step),
            Kind::Gpt2Embedding { tokens } => gpt2_embedding_workload(backend, rng, tokens),
            Kind::Gpt2Head { tokens } => gpt2_head_workload(backend, rng, tokens),
            Kind::Residual { dims, activation } => {
                residual_workload(backend, rng, &dims, activation)
            }
            Kind::Activation { dims, activation } => {
                let x = upload(backend, rng, &dims)?;
                let activation_layer = activation.layer::<B>();
                Ok(Box::new(move || vec![activation_layer.call(&x)]))
            }
            Kind::MaxPool { size, channels } => {
                let x = upload(backend, rng, &[1, size, size, channels])?;
                let padding = Padding2d {
                    top: 1,
                    bottom: 1,
                    left: 1,
                    right: 1,
                };
                Ok(Box::new(move || {
                    vec![functional::max_pool2d(&x, [3, 3], [2, 2], padding)]
                }))
            }
            Kind::GlobalAvgPool { size, channels } => {
                let x = upload(backend, rng, &[1, size, size, channels])?;
                Ok(Box::new(move || vec![functional::global_avg_pool2d(&x)]))
            }
            Kind::Softmax { dims } => {
                let x = upload(backend, rng, &dims)?;
                Ok(Box::new(move || vec![functional::softmax(&x, -1)]))
            }
            Kind::Transpose { dims, perm } => {
                let x = upload(backend, rng, &dims)?;
                Ok(Box::new(move || vec![x.transpose(&perm)]))
            }
            Kind::Matmul {
                m: rows,
                k: inner,
                n: columns,
            } => {
                let lhs = upload(backend, rng, &[rows, inner])?;
                let rhs = upload(backend, rng, &[inner, columns])?;
                Ok(Box::new(move || vec![functional::matmul(&lhs, &rhs)]))
            }
        }
    }
}

/// A `ConvNormActivation` on an NHWC input, with the eps of the model its padding comes from.
fn conv_workload<B: PortableBackend + 'static>(
    backend: &Arc<B>,
    rng: &mut StdRng,
    conv: &Conv,
) -> Result<Workload<B>> {
    let eps = match conv.padding {
        ConvPadding::TfSame => MOBILENET_EPS,
        ConvPadding::Centered => BN_EPS,
    };
    let layer = conv_layer(backend, rng, conv, eps)?;
    let x = upload(backend, rng, &[1, conv.size, conv.size, conv.c_in])?;
    Ok(Box::new(move || vec![layer.call(&x)]))
}

/// ResNet's basic block on a `size`x`size` input of `c_in` channels.
fn resnet_block_workload<B: PortableBackend + 'static>(
    backend: &Arc<B>,
    rng: &mut StdRng,
    size: usize,
    c_in: usize,
    c_out: usize,
    stride: usize,
) -> Result<Workload<B>> {
    let first = conv_layer(backend, rng, &conv(size, c_in, c_out, 3, stride), BN_EPS)?;
    let second_conv = Conv {
        activation: None,
        ..conv(size / stride, c_out, c_out, 3, 1)
    };
    let second = conv_layer(backend, rng, &second_conv, BN_EPS)?;
    let shortcut_conv = Conv {
        activation: None,
        ..conv(size, c_in, c_out, 1, stride)
    };
    let shortcut = match c_in != c_out || stride != 1 {
        true => Some(conv_layer(backend, rng, &shortcut_conv, BN_EPS)?),
        false => None,
    };
    let x = upload(backend, rng, &[1, size, size, c_in])?;
    Ok(Box::new(move || {
        let residual = shortcut
            .as_ref()
            .map_or_else(|| x.clone(), |projection| projection.call(&x));
        let hidden = second.call(&first.call(&x));
        vec![functional::relu(&(hidden + residual))]
    }))
}

/// MobileNetV2's inverted residual on a `size`x`size` input of `c_in` channels.
fn inverted_residual_workload<B: PortableBackend + 'static>(
    backend: &Arc<B>,
    rng: &mut StdRng,
    size: usize,
    c_in: usize,
    c_out: usize,
    stride: usize,
    expansion: usize,
) -> Result<Workload<B>> {
    let hidden = c_in * expansion;
    let relu6 = Some(ActivationFunction::Relu6);
    let expand = conv_layer(
        backend,
        rng,
        &mobilenet_conv(size, c_in, hidden, 1, 1, false, relu6),
        MOBILENET_EPS,
    )?;
    let depthwise = conv_layer(
        backend,
        rng,
        &mobilenet_conv(size, hidden, hidden, 3, stride, true, relu6),
        MOBILENET_EPS,
    )?;
    let project = conv_layer(
        backend,
        rng,
        &mobilenet_conv(size / stride, hidden, c_out, 1, 1, false, None),
        MOBILENET_EPS,
    )?;
    let x = upload(backend, rng, &[1, size, size, c_in])?;
    Ok(Box::new(move || {
        let features = project.call(&depthwise.call(&expand.call(&x)));
        match features.shape() == x.shape() {
            true => vec![&x + features],
            false => vec![features],
        }
    }))
}

/// `nn::Linear` named `proj` on `[rows, input]`.
fn linear_workload<B: PortableBackend + 'static>(
    backend: &Arc<B>,
    rng: &mut StdRng,
    rows: usize,
    input: usize,
    output: usize,
    layout: WeightLayout,
    bias: bool,
) -> Result<Workload<B>> {
    let tensors = linear_tensors(rng, "proj", input, output, layout, bias)?;
    let layer = load_layer(backend, tensors, |params| {
        params.scoped("proj", |p| Linear::with_layout(p, layout))
    })?;
    let x = upload(backend, rng, &[rows, input])?;
    Ok(Box::new(move || vec![layer.call(&x)]))
}

/// `LayerNorm` named `norm` on `[rows, dim]`.
fn layer_norm_workload<B: PortableBackend + 'static>(
    backend: &Arc<B>,
    rng: &mut StdRng,
    rows: usize,
    dim: usize,
) -> Result<Workload<B>> {
    let tensors = norm_tensors(rng, "norm", dim);
    let layer = load_layer(backend, tensors, |params| {
        params.scoped("norm", |p| LayerNorm::new(p, GPT2_EPS))
    })?;
    let x = upload(backend, rng, &[rows, dim])?;
    Ok(Box::new(move || vec![layer.call(&x)]))
}

/// GPT-2's attention layer on the tokens of `step`, with its updated key and value cache.
fn gpt2_attention_workload<B: PortableBackend + 'static>(
    backend: &Arc<B>,
    rng: &mut StdRng,
    step: Step,
) -> Result<Workload<B>> {
    let config = gpt2_attention_config()?;
    let tensors = gpt2_attention_tensors(rng, "")?;
    let layer = load_layer(backend, tensors, |p| {
        CausalSelfAttention::<B, GptConfig>::new(p, config)
    })?;
    let (cache, positions) = gpt2_cache(backend, rng, step)?;
    let x = upload(backend, rng, &[step.tokens, GPT2_DIM])?;
    Ok(Box::new(move || {
        let (output, updated) = layer.call((&x, &cache, &positions));
        vec![output, updated.keys().clone(), updated.values().clone()]
    }))
}

/// GPT-2's MLP on `tokens` tokens.
fn gpt2_mlp_workload<B: PortableBackend + 'static>(
    backend: &Arc<B>,
    rng: &mut StdRng,
    tokens: usize,
) -> Result<Workload<B>> {
    let tensors = gpt2_mlp_tensors(rng, "")?;
    let layer = load_layer(backend, tensors, |p| {
        FeedForward::<B, GptConfig>::new(p, gpt2_mlp_config())
    })?;
    let x = upload(backend, rng, &[tokens, GPT2_DIM])?;
    Ok(Box::new(move || vec![layer.call(&x)]))
}

/// A GPT-2 decoder layer on the tokens of `step`, with its updated key and value cache.
fn gpt2_block_workload<B: PortableBackend + 'static>(
    backend: &Arc<B>,
    rng: &mut StdRng,
    step: Step,
) -> Result<Workload<B>> {
    let mixer = MixerConfig::Attention(gpt2_attention_config()?);
    let tensors = [
        norm_tensors(rng, "ln_1", GPT2_DIM),
        gpt2_attention_tensors(rng, "attn.")?,
        norm_tensors(rng, "ln_2", GPT2_DIM),
        gpt2_mlp_tensors(rng, "mlp.")?,
    ]
    .concat();
    let block = load_layer(backend, tensors, |p| {
        DecoderBlock::<B, GptConfig>::new(
            p,
            NormConfig::Layer { eps: GPT2_EPS },
            &mixer,
            gpt2_mlp_config(),
        )
    })?;
    let (cache, positions) = gpt2_cache(backend, rng, step)?;
    let layer_cache = LayerCache::Attention(cache);
    let x = upload(backend, rng, &[step.tokens, GPT2_DIM])?;
    Ok(Box::new(move || {
        let (output, updated) = block.call((&x, &layer_cache, &positions));
        let LayerCache::Attention(kv) = updated else {
            unreachable!("an attention layer keeps a key and value cache");
        };
        vec![output, kv.keys().clone(), kv.values().clone()]
    }))
}

/// GPT-2's token and position embeddings of `tokens` random token ids.
fn gpt2_embedding_workload<B: PortableBackend + 'static>(
    backend: &Arc<B>,
    rng: &mut StdRng,
    tokens: usize,
) -> Result<Workload<B>> {
    let tensors = vec![
        (
            "wte".to_string(),
            tensor_from_vec(
                &[GPT2_VOCAB, GPT2_DIM],
                random_vec(rng, GPT2_VOCAB * GPT2_DIM),
            ),
        ),
        (
            "wpe".to_string(),
            tensor_from_vec(
                &[GPT2_POSITIONS, GPT2_DIM],
                random_vec(rng, GPT2_POSITIONS * GPT2_DIM),
            ),
        ),
    ];
    let (tokens_table, positions_table) = load_layer(backend, tensors, |params| {
        Ok((
            params.scoped("wte", |p| p.param())?,
            params.scoped("wpe", |p| p.param())?,
        ))
    })?;
    let token_ids: Vec<i32> = (0..tokens)
        .map(|_| rng.gen_range(0..GPT2_VOCAB as i32))
        .collect();
    let ids = DeviceTensor::from_host(Arc::clone(backend), HostTensor::si32([tokens], token_ids)?)?;
    let position_ids: Vec<i32> = (0..tokens as i32).collect();
    let positions = DeviceTensor::from_host(
        Arc::clone(backend),
        HostTensor::si32([tokens], position_ids)?,
    )?;
    Ok(Box::new(move || {
        let token_embeddings = functional::embedding_lookup(&tokens_table, &ids);
        let position_embeddings = functional::embedding_lookup(&positions_table, &positions);
        vec![token_embeddings + position_embeddings]
    }))
}

/// GPT-2's final norm and LM head on `tokens` tokens.
fn gpt2_head_workload<B: PortableBackend + 'static>(
    backend: &Arc<B>,
    rng: &mut StdRng,
    tokens: usize,
) -> Result<Workload<B>> {
    let norm = load_layer(backend, norm_tensors(rng, "ln_f", GPT2_DIM), |params| {
        params.scoped("ln_f", |p| LayerNorm::new(p, GPT2_EPS))
    })?;
    let head = load_layer(
        backend,
        linear_tensors(
            rng,
            "lm_head",
            GPT2_DIM,
            GPT2_VOCAB,
            WeightLayout::OutIn,
            false,
        )?,
        |p| p.scoped("lm_head", Linear::new),
    )?;
    let x = upload(backend, rng, &[tokens, GPT2_DIM])?;
    Ok(Box::new(move || vec![head.call(&norm.call(&x))]))
}

/// `x + y` of the extents `dims`, then `activation`.
fn residual_workload<B: PortableBackend + 'static>(
    backend: &Arc<B>,
    rng: &mut StdRng,
    dims: &[usize],
    activation: Option<ActivationFunction>,
) -> Result<Workload<B>> {
    let x = upload(backend, rng, dims)?;
    let residual = upload(backend, rng, dims)?;
    let activation_layer = activation.map(ActivationFunction::layer::<B>);
    Ok(Box::new(move || {
        let sum = &x + &residual;
        vec![match &activation_layer {
            Some(layer) => layer.call(&sum),
            None => sum,
        }]
    }))
}
