use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{anyhow, Result};
use gpt_rs_macros::capture_ptir;

use super::conv::Conv2d;
use crate::backend::spec::PortableBackend;
use crate::module::{Module, ParamVisitor, ParamVisitorMut};
use crate::nn::layers::Linear;
use crate::ops::functional::common::CaptureIntoDeviceTensor;
use crate::ops::functional::{max_pool2d, relu, reshape, transpose, Padding2d};
use crate::tensor::{DeviceTensor, DeviceTensorOps, IntoDeviceTensor, Tensor};

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ResNet34Config {
    pub num_classes: usize,
}

impl Default for ResNet34Config {
    fn default() -> Self {
        Self { num_classes: 1000 }
    }
}

#[derive(Clone)]
pub struct BasicBlock<B: PortableBackend + 'static> {
    backend: Arc<B>,
    conv1: Conv2d<B>,
    conv2: Conv2d<B>,
    downsample: Option<Conv2d<B>>,
}

impl<B: PortableBackend + 'static> BasicBlock<B> {
    pub fn new(
        backend: Arc<B>,
        conv1: Conv2d<B>,
        conv2: Conv2d<B>,
        downsample: Option<Conv2d<B>>,
    ) -> Self {
        Self {
            backend,
            conv1,
            conv2,
            downsample,
        }
    }

    pub fn forward(&self, x: &DeviceTensor<B>) -> Result<DeviceTensor<B>> {
        let _scope = crate::profiling::layer_scope("ResNet::BasicBlock::forward");
        let identity = if let Some(downsample) = &self.downsample {
            downsample.forward(x)?
        } else {
            x.clone()
        };

        let mut out = self.conv1.forward(x)?;
        out = relu(self.backend.as_ref(), &out)?;
        out = self.conv2.forward(&out)?;
        out = out.add(&identity)?;
        out = relu(self.backend.as_ref(), &out)?;
        Ok(out)
    }
}

impl<B: PortableBackend + 'static> Module<B> for BasicBlock<B> {
    fn visit_params(&self, v: &mut ParamVisitor<'_, B>) -> Result<()> {
        v.scoped("conv1", |v| self.conv1.visit_params(v))?;
        v.scoped("conv2", |v| self.conv2.visit_params(v))?;
        if let Some(downsample) = &self.downsample {
            v.scoped("downsample", |v| downsample.visit_params(v))?;
        }
        Ok(())
    }

    fn visit_params_mut(&mut self, v: &mut ParamVisitorMut<'_, B>) -> Result<()> {
        v.scoped("conv1", |v| self.conv1.visit_params_mut(v))?;
        v.scoped("conv2", |v| self.conv2.visit_params_mut(v))?;
        if let Some(downsample) = &mut self.downsample {
            v.scoped("downsample", |v| downsample.visit_params_mut(v))?;
        }
        Ok(())
    }
}

pub struct ResNet34<B: PortableBackend + 'static> {
    backend: Arc<B>,
    conv1: Conv2d<B>,
    layer1: Vec<BasicBlock<B>>,
    layer2: Vec<BasicBlock<B>>,
    layer3: Vec<BasicBlock<B>>,
    layer4: Vec<BasicBlock<B>>,
    fc: Linear<B>,
}

impl<B: PortableBackend + 'static> ResNet34<B> {
    pub fn new(
        backend: Arc<B>,
        conv1: Conv2d<B>,
        layer1: Vec<BasicBlock<B>>,
        layer2: Vec<BasicBlock<B>>,
        layer3: Vec<BasicBlock<B>>,
        layer4: Vec<BasicBlock<B>>,
        fc: Linear<B>,
    ) -> Self {
        Self {
            backend,
            conv1,
            layer1,
            layer2,
            layer3,
            layer4,
            fc,
        }
    }

    pub fn forward(&self, input_nchw: &DeviceTensor<B>) -> Result<DeviceTensor<B>> {
        let _scope = crate::profiling::layer_scope("ResNet34::forward");
        let mut x = transpose(self.backend.as_ref(), input_nchw, &[0, 2, 3, 1])?;

        x = self.conv1.forward(&x)?;
        x = relu(self.backend.as_ref(), &x)?;

        // maxpool: 3x3 stride 2 padding 1 (NHWC).
        x = max_pool2d(
            self.backend.as_ref(),
            &x,
            [3, 3],
            [2, 2],
            Padding2d {
                top: 1,
                bottom: 1,
                left: 1,
                right: 1,
            },
        )?;

        for block in &self.layer1 {
            x = block.forward(&x)?;
        }
        for block in &self.layer2 {
            x = block.forward(&x)?;
        }
        for block in &self.layer3 {
            x = block.forward(&x)?;
        }
        for block in &self.layer4 {
            x = block.forward(&x)?;
        }

        // Global average pool over H and W to [N, 1, 1, C].
        let spatial_h = x.shape().dims()[1];
        let spatial_w = x.shape().dims()[2];
        let denom = (spatial_h * spatial_w) as f32;
        x = capture_ptir!({ input = &x }, |_session| {
            let sum_h = input.reduce_sum(vec![1], true);
            let sum_hw = sum_h.reduce_sum(vec![2], true);
            Ok(sum_hw.div_scalar(denom).id())
        })?
        .into_device_tensor(x.requires_grad_flag())?;

        let n = x.shape().dims()[0];
        let c = x.shape().dims()[3];
        x = reshape(self.backend.as_ref(), &x, &[n, c])?;

        self.fc.forward(&x)
    }

    pub fn forward_trace(
        &self,
        input_nchw: &DeviceTensor<B>,
    ) -> Result<Vec<(String, DeviceTensor<B>)>> {
        let mut trace = Vec::new();

        let mut x = transpose(self.backend.as_ref(), input_nchw, &[0, 2, 3, 1])?;
        trace.push(("input.nhwc".to_string(), x.clone()));

        x = self.conv1.forward(&x)?;
        trace.push(("stem.conv1".to_string(), x.clone()));
        x = relu(self.backend.as_ref(), &x)?;
        trace.push(("stem.relu".to_string(), x.clone()));

        x = max_pool2d(
            self.backend.as_ref(),
            &x,
            [3, 3],
            [2, 2],
            Padding2d {
                top: 1,
                bottom: 1,
                left: 1,
                right: 1,
            },
        )?;
        trace.push(("stem.maxpool".to_string(), x.clone()));

        for (stage_idx, stage) in [&self.layer1, &self.layer2, &self.layer3, &self.layer4]
            .into_iter()
            .enumerate()
        {
            let stage_num = stage_idx + 1;
            for (block_idx, block) in stage.iter().enumerate() {
                let prefix = format!("layer{stage_num}.{block_idx}");

                let identity = if let Some(downsample) = &block.downsample {
                    let down = downsample.forward(&x)?;
                    trace.push((format!("{prefix}.downsample"), down.clone()));
                    down
                } else {
                    x.clone()
                };

                let mut out = block.conv1.forward(&x)?;
                trace.push((format!("{prefix}.conv1"), out.clone()));
                out = relu(self.backend.as_ref(), &out)?;
                trace.push((format!("{prefix}.relu1"), out.clone()));
                out = block.conv2.forward(&out)?;
                trace.push((format!("{prefix}.conv2"), out.clone()));
                out = out.add(&identity)?;
                trace.push((format!("{prefix}.add"), out.clone()));
                out = relu(self.backend.as_ref(), &out)?;
                trace.push((format!("{prefix}.relu2"), out.clone()));

                x = out;
            }
        }

        let spatial_h = x.shape().dims()[1];
        let spatial_w = x.shape().dims()[2];
        let denom = (spatial_h * spatial_w) as f32;
        x = capture_ptir!({ input = &x }, |_session| {
            let sum_h = input.reduce_sum(vec![1], true);
            let sum_hw = sum_h.reduce_sum(vec![2], true);
            Ok(sum_hw.div_scalar(denom).id())
        })?
        .into_device_tensor(x.requires_grad_flag())?;
        trace.push(("avgpool".to_string(), x.clone()));

        let n = x.shape().dims()[0];
        let c = x.shape().dims()[3];
        x = reshape(self.backend.as_ref(), &x, &[n, c])?;
        trace.push(("flatten".to_string(), x.clone()));

        let logits = self.fc.forward(&x)?;
        trace.push(("logits".to_string(), logits));

        Ok(trace)
    }

    pub fn backend(&self) -> Arc<B> {
        Arc::clone(&self.backend)
    }

    pub fn build_from_params(
        backend: Arc<B>,
        mut get: impl FnMut(&str) -> Result<DeviceTensor<B>>,
    ) -> Result<Self> {
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
        let mut layers: [Vec<BasicBlock<B>>; 4] = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];

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

                layers[stage_idx].push(BasicBlock::new(
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
        Ok(Self::new(
            backend, conv1, layer1, layer2, layer3, layer4, fc,
        ))
    }

    pub fn from_named_tensors(
        backend: Arc<B>,
        mut tensors: HashMap<String, Tensor>,
    ) -> Result<Self> {
        fn take(map: &mut HashMap<String, Tensor>, name: &str) -> Result<Tensor> {
            map.remove(name)
                .ok_or_else(|| anyhow!("missing tensor '{}' for ResNet34", name))
        }

        let conv1_weight = take(&mut tensors, "conv1.weight")?.into_device_tensor(&backend)?;
        let conv1_bias = take(&mut tensors, "conv1.bias")?.into_device_tensor(&backend)?;
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
        let mut layers: [Vec<BasicBlock<B>>; 4] = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];

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

                let block_conv1_weight = take(&mut tensors, &format!("{conv1_key}.weight"))?
                    .into_device_tensor(&backend)?;
                let block_conv1_bias = take(&mut tensors, &format!("{conv1_key}.bias"))?
                    .into_device_tensor(&backend)?;
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

                let block_conv2_weight = take(&mut tensors, &format!("{conv2_key}.weight"))?
                    .into_device_tensor(&backend)?;
                let block_conv2_bias = take(&mut tensors, &format!("{conv2_key}.bias"))?
                    .into_device_tensor(&backend)?;
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
                    let ds_weight = take(&mut tensors, &format!("{ds_key}.weight"))?
                        .into_device_tensor(&backend)?;
                    let ds_bias = take(&mut tensors, &format!("{ds_key}.bias"))?
                        .into_device_tensor(&backend)?;
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

                layers[stage_idx].push(BasicBlock::new(
                    Arc::clone(&backend),
                    conv1,
                    conv2,
                    downsample,
                ));

                in_channels = out_channels;
            }
        }

        let fc_weight = take(&mut tensors, "fc.weight")?.into_device_tensor(&backend)?;
        let fc_bias = take(&mut tensors, "fc.bias")?.into_device_tensor(&backend)?;
        let fc_bias_len = fc_bias.shape().dims()[0];
        let fc_weight_dims = fc_weight.shape().dims();
        let fc_weight = if fc_weight_dims.len() == 2 && fc_weight_dims[0] == fc_bias_len {
            // Accept canonical Torch linear weights [O, I] and pack to gpt-rs layout [I, O].
            transpose(backend.as_ref(), &fc_weight, &[1, 0])?.freeze()?
        } else {
            fc_weight
        };
        let fc = Linear::new(Arc::clone(&backend), fc_weight, Some(fc_bias))?;

        let [layer1, layer2, layer3, layer4] = layers;
        Ok(ResNet34::new(
            backend, conv1, layer1, layer2, layer3, layer4, fc,
        ))
    }
}

impl<B: PortableBackend + 'static> Module<B> for ResNet34<B> {
    fn visit_params(&self, v: &mut ParamVisitor<'_, B>) -> Result<()> {
        v.scoped("conv1", |v| self.conv1.visit_params(v))?;

        for (stage_idx, stage) in [&self.layer1, &self.layer2, &self.layer3, &self.layer4]
            .into_iter()
            .enumerate()
        {
            let stage_name = format!("layer{}", stage_idx + 1);
            v.scoped(&stage_name, |v| {
                for (block_idx, block) in stage.iter().enumerate() {
                    let block_name = block_idx.to_string();
                    v.scoped(&block_name, |v| block.visit_params(v))?;
                }
                Ok(())
            })?;
        }

        v.scoped("fc", |v| self.fc.visit_params(v))?;
        Ok(())
    }

    fn visit_params_mut(&mut self, v: &mut ParamVisitorMut<'_, B>) -> Result<()> {
        v.scoped("conv1", |v| self.conv1.visit_params_mut(v))?;

        for (stage_idx, stage) in [
            &mut self.layer1,
            &mut self.layer2,
            &mut self.layer3,
            &mut self.layer4,
        ]
        .into_iter()
        .enumerate()
        {
            let stage_name = format!("layer{}", stage_idx + 1);
            v.scoped(&stage_name, |v| {
                for (block_idx, block) in stage.iter_mut().enumerate() {
                    let block_name = block_idx.to_string();
                    v.scoped(&block_name, |v| block.visit_params_mut(v))?;
                }
                Ok(())
            })?;
        }

        v.scoped("fc", |v| self.fc.visit_params_mut(v))?;
        Ok(())
    }
}
