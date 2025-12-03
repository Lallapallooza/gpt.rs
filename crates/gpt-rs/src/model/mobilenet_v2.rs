use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{anyhow, Result};
use gpt_rs_macros::capture_ptir;

use super::conv::Conv2d;
use crate::backend::spec::PortableBackend;
use crate::nn::layers::Linear;
use crate::ops::functional::common::CaptureIntoDeviceTensor;
use crate::ops::functional::{relu6, reshape, transpose};
use crate::tensor::{DeviceTensor, DeviceTensorOps, IntoDeviceTensor, Tensor};

#[derive(Clone)]
pub struct InvertedResidual<B: PortableBackend + 'static> {
    backend: Arc<B>,
    expand: Option<Conv2d<B>>,
    depthwise: Conv2d<B>,
    project: Conv2d<B>,
    use_res_connect: bool,
}

impl<B: PortableBackend + 'static> InvertedResidual<B> {
    pub fn new(
        backend: Arc<B>,
        expand: Option<Conv2d<B>>,
        depthwise: Conv2d<B>,
        project: Conv2d<B>,
        use_res_connect: bool,
    ) -> Self {
        Self {
            backend,
            expand,
            depthwise,
            project,
            use_res_connect,
        }
    }

    pub fn forward(&self, x: &DeviceTensor<B>) -> Result<DeviceTensor<B>> {
        let _scope = crate::profiling::layer_scope("MobileNetV2::InvertedResidual::forward");
        let identity = x.clone();
        let mut out = x.clone();

        if let Some(expand) = &self.expand {
            out = expand.forward(&out)?;
            out = relu6(self.backend.as_ref(), &out)?;
        }

        out = self.depthwise.forward(&out)?;
        out = relu6(self.backend.as_ref(), &out)?;

        out = self.project.forward(&out)?;

        if self.use_res_connect {
            out = out.add(&identity)?;
        }

        Ok(out)
    }
}

pub struct MobileNetV2<B: PortableBackend + 'static> {
    backend: Arc<B>,
    stem: Conv2d<B>,
    blocks: Vec<InvertedResidual<B>>,
    head: Conv2d<B>,
    classifier: Linear<B>,
}

impl<B: PortableBackend + 'static> MobileNetV2<B> {
    pub fn new(
        backend: Arc<B>,
        stem: Conv2d<B>,
        blocks: Vec<InvertedResidual<B>>,
        head: Conv2d<B>,
        classifier: Linear<B>,
    ) -> Self {
        Self {
            backend,
            stem,
            blocks,
            head,
            classifier,
        }
    }

    pub fn forward(&self, input_nchw: &DeviceTensor<B>) -> Result<DeviceTensor<B>> {
        let _scope = crate::profiling::layer_scope("MobileNetV2::forward");
        let mut x = transpose(self.backend.as_ref(), input_nchw, &[0, 2, 3, 1])?;

        x = self.stem.forward(&x)?;
        x = relu6(self.backend.as_ref(), &x)?;

        for block in &self.blocks {
            x = block.forward(&x)?;
        }

        x = self.head.forward(&x)?;
        x = relu6(self.backend.as_ref(), &x)?;

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

        self.classifier.forward(&x)
    }

    pub fn forward_trace(
        &self,
        input_nchw: &DeviceTensor<B>,
    ) -> Result<Vec<(String, DeviceTensor<B>)>> {
        let mut trace = Vec::new();

        let mut x = transpose(self.backend.as_ref(), input_nchw, &[0, 2, 3, 1])?;
        trace.push(("input.nhwc".to_string(), x.clone()));

        x = self.stem.forward(&x)?;
        trace.push(("stem.conv".to_string(), x.clone()));
        x = relu6(self.backend.as_ref(), &x)?;
        trace.push(("stem.relu6".to_string(), x.clone()));

        for (block_idx, block) in self.blocks.iter().enumerate() {
            let prefix = format!("blocks.{block_idx}");
            let identity = x.clone();
            let mut out = x.clone();

            if let Some(expand) = &block.expand {
                out = expand.forward(&out)?;
                trace.push((format!("{prefix}.expand"), out.clone()));
                out = relu6(self.backend.as_ref(), &out)?;
                trace.push((format!("{prefix}.expand.relu6"), out.clone()));
            }

            out = block.depthwise.forward(&out)?;
            trace.push((format!("{prefix}.depthwise"), out.clone()));
            out = relu6(self.backend.as_ref(), &out)?;
            trace.push((format!("{prefix}.depthwise.relu6"), out.clone()));

            out = block.project.forward(&out)?;
            trace.push((format!("{prefix}.project"), out.clone()));

            if block.use_res_connect {
                out = out.add(&identity)?;
                trace.push((format!("{prefix}.add"), out.clone()));
            }

            x = out;
        }

        x = self.head.forward(&x)?;
        trace.push(("head.conv".to_string(), x.clone()));
        x = relu6(self.backend.as_ref(), &x)?;
        trace.push(("head.relu6".to_string(), x.clone()));

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

        let logits = self.classifier.forward(&x)?;
        trace.push(("logits".to_string(), logits));

        Ok(trace)
    }

    pub fn backend(&self) -> Arc<B> {
        Arc::clone(&self.backend)
    }

    pub fn from_named_tensors(
        backend: Arc<B>,
        mut tensors: HashMap<String, Tensor>,
    ) -> Result<Self> {
        fn take(map: &mut HashMap<String, Tensor>, name: &str) -> Result<Tensor> {
            map.remove(name)
                .ok_or_else(|| anyhow!("missing tensor '{}' for MobileNetV2", name))
        }

        let stem_weight = take(&mut tensors, "stem.weight")?.into_device_tensor(&backend)?;
        let stem_bias = take(&mut tensors, "stem.bias")?.into_device_tensor(&backend)?;
        let stem = Conv2d::new(
            Arc::clone(&backend),
            stem_weight,
            Some(stem_bias),
            [3, 3],
            [2, 2],
            [1, 1],
            crate::ops::functional::Padding2d {
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
                    let w = take(&mut tensors, &format!("blocks.{block_idx}.expand.weight"))?
                        .into_device_tensor(&backend)?;
                    let b = take(&mut tensors, &format!("blocks.{block_idx}.expand.bias"))?
                        .into_device_tensor(&backend)?;
                    Some(Conv2d::new(
                        Arc::clone(&backend),
                        w,
                        Some(b),
                        [1, 1],
                        [1, 1],
                        [1, 1],
                        crate::ops::functional::Padding2d::zero(),
                    )?)
                } else {
                    None
                };

                let dw_weight = take(
                    &mut tensors,
                    &format!("blocks.{block_idx}.depthwise.weight"),
                )?
                .into_device_tensor(&backend)?;
                let dw_bias = take(&mut tensors, &format!("blocks.{block_idx}.depthwise.bias"))?
                    .into_device_tensor(&backend)?;
                let groups = dw_weight.shape().dims()[0];
                let depthwise = Conv2d::new_grouped(
                    Arc::clone(&backend),
                    dw_weight,
                    Some(dw_bias),
                    [3, 3],
                    [stride, stride],
                    [1, 1],
                    crate::ops::functional::Padding2d {
                        top: 1,
                        bottom: 1,
                        left: 1,
                        right: 1,
                    },
                    groups,
                )?;

                let proj_weight =
                    take(&mut tensors, &format!("blocks.{block_idx}.project.weight"))?
                        .into_device_tensor(&backend)?;
                let proj_bias = take(&mut tensors, &format!("blocks.{block_idx}.project.bias"))?
                    .into_device_tensor(&backend)?;
                let project = Conv2d::new(
                    Arc::clone(&backend),
                    proj_weight,
                    Some(proj_bias),
                    [1, 1],
                    [1, 1],
                    [1, 1],
                    crate::ops::functional::Padding2d::zero(),
                )?;

                let use_res_connect = stride == 1 && input_channels == out_channels;
                blocks.push(InvertedResidual::new(
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

        let head_weight = take(&mut tensors, "head.weight")?.into_device_tensor(&backend)?;
        let head_bias = take(&mut tensors, "head.bias")?.into_device_tensor(&backend)?;
        let head = Conv2d::new(
            Arc::clone(&backend),
            head_weight,
            Some(head_bias),
            [1, 1],
            [1, 1],
            [1, 1],
            crate::ops::functional::Padding2d::zero(),
        )?;

        let classifier_weight =
            take(&mut tensors, "classifier.weight")?.into_device_tensor(&backend)?;
        let classifier_bias =
            take(&mut tensors, "classifier.bias")?.into_device_tensor(&backend)?;
        let bias_len = classifier_bias.shape().dims()[0];
        let weight_dims = classifier_weight.shape().dims();
        let classifier_weight = if weight_dims.len() == 2 && weight_dims[0] == bias_len {
            // Accept canonical Torch linear weights [O, I] and pack to gpt-rs layout [I, O].
            transpose(backend.as_ref(), &classifier_weight, &[1, 0])?.freeze()?
        } else {
            classifier_weight
        };
        let classifier = Linear::new(
            Arc::clone(&backend),
            classifier_weight,
            Some(classifier_bias),
        )?;

        Ok(MobileNetV2::new(backend, stem, blocks, head, classifier))
    }
}
