//! Convolution layers used by image models (NHWC internal layout).

use std::sync::Arc;

use anyhow::{ensure, Result};

use crate::backend::spec::PortableBackend;
use crate::ops::functional::{conv2d, Conv2dParams2d, Padding2d};
use crate::tensor::DeviceTensor;

#[derive(Clone)]
pub struct Conv2d<B: PortableBackend + 'static> {
    backend: Arc<B>,
    weight: DeviceTensor<B>,
    bias: Option<DeviceTensor<B>>,
    params: Conv2dParams2d,
}

impl<B: PortableBackend + 'static> Conv2d<B> {
    pub fn backend(&self) -> Arc<B> {
        Arc::clone(&self.backend)
    }

    pub fn new(
        backend: Arc<B>,
        weight: DeviceTensor<B>,
        bias: Option<DeviceTensor<B>>,
        kernel: [usize; 2],
        stride: [usize; 2],
        dilation: [usize; 2],
        padding: Padding2d,
    ) -> Result<Self> {
        Self::new_grouped(backend, weight, bias, kernel, stride, dilation, padding, 1)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new_grouped(
        backend: Arc<B>,
        weight: DeviceTensor<B>,
        bias: Option<DeviceTensor<B>>,
        kernel: [usize; 2],
        stride: [usize; 2],
        dilation: [usize; 2],
        padding: Padding2d,
        groups: usize,
    ) -> Result<Self> {
        ensure!(groups > 0, "conv2d groups must be > 0");

        let weight = weight.as_param()?;
        let bias = match bias {
            Some(bias) => Some(bias.as_param()?),
            None => None,
        };

        Ok(Self {
            backend,
            weight,
            bias,
            params: Conv2dParams2d {
                kernel,
                stride,
                dilation,
                padding,
                groups,
            },
        })
    }

    pub fn forward(&self, x: &DeviceTensor<B>) -> Result<DeviceTensor<B>> {
        let _scope = crate::profiling::layer_scope("Conv2d::forward");
        conv2d(
            self.backend.as_ref(),
            x,
            &self.weight,
            self.bias.as_ref(),
            self.params,
        )
    }
}
