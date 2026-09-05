//! Builds layers from named parameter tensors or from random initialisation.

use std::sync::Arc;

use anyhow::{ensure, Result};
use rand::RngCore;

use crate::backend::spec::PortableBackend;
use crate::ops::functional::Conv2dParams2d;
use crate::tensor::{DType, DeviceTensor, Shape, Tensor};

use super::{Conv2d, Embedding, LayerNorm, Linear, RmsNorm, RmsNormConfig};

enum TensorSource<'a, B: PortableBackend + 'static> {
    Named(&'a mut dyn FnMut(&str) -> Result<DeviceTensor<B>>),
    Random(&'a mut dyn RngCore),
}

/// Initial values of a randomly initialised tensor.
#[derive(Clone, Copy)]
enum Init {
    Normal,
    Zeros,
    Ones,
}

/// Creates layers. It names each parameter by module path (`model.layers.0.mlp.up_proj`) plus the
/// parameter leaf (`weight`, `bias`). These names match the names that [`crate::module::Module`]
/// visits.
pub struct LayerLoader<'a, B: PortableBackend + 'static> {
    backend: Arc<B>,
    source: TensorSource<'a, B>,
    linear_input_dtype: Option<DType>,
}

impl<'a, B: PortableBackend + 'static> LayerLoader<'a, B> {
    /// Creates a loader that reads parameters by name from `get`. It checks each parameter
    /// against the shape that the layer expects.
    pub fn new(backend: Arc<B>, get: &'a mut dyn FnMut(&str) -> Result<DeviceTensor<B>>) -> Self {
        Self {
            backend,
            source: TensorSource::Named(get),
            linear_input_dtype: None,
        }
    }

    /// Creates a loader that initialises weights from `N(0, 0.02^2)`, biases with zeros and norm
    /// scales with ones.
    pub fn random(backend: Arc<B>, rng: &'a mut dyn RngCore) -> Self {
        Self {
            backend,
            source: TensorSource::Random(rng),
            linear_input_dtype: None,
        }
    }

    /// Sets the input dtype of every [`Linear`] this loader builds.
    pub fn with_linear_input_dtype(mut self, dtype: Option<DType>) -> Self {
        self.linear_input_dtype = dtype;
        self
    }

    /// Returns the parameter `name` with shape `dims`.
    pub fn tensor(&mut self, name: &str, dims: &[usize]) -> Result<DeviceTensor<B>> {
        self.load(name, dims, Init::Normal)
    }

    fn load(&mut self, name: &str, dims: &[usize], init: Init) -> Result<DeviceTensor<B>> {
        match &mut self.source {
            TensorSource::Named(get) => {
                let tensor = get(name)?;
                ensure!(
                    tensor.shape().dims() == dims,
                    "parameter '{name}' has shape {:?}, expected {:?}",
                    tensor.shape().dims(),
                    dims
                );
                Ok(tensor)
            }
            TensorSource::Random(rng) => {
                let shape = Shape::new(dims.to_vec());
                let host = match init {
                    Init::Normal => Tensor::randn(shape, 0.02, rng),
                    Init::Zeros => Tensor::zeros(shape),
                    Init::Ones => Tensor::ones(shape),
                };
                DeviceTensor::from_host(Arc::clone(&self.backend), host)
            }
        }
    }

    /// `{prefix}.weight` (`[out_features, in_features]`) and, with `bias`, `{prefix}.bias`.
    pub fn linear(
        &mut self,
        prefix: &str,
        in_features: usize,
        out_features: usize,
        bias: bool,
    ) -> Result<Linear<B>> {
        let weight = self.tensor(&format!("{prefix}.weight"), &[out_features, in_features])?;
        let bias = if bias {
            Some(self.load(&format!("{prefix}.bias"), &[out_features], Init::Zeros)?)
        } else {
            None
        };
        Ok(Linear {
            weight: weight.as_param()?,
            bias: bias.map(|bias| bias.as_param()).transpose()?,
            input_dtype: self.linear_input_dtype,
        })
    }

    /// `{prefix}.weight` (`[out_channels, in_channels / groups, kh, kw]`) and `{prefix}.bias`
    /// (`[out_channels]`).
    pub fn conv2d(
        &mut self,
        prefix: &str,
        in_channels: usize,
        out_channels: usize,
        params: Conv2dParams2d,
    ) -> Result<Conv2d<B>> {
        ensure!(
            params.groups > 0 && in_channels.is_multiple_of(params.groups),
            "conv2d '{prefix}' input channels {in_channels} must be a multiple of groups {}",
            params.groups
        );
        let [kh, kw] = params.kernel;
        let dims = [out_channels, in_channels / params.groups, kh, kw];
        let weight = self.tensor(&format!("{prefix}.weight"), &dims)?;
        let bias = self.load(&format!("{prefix}.bias"), &[out_channels], Init::Zeros)?;
        Ok(Conv2d {
            weight: weight.as_param()?,
            bias: bias.as_param()?,
            params,
        })
    }

    /// `{prefix}.weight` (`[num_embeddings, dim]`).
    pub fn embedding(
        &mut self,
        prefix: &str,
        num_embeddings: usize,
        dim: usize,
    ) -> Result<Embedding<B>> {
        let weight = self.tensor(&format!("{prefix}.weight"), &[num_embeddings, dim])?;
        Ok(Embedding { weight })
    }

    /// `{prefix}.weight` and `{prefix}.bias` (`[dim]`).
    pub fn layer_norm(&mut self, prefix: &str, dim: usize, eps: f32) -> Result<LayerNorm<B>> {
        let weight = self.load(&format!("{prefix}.weight"), &[dim], Init::Ones)?;
        let bias = self.load(&format!("{prefix}.bias"), &[dim], Init::Zeros)?;
        Ok(LayerNorm { weight, bias, eps })
    }

    /// `{prefix}.weight` (`[dim]`).
    pub fn rms_norm(
        &mut self,
        prefix: &str,
        dim: usize,
        config: RmsNormConfig,
    ) -> Result<RmsNorm<B>> {
        let weight = self.load(&format!("{prefix}.weight"), &[dim], Init::Ones)?;
        Ok(RmsNorm {
            weight,
            eps: config.eps,
            unit_offset: config.unit_offset,
        })
    }
}
