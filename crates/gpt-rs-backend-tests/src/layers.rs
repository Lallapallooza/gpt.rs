//! Layers with random weights under the names of the Hugging Face modules they mirror, loaded as a
//! model's parameters.

use std::sync::Arc;

use anyhow::Result;
use gpt_rs::backend::PortableBackend;
use gpt_rs::nn::layers::Linear;
use gpt_rs::nn::{Params, TensorView};
use gpt_rs::params::{base_param_id, param_key};
use gpt_rs::runtime::next_namespace;
use gpt_rs::tensor::{DeviceTensor, HostTensor};
use rand::rngs::StdRng;
use rand::Rng;

/// Builds a layer with `build`, passing it [`Params`] over the host `tensors`, which are keyed by
/// parameter path. Like a loaded model, the layer reads them as parameters of a fresh namespace,
/// in the views it asks for, so the passes that work on parameters, such as `outline-static`,
/// treat them as a model's.
pub fn load_layer<B: PortableBackend + 'static, L>(
    backend: &Arc<B>,
    tensors: impl IntoIterator<Item = (String, HostTensor)>,
    build: impl FnOnce(&mut Params<B>) -> Result<L>,
) -> Result<L> {
    let host_tensors: Vec<(String, HostTensor)> = tensors.into_iter().collect();
    let namespace = next_namespace();
    let mut get = |name: &str, view: TensorView| -> Result<Option<DeviceTensor<B>>> {
        let Some((_, host_tensor)) = host_tensors.iter().find(|(candidate, _)| candidate == name)
        else {
            return Ok(None);
        };
        let key = param_key(namespace, base_param_id(name)?);
        let ty = view.type_of(&host_tensor.tensor_type())?;
        let viewed = HostTensor::from_le_bytes(&ty, host_tensor.to_le_bytes())?;
        let device_tensor = DeviceTensor::from_host(Arc::clone(backend), viewed)?;
        device_tensor.as_param(key.0).map(Some)
    };
    build(&mut Params::new(Arc::clone(backend), &mut get))
}

/// `len` values drawn uniformly from [-1, 1).
pub fn random_vec(rng: &mut StdRng, len: usize) -> Vec<f32> {
    (0..len).map(|_| rng.gen::<f32>() * 2.0 - 1.0).collect()
}

/// `len` values drawn uniformly from [`lo`, `hi`).
pub fn random_vec_range(rng: &mut StdRng, len: usize, lo: f32, hi: f32) -> Vec<f32> {
    let span = hi - lo;
    (0..len).map(|_| lo + rng.gen::<f32>() * span).collect()
}

/// An f32 host tensor of the extents `shape` that holds `data`.
pub fn tensor_from_vec(shape: &[usize], data: Vec<f32>) -> HostTensor {
    HostTensor::f32(shape, data).unwrap()
}

/// Host weights of a `Linear`: an `[out, in]` weight and an optional bias.
pub struct HostLinear {
    pub(crate) weight: HostTensor,
    pub(crate) bias: Option<HostTensor>,
}

impl HostLinear {
    pub fn random(
        rng: &mut StdRng,
        in_features: usize,
        out_features: usize,
        with_bias: bool,
    ) -> Self {
        let weight = tensor_from_vec(
            &[out_features, in_features],
            random_vec(rng, out_features * in_features),
        );
        let bias =
            with_bias.then(|| tensor_from_vec(&[out_features], random_vec(rng, out_features)));
        Self { weight, bias }
    }

    pub fn tensors(&self, prefix: &str) -> Vec<(String, HostTensor)> {
        let mut tensors = vec![(format!("{prefix}.weight"), self.weight.clone())];
        if let Some(bias) = &self.bias {
            tensors.push((format!("{prefix}.bias"), bias.clone()));
        }
        tensors
    }

    pub fn layer<B: PortableBackend + 'static>(&self, backend: &Arc<B>) -> Linear<B> {
        load_layer(backend, self.tensors("proj"), |p| {
            p.scoped("proj", Linear::new)
        })
        .unwrap()
    }
}

/// Host weights of a `Conv1D`: an `[in, out]` weight and a bias.
pub struct HostConv1D {
    pub(crate) weight: HostTensor,
    pub(crate) bias: HostTensor,
}

impl HostConv1D {
    pub fn random(rng: &mut StdRng, in_features: usize, out_features: usize) -> Self {
        let weight = tensor_from_vec(
            &[in_features, out_features],
            random_vec(rng, in_features * out_features),
        );
        let bias = tensor_from_vec(&[out_features], random_vec(rng, out_features));
        Self { weight, bias }
    }

    pub fn tensors(&self, prefix: &str) -> Vec<(String, HostTensor)> {
        vec![
            (format!("{prefix}.weight"), self.weight.clone()),
            (format!("{prefix}.bias"), self.bias.clone()),
        ]
    }
}

/// Host parameters of a `BatchNorm2d` with a positive running variance.
pub struct HostBatchNorm {
    pub(crate) weight: Vec<f32>,
    pub(crate) bias: Vec<f32>,
    pub(crate) running_mean: Vec<f32>,
    pub(crate) running_var: Vec<f32>,
}

impl HostBatchNorm {
    pub fn random(rng: &mut StdRng, channels: usize) -> Self {
        Self {
            weight: random_vec(rng, channels),
            bias: random_vec(rng, channels),
            running_mean: random_vec(rng, channels),
            running_var: random_vec_range(rng, channels, 0.1, 2.0),
        }
    }

    pub fn tensors(&self, prefix: &str) -> Vec<(String, HostTensor)> {
        let channels = self.weight.len();
        let leaves = [
            ("weight", &self.weight),
            ("bias", &self.bias),
            ("running_mean", &self.running_mean),
            ("running_var", &self.running_var),
        ];
        let mut tensors = Vec::with_capacity(leaves.len());
        for (leaf, data) in leaves {
            let tensor = tensor_from_vec(&[channels], data.clone());
            tensors.push((format!("{prefix}.{leaf}"), tensor));
        }
        tensors
    }
}
