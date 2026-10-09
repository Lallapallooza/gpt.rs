//! Image models: a checkpoint's module tree from NCHW pixels to class logits.
//!
//! A model contributes its configuration ([`VisionConfig`]). [`ImageModel`] loads it and runs it
//! the same way for all models.

use std::sync::Arc;

use anyhow::{anyhow, Result};
use serde::de::DeserializeOwned;

use crate::backend::PortableBackend;
use crate::model::ModelConfig;
use crate::module::Layer;
use crate::nn::Params;
use crate::ops::graph::context::push_default_arena;
use crate::ops::graph::GraphArena;
use crate::runtime::LoadedModel;
use crate::tensor::{DeviceTensor, HostTensor};

/// Checkpoint configuration of an image model: its model kind and its module tree.
pub trait VisionConfig: DeserializeOwned + Send + 'static {
    const KIND: &'static str;

    /// The module tree, whose forward maps NCHW pixels to class logits.
    type Model<B: PortableBackend + 'static>: for<'a> Layer<B, Args<'a> = &'a DeviceTensor<B>, Output = DeviceTensor<B>>
        + Send
        + Sync;

    /// Validates the configuration and builds the module tree.
    fn build<B: PortableBackend + 'static>(&self, p: &mut Params<B>) -> Result<Self::Model<B>>;
}

/// What a loaded image model does.
pub trait VisionModel {
    /// Class logits `[N, num_labels]` of NCHW `pixel_values`, or the pooled features `[N, C]`
    /// when the checkpoint has no classifier.
    fn forward(&self, pixel_values: HostTensor) -> Result<HostTensor>;
}

/// An image model configured by `C`.
pub struct ImageModel<B: PortableBackend + 'static, C: VisionConfig> {
    backend: Arc<B>,
    model: C::Model<B>,
}

/// Builds the image model of `config.kind` from checkpoint tensors.
pub(crate) fn build_from_model_config<B: PortableBackend + 'static, C: VisionConfig>(
    backend: Arc<B>,
    cfg: &ModelConfig,
    get: &mut dyn FnMut(&str) -> Result<Option<DeviceTensor<B>>>,
) -> Result<Box<dyn LoadedModel>> {
    let config: C = serde_json::from_value(cfg.config.clone())
        .map_err(|err| anyhow!("invalid {} config: {err}", C::KIND))?;
    let mut params = Params::new(Arc::clone(&backend), get)
        .with_linear_input_dtype(cfg.runtime.matmul_input_dtype);
    let model = config.build(&mut params)?;
    Ok(Box::new(ImageModel::<B, C> { backend, model }))
}

impl<B: PortableBackend + 'static, C: VisionConfig> VisionModel for ImageModel<B, C> {
    /// Records the forward into one arena, so it runs as one compiled program, including its
    /// computations over parameters alone.
    fn forward(&self, pixel_values: HostTensor) -> Result<HostTensor> {
        let _arena = push_default_arena(GraphArena::new(Arc::clone(&self.backend)));
        let pixel_values = DeviceTensor::from_host(Arc::clone(&self.backend), pixel_values)?;
        self.model.call(&pixel_values).to_host()
    }
}

impl<B: PortableBackend + 'static, C: VisionConfig> LoadedModel for ImageModel<B, C> {
    fn kind(&self) -> &str {
        C::KIND
    }

    fn as_vision(&self) -> Option<&dyn VisionModel> {
        Some(self)
    }
}
