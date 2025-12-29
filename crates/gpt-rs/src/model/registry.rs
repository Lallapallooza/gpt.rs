use std::collections::HashMap;
use std::sync::Arc;

use anyhow::Result;

use crate::backend::spec::PortableBackend;
use crate::runtime::LoadedModel;
use crate::tensor::DeviceTensor;

use super::ModelConfig;

/// Signature for model builders used by the runtime checkpoint loader.
pub type BuildFn<B> = fn(
    Arc<B>,
    &ModelConfig,
    &mut dyn FnMut(&str) -> Result<DeviceTensor<B>>,
) -> Result<Box<dyn LoadedModel<B>>>;

#[derive(Clone, Copy)]
pub struct ModelFactory<B: PortableBackend + 'static> {
    pub kind: &'static str,
    pub build: BuildFn<B>,
}

/// Returns the list of built-in model factories.
///
/// Note: this list is generic over the backend type. It is intentionally defined as a function
/// so models can reference generic build functions (`foo::<B>`) without requiring a type-erased
/// indirection layer.
pub fn model_factories<B: PortableBackend + 'static>() -> &'static [ModelFactory<B>] {
    &[
        ModelFactory {
            kind: super::gpt::KIND,
            build: super::gpt::build_from_model_config::<B>,
        },
        ModelFactory {
            kind: super::resnet::KIND,
            build: super::resnet::build_from_model_config::<B>,
        },
        ModelFactory {
            kind: super::mobilenet_v2::KIND,
            build: super::mobilenet_v2::build_from_model_config::<B>,
        },
    ]
}

pub fn model_factory<B: PortableBackend + 'static>(kind: &str) -> Option<BuildFn<B>> {
    model_factories::<B>()
        .iter()
        .find(|entry| entry.kind == kind)
        .map(|entry| entry.build)
}

/// Builds a lookup table for supported models.
///
/// This is useful when repeated kind lookups are expected; the construction cost is low for the
/// small built-in set but avoids O(N) scans in hot paths.
pub fn model_registry<B: PortableBackend + 'static>() -> HashMap<&'static str, BuildFn<B>> {
    let mut registry = HashMap::new();
    for entry in model_factories::<B>() {
        registry.insert(entry.kind, entry.build);
    }
    registry
}
