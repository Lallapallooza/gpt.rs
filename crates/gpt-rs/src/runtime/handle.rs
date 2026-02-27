use std::sync::Arc;

use anyhow::{bail, Result};

use crate::backend::spec::PortableBackend;
use crate::inference::CausalLanguageModel;
use crate::ops::functional::with_registry;
use crate::tensor::Tensor;

use super::{ModelInput, ModelOutput};

pub trait LoadedModel<B: PortableBackend + 'static>: Send {
    fn kind(&self) -> &str;

    fn forward(&mut self, input: ModelInput<B>) -> Result<ModelOutput>;

    fn debug_token_activations(&mut self, _tokens: &[usize]) -> Result<Vec<(String, Tensor)>> {
        bail!(
            "model kind '{}' does not support debug_token_activations",
            self.kind()
        );
    }

    fn as_causal_lm(&self) -> Option<&dyn CausalLanguageModel<B>> {
        None
    }
}

pub struct ModelHandle<B: PortableBackend + 'static> {
    inner: Box<dyn LoadedModel<B>>,
    registry: crate::ops::functional::FunctionalRegistryHandle<B>,
}

impl<B: PortableBackend + 'static> ModelHandle<B> {
    pub fn new(
        inner: Box<dyn LoadedModel<B>>,
        registry: crate::ops::functional::FunctionalRegistryHandle<B>,
    ) -> Self {
        Self { inner, registry }
    }
}

impl<B: PortableBackend + 'static> LoadedModel<B> for ModelHandle<B> {
    fn kind(&self) -> &str {
        self.inner.kind()
    }

    fn forward(&mut self, input: ModelInput<B>) -> Result<ModelOutput> {
        with_registry(Arc::clone(&self.registry), || self.inner.forward(input))
    }

    fn debug_token_activations(&mut self, tokens: &[usize]) -> Result<Vec<(String, Tensor)>> {
        with_registry(Arc::clone(&self.registry), || {
            self.inner.debug_token_activations(tokens)
        })
    }

    fn as_causal_lm(&self) -> Option<&dyn CausalLanguageModel<B>> {
        self.inner.as_causal_lm().is_some().then_some(self)
    }
}

impl<B: PortableBackend + 'static> CausalLanguageModel<B> for ModelHandle<B> {
    fn context_length(&self) -> usize {
        with_registry(Arc::clone(&self.registry), || {
            self.inner
                .as_causal_lm()
                .expect("ModelHandle::context_length called on a non-causal model")
                .context_length()
        })
    }

    fn num_layers(&self) -> usize {
        with_registry(Arc::clone(&self.registry), || {
            self.inner
                .as_causal_lm()
                .expect("ModelHandle::num_layers called on a non-causal model")
                .num_layers()
        })
    }

    fn forward(&self, tokens: &[usize]) -> Result<Tensor> {
        with_registry(Arc::clone(&self.registry), || {
            self.inner
                .as_causal_lm()
                .expect("ModelHandle::forward called on a non-causal model")
                .forward(tokens)
        })
    }

    fn forward_with_decode_cache(
        &self,
        tokens: &[usize],
        position_offset: usize,
        caches: &mut [Option<crate::ops::functional::DecodeKvCache<B>>],
    ) -> Result<Tensor> {
        with_registry(Arc::clone(&self.registry), || {
            self.inner
                .as_causal_lm()
                .expect("ModelHandle::forward_with_decode_cache called on a non-causal model")
                .forward_with_decode_cache(tokens, position_offset, caches)
        })
    }

    fn forward_with_decode_cache_with_capacity(
        &self,
        tokens: &[usize],
        position_offset: usize,
        caches: &mut [Option<crate::ops::functional::DecodeKvCache<B>>],
        capacity: usize,
    ) -> Result<Tensor> {
        with_registry(Arc::clone(&self.registry), || {
            self.inner
                .as_causal_lm()
                .expect(
                    "ModelHandle::forward_with_decode_cache_with_capacity called on a non-causal model",
                )
                .forward_with_decode_cache_with_capacity(tokens, position_offset, caches, capacity)
        })
    }

    fn forward_with_decode_cache_sample_next(
        &self,
        tokens: &[usize],
        position_offset: usize,
        caches: &mut [Option<crate::ops::functional::DecodeKvCache<B>>],
        request: crate::backend::spec::DecodeSampleRequest,
    ) -> Result<Option<usize>> {
        with_registry(Arc::clone(&self.registry), || {
            self.inner
                .as_causal_lm()
                .expect("ModelHandle::forward_with_decode_cache_sample_next called on a non-causal model")
                .forward_with_decode_cache_sample_next(tokens, position_offset, caches, request)
        })
    }
}
