use anyhow::Result;

use crate::backend::spec::PortableBackend;
use crate::inference::CausalLanguageModel;

use super::{ModelInput, ModelOutput};

pub trait LoadedModel<B: PortableBackend + 'static>: Send {
    fn kind(&self) -> &str;

    fn forward(&mut self, input: ModelInput<B>) -> Result<ModelOutput>;

    fn as_causal_lm(&self) -> Option<&dyn CausalLanguageModel<B>> {
        None
    }
}

/// A model loaded from a checkpoint, with the checkpoint's generation metadata.
pub struct ModelHandle<B: PortableBackend + 'static> {
    inner: Box<dyn LoadedModel<B>>,
    eos_token_ids: Vec<usize>,
}

impl<B: PortableBackend + 'static> ModelHandle<B> {
    pub fn new(inner: Box<dyn LoadedModel<B>>, eos_token_ids: Vec<usize>) -> Self {
        Self {
            inner,
            eos_token_ids,
        }
    }

    /// Token ids that end a generated sequence, as the checkpoint declares them.
    pub fn eos_token_ids(&self) -> &[usize] {
        &self.eos_token_ids
    }
}

impl<B: PortableBackend + 'static> LoadedModel<B> for ModelHandle<B> {
    fn kind(&self) -> &str {
        self.inner.kind()
    }

    fn forward(&mut self, input: ModelInput<B>) -> Result<ModelOutput> {
        self.inner.forward(input)
    }

    fn as_causal_lm(&self) -> Option<&dyn CausalLanguageModel<B>> {
        self.inner.as_causal_lm()
    }
}
