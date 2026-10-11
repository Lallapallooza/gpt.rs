use anyhow::Result;

use crate::inference::vision::VisionModel;
use crate::inference::CausalLanguageModel;
use crate::model::config::ModelRuntimeConfig;

/// A model loaded from a checkpoint. Its capabilities tell what it runs on: token ids for a
/// causal language model, pixels for an image model.
pub trait LoadedModel: Send + Sync {
    /// The Hugging Face `model_type` of the checkpoint.
    fn kind(&self) -> &str;

    fn as_causal_lm(&self) -> Option<&dyn CausalLanguageModel> {
        None
    }

    fn as_vision(&self) -> Option<&dyn VisionModel> {
        None
    }
}

/// A model loaded from a checkpoint, with the checkpoint's generation metadata and the settings it
/// runs with.
pub struct ModelHandle {
    inner: Box<dyn LoadedModel>,
    eos_token_ids: Vec<usize>,
    runtime: ModelRuntimeConfig,
}

impl ModelHandle {
    pub fn new(
        inner: Box<dyn LoadedModel>,
        eos_token_ids: Vec<usize>,
        runtime: ModelRuntimeConfig,
    ) -> Self {
        Self {
            inner,
            eos_token_ids,
            runtime,
        }
    }

    /// Token ids that end a generated sequence, as the checkpoint declares them.
    pub fn eos_token_ids(&self) -> &[usize] {
        &self.eos_token_ids
    }

    /// How the model runs: what the checkpoint declares, or what the load options set.
    pub fn runtime(&self) -> &ModelRuntimeConfig {
        &self.runtime
    }

    /// The model as a causal language model, or an error naming its kind.
    pub fn causal_lm(&self) -> Result<&dyn CausalLanguageModel> {
        self.as_causal_lm().ok_or_else(|| {
            anyhow::anyhow!("model '{}' is not a causal language model", self.kind())
        })
    }

    /// The model as an image model, or an error naming its kind.
    pub fn vision(&self) -> Result<&dyn VisionModel> {
        self.as_vision()
            .ok_or_else(|| anyhow::anyhow!("model '{}' is not an image model", self.kind()))
    }
}

impl LoadedModel for ModelHandle {
    fn kind(&self) -> &str {
        self.inner.kind()
    }

    fn as_causal_lm(&self) -> Option<&dyn CausalLanguageModel> {
        self.inner.as_causal_lm()
    }

    fn as_vision(&self) -> Option<&dyn VisionModel> {
        self.inner.as_vision()
    }
}
