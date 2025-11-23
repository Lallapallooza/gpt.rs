//! Token embedding layer backed by the portable functional embedding lookup.
//!
//! The layer stores the embedding table on the selected backend and provides convenience
//! wrappers for validating index tensors and capturing forward state.

use crate::backend::spec::PortableBackend;
use crate::ops::functional;
use crate::tensor::{DType, DeviceTensor, Tensor};
use anyhow::{bail, ensure, Result};
use std::fmt;
use std::sync::Arc;

/// Embedding layer that maps integer token IDs to dense vectors.
pub struct Embedding<B: PortableBackend + 'static> {
    backend: Arc<B>,
    pub weight: DeviceTensor<B>,
}

/// State bundle returned from [`Embedding::forward_with_state`].
pub struct EmbeddingState<B: PortableBackend + 'static> {
    pub indices: DeviceTensor<B>,
}

/// Placeholder for gradients once the portable backend adds backward support.
pub struct EmbeddingGradients {
    pub weight: Tensor,
}

impl<B: PortableBackend + 'static> Embedding<B> {
    /// Creates a new embedding layer by uploading the weight matrix to the backend.
    pub fn new<W>(backend: Arc<B>, weight: W) -> Result<Self>
    where
        W: crate::tensor::IntoDeviceTensor<B>,
    {
        let weight = weight.into_device_tensor(&backend)?;
        Ok(Self { backend, weight })
    }

    /// Ensures indices are 32-bit integers before launching the lookup program.
    fn validate_indices(&self, indices: &DeviceTensor<B>) -> Result<()> {
        ensure!(
            indices.dtype() == DType::I32,
            "embedding indices must have dtype I32, got {:?}",
            indices.dtype()
        );
        Ok(())
    }

    /// Materialises embeddings for the provided token IDs.
    #[deny(clippy::disallowed_methods, clippy::disallowed_types)]
    ///
    /// The method validates index dtype, opens a profiling scope, and delegates the heavy lifting
    /// to the functional `embedding_lookup` helper.
    pub fn forward(&self, indices: &DeviceTensor<B>) -> Result<DeviceTensor<B>> {
        self.validate_indices(indices)?;
        let _prof_guard = crate::profiling::layer_scope("Embedding::forward");
        functional::embedding_lookup(self.backend.as_ref(), &self.weight, indices)
    }

    /// Runs the embedding lookup and returns the captured indices for potential reuse.
    #[deny(clippy::disallowed_methods, clippy::disallowed_types)]
    pub fn forward_with_state(
        &self,
        indices: &DeviceTensor<B>,
    ) -> Result<(DeviceTensor<B>, EmbeddingState<B>)> {
        self.validate_indices(indices)?;
        let output = self.forward(indices)?;
        Ok((
            output,
            EmbeddingState {
                indices: indices.clone(),
            },
        ))
    }

    /// Placeholder for the backward pass; returns an error until gradients are implemented.
    pub fn backward(
        &self,
        _state: &EmbeddingState<B>,
        _grad_output: &DeviceTensor<B>,
    ) -> Result<EmbeddingGradients> {
        bail!("embedding backward is not available on the portable backend yet")
    }

    /// Returns the backend handle that owns the embedding weights.
    pub fn backend(&self) -> Arc<B> {
        Arc::clone(&self.backend)
    }
}

impl<B: PortableBackend> Clone for Embedding<B> {
    fn clone(&self) -> Self {
        Embedding {
            backend: Arc::clone(&self.backend),
            weight: self.weight.clone(),
        }
    }
}

impl<B: PortableBackend> fmt::Debug for Embedding<B> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Embedding")
            .field("weight", &self.weight)
            .finish()
    }
}
