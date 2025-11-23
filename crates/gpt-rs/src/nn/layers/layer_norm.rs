//! Layer normalization wrapper backed by the functional portable implementation.
//!
//! Stores affine parameters on the backend and hands off the heavy lifting to portable
//! `layer_norm`, returning convenient state objects for reuse.

use crate::backend::spec::PortableBackend;
use crate::ops::functional;
use crate::tensor::{DeviceTensor, Shape, Tensor};
use anyhow::{bail, Result};
use std::fmt;
use std::sync::Arc;

/// Layer normalization with learnable `gamma` and `beta` parameters.
pub struct LayerNorm<B: PortableBackend + 'static> {
    backend: Arc<B>,
    pub gamma: DeviceTensor<B>,
    pub beta: DeviceTensor<B>,
    pub eps: f32,
}

/// State captured during [`LayerNorm::forward_with_state`].
pub struct LayerNormState<B: PortableBackend + 'static> {
    pub normalized: DeviceTensor<B>,
    pub mean: DeviceTensor<B>,
    pub inv_std: DeviceTensor<B>,
    pub input_shape: Shape,
}

/// Placeholder gradient bundle until portable backward support arrives.
pub struct LayerNormGradients {
    pub gamma: Tensor,
    pub beta: Tensor,
    pub input: Tensor,
}

impl<B: PortableBackend + 'static> LayerNorm<B> {
    /// Uploads the affine parameters to the backend and stores the epsilon value.
    pub fn new<G, T>(backend: Arc<B>, gamma: G, beta: T, eps: f32) -> Result<Self>
    where
        G: crate::tensor::IntoDeviceTensor<B>,
        T: crate::tensor::IntoDeviceTensor<B>,
    {
        let gamma = gamma.into_device_tensor(&backend)?;
        let beta = beta.into_device_tensor(&backend)?;
        Ok(Self {
            backend,
            gamma,
            beta,
            eps,
        })
    }

    /// Applies layer normalization without returning intermediate tensors.
    #[deny(clippy::disallowed_methods, clippy::disallowed_types)]
    pub fn forward(&self, x: &DeviceTensor<B>) -> Result<DeviceTensor<B>> {
        let _prof_guard = crate::profiling::layer_scope("LayerNorm::forward");
        let outputs =
            functional::layer_norm(self.backend.as_ref(), x, &self.gamma, &self.beta, self.eps)?;
        Ok(outputs.output)
    }

    /// Applies layer normalization and captures mean/variance buffers for reuse.
    #[deny(clippy::disallowed_methods, clippy::disallowed_types)]
    pub fn forward_with_state(
        &self,
        x: &DeviceTensor<B>,
    ) -> Result<(DeviceTensor<B>, LayerNormState<B>)> {
        let _prof_guard = crate::profiling::layer_scope("LayerNorm::forward_with_state");
        let functional::LayerNormResult {
            output,
            normalized,
            mean,
            inv_std,
        } = functional::layer_norm(self.backend.as_ref(), x, &self.gamma, &self.beta, self.eps)?;

        Ok((
            output,
            LayerNormState {
                normalized,
                mean,
                inv_std,
                input_shape: x.shape().clone(),
            },
        ))
    }

    /// Placeholder backward entry point; returns an error until gradients are implemented.
    pub fn backward(
        &self,
        _state: &LayerNormState<B>,
        _grad_output: &DeviceTensor<B>,
    ) -> Result<LayerNormGradients> {
        bail!("layer norm backward is not available on the portable backend yet")
    }

    /// Returns the backend handle that owns the affine parameters.
    pub fn backend(&self) -> Arc<B> {
        Arc::clone(&self.backend)
    }
}

impl<B: PortableBackend> Clone for LayerNorm<B> {
    fn clone(&self) -> Self {
        LayerNorm {
            backend: Arc::clone(&self.backend),
            gamma: self.gamma.clone(),
            beta: self.beta.clone(),
            eps: self.eps,
        }
    }
}

impl<B: PortableBackend> fmt::Debug for LayerNorm<B> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LayerNorm")
            .field("gamma", &self.gamma)
            .field("beta", &self.beta)
            .field("eps", &self.eps)
            .finish()
    }
}
