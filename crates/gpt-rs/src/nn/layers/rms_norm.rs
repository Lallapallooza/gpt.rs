//! RMS normalization wrapper backed by the functional portable implementation.
//!
//! RMSNorm uses only a learnable scaling vector (`gamma`) and omits bias terms.

use crate::backend::spec::PortableBackend;
use crate::module::{Module, ParamVisitor, ParamVisitorMut, TensorRole};
use crate::ops::functional;
use crate::tensor::DeviceTensor;
use anyhow::Result;
use std::fmt;
use std::sync::Arc;

/// RMS normalization with learnable scaling parameter.
pub struct RmsNorm<B: PortableBackend + 'static> {
    backend: Arc<B>,
    pub gamma: DeviceTensor<B>,
    pub eps: f32,
}

impl<B: PortableBackend + 'static> RmsNorm<B> {
    /// Uploads the scaling vector and stores epsilon.
    pub fn new<G>(backend: Arc<B>, gamma: G, eps: f32) -> Result<Self>
    where
        G: crate::tensor::IntoDeviceTensor<B>,
    {
        let gamma = gamma.into_device_tensor(&backend)?;
        Ok(Self {
            backend,
            gamma,
            eps,
        })
    }

    /// Applies RMS normalization without returning intermediate tensors.
    #[deny(clippy::disallowed_methods, clippy::disallowed_types)]
    pub fn forward(&self, x: &DeviceTensor<B>) -> Result<DeviceTensor<B>> {
        let _prof_guard = crate::profiling::layer_scope("RmsNorm::forward");
        let outputs = functional::rms_norm(self.backend.as_ref(), x, &self.gamma, self.eps)?;
        Ok(outputs.output)
    }

    /// Returns the backend handle that owns the affine parameters.
    pub fn backend(&self) -> Arc<B> {
        Arc::clone(&self.backend)
    }
}

impl<B: PortableBackend> Clone for RmsNorm<B> {
    fn clone(&self) -> Self {
        RmsNorm {
            backend: Arc::clone(&self.backend),
            gamma: self.gamma.clone(),
            eps: self.eps,
        }
    }
}

impl<B: PortableBackend> fmt::Debug for RmsNorm<B> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RmsNorm")
            .field("gamma", &self.gamma)
            .field("eps", &self.eps)
            .finish()
    }
}

impl<B: PortableBackend + 'static> Module<B> for RmsNorm<B> {
    fn visit_params(&self, v: &mut ParamVisitor<'_, B>) -> Result<()> {
        v.param("gamma", TensorRole::Parameter, &self.gamma)?;
        Ok(())
    }

    fn visit_params_mut(&mut self, v: &mut ParamVisitorMut<'_, B>) -> Result<()> {
        v.param("gamma", TensorRole::Parameter, &mut self.gamma)?;
        Ok(())
    }
}
