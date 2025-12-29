//! Linear projection layer wrapping the portable functional operations.
//!
//! Provides `forward` helpers that validate tensor shapes, perform matrix multiplication using
//! the device tensor extension trait, and optionally add a bias via the functional helpers.

use crate::backend::spec::PortableBackend;
use crate::module::{Module, ParamVisitor, ParamVisitorMut, TensorRole};
use crate::ops::functional;
use crate::tensor::{DeviceTensor, DeviceTensorOps};
use anyhow::{ensure, Result};
use std::fmt;
use std::sync::Arc;
/// Fully connected layer `y = x W + b` operating on device tensors.
pub struct Linear<B: PortableBackend + 'static> {
    backend: Arc<B>,
    pub weight: DeviceTensor<B>,
    pub bias: Option<DeviceTensor<B>>,
}

impl<B: PortableBackend + 'static> Linear<B> {
    /// Uploads the weight matrix and optional bias to the backend and returns a linear layer.
    pub fn new<W, Bi>(backend: Arc<B>, weight: W, bias: Bi) -> Result<Self>
    where
        W: crate::tensor::IntoDeviceTensor<B>,
        Bi: crate::tensor::IntoDeviceTensorOption<B>,
    {
        let weight = weight.into_device_tensor(&backend)?.as_param()?;
        let weight_dims = weight.shape().dims();
        ensure!(
            weight_dims.len() == 2,
            "linear weight must be 2D, got {:?}",
            weight_dims
        );

        let bias = match bias.into_device_tensor_option(&backend)? {
            Some(bias) => Some(bias.as_param()?),
            None => None,
        };
        Ok(Self {
            backend,
            weight,
            bias,
        })
    }

    /// Runs the linear projection and returns the resulting device tensor.
    /// The method validates input dimensions, performs the matrix multiply via
    /// [`DeviceTensorOps::matmul`], and applies the optional bias using the functional helper.
    #[deny(clippy::disallowed_methods, clippy::disallowed_types)]
    pub fn forward(&self, input: &DeviceTensor<B>) -> Result<DeviceTensor<B>> {
        let _prof_guard = crate::profiling::layer_scope("Linear::forward");
        let dims = input.shape().dims();
        ensure!(
            dims.len() == 2,
            "linear expects 2D input, got shape {:?}",
            dims
        );
        let weight_dims = self.weight.shape().dims();
        ensure!(
            dims[1] == weight_dims[0],
            "input features ({}) must match weight rows ({})",
            dims[1],
            weight_dims[0]
        );

        let mut output_device = input.matmul(&self.weight)?;
        if let Some(bias) = &self.bias {
            output_device = functional::add_bias(self.backend.as_ref(), &output_device, bias)?;
        }

        let requires_grad = input.requires_grad_flag()
            || self.weight.requires_grad_flag()
            || self
                .bias
                .as_ref()
                .map(|b| b.requires_grad_flag())
                .unwrap_or(false);
        Ok(output_device.requires_grad(requires_grad))
    }

    /// Returns the backend that owns the layer parameters.
    pub fn backend(&self) -> Arc<B> {
        Arc::clone(&self.backend)
    }
}

impl<B: PortableBackend> Clone for Linear<B> {
    fn clone(&self) -> Self {
        Linear {
            backend: Arc::clone(&self.backend),
            weight: self.weight.clone(),
            bias: self.bias.clone(),
        }
    }
}

impl<B: PortableBackend> fmt::Debug for Linear<B> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Linear")
            .field("weight", &self.weight)
            .field("bias", &self.bias)
            .finish()
    }
}

impl<B: PortableBackend + 'static> Module<B> for Linear<B> {
    fn visit_params(&self, v: &mut ParamVisitor<'_, B>) -> Result<()> {
        v.param("weight", TensorRole::Parameter, &self.weight)?;
        if let Some(bias) = &self.bias {
            v.param("bias", TensorRole::Parameter, bias)?;
        }
        Ok(())
    }

    fn visit_params_mut(&mut self, v: &mut ParamVisitorMut<'_, B>) -> Result<()> {
        v.param("weight", TensorRole::Parameter, &mut self.weight)?;
        if let Some(bias) = &mut self.bias {
            v.param("bias", TensorRole::Parameter, bias)?;
        }
        Ok(())
    }
}
