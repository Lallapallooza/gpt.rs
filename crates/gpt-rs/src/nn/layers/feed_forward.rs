//! Position-wise feed-forward block used inside transformer layers.
//!
//! Implements the common two-layer MLP with GELU activation and exposes state capturing hooks
//! similar to the other portable NN layers.

use super::linear::{Linear, LinearGradients, LinearState};
use crate::backend::spec::PortableBackend;
use crate::module::{Module, ParamVisitor, ParamVisitorMut, TensorRole};
use crate::ops::functional;
use crate::tensor::DeviceTensor;
use anyhow::{bail, Result};
use std::sync::Arc;

/// Transformer feed-forward network `Linear -> GELU -> Linear`.
pub struct FeedForward<B: PortableBackend + 'static> {
    backend: Arc<B>,
    pub w_in: Linear<B>,
    pub w_out: Linear<B>,
}

/// State captured during [`FeedForward::forward_with_state`].
pub struct FeedForwardState<B: PortableBackend + 'static> {
    pub input: DeviceTensor<B>,
    pub linear1: LinearState<B>,
    pub activation: DeviceTensor<B>,
    pub linear2: LinearState<B>,
}

/// Placeholder for gradients once the portable backend supports them.
pub struct FeedForwardGradients {
    pub w_in: LinearGradients,
    pub w_out: LinearGradients,
}

impl<B: PortableBackend + 'static> FeedForward<B> {
    /// Creates the feed-forward block by wiring input/output linear layers.
    pub fn new<WIn, WOut, BIn, BOut>(
        backend: Arc<B>,
        w_in: WIn,
        w_out: WOut,
        b_in: BIn,
        b_out: BOut,
    ) -> Result<Self>
    where
        WIn: crate::tensor::IntoDeviceTensor<B>,
        WOut: crate::tensor::IntoDeviceTensor<B>,
        BIn: crate::tensor::IntoDeviceTensorOption<B>,
        BOut: crate::tensor::IntoDeviceTensorOption<B>,
    {
        let w_in_layer = Linear::new(Arc::clone(&backend), w_in, b_in)?;
        let w_out_layer = Linear::new(Arc::clone(&backend), w_out, b_out)?;
        Ok(Self {
            backend,
            w_in: w_in_layer,
            w_out: w_out_layer,
        })
    }

    /// Runs the MLP block without capturing intermediate state.
    /// The method profiles the layer, applies the first linear transformation, runs GELU through
    /// the functional graph helper, and finishes with the output projection.
    #[deny(clippy::disallowed_methods, clippy::disallowed_types)]
    pub fn forward(&self, x: &DeviceTensor<B>) -> Result<DeviceTensor<B>> {
        let _prof_guard = crate::profiling::layer_scope("FeedForward::forward");
        let hidden = self.w_in.forward(x)?;
        let activated = functional::gelu(self.backend.as_ref(), &hidden)?;
        self.w_out.forward(&activated)
    }

    /// Runs the block and returns the captured intermediates.
    #[deny(clippy::disallowed_methods, clippy::disallowed_types)]
    pub fn forward_with_state(
        &self,
        x: &DeviceTensor<B>,
    ) -> Result<(DeviceTensor<B>, FeedForwardState<B>)> {
        let _prof_guard = crate::profiling::layer_scope("FeedForward::forward_with_state");
        let (hidden, linear1_state) = self.w_in.forward_with_state(x)?;
        let activated = functional::gelu(self.backend.as_ref(), &hidden)?;
        let (output, linear2_state) = self.w_out.forward_with_state(&activated)?;

        Ok((
            output,
            FeedForwardState {
                input: x.clone(),
                linear1: linear1_state,
                activation: activated,
                linear2: linear2_state,
            },
        ))
    }

    /// Placeholder for the backward pass; returns an error until gradients are implemented.
    pub fn backward(
        &self,
        _state: &FeedForwardState<B>,
        _grad_output: &DeviceTensor<B>,
    ) -> Result<(FeedForwardGradients, DeviceTensor<B>)> {
        bail!("feed-forward backward is not available on the portable backend")
    }

    /// Returns the backend handle that owns the block's parameters.
    pub fn backend(&self) -> Arc<B> {
        Arc::clone(&self.backend)
    }
}

impl<B: PortableBackend + 'static> Module<B> for FeedForward<B> {
    fn visit_params(&self, v: &mut ParamVisitor<'_, B>) -> Result<()> {
        v.param("w_in", TensorRole::Parameter, &self.w_in.weight)?;
        if let Some(bias) = &self.w_in.bias {
            v.param("b_in", TensorRole::Parameter, bias)?;
        }
        v.param("w_out", TensorRole::Parameter, &self.w_out.weight)?;
        if let Some(bias) = &self.w_out.bias {
            v.param("b_out", TensorRole::Parameter, bias)?;
        }
        Ok(())
    }

    fn visit_params_mut(&mut self, v: &mut ParamVisitorMut<'_, B>) -> Result<()> {
        v.param("w_in", TensorRole::Parameter, &mut self.w_in.weight)?;
        if let Some(bias) = &mut self.w_in.bias {
            v.param("b_in", TensorRole::Parameter, bias)?;
        }
        v.param("w_out", TensorRole::Parameter, &mut self.w_out.weight)?;
        if let Some(bias) = &mut self.w_out.bias {
            v.param("b_out", TensorRole::Parameter, bias)?;
        }
        Ok(())
    }
}
