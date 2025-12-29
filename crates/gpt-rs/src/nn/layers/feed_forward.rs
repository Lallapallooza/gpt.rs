//! Position-wise feed-forward block used inside transformer layers.
//!
//! Implements the common two-layer MLP with GELU activation and exposes state capturing hooks
//! similar to the other portable NN layers.

use super::linear::Linear;
use crate::backend::spec::PortableBackend;
use crate::module::{Module, ParamVisitor, ParamVisitorMut, TensorRole};
use crate::ops::functional;
use crate::tensor::DeviceTensor;
use anyhow::Result;
use std::sync::Arc;

/// Transformer feed-forward network `Linear -> GELU -> Linear`.
pub struct FeedForward<B: PortableBackend + 'static> {
    backend: Arc<B>,
    pub w_in: Linear<B>,
    pub w_out: Linear<B>,
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
