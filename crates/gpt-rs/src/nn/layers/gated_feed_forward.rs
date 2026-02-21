//! Gated feed-forward block for SwiGLU-style transformer MLPs.
//!
//! Computes `down(swiglu(gate(x), up(x)))`, where `gate`, `up`, and `down` are linear layers.

use super::linear::Linear;
use crate::backend::spec::PortableBackend;
use crate::module::{Module, ParamVisitor, ParamVisitorMut, TensorRole};
use crate::ops::functional;
use crate::tensor::DeviceTensor;
use anyhow::Result;
use std::sync::Arc;

/// Transformer gated MLP `Linear(gate) + Linear(up) -> SwiGLU -> Linear(down)`.
pub struct GatedFeedForward<B: PortableBackend + 'static> {
    backend: Arc<B>,
    pub w_gate: Linear<B>,
    pub w_up: Linear<B>,
    pub w_down: Linear<B>,
}

impl<B: PortableBackend + 'static> GatedFeedForward<B> {
    /// Creates a gated feed-forward block from three projections.
    pub fn new<WGate, WUp, WDown, BGate, BUp, BDown>(
        backend: Arc<B>,
        w_gate: WGate,
        w_up: WUp,
        w_down: WDown,
        b_gate: BGate,
        b_up: BUp,
        b_down: BDown,
    ) -> Result<Self>
    where
        WGate: crate::tensor::IntoDeviceTensor<B>,
        WUp: crate::tensor::IntoDeviceTensor<B>,
        WDown: crate::tensor::IntoDeviceTensor<B>,
        BGate: crate::tensor::IntoDeviceTensorOption<B>,
        BUp: crate::tensor::IntoDeviceTensorOption<B>,
        BDown: crate::tensor::IntoDeviceTensorOption<B>,
    {
        let w_gate = Linear::new(Arc::clone(&backend), w_gate, b_gate)?;
        let w_up = Linear::new(Arc::clone(&backend), w_up, b_up)?;
        let w_down = Linear::new(Arc::clone(&backend), w_down, b_down)?;
        Ok(Self {
            backend,
            w_gate,
            w_up,
            w_down,
        })
    }

    /// Runs the gated MLP path and returns the projected output.
    pub fn forward(&self, x: &DeviceTensor<B>) -> Result<DeviceTensor<B>> {
        let _prof_guard = crate::profiling::layer_scope("GatedFeedForward::forward");
        let gate = self.w_gate.forward(x)?;
        let up = self.w_up.forward(x)?;
        let hidden = functional::swiglu(self.backend.as_ref(), &gate, &up)?;
        self.w_down.forward(&hidden)
    }

    /// Returns the backend handle that owns the block's parameters.
    pub fn backend(&self) -> Arc<B> {
        Arc::clone(&self.backend)
    }
}

impl<B: PortableBackend + 'static> Module<B> for GatedFeedForward<B> {
    fn visit_params(&self, v: &mut ParamVisitor<'_, B>) -> Result<()> {
        v.param("w_gate", TensorRole::Parameter, &self.w_gate.weight)?;
        if let Some(bias) = &self.w_gate.bias {
            v.param("b_gate", TensorRole::Parameter, bias)?;
        }
        v.param("w_up", TensorRole::Parameter, &self.w_up.weight)?;
        if let Some(bias) = &self.w_up.bias {
            v.param("b_up", TensorRole::Parameter, bias)?;
        }
        v.param("w_down", TensorRole::Parameter, &self.w_down.weight)?;
        if let Some(bias) = &self.w_down.bias {
            v.param("b_down", TensorRole::Parameter, bias)?;
        }
        Ok(())
    }

    fn visit_params_mut(&mut self, v: &mut ParamVisitorMut<'_, B>) -> Result<()> {
        v.param("w_gate", TensorRole::Parameter, &mut self.w_gate.weight)?;
        if let Some(bias) = &mut self.w_gate.bias {
            v.param("b_gate", TensorRole::Parameter, bias)?;
        }
        v.param("w_up", TensorRole::Parameter, &mut self.w_up.weight)?;
        if let Some(bias) = &mut self.w_up.bias {
            v.param("b_up", TensorRole::Parameter, bias)?;
        }
        v.param("w_down", TensorRole::Parameter, &mut self.w_down.weight)?;
        if let Some(bias) = &mut self.w_down.bias {
            v.param("b_down", TensorRole::Parameter, bias)?;
        }
        Ok(())
    }
}
