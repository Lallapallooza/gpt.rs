//! The MLP of a decoder layer: gated, `down_proj(act(gate_proj(x)) * up_proj(x))` (SwiGLU with
//! SiLU), or plain, `down_proj(act(up_proj(x)))`.

use std::marker::PhantomData;

use super::activation::{Activation, ActivationFunction};
use super::linear::Linear;
use super::names::DecoderNames;
use crate::nn::{self, Params};
use crate::ops::functional::WeightLayout;
use anyhow::Result;

/// How a [`FeedForward`] computes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MlpConfig {
    /// A `gate_proj` whose activation scales the output of `up_proj`.
    pub gated: bool,
    pub act: ActivationFunction,
    /// How the checkpoint stores the projection weights.
    pub weight_layout: WeightLayout,
}

/// The MLP of a decoder layer, with the projection names of the family `N`.
#[nn::module]
pub struct FeedForward<N: DecoderNames> {
    pub gate_proj: Option<Linear>,
    #[module(rename = N::UP_PROJ)]
    pub up_proj: Linear,
    #[module(rename = N::DOWN_PROJ)]
    pub down_proj: Linear,
    pub act: Activation,
    #[module(config)]
    names: PhantomData<N>,
}

#[nn::module]
impl<N: DecoderNames> FeedForward<N> {
    pub fn new(p: &mut Params<B>, config: MlpConfig) -> Result<Self> {
        let layout = config.weight_layout;
        Ok(Self {
            gate_proj: config
                .gated
                .then(|| Linear::with_layout(p, layout))
                .transpose()?,
            up_proj: Linear::with_layout(p, layout)?,
            down_proj: Linear::with_layout(p, layout)?,
            act: config.act.layer(),
            names: PhantomData,
        })
    }

    fn forward(&self, x: &Tensor) -> Tensor {
        let hidden = match self.gate_proj(x) {
            Some(gate) => self.act(&gate) * self.up_proj(x),
            None => self.act(&self.up_proj(x)),
        };
        self.down_proj(&hidden)
    }
}
