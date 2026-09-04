//! Stochastic operators such as dropout expressed through portable graph capture.
//!
//! These helpers centralise RNG handling and ensure deterministic masks when driven by the same
//! backend program.

use anyhow::{ensure, Result};

use crate::backend::spec::DType;
use crate::ops::ptir::{self, scalar_broadcast};
use crate::{capture, ensure_dtype, functional};

/// Applies inverted dropout during training by sampling a Bernoulli mask on the backend.
///
/// In evaluation mode, or with `p == 0`, the functional returns the input unchanged.
#[functional]
pub fn dropout(x: &Tensor, p: f32, training: bool) -> Result<Tensor> {
    if p == 0.0 || !training {
        return Ok(x.clone());
    }
    ensure!(
        (0.0..1.0).contains(&p),
        "{FUNCTIONAL}: probability {p} must be in [0, 1)"
    );
    ensure_dtype!(x, F32);
    let keep_prob = 1.0 - p;
    let shape = x.shape().dims().to_vec();
    capture!(session, |x| {
        // 1. Sample a uniform tensor (same shape as `x`).
        let rng = session.rng_uniform(shape.clone(), DType::F32);
        // 2. Compare against the drop probability to form a boolean keep mask.
        let threshold = scalar_broadcast(&session, p, &shape);
        let mask = rng.greater_equal(&threshold);
        // 3. Convert the boolean mask into a scaling tensor by dividing by keep probability.
        let scale_tensor = scalar_broadcast(&session, 1.0 / keep_prob, &shape);
        let zeros = scalar_broadcast(&session, 0.0, &shape);
        let scaled_mask = ptir::Tensor::select(&mask, &scale_tensor, &zeros);
        // 4. Multiply the scaled mask by the input to produce inverted-dropout activations.
        x * scaled_mask
    })
}
