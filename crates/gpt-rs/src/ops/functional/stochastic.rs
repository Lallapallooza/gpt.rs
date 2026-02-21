//! Stochastic operators such as dropout expressed through portable graph capture.
//!
//! These helpers centralise RNG handling and ensure deterministic masks when driven by the same
//! backend program.

use anyhow::{ensure, Result};
use gpt_rs_macros::{capture_ptir, ptir_pattern, support_runtime_overload};

use crate::backend::spec::PortableBackend;
use crate::ops::functional::common::{
    ensure_dtype_equals, scalar_broadcast, CaptureIntoDeviceTensor,
};
use crate::ops::ptir;
use crate::tensor::{DType as TensorDType, DeviceTensor};

enum DropoutPlan {
    NoOp,
    Apply(DropoutApplyPlan),
}

struct DropoutApplyPlan {
    output_shape: Vec<usize>,
    keep_prob: f32,
    drop_prob: f32,
}

/// Validates dropout inputs and derives reusable metadata.
///
/// Checks keep probability bounds, dtype requirements (only f32 activations today), and whether
/// training is active. Returns a `DropoutPlan` describing the tensor shape alongside the keep prob.
/// Covered by `tests/functional_softmax.rs::dropout_emits_rng_mask_sequence`.
fn validate_dropout<B: PortableBackend + 'static>(
    x: &DeviceTensor<B>,
    p: f32,
    training: bool,
) -> Result<DropoutPlan> {
    if p == 0.0 || !training {
        return Ok(DropoutPlan::NoOp);
    }

    ensure!(
        (0.0..1.0).contains(&p),
        "dropout probability must be in [0, 1)"
    );
    ensure_dtype_equals("dropout input", x, TensorDType::F32)?;

    let keep_prob = 1.0 - p;
    ensure!(keep_prob > 0.0, "dropout keep probability must be positive");

    Ok(DropoutPlan::Apply(DropoutApplyPlan {
        output_shape: x.shape().dims().to_vec(),
        keep_prob,
        drop_prob: p,
    }))
}

/// Applies inverted-dropout during training by sampling a Bernoulli mask on the backend.
/// When evaluation mode is active or the keep probability is one, the original tensor is
/// returned unchanged to avoid unnecessary graph nodes. In training mode the graph performs:
/// - emit a uniform RNG node matching the activation shape;
/// - compare against the drop probability to form a boolean mask;
/// - turn the mask into a scaling tensor that divides by the keep probability;
/// - multiply the input by the scaled mask so expectation stays constant.
/// Coverage: `tests/functional_softmax.rs::dropout_emits_rng_mask_sequence` asserts the captured
/// PTIR graph, while backend parity suites under
/// `crates/gpt-rs-backend-tests/src/torch_parity/` exercise numeric behavior indirectly once wired
/// into training.
#[support_runtime_overload]
#[ptir_pattern(target = "gpt_rs.dropout_f32")]
pub fn dropout<B: PortableBackend + 'static>(
    _backend: &B,
    x: &DeviceTensor<B>,
    p: f32,
    training: bool,
) -> Result<DeviceTensor<B>> {
    let plan = validate_dropout(x, p, training)?;
    match plan {
        DropoutPlan::NoOp => Ok(x.clone()),
        DropoutPlan::Apply(apply) => capture_ptir!({ input = x }, |session| {
            // 1. Sample a uniform tensor (same shape as `input`).
            let rng =
                session.rng_uniform(apply.output_shape.clone(), crate::backend::spec::DType::F32);
            // 2. Compare against the drop probability to form a boolean keep mask.
            let threshold = scalar_broadcast(&session, apply.drop_prob, &apply.output_shape);
            let mask = rng.greater_equal(&threshold);
            // 3. Convert the boolean mask into a scaling tensor by dividing by keep probability.
            let scale_tensor =
                scalar_broadcast(&session, 1.0 / apply.keep_prob, &apply.output_shape);
            let zeros = scalar_broadcast(&session, 0.0, &apply.output_shape);
            let scaled_mask = ptir::Tensor::select(&mask, &scale_tensor, &zeros);
            // 4. Multiply the scaled mask by the input to produce inverted-dropout activations.
            let output = input * scaled_mask;
            Ok(output.id())
        })?
        .into_device_tensor(),
    }
}
