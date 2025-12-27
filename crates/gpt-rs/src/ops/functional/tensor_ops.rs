//! Miscellaneous tensor helpers built on top of the lazy graph interface.
//!
//! These routines combine portable broadcasting and elementwise primitives to implement
//! higher-level conveniences like bias addition.

use anyhow::Result;
use gpt_rs_macros::{capture_ptir, ptir_pattern, support_runtime_overload};

use crate::backend::spec::PortableBackend;
use crate::ops::functional::common::{
    ensure_last_dim, ensure_rank, ensure_rank_at_least, ensure_same_backend, ensure_same_dtype,
    CaptureIntoDeviceTensor,
};
use crate::tensor::DeviceTensor;

struct AddBiasPlan {
    output_shape: Vec<usize>,
    requires_grad: bool,
}

/// Validates bias addition operands and captures broadcast metadata.
///
/// Ensures dtype/back-end match, enforces `[*, last_dim]` semantics, and records the axis used for
/// broadcasting. Tested via the backend Torch parity suites (add_bias coverage).
fn validate_add_bias<B: PortableBackend + 'static>(
    x: &DeviceTensor<B>,
    bias: &DeviceTensor<B>,
) -> Result<AddBiasPlan> {
    ensure_same_dtype("add_bias input", x, "bias", bias)?;
    ensure_rank_at_least("add_bias input", x, 1)?;
    ensure_rank("add_bias bias", bias, 1)?;
    let bias_axis = x.shape().rank() - 1;
    ensure_last_dim("add_bias bias", bias, x.shape().dims()[bias_axis])?;
    ensure_same_backend("add_bias", x, bias)?;
    Ok(AddBiasPlan {
        output_shape: x.shape().dims().to_vec(),
        requires_grad: x.requires_grad_flag() || bias.requires_grad_flag(),
    })
}

/// Adds a bias vector to the last dimension of `x`, broadcasting as needed.
/// Gradient tracking is preserved for both the activation tensor and the bias parameter.
/// Steps: import operands, broadcast the bias across non-last axes, emit an elementwise add, and
/// wrap the resulting value identifier into a lazy tensor.
#[support_runtime_overload]
#[ptir_pattern(target = "gpt_rs.add_bias")]
pub fn add_bias<B: PortableBackend + 'static>(
    _backend: &B,
    x: &DeviceTensor<B>,
    bias: &DeviceTensor<B>,
) -> Result<DeviceTensor<B>> {
    let plan = validate_add_bias(x, bias)?;
    capture_ptir!({ x, bias }, |_session| {
        let bias_broadcast = bias.broadcast_to(plan.output_shape.clone());
        let out = x + bias_broadcast;
        Ok(out.id())
    })?
    .into_device_tensor(plan.requires_grad)
}
