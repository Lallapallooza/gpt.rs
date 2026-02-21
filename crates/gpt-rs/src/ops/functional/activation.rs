//! Activation kernels implemented via portable graph capture.
//!
//! The routines here build backend programs for common nonlinearities such as softmax and GELU
//! while keeping numerical stability tweaks (e.g., max subtraction) close to the graph capture.

use anyhow::Result;
use gpt_rs_macros::{capture_ptir, ptir_pattern, support_runtime_overload};

use crate::backend::spec::PortableBackend;
use crate::ops::functional::common::{
    ensure_rank_at_least, ensure_same_backend, ensure_same_dtype, ensure_shape_matches,
    CaptureIntoDeviceTensor,
};
use crate::ops::ptir;
use crate::tensor::DeviceTensor;

/// Applies ReLU: `max(x, 0)`.
#[support_runtime_overload]
#[ptir_pattern(target = "gpt_rs.relu_f32")]
pub fn relu<B: PortableBackend + 'static>(
    _backend: &B,
    x: &DeviceTensor<B>,
) -> Result<DeviceTensor<B>> {
    let _scope =
        crate::profiling::functional_scope("gpt_rs::ops::functional::activation::relu", "max0");
    capture_ptir!({ x }, |session| {
        let zero = session.scalar(0.0).broadcast_like(&x);
        let out = x.maximum(&zero);
        Ok(out.id())
    })?
    .into_device_tensor()
}

/// Applies ReLU6: `min(max(x, 0), 6)`.
#[support_runtime_overload]
#[ptir_pattern(target = "gpt_rs.relu6_f32")]
pub fn relu6<B: PortableBackend + 'static>(
    _backend: &B,
    x: &DeviceTensor<B>,
) -> Result<DeviceTensor<B>> {
    let _scope = crate::profiling::functional_scope(
        "gpt_rs::ops::functional::activation::relu6",
        "clamp0_6",
    );
    capture_ptir!({ x }, |session| {
        let zero = session.scalar(0.0).broadcast_like(&x);
        let six = session.scalar(6.0).broadcast_like(&x);
        let max0 = x.maximum(&zero);
        let out = max0.minimum(&six);
        Ok(out.id())
    })?
    .into_device_tensor()
}

struct SoftmaxLastDimPlan {
    axis: usize,
}

fn validate_softmax_last_dim<B: PortableBackend + 'static>(
    x: &DeviceTensor<B>,
) -> Result<SoftmaxLastDimPlan> {
    ensure_rank_at_least("softmax_last_dim input", x, 1)?;
    Ok(SoftmaxLastDimPlan {
        axis: x.shape().rank() - 1,
    })
}

/// Computes a numerically stable softmax over the last dimension of `x`.
///
/// The graph is captured in four phases so callers can reason about the generated program:
/// - subtract each row's maximum via `Reduce(Max)` to keep exponentials in range;
/// - exponentiate the shifted tensor to obtain positive weights;
/// - reduce with `Reduce(Sum)` to form per-row denominators and broadcast them back;
/// - divide the exponentials by the broadcasted sums to produce normalized probabilities.
///
/// Reuses an existing [`GraphArena`] whenever one of the inputs already participates in a lazy graph.
#[support_runtime_overload]
#[ptir_pattern(target = "gpt_rs.softmax_last_dim_f32")]
pub fn softmax_last_dim<B: PortableBackend + 'static>(
    _backend: &B,
    x: &DeviceTensor<B>,
) -> Result<DeviceTensor<B>> {
    let plan = validate_softmax_last_dim(x)?;
    capture_ptir!({ input = x }, |_session| {
        let max = input.reduce_max([plan.axis], true);
        let shifted = input - max.broadcast_like(&input);
        let exp_values = shifted.exp();
        let sum = exp_values.reduce_sum([plan.axis], true);
        Ok((exp_values / sum.broadcast_like(&input)).id())
    })?
    .into_device_tensor()
}

/// Applies the exact GELU activation used in GPT models.
///
/// The captured graph matches the analytical definition
/// `0.5 * x * (1 + erf(x / sqrt(2)))`, combining scalar broadcasts with PTIR
/// elementwise operators to remain backend portable.
#[support_runtime_overload]
#[ptir_pattern(target = "gpt_rs.gelu_f32")]
pub fn gelu<B: PortableBackend + 'static>(
    _backend: &B,
    x: &DeviceTensor<B>,
) -> Result<DeviceTensor<B>> {
    ensure_rank_at_least("gelu input", x, 1)?;

    capture_ptir!({ input = x }, |_session| {
        let half = 0.5f32 * input;
        let scaled = input / ptir::sqrt(2.0f32);
        let erf = ptir::erf(scaled);
        let one_plus = 1.0f32 + erf;
        let out = half * one_plus;
        Ok(out.id())
    })?
    .into_device_tensor()
}

/// Applies the SiLU activation (`x * sigmoid(x)`), also known as Swish.
///
/// The captured graph uses the numerically stable identity
/// `sigmoid(x) = 1 / (1 + exp(-x))`.
#[support_runtime_overload]
#[ptir_pattern(target = "gpt_rs.silu_f32")]
pub fn silu<B: PortableBackend + 'static>(
    _backend: &B,
    x: &DeviceTensor<B>,
) -> Result<DeviceTensor<B>> {
    ensure_rank_at_least("silu input", x, 1)?;

    capture_ptir!({ input = x }, |_session| {
        let neg = input * -1.0f32;
        let exp = neg.exp();
        let denom = 1.0f32 + exp;
        let out = input / denom;
        Ok(out.id())
    })?
    .into_device_tensor()
}

/// Applies SwiGLU gating: `silu(gate) * up`.
///
/// This helper is the activation core used by gated MLP blocks in Mistral/Ministral-like models.
/// Both inputs must have identical shape/dtype/backend placement.
#[support_runtime_overload]
#[ptir_pattern(target = "gpt_rs.swiglu_f32")]
pub fn swiglu<B: PortableBackend + 'static>(
    _backend: &B,
    gate: &DeviceTensor<B>,
    up: &DeviceTensor<B>,
) -> Result<DeviceTensor<B>> {
    ensure_same_backend("swiglu", gate, up)?;
    ensure_same_dtype("swiglu gate", gate, "up", up)?;
    ensure_shape_matches("swiglu gate", gate, "up", up)?;
    ensure_rank_at_least("swiglu gate", gate, 1)?;

    capture_ptir!({ gate, up }, |_session| {
        let neg = gate * -1.0f32;
        let exp = neg.exp();
        let denom = 1.0f32 + exp;
        let silu_gate = gate / denom;
        let out = silu_gate * up;
        Ok(out.id())
    })?
    .into_device_tensor()
}
