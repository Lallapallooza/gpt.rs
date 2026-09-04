//! Activation kernels implemented via portable graph capture.
//!
//! The routines here build backend programs for common nonlinearities such as softmax and GELU
//! while keeping numerical stability tweaks (e.g., max subtraction) close to the graph capture.

use anyhow::Result;

use crate::backend::spec::PortableBackend;
use crate::ops::ptir;
use crate::ops::ptir::scalar_broadcast;
use crate::{
    capture, ensure_rank_at_least, ensure_same_backend, ensure_same_dtype, ensure_same_shape,
    functional,
};

/// Builds a numerically stable softmax over `axis` from PTIR tensor ops, so every softmax emits
/// the same IR pattern.
pub(crate) fn softmax_ptir<'ctx, 'gb, B: PortableBackend + 'static>(
    input: &ptir::Tensor<'ctx, 'gb, B>,
    axis: usize,
) -> ptir::Tensor<'ctx, 'gb, B> {
    let max = input.reduce_max([axis], true);
    let shifted = *input - max.broadcast_like(input);
    let exp_values = shifted.exp();
    let sum = exp_values.reduce_sum([axis], true);
    exp_values / sum.broadcast_like(input)
}

/// Applies ReLU: `max(x, 0)`.
#[functional]
pub fn relu(x: &Tensor) -> Result<Tensor> {
    let _scope =
        crate::profiling::functional_scope("gpt_rs::ops::functional::activation::relu", "max0");
    capture!(session, |x| {
        let zero = session.scalar(0.0).broadcast_like(&x);
        x.maximum(&zero)
    })
}

/// Applies ReLU6: `min(max(x, 0), 6)`.
#[functional]
pub fn relu6(x: &Tensor) -> Result<Tensor> {
    let _scope = crate::profiling::functional_scope(
        "gpt_rs::ops::functional::activation::relu6",
        "clamp0_6",
    );
    capture!(session, |x| {
        let zero = session.scalar(0.0).broadcast_like(&x);
        let six = session.scalar(6.0).broadcast_like(&x);
        let max0 = x.maximum(&zero);
        max0.minimum(&six)
    })
}

/// Computes a numerically stable softmax over the last dimension of `x`.
///
/// The graph is captured in four phases so callers can reason about the generated program:
/// - subtract each row's maximum via `Reduce(Max)` to keep exponentials in range;
/// - exponentiate the shifted tensor to obtain positive weights;
/// - reduce with `Reduce(Sum)` to form per-row denominators and broadcast them back;
/// - divide the exponentials by the broadcasted sums to produce normalized probabilities.
#[functional]
pub fn softmax_last_dim(x: &Tensor) -> Result<Tensor> {
    ensure_rank_at_least!(x, 1);
    let axis = x.shape().rank() - 1;
    capture!(|x| softmax_ptir(&x, axis))
}

/// Applies the exact GELU activation `0.5 * x * (1 + erf(x / sqrt(2)))`.
#[functional]
pub fn gelu(x: &Tensor) -> Result<Tensor> {
    ensure_rank_at_least!(x, 1);
    capture!(|x| {
        let half = 0.5f32 * x;
        let erf = ptir::erf(x / ptir::sqrt(2.0f32));
        half * (1.0f32 + erf)
    })
}

/// Applies the tanh approximation of GELU,
/// `0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3)))`. This is `gelu_new` in GPT-2 and
/// `gelu(approximate="tanh")` in Torch.
#[functional]
pub fn gelu_tanh(x: &Tensor) -> Result<Tensor> {
    ensure_rank_at_least!(x, 1);
    let sqrt_2_over_pi = (2.0f32 / std::f32::consts::PI).sqrt();
    capture!(|x| {
        let cubic = x * x * x * 0.044715f32;
        let inner = (x + cubic) * sqrt_2_over_pi;
        0.5f32 * x * (1.0f32 + inner.tanh())
    })
}

/// Applies the SiLU activation (`x * sigmoid(x)`), also known as Swish.
///
/// The captured graph computes `x / (1 + exp(-x))` directly. For very negative `x`, `exp(-x)`
/// overflows to infinity and the result flushes to zero. `x = -inf` gives NaN.
#[functional]
pub fn silu(x: &Tensor) -> Result<Tensor> {
    ensure_rank_at_least!(x, 1);
    capture!(|x| x / (1.0f32 + (x * -1.0f32).exp()))
}

/// Applies SwiGLU gating, `silu(gate) * up`. This is the activation core of gated MLP blocks.
#[functional]
pub fn swiglu(gate: &Tensor, up: &Tensor) -> Result<Tensor> {
    ensure_same_backend!(gate, up);
    ensure_same_dtype!(gate, up);
    ensure_same_shape!(gate, up);
    ensure_rank_at_least!(gate, 1);
    capture!(|gate, up| {
        let silu_gate = gate / (1.0f32 + (gate * -1.0f32).exp());
        silu_gate * up
    })
}

/// Applies the logistic sigmoid `1 / (1 + exp(-x))`.
#[functional]
pub fn sigmoid(x: &Tensor) -> Result<Tensor> {
    ensure_rank_at_least!(x, 1);
    capture!(|x| (1.0f32 + (x * -1.0f32).exp()).reciprocal())
}

/// Applies softplus `log(1 + exp(x))` in the overflow-safe form
/// `max(x, 0) + log(1 + exp(-|x|))`.
#[functional]
pub fn softplus(x: &Tensor) -> Result<Tensor> {
    ensure_rank_at_least!(x, 1);
    let dims = x.shape().dims().to_vec();
    capture!(session, |x| {
        let positive = x.maximum(&scalar_broadcast(&session, 0.0, &dims));
        let tail = (1.0f32 + (x.abs() * -1.0f32).exp()).log();
        positive + tail
    })
}
