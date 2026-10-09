//! Activations: ReLU, softmax, GELU, SiLU and their relatives.

use crate::{ensure_rank_at_least, functional};

/// Applies ReLU: `max(x, 0)`.
#[functional]
pub fn relu(x: &Tensor) -> Tensor {
    x.maximum(0.0)
}

/// Applies ReLU6: `min(max(x, 0), 6)`.
#[functional]
pub fn relu6(x: &Tensor) -> Tensor {
    x.maximum(0.0).minimum(6.0)
}

/// Computes a numerically stable softmax of `x` along `axis`, which counts from the end when it
/// is negative, as in Torch: `exp(x - max(x)) / sum(exp(x - max(x)))`. Subtracting the maximum
/// keeps the exponentials in range.
#[functional]
pub fn softmax(x: &Tensor, axis: i64) -> Tensor {
    let rank = x.shape().rank() as i64;
    assert!(
        (-rank..rank).contains(&axis),
        "{FUNCTIONAL}: axis {axis} is out of range for x of rank {rank}"
    );
    let exp = (x - x.reduce_max([axis], true)).exp();
    &exp / exp.reduce_sum([axis], true)
}

/// Applies the exact GELU activation `0.5 * x * (1 + erf(x / sqrt(2)))`.
#[functional]
pub fn gelu(x: &Tensor) -> Tensor {
    ensure_rank_at_least!(x, 1);
    0.5 * x * (1.0 + (x / 2.0f32.sqrt()).erf())
}

/// Applies the tanh approximation of GELU,
/// `0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3)))`. This is `gelu_new` in GPT-2 and
/// `gelu(approximate="tanh")` in Torch.
#[functional]
pub fn gelu_tanh(x: &Tensor) -> Tensor {
    ensure_rank_at_least!(x, 1);
    let sqrt_2_over_pi = (2.0f32 / std::f32::consts::PI).sqrt();
    let cubic = x * x * x * 0.044715;
    let inner = (x + cubic) * sqrt_2_over_pi;
    0.5 * x * (1.0 + inner.tanh())
}

/// Applies the SiLU activation (`x * sigmoid(x)`), also known as Swish.
///
/// It computes `x / (1 + exp(-x))` directly. For very negative `x`, `exp(-x)` overflows to
/// infinity and the result flushes to zero. `x = -inf` gives NaN.
#[functional]
pub fn silu(x: &Tensor) -> Tensor {
    ensure_rank_at_least!(x, 1);
    x / (1.0 + (-x).exp())
}

/// Applies the logistic sigmoid `1 / (1 + exp(-x))`.
#[functional]
pub fn sigmoid(x: &Tensor) -> Tensor {
    ensure_rank_at_least!(x, 1);
    (1.0 + (-x).exp()).reciprocal()
}

/// Applies softplus `log(1 + exp(x))` in the overflow-safe form
/// `max(x, 0) + log(1 + exp(-|x|))`.
#[functional]
pub fn softplus(x: &Tensor) -> Tensor {
    ensure_rank_at_least!(x, 1);
    x.maximum(0.0) + (1.0 + (-x.abs()).exp()).log()
}
