//! Rotary position embedding (RoPE) for projected head tensors. [`crate::nn::RotaryEmbedding`]
//! computes the cosine and sine tables.

use crate::tensor::DeviceTensor;
use crate::{ensure_dtype, ensure_rank, ensure_same_dtype, ensure_same_shape, functional, s};

/// Applies RoPE to the leading `rotary_dim` channels of `x`.
///
/// `x` must have shape `[num_heads, seq_len, head_dim]`. `cos` and `sin` must have shape
/// `[seq_len, rotary_dim / 2]`.
#[functional]
pub fn apply_rope(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Tensor {
    ensure_rank!(x, 3);
    ensure_rank!(cos, 2);
    ensure_rank!(sin, 2);
    ensure_same_dtype!(x, cos, sin);
    ensure_same_shape!(cos, sin);
    ensure_dtype!(x, F32);
    let (seq, head_dim) = (x.size(1), x.size(2));
    let half = cos.size(1);
    let rotary_dim = half * 2;
    assert!(
        cos.size(0) == seq,
        "{FUNCTIONAL}: cos sequence length {} must match x sequence length {seq}",
        cos.size(0)
    );
    assert!(
        rotary_dim > 0 && rotary_dim <= head_dim,
        "{FUNCTIONAL}: rotary dimension {rotary_dim} must be in (0, head_dim={head_dim}]"
    );

    let x_first = x.slice(s![.., .., ..half]);
    let x_second = x.slice(s![.., .., half..2 * half]);
    let rotated_first = &x_first * cos - &x_second * sin;
    let rotated_second = x_second * cos + x_first * sin;
    let rotated = DeviceTensor::concat(2, &[&rotated_first, &rotated_second]);
    if rotary_dim == head_dim {
        return rotated;
    }
    let passthrough = x.slice(s![.., .., rotary_dim..]);
    DeviceTensor::concat(2, &[&rotated, &passthrough])
}
