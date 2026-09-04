//! Rotary position embedding (RoPE) for projected head tensors. [`crate::nn::RotaryEmbedding`]
//! provides the cosine and sine tables.

use anyhow::{ensure, Result};

use crate::ops::ptir;
use crate::{
    capture, ensure_dtype, ensure_rank, ensure_same_backend, ensure_same_dtype, ensure_same_shape,
    functional,
};

/// Applies RoPE to the leading `rotary_dim` channels of `x`.
///
/// `x` must have shape `[num_heads, seq_len, head_dim]`. `cos` and `sin` must have shape
/// `[seq_len, rotary_dim / 2]`.
#[functional]
pub fn apply_rope(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    ensure_rank!(x, 3);
    ensure_rank!(cos, 2);
    ensure_rank!(sin, 2);
    ensure_same_backend!(x, cos, sin);
    ensure_same_dtype!(x, cos, sin);
    ensure_same_shape!(cos, sin);
    ensure_dtype!(x, F32);
    let (x_dims, cos_dims) = (x.shape().dims(), cos.shape().dims());
    let (heads, seq, head_dim) = (x_dims[0], x_dims[1], x_dims[2]);
    let half = cos_dims[1];
    let rotary_dim = half * 2;
    ensure!(
        cos_dims[0] == seq,
        "{FUNCTIONAL}: cos sequence length {} must match x sequence length {seq}",
        cos_dims[0]
    );
    ensure!(
        rotary_dim > 0 && rotary_dim <= head_dim,
        "{FUNCTIONAL}: rotary dimension {rotary_dim} must be in (0, head_dim={head_dim}]"
    );

    capture!(|x, cos, sin| {
        let x_rot = x.slice(vec![0, 0, 0], vec![heads, seq, rotary_dim]);
        let x_first = x_rot.slice(vec![0, 0, 0], vec![heads, seq, half]);
        let x_second = x_rot.slice(vec![0, 0, half], vec![heads, seq, half]);

        let cos_b = cos
            .reshape(vec![1, seq, half])
            .broadcast_to(vec![heads, seq, half]);
        let sin_b = sin
            .reshape(vec![1, seq, half])
            .broadcast_to(vec![heads, seq, half]);

        let rotated_first = x_first * cos_b - x_second * sin_b;
        let rotated_second = x_second * cos_b + x_first * sin_b;
        let rotated = ptir::Tensor::concat(2, &[rotated_first, rotated_second]);

        if rotary_dim == head_dim {
            rotated
        } else {
            let passthrough = x.slice(
                vec![0, 0, rotary_dim],
                vec![heads, seq, head_dim - rotary_dim],
            );
            ptir::Tensor::concat(2, &[rotated, passthrough])
        }
    })
}
