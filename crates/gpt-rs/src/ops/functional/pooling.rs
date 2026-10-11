//! Pooling over the spatial axes of NHWC activations.

use gpt_rs_ir::dialects::tensor::ReduceWindowMax;

use crate::ops::functional::Padding2d;
use crate::{ensure_rank, functional};

/// 2D max pooling over NHWC activations (`[N, H, W, C]`).
#[functional]
pub fn max_pool2d(
    x: &Tensor,
    window: [usize; 2],
    stride: [usize; 2],
    padding: Padding2d,
) -> Tensor {
    ensure_rank!(x, 4);
    let [height, width] = padding.as_hw_pairs();
    x.reduce_window::<ReduceWindowMax>(
        [1, window[0], window[1], 1],
        [1, stride[0], stride[1], 1],
        [(0, 0), height, width, (0, 0)],
    )
}

/// Global average pooling of NHWC activations (`[N, H, W, C]`) to `[N, C]`.
#[functional]
pub fn global_avg_pool2d(x: &Tensor) -> Tensor {
    ensure_rank!(x, 4);
    let count = x.size(1) * x.size(2);
    x.reduce_sum([1, 2], false) / count as f32
}
