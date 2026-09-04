//! Pooling kernels implemented via portable graph capture.

use anyhow::{ensure, Result};

use crate::backend::spec::{ReduceKind, ReduceWindowSpec};
use crate::ops::functional::Padding2d;
use crate::{capture, functional};

/// 2D max pooling over NHWC activations (`[N, H, W, C]`).
#[functional]
pub fn max_pool2d(
    x: &Tensor,
    window: [usize; 2],
    stride: [usize; 2],
    padding: Padding2d,
) -> Result<Tensor> {
    let _scope = crate::profiling::functional_scope(
        "gpt_rs::ops::functional::pooling::max_pool2d",
        "reduce_window(max)",
    );
    ensure!(
        x.shape().rank() == 4,
        "{FUNCTIONAL}: x must be a rank-4 NHWC tensor, got {:?}",
        x.shape().dims()
    );
    let spec = ReduceWindowSpec {
        window_dims: vec![1, window[0], window[1], 1],
        strides: vec![1, stride[0], stride[1], 1],
        padding: vec![
            (0, 0),
            (padding.top, padding.bottom),
            (padding.left, padding.right),
            (0, 0),
        ],
        base_dilation: vec![1, 1, 1, 1],
        window_dilation: vec![1, 1, 1, 1],
        reduce: ReduceKind::Max,
        accum_dtype: None,
    };
    capture!(|x| x.reduce_window(spec))
}

/// Global average pooling of NHWC activations (`[N, H, W, C]`) to `[N, C]`.
#[functional]
pub fn global_avg_pool2d(x: &Tensor) -> Result<Tensor> {
    ensure!(
        x.shape().rank() == 4,
        "{FUNCTIONAL}: x must be a rank-4 NHWC tensor, got {:?}",
        x.shape().dims()
    );
    let [n, h, w, c] = [0, 1, 2, 3].map(|axis| x.shape().dims()[axis]);
    let denom = (h * w) as f32;
    capture!(|x| {
        let sum_h = x.reduce_sum(vec![1], true);
        let sum_hw = sum_h.reduce_sum(vec![2], true);
        sum_hw.div_scalar(denom).reshape(vec![n, c])
    })
}
