//! Pooling kernels implemented via portable graph capture.

use anyhow::{ensure, Result};
use gpt_rs_macros::{capture_ptir, ptir_pattern, support_runtime_overload};

use crate::backend::spec::{PortableBackend, ReduceKind, ReduceWindowSpec};
use crate::ops::functional::common::CaptureIntoDeviceTensor;
use crate::ops::functional::Padding2d;
use crate::tensor::DeviceTensor;

/// Layout-agnostic max-pooling entrypoint.
#[support_runtime_overload]
#[ptir_pattern(target = "gpt_rs.max_pool2d")]
pub fn max_pool2d<B: PortableBackend + 'static>(
    _backend: &B,
    x: &DeviceTensor<B>,
    window: [usize; 2],
    stride: [usize; 2],
    padding: Padding2d,
) -> Result<DeviceTensor<B>> {
    let _scope = crate::profiling::functional_scope(
        "gpt_rs::ops::functional::pooling::max_pool2d",
        "reduce_window(max)",
    );
    ensure!(
        x.shape().rank() == 4,
        "max_pool2d expects rank-4 NHWC input, got {:?}",
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

    capture_ptir!({ x }, |_session| {
        let out = x.reduce_window(spec);
        Ok(out.id())
    })?
    .into_device_tensor()
}

#[support_runtime_overload]
#[ptir_pattern(target = "gpt_rs.max_pool2d_nhwc")]
pub fn max_pool2d_nhwc<B: PortableBackend + 'static>(
    _backend: &B,
    x: &DeviceTensor<B>,
    window: [usize; 2],
    stride: [usize; 2],
    padding: Padding2d,
) -> Result<DeviceTensor<B>> {
    let _scope = crate::profiling::functional_scope(
        "gpt_rs::ops::functional::pooling::max_pool2d_nhwc",
        "reduce_window(max)",
    );
    ensure!(
        x.shape().rank() == 4,
        "max_pool2d_nhwc expects rank-4 NHWC input, got {:?}",
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

    capture_ptir!({ x }, |_session| {
        let out = x.reduce_window(spec);
        Ok(out.id())
    })?
    .into_device_tensor()
}
