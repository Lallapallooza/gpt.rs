//! The elements that a Python-style index (`s!`) takes, by hand on a tensor that holds 0, 1, 2,
//! ... in row-major order. Torch parity covers the slices that reach other backend kernels.

use std::sync::Arc;

use gpt_rs::s;
use gpt_rs::tensor::{AxisIndex, DeviceTensor, HostTensor};
use gpt_rs_backend_ref_cpu::CpuPortableBackend;

/// The extents and elements of `x[index]`, where `x` of `dims` holds 0, 1, 2, ...
fn take(dims: &[usize], index: impl AsRef<[AxisIndex]>) -> (Vec<usize>, Vec<f32>) {
    let backend = Arc::new(CpuPortableBackend::new());
    let count: usize = dims.iter().product();
    let x = HostTensor::f32(dims, (0..count).map(|i| i as f32).collect()).unwrap();
    let taken = DeviceTensor::from_host(backend, x).unwrap().slice(index);
    let host = taken.to_host().unwrap();
    (host.dims.to_vec(), host.as_f32().unwrap().to_vec())
}

#[test]
fn ellipsis_spans_the_leading_axes_and_a_position_drops_its_axis() {
    // The last element of every row of a [2, 3, 4] tensor.
    assert_eq!(
        take(&[2, 3, 4], s![..., -1]),
        (vec![2, 3], vec![3.0, 7.0, 11.0, 15.0, 19.0, 23.0])
    );
}

#[test]
fn none_adds_an_axis_and_bounds_clamp_to_the_extent() {
    // Column 2 of rows 1 to 3 of a [4, 5] tensor, under a new leading axis.
    assert_eq!(
        take(&[4, 5], s![None, 1..100, 2]),
        (vec![1, 3], vec![7.0, 12.0, 17.0])
    );
}
