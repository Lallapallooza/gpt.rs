//! Python-style indexing (`s!`) against Torch's, where it reaches the strided and empty slices of
//! the backends: negative bounds, steps, inclusive and empty ranges.

use std::sync::Arc;

use gpt_rs::backend::PortableBackend;
use gpt_rs::s;
use gpt_rs::tensor::AxisIndex;
use tch::Tensor as TchTensor;

use super::common::*;

/// Compares `x[index]` of a random `shape` tensor with Torch's `torch_index(x)`.
fn run_case<B: PortableBackend + 'static, const N: usize>(
    backend: &Arc<B>,
    shape: &[usize],
    index: [AxisIndex; N],
    torch_index: impl Fn(&TchTensor) -> TchTensor,
) {
    let data = random_vec(&mut seeded_rng(7), shape.iter().product());
    let (expected, expected_dims) = timed_torch(|| {
        let taken = torch_index(&tch_tensor_from_vec(shape, &data)).contiguous();
        let dims: Vec<usize> = taken.size().iter().map(|&d| d as usize).collect();
        (tensor_to_vec(&taken), dims)
    });
    let (actual, dims) = timed_gpt(|| {
        let taken = device_tensor_from_data(backend, shape, &data).slice(index);
        let dims = taken.shape().dims().to_vec();
        (to_host_vec(&taken), dims)
    });
    assert_eq!(dims, expected_dims);
    assert_close(&expected, &actual);
}

pub fn negative_bounds_and_steps_match_torch<B: PortableBackend + 'static>(backend: &Arc<B>) {
    run_case(backend, &[2, 5, 6], s![.., -3.., 1..;2], |x| {
        x.slice(1, -3, None, 1).slice(2, 1, None, 2)
    });
}

pub fn inclusive_and_empty_ranges_match_torch<B: PortableBackend + 'static>(backend: &Arc<B>) {
    run_case(backend, &[4, 7], s![1..=2, ..;3], |x| {
        x.slice(0, 1, 3, 1).slice(1, 0, None, 3)
    });
    // A start past the end takes nothing, as Python's `x[3:1]`.
    let (start, end) = (3, 1);
    run_case(backend, &[4, 2], s![start..end], |x| {
        x.slice(0, start, end, 1)
    });
}
