//! Contractions over an empty axis, which sum nothing and so give zeros.

use std::sync::Arc;

use gpt_rs::backend::PortableBackend;
use gpt_rs::ops::functional::{self, WeightLayout};

use super::common::*;

fn run_matmul_case<B: PortableBackend + 'static>(
    backend: &Arc<B>,
    lhs_shape: &[usize],
    rhs_shape: &[usize],
) {
    let expected = timed_torch(|| {
        let lhs = tch_tensor_from_vec(lhs_shape, &[]);
        let rhs = tch_tensor_from_vec(rhs_shape, &[]);
        tensor_to_vec(&lhs.matmul(&rhs))
    });

    let actual = timed_gpt(|| {
        let lhs = device_tensor_from_data(backend, lhs_shape, &[]);
        let rhs = device_tensor_from_data(backend, rhs_shape, &[]);
        to_host_vec(&functional::matmul(&lhs, &rhs))
    });

    assert_close(&expected, &actual);
}

pub fn matmul_inner0_matches_torch<B: PortableBackend + 'static>(backend: &Arc<B>) {
    run_matmul_case(backend, &[2, 0], &[0, 3]);
}

pub fn batched_matmul_inner0_matches_torch<B: PortableBackend + 'static>(backend: &Arc<B>) {
    run_matmul_case(backend, &[2, 3, 0], &[2, 0, 4]);
}

pub fn linear_in0_matches_torch<B: PortableBackend + 'static>(backend: &Arc<B>) {
    let expected = timed_torch(|| {
        let x = tch_tensor_from_vec(&[2, 0], &[]);
        let weight = tch_tensor_from_vec(&[3, 0], &[]);
        tensor_to_vec(&x.matmul(&weight.transpose(0, 1)))
    });

    let actual = timed_gpt(|| {
        let x = device_tensor_from_data(backend, &[2, 0], &[]);
        let weight = device_tensor_from_data(backend, &[3, 0], &[]);
        to_host_vec(&functional::linear(&x, &weight, WeightLayout::OutIn))
    });

    assert_close(&expected, &actual);
}
