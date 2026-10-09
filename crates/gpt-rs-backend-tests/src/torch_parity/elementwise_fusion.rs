//! Parity of elementwise chains in the shapes backends fuse.

use std::sync::Arc;

use gpt_rs::backend::PortableBackend;
use gpt_rs::s;
use gpt_rs::tensor::DType;
use tch::Kind;

use super::common::*;

/// Two slices of one source, at nonzero offsets on both axes, feed one chain.
pub fn two_offset_slices_feed_one_chain_matches_torch<B: PortableBackend + 'static>(
    backend: &Arc<B>,
) {
    let mut rng = seeded_rng(0x2101);
    let values = random_vec_range(&mut rng, 6 * 20, -2.0, 2.0);
    let expected = timed_torch(|| {
        let x = tch_tensor_from_vec(&[6, 20], &values);
        let gate = x.narrow(0, 1, 4).narrow(1, 2, 8);
        let up = x.narrow(0, 2, 4).narrow(1, 11, 8);
        tensor_to_vec(&(&gate * gate.exp() * up).contiguous())
    });
    let actual = timed_gpt(|| {
        let x_d = device_tensor_from_data(backend, &[6, 20], &values);
        let gate = x_d.slice(s![1..5, 2..10]);
        let up = x_d.slice(s![2..6, 11..19]);
        to_host_vec(&(&gate * gate.exp() * up))
    });
    assert_close(&expected, &actual);
}

/// A per-row factor and a per-column weight, each computed at its own shape, broadcast into one
/// chain.
pub fn smaller_producers_broadcast_into_chain_matches_torch<B: PortableBackend + 'static>(
    backend: &Arc<B>,
) {
    let mut rng = seeded_rng(0x2102);
    let (rows, cols) = (5, 24);
    let x = random_vec(&mut rng, rows * cols);
    let row_log_scale = random_vec(&mut rng, rows);
    let weight = random_vec(&mut rng, cols);
    let expected = timed_torch(|| {
        let x_t = tch_tensor_from_vec(&[rows, cols], &x);
        let factor = tch_tensor_from_vec(&[rows, 1], &row_log_scale).exp();
        let gain = tch_tensor_from_vec(&[cols], &weight) + 1.0;
        tensor_to_vec(&(x_t * factor * gain))
    });
    let actual = timed_gpt(|| {
        let x_d = device_tensor_from_data(backend, &[rows, cols], &x);
        let s_d = device_tensor_from_data(backend, &[rows, 1], &row_log_scale);
        let w_d = device_tensor_from_data(backend, &[cols], &weight);
        to_host_vec(&(x_d * s_d.exp() * (w_d + 1.0)))
    });
    assert_close(&expected, &actual);
}

/// A chain that stores bf16 rounds its result to nearest-even.
pub fn chain_with_bf16_store_matches_torch<B: PortableBackend + 'static>(backend: &Arc<B>) {
    let mut rng = seeded_rng(0x2103);
    // Exactly halfway between two bf16 values: 1 + 2^-8 rounds down to 1, 1 + 3 * 2^-8 up to
    // 1 + 2^-6, and likewise for the negatives and a larger exponent.
    let ties = [
        1.0 + 1.0 / 256.0,
        1.0 + 3.0 / 256.0,
        -(1.0 + 1.0 / 256.0),
        -(1.0 + 3.0 / 256.0),
        384.0 + 1.0,
        384.0 + 3.0,
    ];
    let n = 64;
    let mut x = random_vec_range(&mut rng, n, -8.0, 8.0);
    let mut y = random_vec_range(&mut rng, n, -8.0, 8.0);
    let mut z = random_vec_range(&mut rng, n, -8.0, 8.0);
    for (index, tie) in ties.iter().enumerate() {
        (x[index], y[index], z[index]) = (*tie, 1.0, 0.0);
    }
    let expected = timed_torch(|| {
        let t = |v: &[f32]| tch_tensor_from_vec(&[4, n / 4], v);
        let result = (t(&x) * t(&y) + t(&z))
            .to_kind(Kind::BFloat16)
            .to_kind(Kind::Float);
        tensor_to_vec(&result)
    });
    let actual = timed_gpt(|| {
        let d = |v: &[f32]| device_tensor_from_data(backend, &[4, n / 4], v);
        let narrowed = (d(&x) * d(&y) + d(&z)).cast(DType::Bf16);
        assert_eq!(narrowed.dtype(), DType::Bf16);
        to_host_vec(&narrowed.cast(DType::F32))
    });
    assert_close_tol(&expected, &actual, 0.0, 0.0);
}
