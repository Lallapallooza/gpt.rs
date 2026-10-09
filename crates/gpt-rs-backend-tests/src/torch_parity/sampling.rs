//! Picking tokens on the device: `argmax`, `top_k`, and the draw of `functional::sample`.

use std::sync::Arc;

use gpt_rs::backend::PortableBackend;
use gpt_rs::gpt_rs_ir::dialects::tensor::state_type;
use gpt_rs::ops::functional;
use gpt_rs::tensor::{DType, DeviceTensor, HostTensor};

use super::common::*;

/// The `si32` elements of `tensor`, widened to `i64` like Torch's indices.
fn indices<B: PortableBackend + 'static>(tensor: &DeviceTensor<B>) -> Vec<i64> {
    let host = tensor.to_host().unwrap();
    host.as_si32()
        .unwrap()
        .iter()
        .map(|&i| i64::from(i))
        .collect()
}

/// Rows with a repeated maximum, whose first one wins, and a row with a NaN, which beats every
/// number.
pub fn argmax_matches_torch<B: PortableBackend + 'static>(backend: &Arc<B>) {
    let mut rng = seeded_rng(0x5A01);
    let (rows, cols) = (3, 37);
    let mut data = random_vec(&mut rng, rows * cols);
    data[5] = 9.0;
    data[20] = 9.0;
    data[cols + 30] = f32::NAN;

    let expected = timed_torch(|| {
        let x = tch_tensor_from_vec(&[rows, cols], &data);
        Vec::<i64>::try_from(x.argmax(-1, false)).unwrap()
    });
    let actual =
        timed_gpt(|| indices(&device_tensor_from_data(backend, &[rows, cols], &data).argmax(-1)));
    assert_eq!(expected, actual);
}

pub fn top_k_matches_torch<B: PortableBackend + 'static>(backend: &Arc<B>) {
    let mut rng = seeded_rng(0x5A02);
    let (rows, cols, k) = (2, 50, 5);
    let data = random_vec(&mut rng, rows * cols);

    let (expected_values, expected_indices) = timed_torch(|| {
        let x = tch_tensor_from_vec(&[rows, cols], &data);
        let (values, indices) = x.topk(k as i64, -1, true, true);
        (
            tensor_to_vec(&values),
            Vec::<i64>::try_from(indices.flatten(0, -1)).unwrap(),
        )
    });
    let (actual_values, actual_indices) = timed_gpt(|| {
        let x = device_tensor_from_data(backend, &[rows, cols], &data);
        let (values, top) = x.top_k(k, -1);
        (to_host_vec(&values), indices(&top))
    });
    assert_eq!(expected_values, actual_values);
    assert_eq!(expected_indices, actual_indices);
}

/// The draw of `functional::sample` against Torch's Gumbel-max on the noise `u` that the backend
/// draws from the same generator state: the scaled logits, cut to the `top_k` largest or not, plus
/// `-log(-log(u))`, and the index of the largest.
pub fn sample_matches_torch_gumbel_max<B: PortableBackend + 'static>(backend: &Arc<B>) {
    let mut rng = seeded_rng(0x5A03);
    let (rows, cols, temperature) = (2, 64, 0.7f32);
    let data = random_vec(&mut rng, rows * cols);
    let state = || {
        let state = HostTensor::from_le_bytes(&state_type(), &[7; 16]).unwrap();
        DeviceTensor::from_host(Arc::clone(backend), state).unwrap()
    };
    for top_k in [Some(5), None] {
        let width = top_k.unwrap_or(cols);
        let noise = to_host_vec(&state().rng_uniform([rows, width], DType::F32).0);
        let expected = timed_torch(|| {
            let scaled = tch_tensor_from_vec(&[rows, cols], &data) / f64::from(temperature);
            let (candidates, columns) = match top_k {
                Some(k) => {
                    let (values, columns) = scaled.topk(k as i64, -1, true, true);
                    (values, Some(columns))
                }
                None => (scaled, None),
            };
            let gumbel = -(-tch_tensor_from_vec(&[rows, width], &noise).log()).log();
            let choice = (candidates + gumbel).argmax(-1, true);
            let token = match columns {
                Some(columns) => columns.gather(-1, &choice, false),
                None => choice,
            };
            Vec::<i64>::try_from(token.flatten(0, -1)).unwrap()
        });
        let actual = timed_gpt(|| {
            let logits = device_tensor_from_data(backend, &[rows, cols], &data);
            let (token, _) = functional::sample(&logits, temperature, top_k, &state());
            indices(&token)
        });
        assert_eq!(expected, actual, "top_k {top_k:?}");
    }
}
