use std::sync::Arc;

use gpt_rs::backend::spec::PortableBackend;
use gpt_rs::nn::layers::RmsNorm;
use gpt_rs::tensor::{DeviceTensor, Shape, Tensor};
use tch::{Kind, Tensor as TchTensor};

use super::common::*;

fn rms_norm_reference(input: &TchTensor, gamma: &TchTensor, eps: f64) -> TchTensor {
    let shape = input.size();
    let rank = shape.len();
    let last_axis = (rank as i64) - 1;
    let feature_dim = *shape.last().expect("input rank >= 1");

    let mean_square = input.square().mean_dim(&[last_axis][..], true, Kind::Float);
    let inv_rms = (&mean_square + eps).rsqrt();

    let mut gamma_shape = vec![1i64; rank];
    gamma_shape[rank - 1] = feature_dim;
    let gamma_broadcast = gamma.reshape(gamma_shape).expand(shape.clone(), true);
    input * inv_rms.expand(shape, true) * gamma_broadcast
}

fn run_rms_norm_case<B: PortableBackend + 'static>(
    backend: &Arc<B>,
    shape: &[usize],
    eps: f64,
    seed: u64,
    input_override: Option<Vec<f32>>,
) {
    let mut rng = seeded_rng(seed);
    let len: usize = shape.iter().product();
    let feature_dim = *shape.last().expect("shape rank >= 1");

    let input_vec = input_override.unwrap_or_else(|| random_vec(&mut rng, len));
    let gamma_vec = random_vec(&mut rng, feature_dim);

    let input_host = Tensor::from_vec(Shape::new(shape.to_vec()), input_vec).unwrap();
    let gamma_host = tensor_from_vec(&[feature_dim], gamma_vec);

    let expected = timed_torch(|| {
        let input_tch = tch_tensor_from_vec(shape, input_host.data());
        let gamma_tch = tch_tensor_from_vec(&[feature_dim], gamma_host.data());
        tensor_to_vec(&rms_norm_reference(&input_tch, &gamma_tch, eps))
    });

    let output_host = timed_gpt(|| {
        let layer = RmsNorm::new(Arc::clone(backend), gamma_host.clone(), eps as f32).unwrap();
        let input_device =
            DeviceTensor::from_host(Arc::clone(backend), input_host.clone()).unwrap();
        let output_device = layer.forward(&input_device).unwrap();
        output_device.to_host().unwrap()
    });

    assert_close(&expected, output_host.data());
}

pub fn rms_norm_layer_matches_torch_basic<B: PortableBackend + 'static>(backend: &Arc<B>) {
    run_rms_norm_case(backend, &[4, 17], 1e-5, 0xCA71_u64, None);
}

pub fn rms_norm_layer_matches_torch_seq_batch<B: PortableBackend + 'static>(backend: &Arc<B>) {
    run_rms_norm_case(backend, &[2, 5, 32], 1e-5, 0xCA72_u64, None);
}

pub fn rms_norm_layer_matches_torch_constant_input<B: PortableBackend + 'static>(backend: &Arc<B>) {
    let shape = [3usize, 7usize, 19usize];
    let len: usize = shape.iter().product();
    run_rms_norm_case(backend, &shape, 1e-5, 0xCA73_u64, Some(const_vec(len, 0.5)));
}
