use std::sync::Arc;

use gpt_rs::backend::PortableBackend;
use gpt_rs::module::Layer;
use gpt_rs::tensor::DeviceTensor;
use tch::Tensor as TchTensor;

use super::common::*;

fn layer_norm_reference(
    input: &TchTensor,
    gamma: &TchTensor,
    beta: &TchTensor,
    eps: f64,
) -> TchTensor {
    input.layer_norm([input.size()[1]], Some(gamma), Some(beta), eps, false)
}

fn run_layer_norm_case<B: PortableBackend + 'static>(
    backend: &Arc<B>,
    batch: usize,
    embed_dim: usize,
    eps: f64,
    seed: u64,
    input_override: Option<Vec<f32>>,
) {
    let mut rng = seeded_rng(seed);
    let input_vec = input_override.unwrap_or_else(|| random_vec(&mut rng, batch * embed_dim));
    let gamma_vec = random_vec(&mut rng, embed_dim);
    let beta_vec = random_vec(&mut rng, embed_dim);

    let input_host = tensor_from_vec(&[batch, embed_dim], input_vec);
    let gamma_host = tensor_from_vec(&[embed_dim], gamma_vec);
    let beta_host = tensor_from_vec(&[embed_dim], beta_vec);

    let expected = timed_torch(|| {
        let input_tch = tch_tensor_from_vec(&[batch, embed_dim], input_host.as_f32().unwrap());
        let gamma_tch = tch_tensor_from_vec(&[embed_dim], gamma_host.as_f32().unwrap());
        let beta_tch = tch_tensor_from_vec(&[embed_dim], beta_host.as_f32().unwrap());
        tensor_to_vec(&layer_norm_reference(
            &input_tch, &gamma_tch, &beta_tch, eps,
        ))
    });

    let output_host = timed_gpt(|| {
        let input_device =
            DeviceTensor::from_host(Arc::clone(backend), input_host.clone()).unwrap();
        let layer = layer_norm_layer(backend, gamma_host.clone(), beta_host.clone(), eps as f32);
        let output_device = layer.call(&input_device);
        output_device.to_host().unwrap()
    });

    assert_close(&expected, output_host.as_f32().unwrap());
}

pub fn layer_norm_matches_torch_basic<B: PortableBackend + 'static>(backend: &Arc<B>) {
    let mut rng = seeded_rng(0xA11CE);
    let batch = 3;
    let embed_dim = 5;
    let eps = 1e-5f64;

    let input_host = tensor_from_vec(&[batch, embed_dim], random_vec(&mut rng, batch * embed_dim));
    let gamma_host = tensor_from_vec(&[embed_dim], random_vec(&mut rng, embed_dim));
    let beta_host = tensor_from_vec(&[embed_dim], random_vec(&mut rng, embed_dim));

    let expected = timed_torch(|| {
        let input_tch = tch_tensor_from_vec(&[batch, embed_dim], input_host.as_f32().unwrap());
        let gamma_tch = tch_tensor_from_vec(&[embed_dim], gamma_host.as_f32().unwrap());
        let beta_tch = tch_tensor_from_vec(&[embed_dim], beta_host.as_f32().unwrap());
        tensor_to_vec(&layer_norm_reference(
            &input_tch, &gamma_tch, &beta_tch, eps,
        ))
    });

    let output_host = timed_gpt(|| {
        let input_device =
            DeviceTensor::from_host(Arc::clone(backend), input_host.clone()).unwrap();
        let layer = layer_norm_layer(backend, gamma_host.clone(), beta_host.clone(), eps as f32);
        let output_device = layer.call(&input_device);
        output_device.to_host().unwrap()
    });

    assert_close(&expected, output_host.as_f32().unwrap());
}

pub fn layer_norm_matches_torch_embed_dim1<B: PortableBackend + 'static>(backend: &Arc<B>) {
    run_layer_norm_case(backend, 8, 1, 1e-5, 0xA11D, None);
}

pub fn layer_norm_matches_torch_prime_embed<B: PortableBackend + 'static>(backend: &Arc<B>) {
    run_layer_norm_case(backend, 7, 13, 1e-5, 0xA11E, None);
}

pub fn layer_norm_matches_torch_large_embed<B: PortableBackend + 'static>(backend: &Arc<B>) {
    run_layer_norm_case(backend, 4, 256, 1e-5, 0xA11F, None);
}

pub fn layer_norm_matches_torch_constant_input<B: PortableBackend + 'static>(backend: &Arc<B>) {
    let input = const_vec(8 * 17, 0.5);
    run_layer_norm_case(backend, 8, 17, 1e-5, 0xA120, Some(input));
}

pub fn layer_norm_matches_torch_eps_1e3<B: PortableBackend + 'static>(backend: &Arc<B>) {
    run_layer_norm_case(backend, 8, 17, 1e-3, 0xA121, None);
}

pub fn layer_norm_matches_torch_eps_1e1<B: PortableBackend + 'static>(backend: &Arc<B>) {
    run_layer_norm_case(backend, 8, 17, 1e-1, 0xA122, None);
}
