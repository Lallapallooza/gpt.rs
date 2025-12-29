use std::sync::Arc;

use gpt_rs::backend::spec::PortableBackend;
use gpt_rs::nn::layers::LayerNorm;
use gpt_rs::ops::functional;
use gpt_rs::tensor::DeviceTensor;
use tch::{Kind, Tensor as TchTensor};

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
        let input_tch = tch_tensor_from_vec(&[batch, embed_dim], input_host.data());
        let gamma_tch = tch_tensor_from_vec(&[embed_dim], gamma_host.data());
        let beta_tch = tch_tensor_from_vec(&[embed_dim], beta_host.data());
        tensor_to_vec(&layer_norm_reference(
            &input_tch, &gamma_tch, &beta_tch, eps,
        ))
    });

    let output_host = timed_gpt(|| {
        let input_device =
            DeviceTensor::from_host(Arc::clone(backend), input_host.clone()).unwrap();
        let layer = LayerNorm::new(
            Arc::clone(backend),
            gamma_host.clone(),
            beta_host.clone(),
            eps as f32,
        )
        .unwrap();
        let output_device = layer.forward(&input_device).unwrap();
        output_device.to_host().unwrap()
    });

    assert_close(&expected, output_host.data());
}

fn run_layer_norm_state_case<B: PortableBackend + 'static>(
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

    let (normalized_vec, mean_vec, inv_std_vec) = timed_torch(|| {
        let input_tch = tch_tensor_from_vec(&[batch, embed_dim], input_host.data());
        let mean = input_tch.mean_dim(-1, true, Kind::Float);
        let var = input_tch.var_dim(-1, false, true);
        let inv_std = (var + eps).rsqrt();
        let normalized = (&input_tch - &mean) * &inv_std;
        (
            tensor_to_vec(&normalized),
            tensor_to_vec(&mean),
            tensor_to_vec(&inv_std),
        )
    });

    let (normalized_host, mean_host, inv_std_host) = timed_gpt(|| {
        let layer = LayerNorm::new(Arc::clone(backend), gamma_host, beta_host, eps as f32).unwrap();
        let input_device = DeviceTensor::from_host(Arc::clone(backend), input_host).unwrap();
        let state = functional::layer_norm(
            backend.as_ref(),
            &input_device,
            &layer.gamma,
            &layer.beta,
            layer.eps,
        )
        .unwrap();
        (
            state.normalized.to_host().unwrap(),
            state.mean.to_host().unwrap(),
            state.inv_std.to_host().unwrap(),
        )
    });

    assert_close(&normalized_vec, normalized_host.data());
    assert_close(&mean_vec, mean_host.data());
    assert_close(&inv_std_vec, inv_std_host.data());
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
        let input_tch = tch_tensor_from_vec(&[batch, embed_dim], input_host.data());
        let gamma_tch = tch_tensor_from_vec(&[embed_dim], gamma_host.data());
        let beta_tch = tch_tensor_from_vec(&[embed_dim], beta_host.data());
        tensor_to_vec(&layer_norm_reference(
            &input_tch, &gamma_tch, &beta_tch, eps,
        ))
    });

    let output_host = timed_gpt(|| {
        let input_device =
            DeviceTensor::from_host(Arc::clone(backend), input_host.clone()).unwrap();
        let layer = LayerNorm::new(
            Arc::clone(backend),
            gamma_host.clone(),
            beta_host.clone(),
            eps as f32,
        )
        .unwrap();
        let output_device = layer.forward(&input_device).unwrap();
        output_device.to_host().unwrap()
    });

    assert_close(&expected, output_host.data());
}

pub fn layer_norm_forward_with_state_matches_moments<B: PortableBackend + 'static>(
    backend: &Arc<B>,
) {
    let mut rng = seeded_rng(0x5EED);
    let batch = 4;
    let embed_dim = 6;
    let eps = 1e-5f64;

    let input_host = tensor_from_vec(&[batch, embed_dim], random_vec(&mut rng, batch * embed_dim));
    let gamma_host = tensor_from_vec(&[embed_dim], random_vec(&mut rng, embed_dim));
    let beta_host = tensor_from_vec(&[embed_dim], random_vec(&mut rng, embed_dim));

    let (normalized_vec, mean_vec, inv_std_vec) = timed_torch(|| {
        let input_tch = tch_tensor_from_vec(&[batch, embed_dim], input_host.data());
        let mean = input_tch.mean_dim(-1, true, Kind::Float);
        let var = input_tch.var_dim(-1, false, true);
        let inv_std = (var + eps).rsqrt();
        let normalized = (&input_tch - &mean) * &inv_std;
        (
            tensor_to_vec(&normalized),
            tensor_to_vec(&mean),
            tensor_to_vec(&inv_std),
        )
    });

    let (normalized_host, mean_host, inv_std_host) = timed_gpt(|| {
        let layer = LayerNorm::new(
            Arc::clone(backend),
            gamma_host.clone(),
            beta_host.clone(),
            eps as f32,
        )
        .unwrap();
        let input_device =
            DeviceTensor::from_host(Arc::clone(backend), input_host.clone()).unwrap();
        let state = functional::layer_norm(
            backend.as_ref(),
            &input_device,
            &layer.gamma,
            &layer.beta,
            layer.eps,
        )
        .unwrap();
        (
            state.normalized.to_host().unwrap(),
            state.mean.to_host().unwrap(),
            state.inv_std.to_host().unwrap(),
        )
    });

    assert_close(&normalized_vec, normalized_host.data());
    assert_close(&mean_vec, mean_host.data());
    assert_close(&inv_std_vec, inv_std_host.data());
}

pub fn layer_norm_forward_propagates_requires_grad<B: PortableBackend + 'static>(backend: &Arc<B>) {
    let mut rng = seeded_rng(0x1234);
    let batch = 2;
    let embed_dim = 3;

    let input = tensor_from_vec(&[batch, embed_dim], random_vec(&mut rng, batch * embed_dim))
        .requires_grad(true);
    let gamma = tensor_from_vec(&[embed_dim], random_vec(&mut rng, embed_dim)).requires_grad(true);
    let beta = tensor_from_vec(&[embed_dim], random_vec(&mut rng, embed_dim)).requires_grad(false);

    timed_gpt(|| {
        let layer = LayerNorm::new(Arc::clone(backend), gamma, beta, 1e-5).unwrap();
        let input_device = DeviceTensor::from_host(Arc::clone(backend), input).unwrap();
        let output = layer.forward(&input_device).unwrap();

        assert!(output.requires_grad_flag());
    });
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

pub fn layer_norm_state_constant_input_matches_moments<B: PortableBackend + 'static>(
    backend: &Arc<B>,
) {
    let input = const_vec(4 * 17, 0.25);
    run_layer_norm_state_case(backend, 4, 17, 1e-5, 0xA123, Some(input));
}
