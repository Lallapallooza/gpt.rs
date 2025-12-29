use std::sync::Arc;

use gpt_rs::backend::spec::PortableBackend;
use gpt_rs::nn::layers::Linear;
use gpt_rs::tensor::{DeviceTensor, Tensor};
use tch::Tensor as TchTensor;

use super::common::*;

fn linear_reference(input: &TchTensor, weight: &TchTensor, bias: Option<&TchTensor>) -> TchTensor {
    let output = input.matmul(weight);
    match bias {
        Some(b) => output + b.unsqueeze(0),
        None => output,
    }
}

fn run_linear_case<B: PortableBackend + 'static>(
    backend: &Arc<B>,
    batch: usize,
    in_features: usize,
    out_features: usize,
    seed: u64,
    bias: bool,
) {
    let mut rng = seeded_rng(seed);
    let input_host = tensor_from_vec(
        &[batch, in_features],
        random_vec(&mut rng, batch * in_features),
    );
    let weight_host = tensor_from_vec(
        &[in_features, out_features],
        random_vec(&mut rng, in_features * out_features),
    );
    let bias_host = if bias {
        Some(tensor_from_vec(
            &[out_features],
            random_vec(&mut rng, out_features),
        ))
    } else {
        None
    };

    let expected = timed_torch(|| {
        let input_tch = tch_tensor_from_vec(&[batch, in_features], input_host.data());
        let weight_tch = tch_tensor_from_vec(&[in_features, out_features], weight_host.data());
        let bias_tch = bias_host
            .as_ref()
            .map(|b| tch_tensor_from_vec(&[out_features], b.data()));
        tensor_to_vec(&linear_reference(
            &input_tch,
            &weight_tch,
            bias_tch.as_ref(),
        ))
    });

    let output_host = timed_gpt(|| {
        let layer =
            Linear::new(Arc::clone(backend), weight_host.clone(), bias_host.clone()).unwrap();
        let input_device =
            DeviceTensor::from_host(Arc::clone(backend), input_host.clone()).unwrap();
        let output = layer.forward(&input_device).unwrap();
        output.to_host().unwrap()
    });

    assert_close(&expected, output_host.data());
}

pub fn linear_matches_torch_with_bias<B: PortableBackend + 'static>(backend: &Arc<B>) {
    let mut rng = seeded_rng(0xCE551u64);
    let batch = 4;
    let in_features = 5;
    let out_features = 3;

    let input_host = tensor_from_vec(
        &[batch, in_features],
        random_vec(&mut rng, batch * in_features),
    );
    let weight_host = tensor_from_vec(
        &[in_features, out_features],
        random_vec(&mut rng, in_features * out_features),
    );
    let bias_host = tensor_from_vec(&[out_features], random_vec(&mut rng, out_features));

    let expected = timed_torch(|| {
        let input_tch = tch_tensor_from_vec(&[batch, in_features], input_host.data());
        let weight_tch = tch_tensor_from_vec(&[in_features, out_features], weight_host.data());
        let bias_tch = tch_tensor_from_vec(&[out_features], bias_host.data());
        tensor_to_vec(&linear_reference(&input_tch, &weight_tch, Some(&bias_tch)))
    });

    let output_host = timed_gpt(|| {
        let layer = Linear::new(
            Arc::clone(backend),
            weight_host.clone(),
            Some(bias_host.clone()),
        )
        .unwrap();
        let input_device =
            DeviceTensor::from_host(Arc::clone(backend), input_host.clone()).unwrap();
        let output = layer.forward(&input_device).unwrap();
        output.to_host().unwrap()
    });

    assert_close(&expected, output_host.data());
}

pub fn linear_matches_torch_without_bias<B: PortableBackend + 'static>(backend: &Arc<B>) {
    let mut rng = seeded_rng(42);
    let batch = 2;
    let in_features = 3;
    let out_features = 4;

    let input_host = tensor_from_vec(
        &[batch, in_features],
        random_vec(&mut rng, batch * in_features),
    );
    let weight_host = tensor_from_vec(
        &[in_features, out_features],
        random_vec(&mut rng, in_features * out_features),
    );

    let expected = timed_torch(|| {
        let input_tch = tch_tensor_from_vec(&[batch, in_features], input_host.data());
        let weight_tch = tch_tensor_from_vec(&[in_features, out_features], weight_host.data());
        tensor_to_vec(&linear_reference(&input_tch, &weight_tch, None))
    });

    let output_host = timed_gpt(|| {
        let layer = Linear::new(
            Arc::clone(backend),
            weight_host.clone(),
            Option::<Tensor>::None,
        )
        .unwrap();
        let input_device =
            DeviceTensor::from_host(Arc::clone(backend), input_host.clone()).unwrap();
        let output = layer.forward(&input_device).unwrap();
        output.to_host().unwrap()
    });

    assert_close(&expected, output_host.data());
}

pub fn linear_matches_torch_batch1_in5_out3_bias<B: PortableBackend + 'static>(backend: &Arc<B>) {
    run_linear_case(backend, 1, 5, 3, 0xCE552, true);
}

pub fn linear_matches_torch_batch64_in64_out64_bias<B: PortableBackend + 'static>(
    backend: &Arc<B>,
) {
    run_linear_case(backend, 64, 64, 64, 0xCE553, true);
}

pub fn linear_matches_torch_batch7_in13_out17_bias<B: PortableBackend + 'static>(backend: &Arc<B>) {
    run_linear_case(backend, 7, 13, 17, 0xCE554, true);
}

pub fn linear_matches_torch_batch4_in1_out1_bias<B: PortableBackend + 'static>(backend: &Arc<B>) {
    run_linear_case(backend, 4, 1, 1, 0xCE555, true);
}

pub fn linear_matches_torch_batch4_in33_out65_no_bias<B: PortableBackend + 'static>(
    backend: &Arc<B>,
) {
    run_linear_case(backend, 4, 33, 65, 0xCE556, false);
}

pub fn linear_rejects_input_dim_mismatch<B: PortableBackend + 'static>(backend: &Arc<B>) {
    let err = timed_gpt(|| {
        let input = device_tensor_from_data(backend, &[2, 8], &[0.0; 16]);
        let weight = tensor_from_vec(&[7, 4], vec![0.0; 28]);
        let layer = Linear::new(Arc::clone(backend), weight, Option::<Tensor>::None).unwrap();
        layer.forward(&input).unwrap_err()
    });
    assert!(err.to_string().contains("input features"));
}
