use std::sync::Arc;

use gpt_rs::backend::spec::PortableBackend;
use gpt_rs::module::Layer;
use gpt_rs::tensor::DeviceTensor;

use super::common::*;

/// `nn::Linear` against `torch.nn.functional.linear` with `weight` `[out_features, in_features]`.
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
        &[out_features, in_features],
        random_vec(&mut rng, out_features * in_features),
    );
    let bias_host =
        bias.then(|| tensor_from_vec(&[out_features], random_vec(&mut rng, out_features)));

    let expected = timed_torch(|| {
        let input_tch = tch_tensor_from_vec(&[batch, in_features], input_host.data());
        let weight_tch = tch_tensor_from_vec(&[out_features, in_features], weight_host.data());
        let bias_tch = bias_host
            .as_ref()
            .map(|b| tch_tensor_from_vec(&[out_features], b.data()));
        tensor_to_vec(&input_tch.linear(&weight_tch, bias_tch.as_ref()))
    });

    let output_host = timed_gpt(|| {
        let tensors = std::iter::once(("proj.weight".to_string(), weight_host.clone())).chain(
            bias_host
                .clone()
                .map(|bias| ("proj.bias".to_string(), bias)),
        );
        let layer = load_layer(backend, tensors, |p| {
            p.linear("proj", in_features, out_features, bias)
        })
        .unwrap();
        let input_device =
            DeviceTensor::from_host(Arc::clone(backend), input_host.clone()).unwrap();
        let output = layer.call(&input_device).unwrap();
        output.to_host().unwrap()
    });

    assert_close(&expected, output_host.data());
}

pub fn linear_matches_torch_with_bias<B: PortableBackend + 'static>(backend: &Arc<B>) {
    run_linear_case(backend, 4, 5, 3, 0xCE551, true);
}

pub fn linear_matches_torch_without_bias<B: PortableBackend + 'static>(backend: &Arc<B>) {
    run_linear_case(backend, 2, 3, 4, 42, false);
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
        let weight = tensor_from_vec(&[4, 7], vec![0.0; 28]);
        let layer = load_layer(backend, [("proj.weight".to_string(), weight)], |p| {
            p.linear("proj", 7, 4, false)
        })
        .unwrap();
        layer.call(&input).unwrap_err()
    });
    assert!(err.to_string().contains("in_features"), "{err}");
}
