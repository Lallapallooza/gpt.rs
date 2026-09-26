use std::sync::Arc;

use gpt_rs::backend::spec::PortableBackend;
use gpt_rs::module::Layer;
use gpt_rs::nn::layers::Conv1D;

use super::common::*;

/// `nn::Conv1D` against `transformers`' `Conv1D` with `weight` `[in_features, out_features]`.
fn run_conv1d_case<B: PortableBackend + 'static>(
    backend: &Arc<B>,
    (batch, in_features, out_features): (usize, usize, usize),
    seed: u64,
) {
    let mut rng = seeded_rng(seed);
    let input = random_vec(&mut rng, batch * in_features);
    let conv = HostConv1D::random(&mut rng, in_features, out_features);

    let expected = timed_torch(|| {
        tensor_to_vec(&conv.torch(&tch_tensor_from_vec(&[batch, in_features], &input)))
    });

    let actual = timed_gpt(|| {
        let layer = load_layer(backend, conv.tensors("proj"), |p| {
            p.scoped("proj", Conv1D::new)
        })
        .unwrap();
        let input = device_tensor_from_data(backend, &[batch, in_features], &input);
        to_host_vec(&layer.call(&input).unwrap())
    });

    assert_close(&expected, &actual);
}

pub fn conv1d_matches_torch<B: PortableBackend + 'static>(backend: &Arc<B>) {
    run_conv1d_case(backend, (4, 5, 3), 0xC1D0);
}

pub fn conv1d_matches_torch_batch1_in32_out96<B: PortableBackend + 'static>(backend: &Arc<B>) {
    run_conv1d_case(backend, (1, 32, 96), 0xC1D1);
}

pub fn conv1d_matches_torch_batch7_in13_out17<B: PortableBackend + 'static>(backend: &Arc<B>) {
    run_conv1d_case(backend, (7, 13, 17), 0xC1D2);
}

pub fn conv1d_matches_torch_square_batch64_in64_out64<B: PortableBackend + 'static>(
    backend: &Arc<B>,
) {
    run_conv1d_case(backend, (64, 64, 64), 0xC1D3);
}
