use std::sync::Arc;

use gpt_rs::backend::PortableBackend;
use gpt_rs::module::Layer;
use gpt_rs::nn::layers::Linear;
use gpt_rs::ops::functional::WeightLayout;

use super::common::*;

/// A [`Linear`] with an `[in_features, out_features]` weight against `transformers`' `Conv1D`.
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
        let layer = load_layer(backend, conv.tensors("proj"), |params| {
            params.scoped("proj", |p| Linear::with_layout(p, WeightLayout::InOut))
        })
        .unwrap();
        let input_device = device_tensor_from_data(backend, &[batch, in_features], &input);
        to_host_vec(&layer.call(&input_device))
    });

    assert_close(&expected, &actual);
}

pub fn conv1d_matches_torch_batch1_in32_out96<B: PortableBackend + 'static>(backend: &Arc<B>) {
    run_conv1d_case(backend, (1, 32, 96), 0xC1D1);
}

pub fn conv1d_matches_torch_batch1_in37_out3105<B: PortableBackend + 'static>(backend: &Arc<B>) {
    run_conv1d_case(backend, (1, 37, 3105), 0xC1D4);
}

pub fn conv1d_matches_torch_batch7_in13_out17<B: PortableBackend + 'static>(backend: &Arc<B>) {
    run_conv1d_case(backend, (7, 13, 17), 0xC1D2);
}

pub fn conv1d_matches_torch_square_batch64_in64_out64<B: PortableBackend + 'static>(
    backend: &Arc<B>,
) {
    run_conv1d_case(backend, (64, 64, 64), 0xC1D3);
}
