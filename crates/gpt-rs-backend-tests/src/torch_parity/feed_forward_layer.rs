use std::sync::Arc;

use gpt_rs::backend::spec::PortableBackend;
use gpt_rs::module::Layer;
use gpt_rs::nn::layers::{ActivationFunction, FeedForward};
use gpt_rs::tensor::DeviceTensor;
use ActivationFunction::{Gelu, GeluTanh};

use super::common::*;

/// `down_proj(act(up_proj(x)))` against Torch. With `activation_only`, the test compares
/// `act(up_proj(x))`.
fn run_feed_forward_case<B: PortableBackend + 'static>(
    backend: &Arc<B>,
    activation: ActivationFunction,
    (batch, embed_dim, hidden_dim): (usize, usize, usize),
    seed: u64,
    bias: bool,
    input_range: f32,
    activation_only: bool,
) {
    let mut rng = seeded_rng(seed);
    let input = random_vec_range(&mut rng, batch * embed_dim, -input_range, input_range);
    let up = HostLinear::random(&mut rng, embed_dim, hidden_dim, bias);
    let down = HostLinear::random(&mut rng, hidden_dim, embed_dim, bias);

    let expected = timed_torch(|| {
        let input = tch_tensor_from_vec(&[batch, embed_dim], &input);
        let approximate = match activation {
            ActivationFunction::Gelu => "none",
            ActivationFunction::GeluTanh => "tanh",
        };
        let hidden = up.torch(&input).gelu(approximate);
        tensor_to_vec(&if activation_only {
            hidden
        } else {
            down.torch(&hidden)
        })
    });

    let actual = timed_gpt(|| {
        let tensors = [up.tensors("mlp.up_proj"), down.tensors("mlp.down_proj")];
        let layer = load_layer(backend, tensors.concat(), |p| {
            FeedForward::load(p, "mlp", embed_dim, hidden_dim, activation, bias)
        })
        .unwrap();
        let input = device_tensor_from_data(backend, &[batch, embed_dim], &input);
        let output: DeviceTensor<B> = if activation_only {
            let hidden = layer.up_proj.call(&input).unwrap();
            layer.activation.call(&hidden).unwrap()
        } else {
            layer.call(&input).unwrap()
        };
        to_host_vec(&output)
    });

    assert_close(&expected, &actual);
}

pub fn feed_forward_matches_torch_with_bias<B: PortableBackend + 'static>(backend: &Arc<B>) {
    run_feed_forward_case(backend, Gelu, (4, 6, 10), 0xF00D, true, 1.0, false);
}

pub fn feed_forward_matches_torch_without_bias<B: PortableBackend + 'static>(backend: &Arc<B>) {
    run_feed_forward_case(backend, Gelu, (3, 4, 12), 0xCAFE, false, 1.0, false);
}

pub fn feed_forward_state_records_activation<B: PortableBackend + 'static>(backend: &Arc<B>) {
    run_feed_forward_case(backend, Gelu, (2, 5, 7), 0xBEEF, true, 1.0, true);
}

pub fn feed_forward_matches_torch_batch1_embed32_hidden128_bias<B: PortableBackend + 'static>(
    backend: &Arc<B>,
) {
    run_feed_forward_case(backend, Gelu, (1, 32, 128), 0xF010, true, 1.0, false);
}

pub fn feed_forward_matches_torch_batch4_embed64_hidden256_bias<B: PortableBackend + 'static>(
    backend: &Arc<B>,
) {
    run_feed_forward_case(backend, Gelu, (4, 64, 256), 0xF011, true, 1.0, false);
}

pub fn feed_forward_matches_torch_batch7_embed13_hidden31_no_bias<B: PortableBackend + 'static>(
    backend: &Arc<B>,
) {
    run_feed_forward_case(backend, Gelu, (7, 13, 31), 0xF012, false, 1.0, false);
}

pub fn feed_forward_state_records_activation_batch4_embed32_hidden128<
    B: PortableBackend + 'static,
>(
    backend: &Arc<B>,
) {
    run_feed_forward_case(backend, Gelu, (4, 32, 128), 0xF013, true, 1.0, true);
}

pub fn feed_forward_state_records_activation_extreme_inputs<B: PortableBackend + 'static>(
    backend: &Arc<B>,
) {
    run_feed_forward_case(backend, Gelu, (2, 16, 32), 0xF014, true, 10.0, true);
}

pub fn feed_forward_gelu_tanh_matches_torch<B: PortableBackend + 'static>(backend: &Arc<B>) {
    run_feed_forward_case(backend, GeluTanh, (4, 32, 128), 0xF015, true, 1.0, false);
}

pub fn feed_forward_gelu_tanh_activation_extreme_inputs<B: PortableBackend + 'static>(
    backend: &Arc<B>,
) {
    run_feed_forward_case(backend, GeluTanh, (2, 16, 32), 0xF016, true, 10.0, true);
}
