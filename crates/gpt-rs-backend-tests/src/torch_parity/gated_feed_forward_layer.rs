use std::sync::Arc;

use gpt_rs::backend::PortableBackend;
use gpt_rs::module::Layer;
use gpt_rs::nn::layers::{ActivationFunction, FeedForward, LlamaNames, MlpConfig};
use gpt_rs::ops::functional::WeightLayout;

use super::common::*;

/// `down_proj(silu(gate_proj(x)) * up_proj(x))` against Torch.
fn run_gated_feed_forward_case<B: PortableBackend + 'static>(
    backend: &Arc<B>,
    (batch, embed_dim, hidden_dim): (usize, usize, usize),
    seed: u64,
    bias: bool,
) {
    let mut rng = seeded_rng(seed);
    let input = random_vec(&mut rng, batch * embed_dim);
    let gate = HostLinear::random(&mut rng, embed_dim, hidden_dim, bias);
    let up = HostLinear::random(&mut rng, embed_dim, hidden_dim, bias);
    let down = HostLinear::random(&mut rng, hidden_dim, embed_dim, bias);

    let expected = timed_torch(|| {
        let input_tch = tch_tensor_from_vec(&[batch, embed_dim], &input);
        let hidden = gate.torch(&input_tch).silu() * up.torch(&input_tch);
        tensor_to_vec(&down.torch(&hidden))
    });

    let actual = timed_gpt(|| {
        let tensors = [
            gate.tensors("mlp.gate_proj"),
            up.tensors("mlp.up_proj"),
            down.tensors("mlp.down_proj"),
        ];
        let config = MlpConfig {
            gated: true,
            act: ActivationFunction::Silu,
            weight_layout: WeightLayout::OutIn,
        };
        let layer = load_layer(backend, tensors.concat(), |params| {
            params.scoped("mlp", |p| FeedForward::<_, LlamaNames>::new(p, config))
        })
        .unwrap();
        let input_device = device_tensor_from_data(backend, &[batch, embed_dim], &input);
        to_host_vec(&layer.call(&input_device))
    });

    assert_close(&expected, &actual);
}

pub fn gated_feed_forward_matches_torch_with_bias<B: PortableBackend + 'static>(backend: &Arc<B>) {
    run_gated_feed_forward_case(backend, (3, 8, 16), 0x61A1_u64, true);
}

pub fn gated_feed_forward_matches_torch_without_bias<B: PortableBackend + 'static>(
    backend: &Arc<B>,
) {
    run_gated_feed_forward_case(backend, (4, 7, 15), 0x61A2_u64, false);
}

pub fn gated_feed_forward_matches_torch_batch4_embed64_hidden256_bias<
    B: PortableBackend + 'static,
>(
    backend: &Arc<B>,
) {
    run_gated_feed_forward_case(backend, (4, 64, 256), 0x61A3_u64, true);
}
