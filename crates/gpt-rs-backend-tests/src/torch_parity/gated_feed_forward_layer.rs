use std::sync::Arc;

use gpt_rs::backend::spec::PortableBackend;
use gpt_rs::module::Layer;
use gpt_rs::nn::layers::GatedFeedForward;
use gpt_rs::ops::functional;
use gpt_rs::tensor::DeviceTensor;

use super::common::*;

/// `down_proj(silu(gate_proj(x)) * up_proj(x))` against Torch. With `hidden_only`, the test
/// compares the SwiGLU hidden state before `down_proj`.
fn run_gated_feed_forward_case<B: PortableBackend + 'static>(
    backend: &Arc<B>,
    (batch, embed_dim, hidden_dim): (usize, usize, usize),
    seed: u64,
    bias: bool,
    hidden_only: bool,
) {
    let mut rng = seeded_rng(seed);
    let input = random_vec(&mut rng, batch * embed_dim);
    let gate = HostLinear::random(&mut rng, embed_dim, hidden_dim, bias);
    let up = HostLinear::random(&mut rng, embed_dim, hidden_dim, bias);
    let down = HostLinear::random(&mut rng, hidden_dim, embed_dim, bias);

    let expected = timed_torch(|| {
        let input = tch_tensor_from_vec(&[batch, embed_dim], &input);
        let hidden = gate.torch(&input).silu() * up.torch(&input);
        tensor_to_vec(&if hidden_only {
            hidden
        } else {
            down.torch(&hidden)
        })
    });

    let actual = timed_gpt(|| {
        let tensors = [
            gate.tensors("mlp.gate_proj"),
            up.tensors("mlp.up_proj"),
            down.tensors("mlp.down_proj"),
        ];
        let layer = load_layer(backend, tensors.concat(), |p| {
            GatedFeedForward::load(p, "mlp", embed_dim, hidden_dim, bias)
        })
        .unwrap();
        let input = device_tensor_from_data(backend, &[batch, embed_dim], &input);
        let output: DeviceTensor<B> = if hidden_only {
            let gate = layer.gate_proj.call(&input).unwrap();
            let up = layer.up_proj.call(&input).unwrap();
            functional::swiglu(&gate, &up).unwrap()
        } else {
            layer.call(&input).unwrap()
        };
        to_host_vec(&output)
    });

    assert_close(&expected, &actual);
}

pub fn gated_feed_forward_matches_torch_with_bias<B: PortableBackend + 'static>(backend: &Arc<B>) {
    run_gated_feed_forward_case(backend, (3, 8, 16), 0x61A1_u64, true, false);
}

pub fn gated_feed_forward_matches_torch_without_bias<B: PortableBackend + 'static>(
    backend: &Arc<B>,
) {
    run_gated_feed_forward_case(backend, (4, 7, 15), 0x61A2_u64, false, false);
}

pub fn gated_feed_forward_matches_torch_batch4_embed64_hidden256_bias<
    B: PortableBackend + 'static,
>(
    backend: &Arc<B>,
) {
    run_gated_feed_forward_case(backend, (4, 64, 256), 0x61A3_u64, true, false);
}

pub fn gated_feed_forward_state_records_swiglu_hidden<B: PortableBackend + 'static>(
    backend: &Arc<B>,
) {
    run_gated_feed_forward_case(backend, (2, 5, 11), 0x61A4_u64, true, true);
}
