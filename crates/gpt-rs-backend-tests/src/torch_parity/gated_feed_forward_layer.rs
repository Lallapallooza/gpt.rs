use std::sync::Arc;

use gpt_rs::backend::spec::PortableBackend;
use gpt_rs::nn::layers::GatedFeedForward;
use gpt_rs::ops::functional;
use gpt_rs::tensor::DeviceTensor;
use tch::Tensor as TchTensor;

use super::common::*;

fn gated_feed_forward_reference(
    input: &TchTensor,
    w_gate: &TchTensor,
    b_gate: Option<&TchTensor>,
    w_up: &TchTensor,
    b_up: Option<&TchTensor>,
    w_down: &TchTensor,
    b_down: Option<&TchTensor>,
) -> TchTensor {
    let mut gate = input.matmul(w_gate);
    if let Some(bias) = b_gate {
        gate += bias.unsqueeze(0);
    }

    let mut up = input.matmul(w_up);
    if let Some(bias) = b_up {
        up += bias.unsqueeze(0);
    }

    let hidden = gate.silu() * up;
    let mut output = hidden.matmul(w_down);
    if let Some(bias) = b_down {
        output += bias.unsqueeze(0);
    }
    output
}

fn run_gated_feed_forward_case<B: PortableBackend + 'static>(
    backend: &Arc<B>,
    batch: usize,
    embed_dim: usize,
    hidden_dim: usize,
    seed: u64,
    bias: bool,
) {
    let mut rng = seeded_rng(seed);
    let input_host = tensor_from_vec(&[batch, embed_dim], random_vec(&mut rng, batch * embed_dim));
    let w_gate_host = tensor_from_vec(
        &[embed_dim, hidden_dim],
        random_vec(&mut rng, embed_dim * hidden_dim),
    );
    let w_up_host = tensor_from_vec(
        &[embed_dim, hidden_dim],
        random_vec(&mut rng, embed_dim * hidden_dim),
    );
    let w_down_host = tensor_from_vec(
        &[hidden_dim, embed_dim],
        random_vec(&mut rng, hidden_dim * embed_dim),
    );

    let b_gate_host = if bias {
        Some(tensor_from_vec(
            &[hidden_dim],
            random_vec(&mut rng, hidden_dim),
        ))
    } else {
        None
    };
    let b_up_host = if bias {
        Some(tensor_from_vec(
            &[hidden_dim],
            random_vec(&mut rng, hidden_dim),
        ))
    } else {
        None
    };
    let b_down_host = if bias {
        Some(tensor_from_vec(
            &[embed_dim],
            random_vec(&mut rng, embed_dim),
        ))
    } else {
        None
    };

    let expected = timed_torch(|| {
        let input_tch = tch_tensor_from_vec(&[batch, embed_dim], input_host.data());
        let w_gate_tch = tch_tensor_from_vec(&[embed_dim, hidden_dim], w_gate_host.data());
        let w_up_tch = tch_tensor_from_vec(&[embed_dim, hidden_dim], w_up_host.data());
        let w_down_tch = tch_tensor_from_vec(&[hidden_dim, embed_dim], w_down_host.data());
        let b_gate_tch = b_gate_host
            .as_ref()
            .map(|b| tch_tensor_from_vec(&[hidden_dim], b.data()));
        let b_up_tch = b_up_host
            .as_ref()
            .map(|b| tch_tensor_from_vec(&[hidden_dim], b.data()));
        let b_down_tch = b_down_host
            .as_ref()
            .map(|b| tch_tensor_from_vec(&[embed_dim], b.data()));

        tensor_to_vec(&gated_feed_forward_reference(
            &input_tch,
            &w_gate_tch,
            b_gate_tch.as_ref(),
            &w_up_tch,
            b_up_tch.as_ref(),
            &w_down_tch,
            b_down_tch.as_ref(),
        ))
    });

    let output_host = timed_gpt(|| {
        let layer = GatedFeedForward::new(
            Arc::clone(backend),
            w_gate_host.clone(),
            w_up_host.clone(),
            w_down_host.clone(),
            b_gate_host.clone(),
            b_up_host.clone(),
            b_down_host.clone(),
        )
        .unwrap();
        let input_device =
            DeviceTensor::from_host(Arc::clone(backend), input_host.clone()).unwrap();
        let output_device = layer.forward(&input_device).unwrap();
        output_device.to_host().unwrap()
    });

    assert_close(&expected, output_host.data());
}

pub fn gated_feed_forward_matches_torch_with_bias<B: PortableBackend + 'static>(backend: &Arc<B>) {
    run_gated_feed_forward_case(backend, 3, 8, 16, 0x61A1_u64, true);
}

pub fn gated_feed_forward_matches_torch_without_bias<B: PortableBackend + 'static>(
    backend: &Arc<B>,
) {
    run_gated_feed_forward_case(backend, 4, 7, 15, 0x61A2_u64, false);
}

pub fn gated_feed_forward_matches_torch_batch4_embed64_hidden256_bias<
    B: PortableBackend + 'static,
>(
    backend: &Arc<B>,
) {
    run_gated_feed_forward_case(backend, 4, 64, 256, 0x61A3_u64, true);
}

pub fn gated_feed_forward_state_records_swiglu_hidden<B: PortableBackend + 'static>(
    backend: &Arc<B>,
) {
    let mut rng = seeded_rng(0x61A4_u64);
    let batch = 2usize;
    let embed_dim = 5usize;
    let hidden_dim = 11usize;

    let input_host = tensor_from_vec(&[batch, embed_dim], random_vec(&mut rng, batch * embed_dim));
    let w_gate_host = tensor_from_vec(
        &[embed_dim, hidden_dim],
        random_vec(&mut rng, embed_dim * hidden_dim),
    );
    let w_up_host = tensor_from_vec(
        &[embed_dim, hidden_dim],
        random_vec(&mut rng, embed_dim * hidden_dim),
    );
    let w_down_host = tensor_from_vec(
        &[hidden_dim, embed_dim],
        random_vec(&mut rng, hidden_dim * embed_dim),
    );
    let b_gate_host = tensor_from_vec(&[hidden_dim], random_vec(&mut rng, hidden_dim));
    let b_up_host = tensor_from_vec(&[hidden_dim], random_vec(&mut rng, hidden_dim));
    let b_down_host = tensor_from_vec(&[embed_dim], random_vec(&mut rng, embed_dim));

    let expected = timed_torch(|| {
        let input_tch = tch_tensor_from_vec(&[batch, embed_dim], input_host.data());
        let w_gate_tch = tch_tensor_from_vec(&[embed_dim, hidden_dim], w_gate_host.data());
        let w_up_tch = tch_tensor_from_vec(&[embed_dim, hidden_dim], w_up_host.data());
        let b_gate_tch = tch_tensor_from_vec(&[hidden_dim], b_gate_host.data());
        let b_up_tch = tch_tensor_from_vec(&[hidden_dim], b_up_host.data());
        let gate = input_tch.matmul(&w_gate_tch) + b_gate_tch.unsqueeze(0);
        let up = input_tch.matmul(&w_up_tch) + b_up_tch.unsqueeze(0);
        tensor_to_vec(&(gate.silu() * up))
    });

    let actual = timed_gpt(|| {
        let layer = GatedFeedForward::new(
            Arc::clone(backend),
            w_gate_host.clone(),
            w_up_host.clone(),
            w_down_host.clone(),
            Some(b_gate_host.clone()),
            Some(b_up_host.clone()),
            Some(b_down_host.clone()),
        )
        .unwrap();
        let input_device =
            DeviceTensor::from_host(Arc::clone(backend), input_host.clone()).unwrap();
        let gate = layer.w_gate.forward(&input_device).unwrap();
        let up = layer.w_up.forward(&input_device).unwrap();
        let hidden = functional::swiglu(backend.as_ref(), &gate, &up).unwrap();
        hidden.to_host().unwrap().data().to_vec()
    });

    assert_close(&expected, &actual);
}
