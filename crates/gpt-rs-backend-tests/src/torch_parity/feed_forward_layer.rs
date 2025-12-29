use std::sync::Arc;

use gpt_rs::backend::spec::PortableBackend;
use gpt_rs::nn::layers::FeedForward;
use gpt_rs::ops::functional;
use gpt_rs::tensor::{DeviceTensor, Tensor};
use tch::Tensor as TchTensor;

use super::common::*;

fn feed_forward_reference(
    input: &TchTensor,
    w_in: &TchTensor,
    b_in: Option<&TchTensor>,
    w_out: &TchTensor,
    b_out: Option<&TchTensor>,
) -> TchTensor {
    let mut hidden = input.matmul(w_in);
    if let Some(bias) = b_in {
        hidden += bias.unsqueeze(0);
    }
    let hidden = hidden.gelu("none");
    let mut output = hidden.matmul(w_out);
    if let Some(bias) = b_out {
        output += bias.unsqueeze(0);
    }
    output
}

fn run_feed_forward_case<B: PortableBackend + 'static>(
    backend: &Arc<B>,
    batch: usize,
    embed_dim: usize,
    hidden_dim: usize,
    seed: u64,
    bias: bool,
) {
    let mut rng = seeded_rng(seed);
    let input_host = tensor_from_vec(&[batch, embed_dim], random_vec(&mut rng, batch * embed_dim));
    let w_in_host = tensor_from_vec(
        &[embed_dim, hidden_dim],
        random_vec(&mut rng, embed_dim * hidden_dim),
    );
    let w_out_host = tensor_from_vec(
        &[hidden_dim, embed_dim],
        random_vec(&mut rng, hidden_dim * embed_dim),
    );
    let b_in_host = if bias {
        Some(tensor_from_vec(
            &[hidden_dim],
            random_vec(&mut rng, hidden_dim),
        ))
    } else {
        None
    };
    let b_out_host = if bias {
        Some(tensor_from_vec(
            &[embed_dim],
            random_vec(&mut rng, embed_dim),
        ))
    } else {
        None
    };

    let expected = timed_torch(|| {
        let input_tch = tch_tensor_from_vec(&[batch, embed_dim], input_host.data());
        let w_in_tch = tch_tensor_from_vec(&[embed_dim, hidden_dim], w_in_host.data());
        let w_out_tch = tch_tensor_from_vec(&[hidden_dim, embed_dim], w_out_host.data());
        let b_in_tch = b_in_host
            .as_ref()
            .map(|b| tch_tensor_from_vec(&[hidden_dim], b.data()));
        let b_out_tch = b_out_host
            .as_ref()
            .map(|b| tch_tensor_from_vec(&[embed_dim], b.data()));
        tensor_to_vec(&feed_forward_reference(
            &input_tch,
            &w_in_tch,
            b_in_tch.as_ref(),
            &w_out_tch,
            b_out_tch.as_ref(),
        ))
    });

    let output_host = timed_gpt(|| {
        let layer = FeedForward::new(
            Arc::clone(backend),
            w_in_host.clone(),
            w_out_host.clone(),
            b_in_host.clone(),
            b_out_host.clone(),
        )
        .unwrap();
        let input_device =
            DeviceTensor::from_host(Arc::clone(backend), input_host.clone()).unwrap();
        let output_device = layer.forward(&input_device).unwrap();
        output_device.to_host().unwrap()
    });

    assert_close(&expected, output_host.data());
}

pub fn feed_forward_matches_torch_with_bias<B: PortableBackend + 'static>(backend: &Arc<B>) {
    let mut rng = seeded_rng(0xF00D);
    let batch = 4;
    let embed_dim = 6;
    let hidden_dim = 10;

    let input_host = tensor_from_vec(&[batch, embed_dim], random_vec(&mut rng, batch * embed_dim));
    let w_in_host = tensor_from_vec(
        &[embed_dim, hidden_dim],
        random_vec(&mut rng, embed_dim * hidden_dim),
    );
    let b_in_host = tensor_from_vec(&[hidden_dim], random_vec(&mut rng, hidden_dim));
    let w_out_host = tensor_from_vec(
        &[hidden_dim, embed_dim],
        random_vec(&mut rng, hidden_dim * embed_dim),
    );
    let b_out_host = tensor_from_vec(&[embed_dim], random_vec(&mut rng, embed_dim));

    let expected = timed_torch(|| {
        let input_tch = tch_tensor_from_vec(&[batch, embed_dim], input_host.data());
        let w_in_tch = tch_tensor_from_vec(&[embed_dim, hidden_dim], w_in_host.data());
        let b_in_tch = tch_tensor_from_vec(&[hidden_dim], b_in_host.data());
        let w_out_tch = tch_tensor_from_vec(&[hidden_dim, embed_dim], w_out_host.data());
        let b_out_tch = tch_tensor_from_vec(&[embed_dim], b_out_host.data());
        tensor_to_vec(&feed_forward_reference(
            &input_tch,
            &w_in_tch,
            Some(&b_in_tch),
            &w_out_tch,
            Some(&b_out_tch),
        ))
    });

    let output_host = timed_gpt(|| {
        let layer = FeedForward::new(
            Arc::clone(backend),
            w_in_host.clone(),
            w_out_host.clone(),
            Some(b_in_host.clone()),
            Some(b_out_host.clone()),
        )
        .unwrap();
        let input_device =
            DeviceTensor::from_host(Arc::clone(backend), input_host.clone()).unwrap();
        let output_device = layer.forward(&input_device).unwrap();
        output_device.to_host().unwrap()
    });

    assert_close(&expected, output_host.data());
}

pub fn feed_forward_matches_torch_without_bias<B: PortableBackend + 'static>(backend: &Arc<B>) {
    let mut rng = seeded_rng(0xCAFE);
    let batch = 3;
    let embed_dim = 4;
    let hidden_dim = 12;

    let input_host = tensor_from_vec(&[batch, embed_dim], random_vec(&mut rng, batch * embed_dim));
    let w_in_host = tensor_from_vec(
        &[embed_dim, hidden_dim],
        random_vec(&mut rng, embed_dim * hidden_dim),
    );
    let w_out_host = tensor_from_vec(
        &[hidden_dim, embed_dim],
        random_vec(&mut rng, hidden_dim * embed_dim),
    );

    let expected = timed_torch(|| {
        let input_tch = tch_tensor_from_vec(&[batch, embed_dim], input_host.data());
        let w_in_tch = tch_tensor_from_vec(&[embed_dim, hidden_dim], w_in_host.data());
        let w_out_tch = tch_tensor_from_vec(&[hidden_dim, embed_dim], w_out_host.data());
        tensor_to_vec(&feed_forward_reference(
            &input_tch, &w_in_tch, None, &w_out_tch, None,
        ))
    });

    let output_host = timed_gpt(|| {
        let layer = FeedForward::new(
            Arc::clone(backend),
            w_in_host.clone(),
            w_out_host.clone(),
            Option::<Tensor>::None,
            Option::<Tensor>::None,
        )
        .unwrap();
        let input_device =
            DeviceTensor::from_host(Arc::clone(backend), input_host.clone()).unwrap();
        let output_device = layer.forward(&input_device).unwrap();
        output_device.to_host().unwrap()
    });

    assert_close(&expected, output_host.data());
}

pub fn feed_forward_state_records_activation<B: PortableBackend + 'static>(backend: &Arc<B>) {
    let mut rng = seeded_rng(0xBEEF);
    let batch = 2;
    let embed_dim = 5;
    let hidden_dim = 7;

    let input_host = tensor_from_vec(&[batch, embed_dim], random_vec(&mut rng, batch * embed_dim));
    let w_in_host = tensor_from_vec(
        &[embed_dim, hidden_dim],
        random_vec(&mut rng, embed_dim * hidden_dim),
    );
    let b_in_host = tensor_from_vec(&[hidden_dim], random_vec(&mut rng, hidden_dim));
    let w_out_host = tensor_from_vec(
        &[hidden_dim, embed_dim],
        random_vec(&mut rng, hidden_dim * embed_dim),
    );
    let b_out_host = tensor_from_vec(&[embed_dim], random_vec(&mut rng, embed_dim));

    let expected = timed_torch(|| {
        let input_tch = tch_tensor_from_vec(&[batch, embed_dim], input_host.data());
        let w_in_tch = tch_tensor_from_vec(&[embed_dim, hidden_dim], w_in_host.data());
        let b_in_tch = tch_tensor_from_vec(&[hidden_dim], b_in_host.data());
        let hidden = input_tch.matmul(&w_in_tch) + b_in_tch.unsqueeze(0);
        tensor_to_vec(&hidden.gelu("none"))
    });

    let activation_host = timed_gpt(|| {
        let layer = FeedForward::new(
            Arc::clone(backend),
            w_in_host.clone(),
            w_out_host.clone(),
            Some(b_in_host.clone()),
            Some(b_out_host.clone()),
        )
        .unwrap();
        let input_device =
            DeviceTensor::from_host(Arc::clone(backend), input_host.clone()).unwrap();
        let hidden = layer.w_in.forward(&input_device).unwrap();
        let activation = functional::gelu(backend.as_ref(), &hidden).unwrap();
        activation.to_host().unwrap()
    });

    assert_close(&expected, activation_host.data());
}

pub fn feed_forward_matches_torch_batch1_embed32_hidden128_bias<B: PortableBackend + 'static>(
    backend: &Arc<B>,
) {
    run_feed_forward_case(backend, 1, 32, 128, 0xF010, true);
}

pub fn feed_forward_matches_torch_batch4_embed64_hidden256_bias<B: PortableBackend + 'static>(
    backend: &Arc<B>,
) {
    run_feed_forward_case(backend, 4, 64, 256, 0xF011, true);
}

pub fn feed_forward_matches_torch_batch7_embed13_hidden31_no_bias<B: PortableBackend + 'static>(
    backend: &Arc<B>,
) {
    run_feed_forward_case(backend, 7, 13, 31, 0xF012, false);
}

pub fn feed_forward_state_records_activation_batch4_embed32_hidden128<
    B: PortableBackend + 'static,
>(
    backend: &Arc<B>,
) {
    let mut rng = seeded_rng(0xF013);
    let batch = 4;
    let embed_dim = 32;
    let hidden_dim = 128;

    let input_host = tensor_from_vec(&[batch, embed_dim], random_vec(&mut rng, batch * embed_dim));
    let w_in_host = tensor_from_vec(
        &[embed_dim, hidden_dim],
        random_vec(&mut rng, embed_dim * hidden_dim),
    );
    let b_in_host = tensor_from_vec(&[hidden_dim], random_vec(&mut rng, hidden_dim));

    let expected = timed_torch(|| {
        let input_tch = tch_tensor_from_vec(&[batch, embed_dim], input_host.data());
        let w_in_tch = tch_tensor_from_vec(&[embed_dim, hidden_dim], w_in_host.data());
        let b_in_tch = tch_tensor_from_vec(&[hidden_dim], b_in_host.data());
        let hidden = input_tch.matmul(&w_in_tch) + b_in_tch.unsqueeze(0);
        tensor_to_vec(&hidden.gelu("none"))
    });

    let activation_host = timed_gpt(|| {
        let w_out_host = tensor_from_vec(
            &[hidden_dim, embed_dim],
            random_vec(&mut rng, hidden_dim * embed_dim),
        );
        let b_out_host = tensor_from_vec(&[embed_dim], random_vec(&mut rng, embed_dim));
        let layer = FeedForward::new(
            Arc::clone(backend),
            w_in_host.clone(),
            w_out_host,
            Some(b_in_host.clone()),
            Some(b_out_host),
        )
        .unwrap();
        let input_device =
            DeviceTensor::from_host(Arc::clone(backend), input_host.clone()).unwrap();
        let hidden = layer.w_in.forward(&input_device).unwrap();
        let activation = functional::gelu(backend.as_ref(), &hidden).unwrap();
        activation.to_host().unwrap()
    });

    assert_close(&expected, activation_host.data());
}

pub fn feed_forward_state_records_activation_extreme_inputs<B: PortableBackend + 'static>(
    backend: &Arc<B>,
) {
    let mut rng = seeded_rng(0xF014);
    let batch = 2;
    let embed_dim = 16;
    let hidden_dim = 32;

    let input_host = tensor_from_vec(
        &[batch, embed_dim],
        random_vec_range(&mut rng, batch * embed_dim, -10.0, 10.0),
    );
    let w_in_host = tensor_from_vec(
        &[embed_dim, hidden_dim],
        random_vec(&mut rng, embed_dim * hidden_dim),
    );
    let b_in_host = tensor_from_vec(&[hidden_dim], random_vec(&mut rng, hidden_dim));

    let expected = timed_torch(|| {
        let input_tch = tch_tensor_from_vec(&[batch, embed_dim], input_host.data());
        let w_in_tch = tch_tensor_from_vec(&[embed_dim, hidden_dim], w_in_host.data());
        let b_in_tch = tch_tensor_from_vec(&[hidden_dim], b_in_host.data());
        let hidden = input_tch.matmul(&w_in_tch) + b_in_tch.unsqueeze(0);
        tensor_to_vec(&hidden.gelu("none"))
    });

    let activation_host = timed_gpt(|| {
        let w_out_host = tensor_from_vec(
            &[hidden_dim, embed_dim],
            random_vec(&mut rng, hidden_dim * embed_dim),
        );
        let b_out_host = tensor_from_vec(&[embed_dim], random_vec(&mut rng, embed_dim));
        let layer = FeedForward::new(
            Arc::clone(backend),
            w_in_host.clone(),
            w_out_host,
            Some(b_in_host.clone()),
            Some(b_out_host),
        )
        .unwrap();
        let input_device =
            DeviceTensor::from_host(Arc::clone(backend), input_host.clone()).unwrap();
        let hidden = layer.w_in.forward(&input_device).unwrap();
        let activation = functional::gelu(backend.as_ref(), &hidden).unwrap();
        activation.to_host().unwrap()
    });

    assert_close(&expected, activation_host.data());
}
