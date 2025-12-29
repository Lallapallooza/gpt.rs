use std::sync::Arc;

use gpt_rs::backend::spec::PortableBackend;
use gpt_rs::nn::layers::{AttentionConfig, CausalSelfAttention};
use gpt_rs::ops::functional;
use gpt_rs::tensor::{DeviceTensor, Tensor};
use tch::{Kind, Tensor as TchTensor};

use super::common::*;

fn attention_reference(
    input: &TchTensor,
    w_qkv: &TchTensor,
    b_qkv: Option<&TchTensor>,
    w_out: &TchTensor,
    b_out: Option<&TchTensor>,
    config: &AttentionConfig,
) -> TchTensor {
    let seq_len = input.size()[0];
    let embed_dim = input.size()[1];
    let qkv = if let Some(bias) = b_qkv {
        input.matmul(w_qkv) + bias.unsqueeze(0)
    } else {
        input.matmul(w_qkv)
    };
    let q_proj_dim = config.query_projection_dim() as i64;
    let kv_proj_dim = config.key_value_projection_dim() as i64;
    let q = qkv.narrow(1, 0, q_proj_dim);
    let k = qkv.narrow(1, q_proj_dim, kv_proj_dim);
    let v = qkv.narrow(1, q_proj_dim + kv_proj_dim, kv_proj_dim);

    let num_heads = config.num_heads() as i64;
    let kv_heads = config.num_key_value_heads as i64;
    let head_dim = q_proj_dim / num_heads;
    let kv_head_dim = kv_proj_dim / kv_heads;
    assert_eq!(head_dim, kv_head_dim, "reference assumes shared head dim");
    let group_size = num_heads / kv_heads;
    let q = q.reshape([seq_len, num_heads, head_dim]).transpose(0, 1);
    let k = k
        .reshape([seq_len, kv_heads, kv_head_dim])
        .transpose(0, 1)
        .repeat_interleave_self_int(group_size, 0, None::<i64>);
    let v = v
        .reshape([seq_len, kv_heads, kv_head_dim])
        .transpose(0, 1)
        .repeat_interleave_self_int(group_size, 0, None::<i64>);

    let scale = (head_dim as f64).sqrt();
    let scores = q.matmul(&k.transpose(-2, -1)) / scale;
    let causal_mask = TchTensor::ones(scores.size().as_slice(), (Kind::Bool, scores.device()))
        .tril(0)
        .logical_not();
    let masked = scores.masked_fill(&causal_mask, f64::NEG_INFINITY);
    let attn = masked.softmax(-1, Kind::Float);
    let context = attn
        .matmul(&v)
        .transpose(0, 1)
        .reshape([seq_len, embed_dim]);

    let mut output = context.matmul(w_out);
    if let Some(bias) = b_out {
        output += bias.unsqueeze(0);
    }
    output
}

fn build_attention_layer<B: PortableBackend + 'static>(
    backend: &Arc<B>,
    config: AttentionConfig,
    w_qkv: Tensor,
    w_out: Tensor,
    b_qkv: Option<Tensor>,
    b_out: Option<Tensor>,
) -> CausalSelfAttention<B> {
    CausalSelfAttention::new(Arc::clone(backend), config, w_qkv, w_out, b_qkv, b_out).unwrap()
}

fn run_mha_case<B: PortableBackend + 'static>(
    backend: &Arc<B>,
    seq_len: usize,
    embed_dim: usize,
    num_heads: usize,
    num_kv_heads: usize,
    seed: u64,
    bias: bool,
) {
    let mut rng = seeded_rng(seed);
    let config = if num_heads == num_kv_heads {
        AttentionConfig::with_equal_heads(embed_dim, num_heads)
    } else {
        AttentionConfig::with_kv(embed_dim, num_heads, num_kv_heads)
    };
    let total_dim = config.total_projection_dim();

    let input_host = tensor_from_vec(
        &[seq_len, embed_dim],
        random_vec(&mut rng, seq_len * embed_dim),
    );
    let w_qkv_host = tensor_from_vec(
        &[embed_dim, total_dim],
        random_vec(&mut rng, embed_dim * total_dim),
    );
    let w_out_host = tensor_from_vec(
        &[embed_dim, embed_dim],
        random_vec(&mut rng, embed_dim * embed_dim),
    );
    let b_qkv_host = if bias {
        Some(tensor_from_vec(
            &[total_dim],
            random_vec(&mut rng, total_dim),
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
        let input_tch = tch_tensor_from_vec(&[seq_len, embed_dim], input_host.data());
        let w_qkv_tch = tch_tensor_from_vec(&[embed_dim, total_dim], w_qkv_host.data());
        let w_out_tch = tch_tensor_from_vec(&[embed_dim, embed_dim], w_out_host.data());
        let b_qkv_tch = b_qkv_host
            .as_ref()
            .map(|b| tch_tensor_from_vec(&[total_dim], b.data()));
        let b_out_tch = b_out_host
            .as_ref()
            .map(|b| tch_tensor_from_vec(&[embed_dim], b.data()));
        tensor_to_vec(&attention_reference(
            &input_tch,
            &w_qkv_tch,
            b_qkv_tch.as_ref(),
            &w_out_tch,
            b_out_tch.as_ref(),
            &config,
        ))
    });

    let output_host = timed_gpt(|| {
        let layer = build_attention_layer(
            backend,
            config.clone(),
            w_qkv_host.clone(),
            w_out_host.clone(),
            b_qkv_host.clone(),
            b_out_host.clone(),
        );

        let input_device =
            DeviceTensor::from_host(Arc::clone(backend), input_host.clone()).unwrap();
        let output_device = layer.forward(&input_device).unwrap();
        output_device.to_host().unwrap()
    });

    assert_close(&expected, output_host.data());
}

fn run_mha_prefill_decode_case<B: PortableBackend + 'static>(
    backend: &Arc<B>,
    prefill_len: usize,
    decode_steps: usize,
    embed_dim: usize,
    num_heads: usize,
    num_kv_heads: usize,
    seed: u64,
) {
    let mut rng = seeded_rng(seed);
    let total_seq = prefill_len + decode_steps;
    let config = if num_heads == num_kv_heads {
        AttentionConfig::with_equal_heads(embed_dim, num_heads)
    } else {
        AttentionConfig::with_kv(embed_dim, num_heads, num_kv_heads)
    };
    let total_dim = config.total_projection_dim();

    let input_host = tensor_from_vec(
        &[total_seq, embed_dim],
        random_vec(&mut rng, total_seq * embed_dim),
    );
    let w_qkv_host = tensor_from_vec(
        &[embed_dim, total_dim],
        random_vec(&mut rng, embed_dim * total_dim),
    );
    let b_qkv_host = tensor_from_vec(&[total_dim], random_vec(&mut rng, total_dim));
    let w_out_host = tensor_from_vec(
        &[embed_dim, embed_dim],
        random_vec(&mut rng, embed_dim * embed_dim),
    );
    let b_out_host = tensor_from_vec(&[embed_dim], random_vec(&mut rng, embed_dim));

    let expected = timed_torch(|| {
        let input_tch = tch_tensor_from_vec(&[total_seq, embed_dim], input_host.data());
        let w_qkv_tch = tch_tensor_from_vec(&[embed_dim, total_dim], w_qkv_host.data());
        let b_qkv_tch = tch_tensor_from_vec(&[total_dim], b_qkv_host.data());
        let w_out_tch = tch_tensor_from_vec(&[embed_dim, embed_dim], w_out_host.data());
        let b_out_tch = tch_tensor_from_vec(&[embed_dim], b_out_host.data());
        tensor_to_vec(&attention_reference(
            &input_tch,
            &w_qkv_tch,
            Some(&b_qkv_tch),
            &w_out_tch,
            Some(&b_out_tch),
            &config,
        ))
    });

    timed_gpt(|| {
        let layer = build_attention_layer(
            backend,
            config.clone(),
            w_qkv_host.clone(),
            w_out_host.clone(),
            Some(b_qkv_host.clone()),
            Some(b_out_host.clone()),
        );

        let prefill_input = tensor_from_vec(
            &[prefill_len, embed_dim],
            input_host.data()[0..prefill_len * embed_dim].to_vec(),
        );
        let prefill_device = DeviceTensor::from_host(Arc::clone(backend), prefill_input).unwrap();
        let (_prefill_out, mut cache) = layer.forward_with_cache(&prefill_device, None).unwrap();

        for step in 0..decode_steps {
            let start = (prefill_len + step) * embed_dim;
            let end = start + embed_dim;
            let token = tensor_from_vec(&[1, embed_dim], input_host.data()[start..end].to_vec());
            let token_device = DeviceTensor::from_host(Arc::clone(backend), token).unwrap();
            let (output, next_cache) = layer
                .forward_with_cache(&token_device, Some(&cache))
                .unwrap();

            let output_host = output.to_host().unwrap();
            assert_close(&expected[start..end], output_host.data());
            cache = next_cache;
        }
    });
}

pub fn multi_head_attention_matches_torch_with_bias<B: PortableBackend + 'static>(
    backend: &Arc<B>,
) {
    let mut rng = seeded_rng(0xA57A);
    let seq_len = 5;
    let embed_dim = 8;
    let num_heads = 4;

    let input_host = tensor_from_vec(
        &[seq_len, embed_dim],
        random_vec(&mut rng, seq_len * embed_dim),
    );
    let w_qkv_host = tensor_from_vec(
        &[embed_dim, 3 * embed_dim],
        random_vec(&mut rng, embed_dim * 3 * embed_dim),
    );
    let b_qkv_host = tensor_from_vec(&[3 * embed_dim], random_vec(&mut rng, 3 * embed_dim));
    let w_out_host = tensor_from_vec(
        &[embed_dim, embed_dim],
        random_vec(&mut rng, embed_dim * embed_dim),
    );
    let b_out_host = tensor_from_vec(&[embed_dim], random_vec(&mut rng, embed_dim));

    let config = AttentionConfig::with_equal_heads(embed_dim, num_heads);
    let expected = timed_torch(|| {
        let input_tch = tch_tensor_from_vec(&[seq_len, embed_dim], input_host.data());
        let w_qkv_tch = tch_tensor_from_vec(&[embed_dim, 3 * embed_dim], w_qkv_host.data());
        let b_qkv_tch = tch_tensor_from_vec(&[3 * embed_dim], b_qkv_host.data());
        let w_out_tch = tch_tensor_from_vec(&[embed_dim, embed_dim], w_out_host.data());
        let b_out_tch = tch_tensor_from_vec(&[embed_dim], b_out_host.data());
        tensor_to_vec(&attention_reference(
            &input_tch,
            &w_qkv_tch,
            Some(&b_qkv_tch),
            &w_out_tch,
            Some(&b_out_tch),
            &config,
        ))
    });

    let output_host = timed_gpt(|| {
        let layer = build_attention_layer(
            backend,
            config.clone(),
            w_qkv_host.clone(),
            w_out_host.clone(),
            Some(b_qkv_host.clone()),
            Some(b_out_host.clone()),
        );

        let input_device =
            DeviceTensor::from_host(Arc::clone(backend), input_host.clone()).unwrap();
        let output_device = layer.forward(&input_device).unwrap();
        output_device.to_host().unwrap()
    });

    assert_close(&expected, output_host.data());
}

pub fn multi_head_attention_matches_torch_grouped<B: PortableBackend + 'static>(backend: &Arc<B>) {
    let mut rng = seeded_rng(0xC0FE);
    let seq_len = 4;
    let embed_dim = 8;
    let num_heads = 4;
    let num_kv_heads = 2;

    let config = AttentionConfig::with_kv(embed_dim, num_heads, num_kv_heads);
    let total_dim = config.total_projection_dim();

    let input_host = tensor_from_vec(
        &[seq_len, embed_dim],
        random_vec(&mut rng, seq_len * embed_dim),
    );
    let w_qkv_host = tensor_from_vec(
        &[embed_dim, total_dim],
        random_vec(&mut rng, embed_dim * total_dim),
    );
    let b_qkv_host = tensor_from_vec(&[total_dim], random_vec(&mut rng, total_dim));
    let w_out_host = tensor_from_vec(
        &[embed_dim, embed_dim],
        random_vec(&mut rng, embed_dim * embed_dim),
    );
    let b_out_host = tensor_from_vec(&[embed_dim], random_vec(&mut rng, embed_dim));

    let expected = timed_torch(|| {
        let input_tch = tch_tensor_from_vec(&[seq_len, embed_dim], input_host.data());
        let w_qkv_tch = tch_tensor_from_vec(&[embed_dim, total_dim], w_qkv_host.data());
        let b_qkv_tch = tch_tensor_from_vec(&[total_dim], b_qkv_host.data());
        let w_out_tch = tch_tensor_from_vec(&[embed_dim, embed_dim], w_out_host.data());
        let b_out_tch = tch_tensor_from_vec(&[embed_dim], b_out_host.data());
        tensor_to_vec(&attention_reference(
            &input_tch,
            &w_qkv_tch,
            Some(&b_qkv_tch),
            &w_out_tch,
            Some(&b_out_tch),
            &config,
        ))
    });

    let output_host = timed_gpt(|| {
        let layer = build_attention_layer(
            backend,
            config.clone(),
            w_qkv_host.clone(),
            w_out_host.clone(),
            Some(b_qkv_host.clone()),
            Some(b_out_host.clone()),
        );

        let input_device =
            DeviceTensor::from_host(Arc::clone(backend), input_host.clone()).unwrap();
        let (output_device, state) = layer.forward_with_cache(&input_device, None).unwrap();
        let output_host = output_device.to_host().unwrap();

        assert_eq!(
            state.keys().shape().dims(),
            &[config.num_key_value_heads, seq_len, config.kv_head_dim]
        );

        let last_token_host = tensor_from_vec(
            &[1, embed_dim],
            input_host.data()[embed_dim * (seq_len - 1)..].to_vec(),
        );
        let last_token_device =
            DeviceTensor::from_host(Arc::clone(backend), last_token_host).unwrap();
        let (_, second_state) = layer
            .forward_with_cache(&last_token_device, Some(&state))
            .unwrap();
        assert_eq!(second_state.len(), seq_len + 1);

        output_host
    });

    assert_close(&expected, output_host.data());
}

pub fn multi_head_attention_matches_torch_without_bias<B: PortableBackend + 'static>(
    backend: &Arc<B>,
) {
    let mut rng = seeded_rng(0xDEAD);
    let seq_len = 3;
    let embed_dim = 6;
    let num_heads = 3;
    let config = AttentionConfig::with_equal_heads(embed_dim, num_heads);

    let input_host = tensor_from_vec(
        &[seq_len, embed_dim],
        random_vec(&mut rng, seq_len * embed_dim),
    );
    let w_qkv_host = tensor_from_vec(
        &[embed_dim, 3 * embed_dim],
        random_vec(&mut rng, embed_dim * 3 * embed_dim),
    );
    let w_out_host = tensor_from_vec(
        &[embed_dim, embed_dim],
        random_vec(&mut rng, embed_dim * embed_dim),
    );

    let expected = timed_torch(|| {
        let input_tch = tch_tensor_from_vec(&[seq_len, embed_dim], input_host.data());
        let w_qkv_tch = tch_tensor_from_vec(&[embed_dim, 3 * embed_dim], w_qkv_host.data());
        let w_out_tch = tch_tensor_from_vec(&[embed_dim, embed_dim], w_out_host.data());
        tensor_to_vec(&attention_reference(
            &input_tch, &w_qkv_tch, None, &w_out_tch, None, &config,
        ))
    });

    let output_host = timed_gpt(|| {
        let layer = build_attention_layer(
            backend,
            config.clone(),
            w_qkv_host.clone(),
            w_out_host.clone(),
            Option::<Tensor>::None,
            Option::<Tensor>::None,
        );

        let input_device =
            DeviceTensor::from_host(Arc::clone(backend), input_host.clone()).unwrap();
        let output_device = layer.forward(&input_device).unwrap();
        output_device.to_host().unwrap()
    });

    assert_close(&expected, output_host.data());
}

pub fn multi_head_attention_state_records_context<B: PortableBackend + 'static>(backend: &Arc<B>) {
    let mut rng = seeded_rng(0xFACE);
    let seq_len = 4;
    let embed_dim = 8;
    let num_heads = 2;
    let config = AttentionConfig::with_equal_heads(embed_dim, num_heads);

    let input_host = tensor_from_vec(
        &[seq_len, embed_dim],
        random_vec(&mut rng, seq_len * embed_dim),
    );
    let w_qkv_host = tensor_from_vec(
        &[embed_dim, 3 * embed_dim],
        random_vec(&mut rng, embed_dim * 3 * embed_dim),
    );
    let b_qkv_host = tensor_from_vec(&[3 * embed_dim], random_vec(&mut rng, 3 * embed_dim));
    let w_out_host = tensor_from_vec(
        &[embed_dim, embed_dim],
        random_vec(&mut rng, embed_dim * embed_dim),
    );
    let b_out_host = tensor_from_vec(&[embed_dim], random_vec(&mut rng, embed_dim));

    let expected = timed_torch(|| {
        let input_tch = tch_tensor_from_vec(&[seq_len, embed_dim], input_host.data());
        let w_qkv_tch = tch_tensor_from_vec(&[embed_dim, 3 * embed_dim], w_qkv_host.data());
        let b_qkv_tch = tch_tensor_from_vec(&[3 * embed_dim], b_qkv_host.data());
        tensor_to_vec(&attention_reference(
            &input_tch,
            &w_qkv_tch,
            Some(&b_qkv_tch),
            &TchTensor::eye(embed_dim as i64, (Kind::Float, input_tch.device())),
            None,
            &config,
        ))
    });

    let context_host = timed_gpt(|| {
        let layer = build_attention_layer(
            backend,
            config.clone(),
            w_qkv_host.clone(),
            w_out_host.clone(),
            Some(b_qkv_host.clone()),
            Some(b_out_host.clone()),
        );

        let input_device =
            DeviceTensor::from_host(Arc::clone(backend), input_host.clone()).unwrap();
        let qkv = layer.proj_qkv.forward(&input_device).unwrap();
        let functional::AttentionComputation {
            output: attention_context,
            ..
        } = functional::attention(backend.as_ref(), &config, &qkv, None).unwrap();
        attention_context.to_host().unwrap()
    });

    assert_close(&expected, context_host.data());
}

pub fn multi_head_attention_seq1_embed32_heads4_bias_matches_torch<B: PortableBackend + 'static>(
    backend: &Arc<B>,
) {
    run_mha_case(backend, 1, 32, 4, 4, 0xA500, true);
}

pub fn multi_head_attention_seq8_embed32_heads4_bias_matches_torch<B: PortableBackend + 'static>(
    backend: &Arc<B>,
) {
    run_mha_case(backend, 8, 32, 4, 4, 0xA501, true);
}

pub fn multi_head_attention_seq8_embed32_heads8_kv1_bias_matches_torch<
    B: PortableBackend + 'static,
>(
    backend: &Arc<B>,
) {
    run_mha_case(backend, 8, 32, 8, 1, 0xA502, true);
}

pub fn multi_head_attention_seq8_embed32_heads8_kv2_bias_matches_torch<
    B: PortableBackend + 'static,
>(
    backend: &Arc<B>,
) {
    run_mha_case(backend, 8, 32, 8, 2, 0xA503, true);
}

pub fn multi_head_attention_head_dim1_embed8_heads8_bias_matches_torch<
    B: PortableBackend + 'static,
>(
    backend: &Arc<B>,
) {
    run_mha_case(backend, 4, 8, 8, 8, 0xA504, true);
}

pub fn multi_head_attention_prefill4_decode3_matches_full_concat<B: PortableBackend + 'static>(
    backend: &Arc<B>,
) {
    run_mha_prefill_decode_case(backend, 4, 3, 32, 4, 4, 0xA505);
}

pub fn multi_head_attention_prefill4_decode3_grouped_kv2_matches_full_concat<
    B: PortableBackend + 'static,
>(
    backend: &Arc<B>,
) {
    run_mha_prefill_decode_case(backend, 4, 3, 32, 8, 2, 0xA506);
}
