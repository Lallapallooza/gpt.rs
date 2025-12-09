use std::sync::Arc;

use gpt_rs::backend::spec::PortableBackend;
use gpt_rs::nn::layers::AttentionConfig;
use gpt_rs::ops::functional::{
    attention, build_registry, with_registry, AttentionCache, FunctionalOverrides,
};
use gpt_rs::tensor::DeviceTensor;
use tch::{Kind, Tensor as TchTensor};

use super::common::*;

fn attention_reference(
    input: &TchTensor,
    w_qkv: &TchTensor,
    b_qkv: Option<&TchTensor>,
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
    attn.matmul(&v)
        .transpose(0, 1)
        .reshape([seq_len, embed_dim])
}

fn run_attention_case<B: PortableBackend + 'static>(
    backend: &Arc<B>,
    seq_len: usize,
    embed_dim: usize,
    num_heads: usize,
    num_kv_heads: usize,
    seed: u64,
) {
    let mut rng = seeded_rng(seed);
    let config = if num_heads == num_kv_heads {
        AttentionConfig::with_equal_heads(embed_dim, num_heads)
    } else {
        AttentionConfig::with_kv(embed_dim, num_heads, num_kv_heads)
    };
    let total_dim = config.total_projection_dim();

    let input_vec = random_vec(&mut rng, seq_len * embed_dim);
    let w_qkv_vec = random_vec(&mut rng, embed_dim * total_dim);
    let b_qkv_vec = random_vec(&mut rng, total_dim);

    let (expected, qkv_data) = timed_torch(|| {
        let input_t = tch_tensor_from_vec(&[seq_len, embed_dim], &input_vec);
        let w_qkv_t = tch_tensor_from_vec(&[embed_dim, total_dim], &w_qkv_vec);
        let b_qkv_t = tch_tensor_from_vec(&[total_dim], &b_qkv_vec);

        let expected = attention_reference(&input_t, &w_qkv_t, Some(&b_qkv_t), &config);
        let qkv_t = input_t.matmul(&w_qkv_t) + b_qkv_t.unsqueeze(0);

        (tensor_to_vec(&expected), tensor_to_vec(&qkv_t))
    });

    let qkv_device = DeviceTensor::from_host(
        Arc::clone(backend),
        tensor_from_vec(&[seq_len, total_dim], qkv_data),
    )
    .unwrap();

    let output = timed_gpt(|| {
        let registry = build_registry::<B>(&FunctionalOverrides::default());
        let computation = with_registry(Arc::clone(&registry), || {
            attention(backend.as_ref(), &config, &qkv_device, None)
        })
        .expect("attention forward should succeed");

        assert_eq!(
            computation.cache.keys().shape().dims(),
            &[config.num_key_value_heads, seq_len, config.kv_head_dim]
        );
        computation.output.to_host().unwrap()
    });

    assert_close(&expected, output.data());
}

fn run_attention_prefill_decode_case<B: PortableBackend + 'static>(
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

    let input_vec = random_vec(&mut rng, total_seq * embed_dim);
    let w_qkv_vec = random_vec(&mut rng, embed_dim * total_dim);
    let b_qkv_vec = random_vec(&mut rng, total_dim);

    let (expected_full, qkv_full) = timed_torch(|| {
        let input_t = tch_tensor_from_vec(&[total_seq, embed_dim], &input_vec);
        let w_qkv_t = tch_tensor_from_vec(&[embed_dim, total_dim], &w_qkv_vec);
        let b_qkv_t = tch_tensor_from_vec(&[total_dim], &b_qkv_vec);

        let expected = attention_reference(&input_t, &w_qkv_t, Some(&b_qkv_t), &config);
        let qkv_t = input_t.matmul(&w_qkv_t) + b_qkv_t.unsqueeze(0);

        (tensor_to_vec(&expected), tensor_to_vec(&qkv_t))
    });

    timed_gpt(|| {
        let registry = build_registry::<B>(&FunctionalOverrides::default());

        let prefill_len_usize = prefill_len;
        let prefill_slice = qkv_full[0..prefill_len_usize * total_dim].to_vec();
        let qkv_prefill = DeviceTensor::from_host(
            Arc::clone(backend),
            tensor_from_vec(&[prefill_len_usize, total_dim], prefill_slice),
        )
        .unwrap();
        let prefill = with_registry(Arc::clone(&registry), || {
            attention(backend.as_ref(), &config, &qkv_prefill, None)
        })
        .expect("prefill attention should succeed");
        let mut cache = Some(prefill.cache);

        for step in 0..decode_steps {
            let start = (prefill_len + step) * total_dim;
            let end = start + total_dim;
            let token_slice = qkv_full[start..end].to_vec();
            let token_device = DeviceTensor::from_host(
                Arc::clone(backend),
                tensor_from_vec(&[1, total_dim], token_slice),
            )
            .unwrap();

            let result = with_registry(Arc::clone(&registry), || {
                attention(backend.as_ref(), &config, &token_device, cache.as_ref())
            })
            .expect("decode attention should succeed");

            let output = result.output.to_host().unwrap();
            let expected_start = (prefill_len + step) * embed_dim;
            let expected_end = expected_start + embed_dim;
            assert_close(&expected_full[expected_start..expected_end], output.data());

            cache = Some(result.cache);
        }
    });
}

pub fn attention_matches_torch<B: PortableBackend + 'static>(backend: &Arc<B>) {
    let mut rng = seeded_rng(555);
    let seq_len = 4;
    let embed_dim = 6;
    let num_heads = 3;

    let input_vec = random_vec(&mut rng, seq_len * embed_dim);
    let w_qkv_vec = random_vec(&mut rng, embed_dim * 3 * embed_dim);
    let b_qkv_vec = random_vec(&mut rng, 3 * embed_dim);

    let config = AttentionConfig::with_equal_heads(embed_dim, num_heads);

    let (expected, qkv_data) = timed_torch(|| {
        let input_t = tch_tensor_from_vec(&[seq_len, embed_dim], &input_vec);
        let w_qkv_t = tch_tensor_from_vec(&[embed_dim, 3 * embed_dim], &w_qkv_vec);
        let b_qkv_t = tch_tensor_from_vec(&[3 * embed_dim], &b_qkv_vec);

        let expected = attention_reference(&input_t, &w_qkv_t, Some(&b_qkv_t), &config);
        let qkv_t = input_t.matmul(&w_qkv_t) + b_qkv_t.unsqueeze(0);

        (tensor_to_vec(&expected), tensor_to_vec(&qkv_t))
    });

    let qkv_device = DeviceTensor::from_host(
        Arc::clone(backend),
        tensor_from_vec(&[seq_len, 3 * embed_dim], qkv_data),
    )
    .unwrap();

    let output_host = timed_gpt(|| {
        let registry = build_registry::<B>(&FunctionalOverrides::default());
        let computation = with_registry(Arc::clone(&registry), || {
            attention(backend.as_ref(), &config, &qkv_device, None)
        })
        .expect("attention forward should succeed");
        let output = computation.output;
        assert_eq!(
            computation.cache.keys().shape().dims(),
            &[config.num_key_value_heads, seq_len, config.kv_head_dim]
        );
        output.to_host().unwrap()
    });

    assert_close(&expected, output_host.data());
}

pub fn attention_matches_torch_grouped<B: PortableBackend + 'static>(backend: &Arc<B>) {
    let mut rng = seeded_rng(777);
    let seq_len = 3;
    let embed_dim = 8;
    let num_heads = 4;
    let num_kv_heads = 2;

    let config = AttentionConfig::with_kv(embed_dim, num_heads, num_kv_heads);
    let total_dim = config.total_projection_dim();

    let input_vec = random_vec(&mut rng, seq_len * embed_dim);
    let w_qkv_vec = random_vec(&mut rng, embed_dim * total_dim);
    let b_qkv_vec = random_vec(&mut rng, total_dim);

    let (expected, qkv_data) = timed_torch(|| {
        let input_t = tch_tensor_from_vec(&[seq_len, embed_dim], &input_vec);
        let w_qkv_t = tch_tensor_from_vec(&[embed_dim, total_dim], &w_qkv_vec);
        let b_qkv_t = tch_tensor_from_vec(&[total_dim], &b_qkv_vec);
        let expected = attention_reference(&input_t, &w_qkv_t, Some(&b_qkv_t), &config);
        let qkv_t = input_t.matmul(&w_qkv_t) + b_qkv_t.unsqueeze(0);
        (tensor_to_vec(&expected), tensor_to_vec(&qkv_t))
    });
    let qkv_device = DeviceTensor::from_host(
        Arc::clone(backend),
        tensor_from_vec(&[seq_len, total_dim], qkv_data),
    )
    .unwrap();

    let output = timed_gpt(|| {
        let registry = build_registry::<B>(&FunctionalOverrides::default());
        let computation = with_registry(Arc::clone(&registry), || {
            attention(backend.as_ref(), &config, &qkv_device, None)
        })
        .expect("attention forward should succeed");

        assert_eq!(
            computation.cache.keys().shape().dims(),
            &[config.num_key_value_heads, seq_len, config.kv_head_dim]
        );
        computation.output.to_host().unwrap()
    });

    assert_close(&expected, output.data());
}

pub fn attention_appends_cache<B: PortableBackend + 'static>(backend: &Arc<B>) {
    let mut rng = seeded_rng(2025);
    let total_seq = 3;
    let embed_dim = 6;
    let num_heads = 3;

    let config = AttentionConfig::with_equal_heads(embed_dim, num_heads);
    let total_dim = config.total_projection_dim();

    let input_vec = random_vec(&mut rng, total_seq * embed_dim);
    let w_qkv_vec = random_vec(&mut rng, embed_dim * total_dim);
    let b_qkv_vec = random_vec(&mut rng, total_dim);

    let input_t = tch_tensor_from_vec(&[total_seq, embed_dim], &input_vec);
    let w_qkv_t = tch_tensor_from_vec(&[embed_dim, total_dim], &w_qkv_vec);
    let b_qkv_t = tch_tensor_from_vec(&[total_dim], &b_qkv_vec);
    let qkv_total = input_t.matmul(&w_qkv_t) + b_qkv_t.unsqueeze(0);

    let qkv_first = qkv_total.narrow(0, 0, 2);
    let qkv_second = qkv_total.narrow(0, 2, 1);

    timed_gpt(|| {
        let qkv_device_first = DeviceTensor::from_host(
            Arc::clone(backend),
            tensor_from_vec(&[2, total_dim], tensor_to_vec(&qkv_first)),
        )
        .unwrap();
        let qkv_device_second = DeviceTensor::from_host(
            Arc::clone(backend),
            tensor_from_vec(&[1, total_dim], tensor_to_vec(&qkv_second)),
        )
        .unwrap();

        let registry = build_registry::<B>(&FunctionalOverrides::default());
        let first = with_registry(Arc::clone(&registry), || {
            attention(backend.as_ref(), &config, &qkv_device_first, None)
        })
        .expect("initial attention computation should succeed");

        assert_eq!(first.present.len(), 2);
        assert_eq!(first.cache.len(), 2);

        let second = with_registry(Arc::clone(&registry), || {
            attention(
                backend.as_ref(),
                &config,
                &qkv_device_second,
                Some(&first.cache),
            )
        })
        .expect("second attention computation should succeed");

        assert_eq!(second.present.len(), 1);
        assert_eq!(second.cache.len(), 3);
        assert_eq!(
            second.cache.keys().shape().dims(),
            &[config.num_key_value_heads, 3, config.kv_head_dim]
        );
    });
}

pub fn attention_extends_cache_multiple_steps<B: PortableBackend + 'static>(backend: &Arc<B>) {
    let mut rng = seeded_rng(4242);
    let initial_seq = 3;
    let steps = 40;
    let embed_dim = 6;
    let num_heads = 3;

    let config = AttentionConfig::with_equal_heads(embed_dim, num_heads);
    let total_dim = config.total_projection_dim();

    timed_gpt(|| {
        let registry = build_registry::<B>(&FunctionalOverrides::default());
        let mut cache: Option<AttentionCache<B>> = None;
        let mut accumulated = 0usize;

        for step in 0..steps {
            let current_len = if step == 0 { initial_seq } else { 1 };
            let qkv_vec = random_vec(&mut rng, current_len * total_dim);
            let qkv_device = DeviceTensor::from_host(
                Arc::clone(backend),
                tensor_from_vec(&[current_len, total_dim], qkv_vec.clone()),
            )
            .expect("device tensor conversion should succeed");

            let result = with_registry(Arc::clone(&registry), || {
                attention(backend.as_ref(), &config, &qkv_device, cache.as_ref())
            })
            .expect("attention with cache should succeed across steps");

            assert_eq!(result.present.len(), current_len);
            accumulated += current_len;
            assert_eq!(result.cache.len(), accumulated);

            cache = Some(result.cache);
        }
    });
}

pub fn attention_incremental_matches_full<B: PortableBackend + 'static>(backend: &Arc<B>) {
    let mut rng = seeded_rng(1337);
    let seq_len = 4;
    let embed_dim = 8;
    let num_heads = 4;

    let config = AttentionConfig::with_equal_heads(embed_dim, num_heads);
    let total_dim = config.total_projection_dim();

    timed_gpt(|| {
        let qkv_vec = random_vec(&mut rng, seq_len * total_dim);
        let qkv_device = DeviceTensor::from_host(
            Arc::clone(backend),
            tensor_from_vec(&[seq_len, total_dim], qkv_vec.clone()),
        )
        .unwrap();

        let registry = build_registry::<B>(&FunctionalOverrides::default());
        let full = with_registry(Arc::clone(&registry), || {
            attention(backend.as_ref(), &config, &qkv_device, None)
        })
        .expect("full attention should succeed");
        let full_host = full.output.to_host().unwrap();
        let full_data = full_host.data().to_vec();

        let mut incremental_cache: Option<_> = None;
        for token in 0..seq_len {
            let start = token * total_dim;
            let end = start + total_dim;
            let token_tensor = tensor_from_vec(&[1, total_dim], qkv_vec[start..end].to_vec());
            let token_device =
                DeviceTensor::from_host(Arc::clone(backend), token_tensor).expect("token upload");

            let computation = with_registry(Arc::clone(&registry), || {
                attention(
                    backend.as_ref(),
                    &config,
                    &token_device,
                    incremental_cache.as_ref(),
                )
            })
            .expect("incremental attention should succeed");

            let incremental_output = computation.output.to_host().unwrap();
            assert_close(
                &full_data[token * embed_dim..(token + 1) * embed_dim],
                incremental_output.data(),
            );

            incremental_cache = Some(computation.cache);
        }
    });
}

pub fn attention_seq1_embed8_heads2_matches_torch<B: PortableBackend + 'static>(backend: &Arc<B>) {
    run_attention_case(backend, 1, 8, 2, 2, 600);
}

pub fn attention_seq8_embed32_heads4_matches_torch<B: PortableBackend + 'static>(backend: &Arc<B>) {
    run_attention_case(backend, 8, 32, 4, 4, 601);
}

pub fn attention_head_dim1_embed4_heads4_matches_torch<B: PortableBackend + 'static>(
    backend: &Arc<B>,
) {
    run_attention_case(backend, 4, 4, 4, 4, 602);
}

pub fn attention_multi_query_kv1_matches_torch<B: PortableBackend + 'static>(backend: &Arc<B>) {
    run_attention_case(backend, 8, 32, 8, 1, 603);
}

pub fn attention_grouped_kv2_matches_torch<B: PortableBackend + 'static>(backend: &Arc<B>) {
    run_attention_case(backend, 8, 32, 8, 2, 604);
}

pub fn attention_prefill4_decode3_matches_full_concat<B: PortableBackend + 'static>(
    backend: &Arc<B>,
) {
    run_attention_prefill_decode_case(backend, 4, 3, 32, 4, 4, 605);
}

pub fn attention_prefill4_decode3_grouped_kv2_matches_full_concat<B: PortableBackend + 'static>(
    backend: &Arc<B>,
) {
    run_attention_prefill_decode_case(backend, 4, 3, 32, 8, 2, 606);
}
