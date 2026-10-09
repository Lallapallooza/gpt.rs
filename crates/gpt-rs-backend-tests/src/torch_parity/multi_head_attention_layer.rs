use std::sync::Arc;

use gpt_rs::backend::PortableBackend;
use gpt_rs::module::Layer;
use gpt_rs::nn::layers::{
    AttentionConfig, CausalSelfAttention, LlamaNames, RmsNormConfig, RopeConfig, RopeScaling,
    RotaryEmbedding,
};
use gpt_rs::ops::functional::{AttentionPositions, WeightLayout};
use gpt_rs::tensor::HostTensor;
use tch::{Device, Kind, Tensor as TchTensor};

use super::common::*;

const ROPE_THETA: f32 = 10_000.0;

/// Host weights of a [`CausalSelfAttention`] layer.
struct HostAttention {
    q: HostLinear,
    k: HostLinear,
    v: HostLinear,
    o: HostLinear,
    /// `(q_norm weight, k_norm weight)`, each `[head_dim]`.
    qk_norm: Option<(Vec<f32>, Vec<f32>)>,
}

fn rms_norm_reference(x: &TchTensor, weight: &[f32], norm: RmsNormConfig) -> TchTensor {
    let weight = tch_tensor_from_vec(&[weight.len()], weight);
    let scale = if norm.unit_offset {
        weight + 1.0
    } else {
        weight
    };
    let mean_square = (x * x).mean_dim(Some([-1i64].as_slice()), true, Kind::Float);
    x * (mean_square + norm.eps as f64).rsqrt() * scale
}

fn rope(rotary_dim: usize) -> RopeConfig {
    RopeConfig {
        rotary_dim,
        theta: ROPE_THETA,
        scaling: RopeScaling::None,
    }
}

/// Hugging Face `apply_rotary_pos_emb` on the leading `rotary_dim` channels of `x`, a
/// `[heads, T, head_dim]` tensor, at positions `0..T`.
fn rope_reference(x: &TchTensor, rotary_dim: usize) -> TchTensor {
    let size = x.size();
    let (seq, head_dim, rotary) = (size[1], size[2], rotary_dim as i64);
    let exponent = TchTensor::arange_start_step(0, rotary, 2, (Kind::Float, Device::Cpu)) / rotary;
    let inv_freq = (exponent * (ROPE_THETA as f64).ln()).exp().reciprocal();
    let freqs =
        TchTensor::arange(seq, (Kind::Float, Device::Cpu)).unsqueeze(1) * inv_freq.unsqueeze(0);
    let emb = TchTensor::cat(&[&freqs, &freqs], 1);
    let (cos, sin) = (emb.cos(), emb.sin());
    let x_rot = x.narrow(2, 0, rotary);
    let half = rotary / 2;
    let rotate_half = TchTensor::cat(&[-x_rot.narrow(2, half, half), x_rot.narrow(2, 0, half)], 2);
    let rotated = &x_rot * &cos + rotate_half * &sin;
    TchTensor::cat(&[rotated, x.narrow(2, rotary, head_dim - rotary)], 2)
}

fn attention_reference(
    input: &TchTensor,
    weights: &HostAttention,
    config: &AttentionConfig,
    rope: Option<RopeConfig>,
) -> TchTensor {
    let seq_len = input.size()[0];
    let num_heads = config.num_query_heads as i64;
    let kv_heads = config.num_key_value_heads as i64;
    let head_dim = config.head_dim as i64;
    let group_size = num_heads / kv_heads;

    let q_width = if config.output_gate {
        2 * head_dim
    } else {
        head_dim
    };
    let q_and_gate = weights
        .q
        .torch(input)
        .reshape([seq_len, num_heads, q_width]);
    let mut q = q_and_gate.narrow(2, 0, head_dim);
    let mut k = weights
        .k
        .torch(input)
        .reshape([seq_len, kv_heads, head_dim]);
    let v = weights
        .v
        .torch(input)
        .reshape([seq_len, kv_heads, head_dim]);
    if let (Some(norm), Some((q_weight, k_weight))) = (config.qk_norm, &weights.qk_norm) {
        q = rms_norm_reference(&q, q_weight, norm);
        k = rms_norm_reference(&k, k_weight, norm);
    }
    let (mut q, mut k, v) = (q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1));
    if let Some(rope) = rope {
        q = rope_reference(&q, rope.rotary_dim);
        k = rope_reference(&k, rope.rotary_dim);
    }
    let k = k.repeat_interleave_self_int(group_size, 0, None::<i64>);
    let v = v.repeat_interleave_self_int(group_size, 0, None::<i64>);

    let scale = (head_dim as f64).sqrt();
    let scores = q.matmul(&k.transpose(-2, -1)) / scale;
    let causal_mask = TchTensor::ones(scores.size().as_slice(), (Kind::Bool, scores.device()))
        .tril(0)
        .logical_not();
    let masked = scores.masked_fill(&causal_mask, f64::NEG_INFINITY);
    let attn = masked.softmax(-1, Kind::Float);
    let mut context = attn
        .matmul(&v)
        .transpose(0, 1)
        .reshape([seq_len, num_heads * head_dim]);
    if config.output_gate {
        let gate = q_and_gate
            .narrow(2, head_dim, head_dim)
            .reshape([seq_len, num_heads * head_dim]);
        context *= gate.sigmoid();
    }
    weights.o.torch(&context)
}

/// The `[in_features, out_features]` weight and the bias of `linears` fused along their outputs,
/// as `prefix.weight` and `prefix.bias`: a `Conv1D` that computes all of them.
fn fused_in_out(prefix: &str, linears: &[&HostLinear]) -> Vec<(String, HostTensor)> {
    let parts: Vec<Vec<(String, HostTensor)>> =
        linears.iter().map(|linear| linear.tensors("p")).collect();
    let in_features = parts[0][0].1.dims[1];
    let out_features: usize = parts.iter().map(|part| part[0].1.dims[0]).sum();
    let mut weight = vec![0.0; in_features * out_features];
    let mut bias = Vec::with_capacity(out_features);
    let mut column = 0;
    for part in &parts {
        let (rows, values) = (part[0].1.dims[0], part[0].1.as_f32().unwrap());
        for row in 0..rows {
            for input in 0..in_features {
                weight[input * out_features + column + row] = values[row * in_features + input];
            }
        }
        match part.get(1) {
            Some((_, part_bias)) => bias.extend_from_slice(part_bias.as_f32().unwrap()),
            None => bias.extend(std::iter::repeat_n(0.0, rows)),
        }
        column += rows;
    }
    vec![
        (
            format!("{prefix}.weight"),
            tensor_from_vec(&[in_features, out_features], weight),
        ),
        (
            format!("{prefix}.bias"),
            tensor_from_vec(&[out_features], bias),
        ),
    ]
}

/// Runs `prefill_len` tokens as one chunk into a fresh cache, then `decode_steps` single tokens,
/// and compares every output row with full-sequence Torch attention.
fn run_attention_case<B: PortableBackend + 'static>(
    backend: &Arc<B>,
    config: AttentionConfig,
    rope: Option<RopeConfig>,
    bias: bool,
    (prefill_len, decode_steps): (usize, usize),
    seed: u64,
) {
    let mut rng = seeded_rng(seed);
    let total_seq = prefill_len + decode_steps;
    let (embed_dim, kv_dim) = (config.embed_dim, config.key_value_projection_dim());

    let input = random_vec(&mut rng, total_seq * embed_dim);
    let weights = HostAttention {
        q: HostLinear::random(&mut rng, embed_dim, config.query_projection_dim(), bias),
        k: HostLinear::random(&mut rng, embed_dim, kv_dim, bias),
        v: HostLinear::random(&mut rng, embed_dim, kv_dim, bias),
        o: HostLinear::random(&mut rng, config.query_dim(), embed_dim, bias),
        qk_norm: config.qk_norm.map(|_| {
            let d = config.head_dim;
            let mut weight = || random_vec_range(&mut rng, d, 0.5, 1.5);
            (weight(), weight())
        }),
    };

    let expected = timed_torch(|| {
        tensor_to_vec(&attention_reference(
            &tch_tensor_from_vec(&[total_seq, embed_dim], &input),
            &weights,
            &config,
            rope,
        ))
    });

    timed_gpt(|| {
        let mut tensors = if config.fused_qkv {
            let qkv = [&weights.q, &weights.k, &weights.v];
            [
                fused_in_out("attn.qkv_proj", &qkv),
                fused_in_out("attn.o_proj", &[&weights.o]),
            ]
            .concat()
        } else {
            [
                weights.q.tensors("attn.q_proj"),
                weights.k.tensors("attn.k_proj"),
                weights.v.tensors("attn.v_proj"),
                weights.o.tensors("attn.o_proj"),
            ]
            .concat()
        };
        if let Some((q_weight, k_weight)) = &weights.qk_norm {
            for (name, weight) in [
                ("attn.q_norm.weight", q_weight),
                ("attn.k_norm.weight", k_weight),
            ] {
                tensors.push((
                    name.to_string(),
                    tensor_from_vec(&[weight.len()], weight.clone()),
                ));
            }
        }
        let layer = load_layer(backend, tensors, |p| {
            p.scoped("attn", |p| {
                CausalSelfAttention::<_, LlamaNames>::new(p, config.clone())
            })
        })
        .unwrap();
        let rotary = rope.map(|rope| RotaryEmbedding::new(rope).unwrap());
        let mut cache = layer.config.empty_cache(backend, total_seq).unwrap();
        let mut position = 0;
        for len in std::iter::once(prefill_len).chain(std::iter::repeat_n(1, decode_steps)) {
            let rows = position * embed_dim..(position + len) * embed_dim;
            let x = device_tensor_from_data(backend, &[len, embed_dim], &input[rows.clone()]);
            let ids = positions_tensor(backend, position..position + len);
            let tables = rotary.as_ref().map(|rope| rope.tables(&ids));
            let positions = AttentionPositions::new(ids, tables);
            let (output, next_cache) = layer.call((&x, &cache, &positions));
            assert_close(&expected[rows], &to_host_vec(&output));
            cache = next_cache;
            position += len;
        }
        assert_eq!(cache.len(), total_seq);
    });
}

/// Plain multi- or grouped-query attention.
fn run_mha_case<B: PortableBackend + 'static>(
    backend: &Arc<B>,
    steps: (usize, usize),
    (embed_dim, num_heads, num_kv_heads): (usize, usize, usize),
    seed: u64,
    bias: bool,
) {
    let config = AttentionConfig::with_kv(embed_dim, num_heads, num_kv_heads).unwrap();
    run_attention_case(backend, config, None, bias, steps, seed);
}

pub fn multi_head_attention_matches_torch_without_bias<B: PortableBackend + 'static>(
    backend: &Arc<B>,
) {
    run_mha_case(backend, (3, 0), (6, 3, 3), 0xDEAD, false);
}

pub fn multi_head_attention_seq1_embed32_heads4_bias_matches_torch<B: PortableBackend + 'static>(
    backend: &Arc<B>,
) {
    run_mha_case(backend, (1, 0), (32, 4, 4), 0xA500, true);
}

pub fn multi_head_attention_seq8_embed32_heads4_bias_matches_torch<B: PortableBackend + 'static>(
    backend: &Arc<B>,
) {
    run_mha_case(backend, (8, 0), (32, 4, 4), 0xA501, true);
}

pub fn multi_head_attention_seq8_embed32_heads8_kv1_bias_matches_torch<
    B: PortableBackend + 'static,
>(
    backend: &Arc<B>,
) {
    run_mha_case(backend, (8, 0), (32, 8, 1), 0xA502, true);
}

pub fn multi_head_attention_seq8_embed32_heads8_kv2_bias_matches_torch<
    B: PortableBackend + 'static,
>(
    backend: &Arc<B>,
) {
    run_mha_case(backend, (8, 0), (32, 8, 2), 0xA503, true);
}

pub fn multi_head_attention_head_dim1_embed8_heads8_bias_matches_torch<
    B: PortableBackend + 'static,
>(
    backend: &Arc<B>,
) {
    run_mha_case(backend, (4, 0), (8, 8, 8), 0xA504, true);
}

pub fn multi_head_attention_prefill4_decode3_matches_full_sequence<B: PortableBackend + 'static>(
    backend: &Arc<B>,
) {
    run_mha_case(backend, (4, 3), (32, 4, 4), 0xA505, true);
}

pub fn multi_head_attention_prefill4_decode3_grouped_kv2_matches_full_sequence<
    B: PortableBackend + 'static,
>(
    backend: &Arc<B>,
) {
    run_mha_case(backend, (4, 3), (32, 8, 2), 0xA506, true);
}

pub fn attention_full_rotary_grouped_kv2_prefill4_decode3_matches_torch<
    B: PortableBackend + 'static,
>(
    backend: &Arc<B>,
) {
    let config = AttentionConfig::with_kv(32, 4, 2).unwrap();
    run_attention_case(backend, config, Some(rope(8)), false, (4, 3), 0xA507);
}

pub fn attention_qk_norm_matches_torch<B: PortableBackend + 'static>(backend: &Arc<B>) {
    let config = AttentionConfig::with_kv(32, 4, 4)
        .unwrap()
        .with_qk_norm(RmsNormConfig {
            eps: 1e-6,
            unit_offset: false,
        });
    run_attention_case(backend, config, None, false, (6, 0), 0xA508);
}

pub fn attention_output_gate_grouped_kv2_prefill5_decode2_matches_torch<
    B: PortableBackend + 'static,
>(
    backend: &Arc<B>,
) {
    let config = AttentionConfig::with_kv(32, 4, 2)
        .unwrap()
        .with_output_gate();
    run_attention_case(backend, config, None, true, (5, 2), 0xA509);
}

pub fn attention_partial_rotary_qk_norm_output_gate_prefill5_decode3_matches_torch<
    B: PortableBackend + 'static,
>(
    backend: &Arc<B>,
) {
    let config = AttentionConfig::with_projection_dims(24, 4, 2, 8)
        .unwrap()
        .with_qk_norm(RmsNormConfig {
            eps: 1e-6,
            unit_offset: true,
        })
        .with_output_gate();
    run_attention_case(backend, config, Some(rope(4)), false, (5, 3), 0xA50A);
}

/// One projection computes the queries, keys and values, and every projection stores its weight
/// as `[in_features, out_features]`, as in GPT-2.
pub fn attention_fused_qkv_in_out_prefill4_decode3_matches_torch<B: PortableBackend + 'static>(
    backend: &Arc<B>,
) {
    let config = AttentionConfig::with_equal_heads(32, 4)
        .unwrap()
        .with_fused_qkv()
        .with_weight_layout(WeightLayout::InOut);
    run_attention_case(backend, config, None, true, (4, 3), 0xA50B);
}
