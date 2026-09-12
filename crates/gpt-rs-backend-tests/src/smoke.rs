use std::sync::Arc;

use gpt_rs::backend::spec::PortableBackend;
use gpt_rs::inference::decoder::PREFILL_CHUNK;
use gpt_rs::inference::generate::Generator;
use gpt_rs::inference::sampler::Sampler;
use gpt_rs::inference::CausalLanguageModel;
use gpt_rs::model::qwen3_5::Qwen35LayerType;
use gpt_rs::model::{Gpt, GptConfig, Qwen35, Qwen35Config};
use gpt_rs::nn::{ActivationFunction, LayerLoader, RopeParameters, RopeScaling};
use gpt_rs::ops::functional;
use gpt_rs::tensor::{DType, DeviceTensor, Shape, Tensor};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

pub fn matmul_matches_expected<B: PortableBackend + 'static>(backend: &Arc<B>) {
    let a = Tensor::from_vec(Shape::new([2, 2]), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let b = Tensor::from_vec(Shape::new([2, 2]), vec![5.0, 6.0, 7.0, 8.0]).unwrap();

    let a_device = DeviceTensor::from_host(Arc::clone(backend), a.clone()).unwrap();
    let b_device = DeviceTensor::from_host(Arc::clone(backend), b.clone()).unwrap();

    let result = functional::matmul(&a_device, &b_device).unwrap();
    let host = result.to_host().unwrap();

    let expected = vec![19.0, 22.0, 43.0, 50.0];
    assert_eq!(host.data(), expected.as_slice());
}

pub fn gpt_forward_shape<B: PortableBackend + 'static>(backend: &Arc<B>) {
    let mut rng = StdRng::seed_from_u64(42);
    let config = GptConfig {
        vocab_size: 32,
        n_positions: 16,
        n_embd: 8,
        n_layer: 2,
        n_head: 2,
        n_inner: Some(16),
        layer_norm_epsilon: 1e-5,
        activation_function: ActivationFunction::GeluTanh,
    };
    let model = Gpt::random(config.clone(), Arc::clone(backend), &mut rng).unwrap();
    let tokens = vec![1, 2, 3, 4];
    let result = model.forward(&tokens);
    assert!(result.is_ok());
}

pub fn gpt_kv_cache_matches_full_context_decode<B: PortableBackend + 'static>(backend: &Arc<B>) {
    let mut rng = StdRng::seed_from_u64(7);
    let config = GptConfig {
        vocab_size: 64,
        n_positions: 32,
        n_embd: 32,
        n_layer: 2,
        n_head: 4,
        n_inner: Some(64),
        layer_norm_epsilon: 1e-5,
        activation_function: ActivationFunction::GeluTanh,
    };
    let model = Gpt::random(config, Arc::clone(backend), &mut rng).unwrap();
    let sampler = Sampler::new(0.0);
    let prompt = vec![1usize, 2, 3, 4];
    let steps = 16usize;

    let mut kv_gen = Generator::new(&model, &sampler, prompt.as_slice(), true, None).unwrap();
    let mut full_gen = Generator::new(&model, &sampler, prompt.as_slice(), false, None).unwrap();
    for step in 0..steps {
        let kv_next = if step + 1 == steps {
            kv_gen.step_final().unwrap()
        } else {
            kv_gen.step().unwrap()
        };
        let full_next = if step + 1 == steps {
            full_gen.step_final().unwrap()
        } else {
            full_gen.step().unwrap()
        };
        assert_eq!(
            kv_next, full_next,
            "kv-cache decode diverged from full-context decode at step {step}"
        );
    }
    assert_eq!(
        kv_gen.tokens(),
        full_gen.tokens(),
        "kv-cache and full-context token sequences differ"
    );
}

/// A Qwen3.5 with one linear-attention and one full-attention layer, random weights and
/// checkpoint dtypes.
fn tiny_qwen3_5<B: PortableBackend + 'static>(backend: &Arc<B>) -> Qwen35<B> {
    let cfg = Qwen35Config {
        vocab_size: 64,
        max_position_embeddings: 512,
        hidden_size: 32,
        layer_types: vec![
            Qwen35LayerType::LinearAttention,
            Qwen35LayerType::FullAttention,
        ],
        intermediate_size: 64,
        rms_norm_eps: 1e-6,
        num_attention_heads: 4,
        num_key_value_heads: 2,
        head_dim: 8,
        attention_bias: false,
        rope_parameters: RopeParameters {
            rope_theta: 10_000.0,
            partial_rotary_factor: 0.5,
            scaling: RopeScaling::None,
        },
        linear_num_key_heads: 2,
        linear_num_value_heads: 4,
        linear_key_head_dim: 8,
        linear_value_head_dim: 8,
        linear_conv_kernel_dim: 4,
    };
    let (h, m) = (cfg.hidden_size, cfg.intermediate_size);
    let (hq, hkv, d) = (
        cfg.num_attention_heads,
        cfg.num_key_value_heads,
        cfg.head_dim,
    );
    let hv = cfg.linear_num_value_heads;
    let key_dim = cfg.linear_num_key_heads * cfg.linear_key_head_dim;
    let value_dim = hv * cfg.linear_value_head_dim;
    let conv_dim = 2 * key_dim + value_dim;

    let mut rng = StdRng::seed_from_u64(0x9e35);
    let mut tensors = std::collections::HashMap::new();
    let mut add = |name: String, shape: Vec<usize>, dtype: DType, range: (f32, f32)| {
        let len = shape.iter().product();
        let data: Vec<f32> = (0..len).map(|_| rng.gen_range(range.0..range.1)).collect();
        let host = crate::tensor_as(&shape, &data, dtype);
        let tensor = DeviceTensor::from_host(Arc::clone(backend), host).unwrap();
        tensors.insert(name, tensor);
    };
    // Qwen3.5 norm weights are zero-centred, so the scale is `1 + weight`.
    // The DeltaNet norm is not zero-centred.
    let (matrix, offset_norm, norm) = ((-0.3, 0.3), (-0.2, 0.2), (0.8, 1.2));
    add(
        "model.embed_tokens.weight".into(),
        vec![cfg.vocab_size, h],
        DType::BF16,
        (-1.0, 1.0),
    );
    add(
        "lm_head.weight".into(),
        vec![cfg.vocab_size, h],
        DType::BF16,
        matrix,
    );
    add("model.norm.weight".into(), vec![h], DType::F32, offset_norm);
    for (layer, layer_type) in cfg.layer_types.iter().enumerate() {
        let mut add = |name: &str, shape: Vec<usize>, dtype: DType, range: (f32, f32)| {
            add(format!("model.layers.{layer}.{name}"), shape, dtype, range)
        };
        add("input_layernorm.weight", vec![h], DType::F32, offset_norm);
        add(
            "post_attention_layernorm.weight",
            vec![h],
            DType::F32,
            offset_norm,
        );
        add("mlp.gate_proj.weight", vec![m, h], DType::BF16, matrix);
        add("mlp.up_proj.weight", vec![m, h], DType::BF16, matrix);
        add("mlp.down_proj.weight", vec![h, m], DType::BF16, matrix);
        match layer_type {
            Qwen35LayerType::FullAttention => {
                add(
                    "self_attn.q_proj.weight",
                    vec![hq * 2 * d, h],
                    DType::BF16,
                    matrix,
                );
                add(
                    "self_attn.k_proj.weight",
                    vec![hkv * d, h],
                    DType::BF16,
                    matrix,
                );
                add(
                    "self_attn.v_proj.weight",
                    vec![hkv * d, h],
                    DType::BF16,
                    matrix,
                );
                add(
                    "self_attn.o_proj.weight",
                    vec![h, hq * d],
                    DType::BF16,
                    matrix,
                );
                add("self_attn.q_norm.weight", vec![d], DType::F32, offset_norm);
                add("self_attn.k_norm.weight", vec![d], DType::F32, offset_norm);
            }
            Qwen35LayerType::LinearAttention => {
                let la = |name: &str| format!("linear_attn.{name}");
                add(
                    &la("in_proj_qkv.weight"),
                    vec![conv_dim, h],
                    DType::BF16,
                    matrix,
                );
                add(
                    &la("in_proj_z.weight"),
                    vec![value_dim, h],
                    DType::BF16,
                    matrix,
                );
                add(&la("in_proj_b.weight"), vec![hv, h], DType::BF16, matrix);
                add(&la("in_proj_a.weight"), vec![hv, h], DType::BF16, matrix);
                let conv = vec![conv_dim, 1, cfg.linear_conv_kernel_dim];
                add(&la("conv1d.weight"), conv, DType::F32, (-0.5, 0.5));
                add(&la("dt_bias"), vec![hv], DType::F32, (-0.5, 0.5));
                add(&la("A_log"), vec![hv], DType::F32, (-2.3, 0.0));
                let norm_dim = vec![cfg.linear_value_head_dim];
                add(&la("norm.weight"), norm_dim, DType::F32, norm);
                add(
                    &la("out_proj.weight"),
                    vec![h, value_dim],
                    DType::BF16,
                    matrix,
                );
            }
        }
    }
    let mut get = |name: &str| {
        tensors
            .remove(name)
            .ok_or_else(|| anyhow::anyhow!("no test tensor named {name}"))
    };
    Qwen35::build(cfg, &mut LayerLoader::new(Arc::clone(backend), &mut get)).unwrap()
}

fn assert_logits_close(label: &str, expected: &[f32], actual: &[f32]) {
    assert_eq!(expected.len(), actual.len(), "{label}: length mismatch");
    for (idx, (&e, &a)) in expected.iter().zip(actual).enumerate() {
        let tol = 1e-4 + 1e-4 * e.abs().max(a.abs());
        assert!(
            (e - a).abs() <= tol,
            "{label}: logit {idx} differs: expected {e}, actual {a}"
        );
    }
}

pub fn qwen3_5_prefill_chunking_and_cache_growth_agree<B: PortableBackend + 'static>(
    backend: &Arc<B>,
) {
    let model = tiny_qwen3_5(backend);
    let prompt: Vec<usize> = (0..264).map(|i| (i * 7 + 3) % 64).collect();
    assert!(prompt.len() > PREFILL_CHUNK);

    let full = model.forward(&prompt).unwrap();
    let vocab = full.shape().dims()[1];
    let expected = &full.data()[(prompt.len() - 1) * vocab..];

    let mut chunked_caches = vec![None; model.num_layers()];
    let chunked = model
        .forward_with_decode_cache(&prompt, 0, &mut chunked_caches, None)
        .unwrap();
    assert_logits_close("chunked prefill", expected, chunked.data());

    let prefill = 120;
    let mut stepped_caches = vec![None; model.num_layers()];
    let mut stepped = model
        .forward_with_decode_cache(&prompt[..prefill], 0, &mut stepped_caches, None)
        .unwrap();
    for position in prefill..prompt.len() {
        stepped = model
            .forward_with_decode_cache(
                &prompt[position..=position],
                position,
                &mut stepped_caches,
                None,
            )
            .unwrap();
    }
    assert_logits_close("decode with cache growth", expected, stepped.data());

    for (offset, token) in [5usize, 9].into_iter().enumerate() {
        let position = prompt.len() + offset;
        let after_chunked = model
            .forward_with_decode_cache(&[token], position, &mut chunked_caches, None)
            .unwrap();
        let after_stepped = model
            .forward_with_decode_cache(&[token], position, &mut stepped_caches, None)
            .unwrap();
        assert_logits_close(
            "decode after prefill",
            after_stepped.data(),
            after_chunked.data(),
        );
    }
}
