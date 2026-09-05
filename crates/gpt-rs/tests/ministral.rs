use std::sync::Arc;

use anyhow::Result;
use gpt_rs::inference::{CausalLanguageModel, LayerCache};
use gpt_rs::model::{Ministral, MinistralConfig};
use gpt_rs::nn::{RopeParameters, RopeScaling};
use gpt_rs_backend_ref_cpu::CpuPortableBackend;
use rand::rngs::StdRng;
use rand::SeedableRng;

fn assert_close_row(expected: &[f32], actual: &[f32], atol: f32, rtol: f32) {
    assert_eq!(
        expected.len(),
        actual.len(),
        "row length mismatch: expected {}, got {}",
        expected.len(),
        actual.len()
    );
    for (i, (exp, act)) in expected.iter().zip(actual.iter()).enumerate() {
        let tol = atol + rtol * exp.abs();
        let diff = (exp - act).abs();
        assert!(
            diff <= tol,
            "index {} mismatch: expected {:.8}, got {:.8}, diff {:.8}, tol {:.8}",
            i,
            exp,
            act,
            diff,
            tol
        );
    }
}

fn tiny_config() -> MinistralConfig {
    MinistralConfig {
        vocab_size: 64,
        max_position_embeddings: 32,
        hidden_size: 16,
        num_hidden_layers: 2,
        num_attention_heads: 4,
        num_key_value_heads: 2,
        head_dim: 4,
        intermediate_size: 48,
        rms_norm_eps: 1e-5,
        rope_parameters: RopeParameters {
            rope_theta: 10_000.0,
            partial_rotary_factor: 1.0,
            scaling: RopeScaling::None,
        },
    }
}

#[test]
fn ministral_forward_shape() -> Result<()> {
    let backend = Arc::new(CpuPortableBackend::new());
    let mut rng = StdRng::seed_from_u64(7);
    let model = Ministral::random(tiny_config(), backend, &mut rng)?;
    let tokens = [1, 2, 3, 4, 5];

    let logits = model.forward(&tokens)?;
    assert_eq!(
        logits.shape().dims(),
        &[tokens.len(), model.config.vocab_size]
    );
    Ok(())
}

#[test]
fn ministral_decode_cache_matches_full_forward_stepwise() -> Result<()> {
    let backend = Arc::new(CpuPortableBackend::new());
    let mut rng = StdRng::seed_from_u64(42);
    let model = Ministral::random(tiny_config(), backend, &mut rng)?;
    let tokens = [3, 5, 8, 13, 21, 34];

    let full = model.forward(&tokens)?;
    let full_data = full.data().to_vec();
    let vocab = model.config.vocab_size;

    let mut caches: Vec<Option<LayerCache<CpuPortableBackend>>> =
        vec![None; model.config.num_hidden_layers];
    for (pos, token) in tokens.iter().copied().enumerate() {
        let step = model.forward_with_decode_cache(&[token], pos, &mut caches, None)?;
        let expected = &full_data[pos * vocab..(pos + 1) * vocab];
        assert_close_row(expected, step.data(), 1e-4, 1e-4);

        for cache in &caches {
            let Some(LayerCache::Attention(cache)) = cache else {
                panic!("every layer should hold a KV cache");
            };
            assert_eq!(cache.len(), pos + 1);
        }
    }

    Ok(())
}

#[test]
fn ministral_decode_cache_respects_fixed_capacity() -> Result<()> {
    let backend = Arc::new(CpuPortableBackend::new());
    let mut rng = StdRng::seed_from_u64(99);
    let model = Ministral::random(tiny_config(), backend, &mut rng)?;
    let tokens = [2, 4, 6, 8];

    let mut caches: Vec<Option<LayerCache<CpuPortableBackend>>> =
        vec![None; model.config.num_hidden_layers];
    let capacity = 16usize;
    for (pos, token) in tokens.iter().copied().enumerate() {
        model.forward_with_decode_cache(&[token], pos, &mut caches, Some(capacity))?;
        for cache in &caches {
            let Some(LayerCache::Attention(cache)) = cache else {
                panic!("every layer should hold a KV cache");
            };
            assert_eq!(cache.capacity(), capacity);
            assert_eq!(cache.len(), pos + 1);
        }
    }

    Ok(())
}
