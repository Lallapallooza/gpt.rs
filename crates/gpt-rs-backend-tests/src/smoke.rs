use std::sync::Arc;

use gpt_rs::backend::spec::PortableBackend;
use gpt_rs::inference::generate::Generator;
use gpt_rs::inference::sampler::Sampler;
use gpt_rs::model::{Gpt, GptConfig};
use gpt_rs::ops::functional;
use gpt_rs::tensor::{DeviceTensor, Shape, Tensor};
use rand::rngs::StdRng;
use rand::SeedableRng;

pub fn matmul_matches_expected<B: PortableBackend + 'static>(backend: &Arc<B>) {
    let a = Tensor::from_vec(Shape::new([2, 2]), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let b = Tensor::from_vec(Shape::new([2, 2]), vec![5.0, 6.0, 7.0, 8.0]).unwrap();

    let a_device = DeviceTensor::from_host(Arc::clone(backend), a.clone()).unwrap();
    let b_device = DeviceTensor::from_host(Arc::clone(backend), b.clone()).unwrap();

    let result = functional::matmul(backend.as_ref(), &a_device, &b_device).unwrap();
    let host = result.to_host().unwrap();

    let expected = vec![19.0, 22.0, 43.0, 50.0];
    assert_eq!(host.data(), expected.as_slice());
}

pub fn gpt_forward_shape<B: PortableBackend + 'static>(backend: &Arc<B>) {
    let mut rng = StdRng::seed_from_u64(42);
    let config = GptConfig {
        vocab_size: 32,
        context_length: 16,
        embed_dim: 8,
        num_layers: 2,
        num_heads: 2,
        mlp_ratio: 2,
        dropout: 0.0,
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
        context_length: 32,
        embed_dim: 32,
        num_layers: 2,
        num_heads: 4,
        mlp_ratio: 2,
        dropout: 0.0,
    };
    let model = Gpt::random(config, Arc::clone(backend), &mut rng).unwrap();
    let sampler = Sampler::new(0.0);
    let prompt = vec![1usize, 2, 3, 4];
    let steps = 16usize;

    let mut kv_gen = Generator::new(&model, &sampler, prompt.as_slice(), true).unwrap();
    let mut full_gen = Generator::new(&model, &sampler, prompt.as_slice(), false).unwrap();
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
