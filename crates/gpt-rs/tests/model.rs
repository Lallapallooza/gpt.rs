use std::sync::Arc;

use gpt_rs::model::{Gpt, GptConfig};
use gpt_rs::nn::ActivationFunction;
use gpt_rs_backend_ref_cpu::CpuPortableBackend;
use rand::rngs::StdRng;
use rand::SeedableRng;

fn cpu_backend() -> Arc<CpuPortableBackend> {
    Arc::new(CpuPortableBackend::new())
}

#[test]
fn gpt_forward_shape() {
    let backend = cpu_backend();
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
    let model = Gpt::random(config, Arc::clone(&backend), &mut rng).unwrap();
    let tokens = vec![1, 2, 3, 4];
    let result = model.forward(&tokens);
    assert!(result.is_ok(), "forward failed: {:?}", result.err());
}
