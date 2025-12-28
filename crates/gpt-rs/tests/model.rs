use std::sync::Arc;

use gpt_rs::model::{Gpt, GptConfig};
use gpt_rs::ops::functional::FunctionalOverrides;
use gpt_rs::train::trainer::Trainer;
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
        context_length: 16,
        embed_dim: 8,
        num_layers: 2,
        num_heads: 2,
        mlp_ratio: 2,
        dropout: 0.0,
        functional_overrides: FunctionalOverrides::default(),
    };
    let model = Gpt::random(config, Arc::clone(&backend), &mut rng).unwrap();
    let tokens = vec![1, 2, 3, 4];
    let result = model.forward(&tokens);
    assert!(result.is_ok(), "forward failed: {:?}", result.err());
}

#[test]
fn trainer_updates_lm_head() {
    let backend = cpu_backend();
    let mut rng = StdRng::seed_from_u64(7);
    let config = GptConfig {
        vocab_size: 32,
        context_length: 8,
        embed_dim: 8,
        num_layers: 1,
        num_heads: 1,
        mlp_ratio: 2,
        dropout: 0.0,
        functional_overrides: FunctionalOverrides::default(),
    };
    let model = Gpt::random(config, Arc::clone(&backend), &mut rng).unwrap();
    let mut trainer = Trainer::new(model, 1e-2);
    let tokens = vec![1, 2, 3, 4];
    let result = trainer.train_step(&tokens, &[2, 3, 4, 5]);
    assert!(result.is_err());
}
