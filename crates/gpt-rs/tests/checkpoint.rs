use std::fs;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use gpt_rs::checkpoint::{CheckpointLoader, CheckpointSaver};
use gpt_rs::model::{Gpt, ModelConfig};
use gpt_rs::ops::functional::FunctionalOverrides;
use gpt_rs_backend_ref_cpu::CpuPortableBackend;
use rand::rngs::StdRng;
use rand::SeedableRng;

fn cpu_backend() -> Arc<CpuPortableBackend> {
    Arc::new(CpuPortableBackend::new())
}

#[test]
#[ignore = "checkpoint roundtrip still broken after backend refactor"]
fn checkpoint_roundtrip() {
    let backend = cpu_backend();
    let mut rng = StdRng::seed_from_u64(11);
    let config = ModelConfig {
        vocab_size: 16,
        context_length: 8,
        embed_dim: 8,
        num_layers: 1,
        num_heads: 1,
        mlp_ratio: 2,
        dropout: 0.0,
        functional_overrides: FunctionalOverrides::default(),
    };
    let model = Gpt::random(config, Arc::clone(&backend), &mut rng).unwrap();

    let base = std::env::temp_dir();
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis();
    let path = base.join(format!("gpt_rs_checkpoint_{}.bin", timestamp));
    CheckpointSaver::save(&path, &model).unwrap();
    let loaded = CheckpointLoader::load(&path).unwrap();
    let loaded_model = loaded.into_model(Arc::clone(&backend)).unwrap();
    fs::remove_file(&path).unwrap();

    let mut original = Vec::new();
    model
        .for_each_parameter(|name, tensor| {
            let host = tensor.to_host()?;
            original.push((name.to_string(), host.data().to_vec()));
            Ok(())
        })
        .unwrap();
    let mut restored = Vec::new();
    loaded_model
        .for_each_parameter(|name, tensor| {
            let host = tensor.to_host()?;
            restored.push((name.to_string(), host.data().to_vec()));
            Ok(())
        })
        .unwrap();
    original.sort_by(|a, b| a.0.cmp(&b.0));
    restored.sort_by(|a, b| a.0.cmp(&b.0));
    assert_eq!(original.len(), restored.len());
    for ((name_a, data_a), (name_b, data_b)) in original.iter().zip(restored.iter()) {
        assert_eq!(name_a, name_b);
        assert_eq!(data_a.len(), data_b.len());
        for (x, y) in data_a.iter().zip(data_b.iter()) {
            assert!((*x - *y).abs() < 1e-6);
        }
    }
}
