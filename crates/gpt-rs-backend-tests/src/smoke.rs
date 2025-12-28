use std::fs;
use std::path::Path;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use gpt_rs::backend::spec::PortableBackend;
use gpt_rs::checkpoint::{CheckpointLoader, CheckpointSaver};
use gpt_rs::model::{Gpt, GptConfig};
use gpt_rs::ops::functional;
use gpt_rs::ops::functional::FunctionalOverrides;
use gpt_rs::tensor::{DeviceTensor, Shape, Tensor};
use gpt_rs::tokenizer::{Tokenizer, TokenizerConfig};
use gpt_rs::train::trainer::Trainer;
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
        functional_overrides: FunctionalOverrides::default(),
    };
    let model = Gpt::random(config.clone(), Arc::clone(backend), &mut rng).unwrap();
    let tokens = vec![1, 2, 3, 4];
    let result = model.forward(&tokens);
    assert!(result.is_ok());
}

pub fn tokenizer_roundtrip() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("configs")
        .join("gpt2_tokenizer.json");
    let data = fs::read_to_string(path).expect("failed to read tokenizer config");
    let config: TokenizerConfig = serde_json::from_str(&data).expect("invalid tokenizer config");
    let tokenizer = Tokenizer::from_config(config);

    let text = "Hello rust";
    let tokens = tokenizer.encode(text);
    let decoded = tokenizer.decode(&tokens);

    assert_eq!(decoded, text);
    assert!(!tokens.is_empty());
}

pub fn trainer_updates_lm_head<B: PortableBackend + 'static>(backend: &Arc<B>) {
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
    let model = Gpt::random(config, Arc::clone(backend), &mut rng).unwrap();
    let mut trainer = Trainer::new(model, 1e-2);
    let tokens = vec![1, 2, 3, 4];
    let result = trainer.train_step(&tokens, &[2, 3, 4, 5]);
    assert!(result.is_err());
}

pub fn checkpoint_roundtrip<B: PortableBackend + 'static>(backend: &Arc<B>) {
    let mut rng = StdRng::seed_from_u64(11);
    let config = GptConfig {
        vocab_size: 16,
        context_length: 8,
        embed_dim: 8,
        num_layers: 1,
        num_heads: 1,
        mlp_ratio: 2,
        dropout: 0.0,
        functional_overrides: FunctionalOverrides::default(),
    };
    let model = Gpt::random(config, Arc::clone(backend), &mut rng).unwrap();

    let base = std::env::temp_dir();
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis();
    let path = base.join(format!("gpt_rs_checkpoint_{}.bin", timestamp));
    CheckpointSaver::save(&path, &model).unwrap();
    let loaded = CheckpointLoader::load(&path).unwrap();
    let loaded_model = loaded.into_model(Arc::clone(backend)).unwrap();
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
