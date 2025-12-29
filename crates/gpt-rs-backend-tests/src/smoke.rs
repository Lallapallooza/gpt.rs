use std::sync::Arc;

use gpt_rs::backend::spec::PortableBackend;
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
