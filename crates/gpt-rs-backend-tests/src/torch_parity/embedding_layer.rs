use std::sync::Arc;

use gpt_rs::backend::spec::PortableBackend;
use gpt_rs::nn::layers::Embedding;
use gpt_rs::tensor::{DeviceTensor, Shape, Tensor};
use rand::Rng;
use tch::{Kind, Tensor as TchTensor};

use super::common::*;

fn embedding_reference(weight: &TchTensor, indices: &[usize]) -> TchTensor {
    let idx: Vec<i64> = indices.iter().map(|&i| i as i64).collect();
    let idx = TchTensor::from_slice(&idx)
        .to_kind(Kind::Int64)
        .reshape([indices.len() as i64]);
    weight.index_select(0, &idx)
}

fn run_embedding_case<B: PortableBackend + 'static>(
    backend: &Arc<B>,
    vocab: usize,
    embed_dim: usize,
    seq_len: usize,
    seed: u64,
    rank2_indices: bool,
) {
    let mut rng = seeded_rng(seed);
    let weight_host = tensor_from_vec(&[vocab, embed_dim], random_vec(&mut rng, vocab * embed_dim));
    let indices: Vec<usize> = (0..seq_len).map(|_| rng.gen_range(0..vocab)).collect();

    let expected = timed_torch(|| {
        let weight_tch = tch_tensor_from_vec(&[vocab, embed_dim], weight_host.data());
        tensor_to_vec(&embedding_reference(&weight_tch, &indices))
    });

    let output_host = timed_gpt(|| {
        let weight_device =
            DeviceTensor::from_host(Arc::clone(backend), weight_host.clone()).unwrap();
        let shape = if rank2_indices {
            Shape::new([indices.len(), 1])
        } else {
            Shape::new([indices.len()])
        };
        let indices_tensor =
            Tensor::from_i32(shape, indices.iter().map(|&idx| idx as i32).collect()).unwrap();
        let indices_device = DeviceTensor::from_host(Arc::clone(backend), indices_tensor).unwrap();
        let layer = Embedding::new(Arc::clone(backend), weight_device.clone()).unwrap();
        let output_device = layer.forward(&indices_device).unwrap();
        output_device.to_host().unwrap()
    });

    assert_close(&expected, output_host.data());
}

pub fn embedding_matches_torch_basic<B: PortableBackend + 'static>(backend: &Arc<B>) {
    let mut rng = seeded_rng(0xB00B);
    let vocab = 16;
    let embed_dim = 8;
    let seq_len = 5;

    let weight_host = tensor_from_vec(&[vocab, embed_dim], random_vec(&mut rng, vocab * embed_dim));
    let indices: Vec<usize> = (0..seq_len).map(|_| rng.gen_range(0..vocab)).collect();

    let expected = timed_torch(|| {
        let weight_tch = tch_tensor_from_vec(&[vocab, embed_dim], weight_host.data());
        tensor_to_vec(&embedding_reference(&weight_tch, &indices))
    });

    let output_host = timed_gpt(|| {
        let weight_device =
            DeviceTensor::from_host(Arc::clone(backend), weight_host.clone()).unwrap();
        let indices_tensor = Tensor::from_i32(
            Shape::new([indices.len(), 1]),
            indices.iter().map(|&idx| idx as i32).collect(),
        )
        .unwrap();
        let indices_device = DeviceTensor::from_host(Arc::clone(backend), indices_tensor).unwrap();
        let layer = Embedding::new(Arc::clone(backend), weight_device.clone()).unwrap();
        let output_device = layer.forward(&indices_device).unwrap();
        output_device.to_host().unwrap()
    });

    assert_close(&expected, output_host.data());
}

pub fn embedding_supports_duplicate_indices<B: PortableBackend + 'static>(backend: &Arc<B>) {
    let mut rng = seeded_rng(0xFEED);
    let vocab = 10;
    let embed_dim = 6;

    let weight_host = tensor_from_vec(&[vocab, embed_dim], random_vec(&mut rng, vocab * embed_dim));
    let indices = vec![2, 2, 7, 2];

    let expected = timed_torch(|| {
        let weight_tch = tch_tensor_from_vec(&[vocab, embed_dim], weight_host.data());
        tensor_to_vec(&embedding_reference(&weight_tch, &indices))
    });

    let output_host = timed_gpt(|| {
        let weight_device =
            DeviceTensor::from_host(Arc::clone(backend), weight_host.clone()).unwrap();
        let indices_tensor = Tensor::from_i32(
            Shape::new([indices.len(), 1]),
            indices.iter().map(|&idx| idx as i32).collect(),
        )
        .unwrap();
        let indices_device = DeviceTensor::from_host(Arc::clone(backend), indices_tensor).unwrap();
        let layer = Embedding::new(Arc::clone(backend), weight_device.clone()).unwrap();
        let output_device = layer.forward(&indices_device).unwrap();
        output_device.to_host().unwrap()
    });

    assert_close(&expected, output_host.data());
}

pub fn embedding_forward_preserves_requires_grad<B: PortableBackend + 'static>(backend: &Arc<B>) {
    let mut rng = seeded_rng(0xC0DE);
    let vocab = 12;
    let embed_dim = 4;

    timed_gpt(|| {
        let weight = tensor_from_vec(&[vocab, embed_dim], random_vec(&mut rng, vocab * embed_dim))
            .requires_grad(true);
        let device_weight = DeviceTensor::from_host(Arc::clone(backend), weight).unwrap();
        let layer = Embedding::new(Arc::clone(backend), device_weight.clone()).unwrap();
        let indices = [1usize, 3, 5];
        let indices_tensor = Tensor::from_i32(
            Shape::new([indices.len(), 1]),
            indices.iter().map(|&idx| idx as i32).collect(),
        )
        .unwrap();
        let indices_device = DeviceTensor::from_host(Arc::clone(backend), indices_tensor).unwrap();
        let output = layer.forward(&indices_device).unwrap();

        assert!(output.requires_grad_flag());
    });
}

pub fn embedding_matches_torch_vocab64_embed32_seq16_rank1<B: PortableBackend + 'static>(
    backend: &Arc<B>,
) {
    run_embedding_case(backend, 64, 32, 16, 0xB010, false);
}

pub fn embedding_matches_torch_vocab64_embed32_seq16_rank2<B: PortableBackend + 'static>(
    backend: &Arc<B>,
) {
    run_embedding_case(backend, 64, 32, 16, 0xB011, true);
}

pub fn embedding_matches_torch_vocab32_embed8_seq5<B: PortableBackend + 'static>(backend: &Arc<B>) {
    run_embedding_case(backend, 32, 8, 5, 0xB012, false);
}

pub fn embedding_matches_torch_vocab32_embed128_seq8<B: PortableBackend + 'static>(
    backend: &Arc<B>,
) {
    run_embedding_case(backend, 32, 128, 8, 0xB013, false);
}

pub fn embedding_rejects_indices_last_dim_not_one<B: PortableBackend + 'static>(backend: &Arc<B>) {
    let err = timed_gpt(|| {
        let weight = tensor_from_vec(&[8, 4], vec![0.0; 32]);
        let weight_device = DeviceTensor::from_host(Arc::clone(backend), weight).unwrap();
        let indices = Tensor::from_i32(Shape::new([3, 2]), vec![0, 1, 2, 3, 4, 5]).unwrap();
        let indices_device = DeviceTensor::from_host(Arc::clone(backend), indices).unwrap();
        let layer = Embedding::new(Arc::clone(backend), weight_device).unwrap();
        layer.forward(&indices_device).unwrap_err()
    });
    assert!(err.to_string().contains("last dimension"));
}

pub fn embedding_rejects_indices_rank3<B: PortableBackend + 'static>(backend: &Arc<B>) {
    let err = timed_gpt(|| {
        let weight = tensor_from_vec(&[8, 4], vec![0.0; 32]);
        let weight_device = DeviceTensor::from_host(Arc::clone(backend), weight).unwrap();
        let indices = Tensor::from_i32(Shape::new([2, 2, 1]), vec![0, 1, 2, 3]).unwrap();
        let indices_device = DeviceTensor::from_host(Arc::clone(backend), indices).unwrap();
        let layer = Embedding::new(Arc::clone(backend), weight_device).unwrap();
        layer.forward(&indices_device).unwrap_err()
    });
    assert!(err.to_string().contains("embedding indices must be rank 1"));
}
