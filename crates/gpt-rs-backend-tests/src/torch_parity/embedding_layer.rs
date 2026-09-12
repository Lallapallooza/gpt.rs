use std::sync::Arc;

use gpt_rs::backend::spec::PortableBackend;
use gpt_rs::module::Layer;
use gpt_rs::tensor::{DType, DeviceTensor, Shape, Tensor};
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
) {
    let mut rng = seeded_rng(seed);
    let weight_host = tensor_from_vec(&[vocab, embed_dim], random_vec(&mut rng, vocab * embed_dim));
    let indices: Vec<usize> = (0..seq_len).map(|_| rng.gen_range(0..vocab)).collect();

    let expected = timed_torch(|| {
        let weight_tch = tch_tensor_from_vec(&[vocab, embed_dim], weight_host.data());
        tensor_to_vec(&embedding_reference(&weight_tch, &indices))
    });

    let output_host = timed_gpt(|| {
        let indices_tensor = Tensor::from_i32(
            Shape::new([indices.len()]),
            indices.iter().map(|&idx| idx as i32).collect(),
        )
        .unwrap();
        let indices_device = DeviceTensor::from_host(Arc::clone(backend), indices_tensor).unwrap();
        let layer = embedding_layer(backend, weight_host.clone());
        let output_device = layer.call(&indices_device).unwrap();
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
        let indices_tensor = Tensor::from_i32(
            Shape::new([indices.len()]),
            indices.iter().map(|&idx| idx as i32).collect(),
        )
        .unwrap();
        let indices_device = DeviceTensor::from_host(Arc::clone(backend), indices_tensor).unwrap();
        let layer = embedding_layer(backend, weight_host.clone());
        let output_device = layer.call(&indices_device).unwrap();
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
        let indices_tensor = Tensor::from_i32(
            Shape::new([indices.len()]),
            indices.iter().map(|&idx| idx as i32).collect(),
        )
        .unwrap();
        let indices_device = DeviceTensor::from_host(Arc::clone(backend), indices_tensor).unwrap();
        let layer = embedding_layer(backend, weight_host.clone());
        let output_device = layer.call(&indices_device).unwrap();
        output_device.to_host().unwrap()
    });

    assert_close(&expected, output_host.data());
}

pub fn embedding_matches_torch_vocab64_embed32_seq16_rank1<B: PortableBackend + 'static>(
    backend: &Arc<B>,
) {
    run_embedding_case(backend, 64, 32, 16, 0xB010);
}

pub fn embedding_matches_torch_vocab32_embed8_seq5<B: PortableBackend + 'static>(backend: &Arc<B>) {
    run_embedding_case(backend, 32, 8, 5, 0xB012);
}

pub fn embedding_matches_torch_vocab32_embed128_seq8<B: PortableBackend + 'static>(
    backend: &Arc<B>,
) {
    run_embedding_case(backend, 32, 128, 8, 0xB013);
}

pub fn embedding_bf16_table_returns_f32_rows<B: PortableBackend + 'static>(backend: &Arc<B>) {
    let mut rng = seeded_rng(0xB014);
    let (vocab, embed_dim) = (32, 24);
    let weights: Vec<f32> = random_vec(&mut rng, vocab * embed_dim)
        .into_iter()
        .map(|v| half::bf16::from_f32(v).to_f32())
        .collect();
    let indices = [3usize, 0, 31, 3, 17];

    let expected = timed_torch(|| {
        let weight_tch = tch_tensor_from_vec(&[vocab, embed_dim], &weights);
        tensor_to_vec(&embedding_reference(&weight_tch, &indices))
    });

    let actual = timed_gpt(|| {
        let table = crate::tensor_as(&[vocab, embed_dim], &weights, DType::BF16);
        let ids = indices.iter().map(|&idx| idx as i32).collect();
        let ids = Tensor::from_i32(Shape::new([indices.len()]), ids).unwrap();
        let ids = DeviceTensor::from_host(Arc::clone(backend), ids).unwrap();
        let layer = embedding_layer(backend, table);
        let rows = layer.call(&ids).unwrap();
        assert_eq!(rows.dtype(), DType::F32);
        to_host_vec(&rows)
    });

    assert_close(&expected, &actual);
}

pub fn embedding_rejects_indices_rank2<B: PortableBackend + 'static>(backend: &Arc<B>) {
    let err = timed_gpt(|| {
        let weight = tensor_from_vec(&[8, 4], vec![0.0; 32]);
        let indices = Tensor::from_i32(Shape::new([3, 2]), vec![0, 1, 2, 3, 4, 5]).unwrap();
        let indices_device = DeviceTensor::from_host(Arc::clone(backend), indices).unwrap();
        let layer = embedding_layer(backend, weight);
        layer.call(&indices_device).unwrap_err()
    });
    assert!(
        err.to_string()
            .contains("embedding_lookup: indices must have rank 1, got [3, 2]"),
        "{err}"
    );
}

pub fn embedding_rejects_indices_rank3<B: PortableBackend + 'static>(backend: &Arc<B>) {
    let err = timed_gpt(|| {
        let weight = tensor_from_vec(&[8, 4], vec![0.0; 32]);
        let indices = Tensor::from_i32(Shape::new([2, 2, 1]), vec![0, 1, 2, 3]).unwrap();
        let indices_device = DeviceTensor::from_host(Arc::clone(backend), indices).unwrap();
        let layer = embedding_layer(backend, weight);
        layer.call(&indices_device).unwrap_err()
    });
    assert!(
        err.to_string()
            .contains("embedding_lookup: indices must have rank 1, got [2, 2, 1]"),
        "{err}"
    );
}
