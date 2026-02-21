use std::sync::Arc;

use gpt_rs::backend::spec::PortableBackend;
use gpt_rs::nn::layers::{
    AttentionConfig, CausalSelfAttention, Embedding, FeedForward, LayerNorm, Linear,
};
use gpt_rs::tensor::{DeviceTensor, Shape, Tensor};

pub fn linear_initializes_from_host_and_device_weights<B: PortableBackend + 'static>(
    backend: &Arc<B>,
) {
    let weight = Tensor::ones(Shape::new([4, 6]));
    let bias = Tensor::zeros(Shape::new([6]));

    let host_layer = Linear::new(Arc::clone(backend), weight.clone(), Some(bias.clone())).unwrap();
    let weight_device = DeviceTensor::from_host(Arc::clone(backend), weight.clone()).unwrap();
    let bias_device = DeviceTensor::from_host(Arc::clone(backend), bias.clone()).unwrap();
    let device_layer = Linear::new(
        Arc::clone(backend),
        weight_device.clone(),
        Some(bias_device.clone()),
    )
    .unwrap();
    let refs_layer = Linear::new(Arc::clone(backend), &weight_device, Some(&bias_device)).unwrap();

    assert_eq!(host_layer.weight.shape().dims(), &[4, 6]);
    assert_eq!(device_layer.weight.shape().dims(), &[4, 6]);
    let refs_backend = refs_layer.backend();
    assert!(Arc::ptr_eq(&refs_backend, backend));
}

pub fn layer_norm_initializes_from_host_and_device_tensors<B: PortableBackend + 'static>(
    backend: &Arc<B>,
) {
    let gamma = Tensor::ones(Shape::new([8]));
    let beta = Tensor::zeros(Shape::new([8]));

    let _ = LayerNorm::new(Arc::clone(backend), gamma.clone(), beta.clone(), 1e-5).unwrap();
    let gamma_device = DeviceTensor::from_host(Arc::clone(backend), gamma.clone()).unwrap();
    let beta_device = DeviceTensor::from_host(Arc::clone(backend), beta.clone()).unwrap();
    let layer = LayerNorm::new(Arc::clone(backend), &gamma_device, &beta_device, 1e-5).unwrap();

    assert_eq!(layer.gamma.shape().dims(), &[8]);
    assert_eq!(layer.beta.shape().dims(), &[8]);
}

pub fn embedding_initializes_from_device_tensor<B: PortableBackend + 'static>(backend: &Arc<B>) {
    let weight = Tensor::ones(Shape::new([32, 16]));

    let device_weight = DeviceTensor::from_host(Arc::clone(backend), weight).unwrap();
    let first = Embedding::new(Arc::clone(backend), device_weight.clone()).unwrap();
    let second = Embedding::new(Arc::clone(backend), device_weight).unwrap();

    assert_eq!(first.weight.shape().dims(), &[32, 16]);
    assert_eq!(second.weight.shape().dims(), &[32, 16]);
}

pub fn feed_forward_accepts_mixed_tensor_inputs<B: PortableBackend + 'static>(backend: &Arc<B>) {
    let w_in = Tensor::ones(Shape::new([8, 16]));
    let w_out = Tensor::ones(Shape::new([16, 8]));
    let b_in = Tensor::zeros(Shape::new([16]));
    let b_out = Tensor::zeros(Shape::new([8]));

    let w_out_device = DeviceTensor::from_host(Arc::clone(backend), w_out).unwrap();
    let b_out_device = DeviceTensor::from_host(Arc::clone(backend), b_out).unwrap();
    let layer = FeedForward::new(
        Arc::clone(backend),
        w_in,
        &w_out_device,
        Some(b_in),
        Some(&b_out_device),
    )
    .unwrap();

    assert_eq!(layer.w_in.weight.shape().dims(), &[8, 16]);
    assert_eq!(layer.w_out.weight.shape().dims(), &[16, 8]);
}

pub fn multi_head_attention_accepts_device_parameters<B: PortableBackend + 'static>(
    backend: &Arc<B>,
) {
    let config = AttentionConfig::with_equal_heads(8, 2);

    let w_qkv = Tensor::ones(Shape::new([8, 24]));
    let w_out = Tensor::ones(Shape::new([8, 8]));
    let b_qkv = Tensor::zeros(Shape::new([24]));
    let b_out = Tensor::zeros(Shape::new([8]));

    let w_qkv_device = DeviceTensor::from_host(Arc::clone(backend), w_qkv).unwrap();
    let w_out_device = DeviceTensor::from_host(Arc::clone(backend), w_out).unwrap();
    let b_qkv_device = DeviceTensor::from_host(Arc::clone(backend), b_qkv).unwrap();
    let b_out_device = DeviceTensor::from_host(Arc::clone(backend), b_out).unwrap();

    let attention = CausalSelfAttention::new(
        Arc::clone(backend),
        config,
        &w_qkv_device,
        &w_out_device,
        Some(&b_qkv_device),
        Some(&b_out_device),
    )
    .unwrap();

    assert_eq!(attention.proj_qkv.weight.shape().dims(), &[8, 24]);
    assert_eq!(attention.proj_out.weight.shape().dims(), &[8, 8]);
    let attention_backend = attention.backend();
    assert!(Arc::ptr_eq(&attention_backend, backend));
}
