#![cfg(feature = "torch")]

use std::sync::Arc;

use gpt_rs_backend_ref_cpu::CpuPortableBackend;
use gpt_rs_backend_tests::torch_parity::vision_ops;
use gpt_rs_backend_tests::torch_parity::{functional_ops, layer_norm_layer};

fn cpu_backend() -> Arc<CpuPortableBackend> {
    Arc::new(CpuPortableBackend::new())
}

#[test]
fn functional_softmax_last_dim_matches_torch_reference() {
    let backend = cpu_backend();
    functional_ops::softmax_last_dim_matches_torch(&backend);
}

#[test]
fn functional_gelu_matches_torch_reference() {
    let backend = cpu_backend();
    functional_ops::gelu_matches_torch(&backend);
}

#[test]
fn functional_matmul_matches_torch_reference() {
    let backend = cpu_backend();
    functional_ops::matmul_matches_torch(&backend);
}

#[test]
fn layer_norm_state_matches_torch_reference() {
    let backend = cpu_backend();
    layer_norm_layer::layer_norm_forward_with_state_matches_moments(&backend);
}

#[test]
fn functional_attention_matches_torch_reference() {
    let backend = cpu_backend();
    functional_ops::attention_matches_torch(&backend);
}

#[test]
fn vision_conv2d_nhwc_matches_torch_reference() {
    let backend = cpu_backend();
    vision_ops::conv2d_nhwc_matches_torch(&backend);
}

#[test]
fn vision_conv2d_nhwc_kernel7_matches_torch_reference() {
    let backend = cpu_backend();
    vision_ops::conv2d_nhwc_kernel7_matches_torch(&backend);
}

#[test]
fn vision_conv2d_nhwc_kernel1_stride2_matches_torch_reference() {
    let backend = cpu_backend();
    vision_ops::conv2d_nhwc_kernel1_stride2_matches_torch(&backend);
}

#[test]
fn vision_depthwise_conv2d_nhwc_matches_torch_reference() {
    let backend = cpu_backend();
    vision_ops::depthwise_conv2d_nhwc_matches_torch(&backend);
}

#[test]
fn vision_max_pool2d_nhwc_matches_torch_reference() {
    let backend = cpu_backend();
    vision_ops::max_pool2d_nhwc_matches_torch(&backend);
}

#[test]
fn vision_relu6_matches_torch_reference() {
    let backend = cpu_backend();
    vision_ops::relu6_matches_torch(&backend);
}
