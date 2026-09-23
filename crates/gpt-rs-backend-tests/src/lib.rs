pub mod api_invariants;
pub mod recording_backend;
pub mod smoke;
#[cfg(feature = "torch")]
pub mod torch_parity;

use std::sync::Arc;

use anyhow::{anyhow, Result};
use gpt_rs::backend::spec::PortableBackend;
use gpt_rs::nn::LayerLoader;
use gpt_rs::tensor::{DType, DeviceTensor, Shape, Tensor};

/// Builds a layer with `build`, passing it a [`LayerLoader`] over the host `tensors`, which are
/// keyed by parameter path.
pub fn load_layer<B: PortableBackend + 'static, L>(
    backend: &Arc<B>,
    tensors: impl IntoIterator<Item = (String, Tensor)>,
    build: impl FnOnce(&mut LayerLoader<'_, B>) -> Result<L>,
) -> Result<L> {
    let tensors: Vec<(String, Tensor)> = tensors.into_iter().collect();
    let mut get = |name: &str| {
        let (_, tensor) = tensors
            .iter()
            .find(|(candidate, _)| candidate == name)
            .ok_or_else(|| anyhow!("missing tensor '{name}'"))?;
        DeviceTensor::from_host(Arc::clone(backend), tensor.clone())
    };
    build(&mut LayerLoader::new(Arc::clone(backend), &mut get))
}

/// Host tensor holding `data` rounded to nearest-even in the floating `dtype`.
pub fn tensor_as(shape: &[usize], data: &[f32], dtype: DType) -> Tensor {
    let bytes: Vec<u8> = match dtype {
        DType::F32 => data.iter().flat_map(|v| v.to_le_bytes()).collect(),
        DType::BF16 => data
            .iter()
            .flat_map(|&v| half::bf16::from_f32(v).to_le_bytes())
            .collect(),
        DType::F16 => data
            .iter()
            .flat_map(|&v| half::f16::from_f32(v).to_le_bytes())
            .collect(),
        DType::I32 => panic!("tensor_as only rounds to floating dtypes"),
    };
    Tensor::from_le_bytes(Shape::new(shape.to_vec()), dtype, &bytes).unwrap()
}

#[macro_export]
macro_rules! define_backend_tests {
    ($module:ident, $backend_ctor:expr) => {
        $crate::define_backend_tests!($module, $backend_ctor, test);
    };
    ($module:ident, $backend_ctor:expr, $test_attr:meta) => {
        #[cfg(test)]
        mod $module {
            use std::sync::Arc;

            use $crate::{api_invariants, smoke};

            #[$test_attr]
            fn smoke_matmul_matches_expected() {
                let backend = ($backend_ctor)();
                smoke::matmul_matches_expected(&backend);
            }

            #[$test_attr]
            fn smoke_gpt_forward_shape() {
                let backend = ($backend_ctor)();
                smoke::gpt_forward_shape(&backend);
            }

            #[$test_attr]
            fn smoke_gpt_kv_cache_matches_full_context_decode() {
                let backend = ($backend_ctor)();
                smoke::gpt_kv_cache_matches_full_context_decode(&backend);
            }

            #[$test_attr]
            fn smoke_qwen3_5_prefill_chunking_and_cache_growth_agree() {
                let backend = ($backend_ctor)();
                smoke::qwen3_5_prefill_chunking_and_cache_growth_agree(&backend);
            }

            #[$test_attr]
            fn api_linear_loads_named_parameters() {
                let backend = ($backend_ctor)();
                api_invariants::linear_loads_named_parameters(&backend);
            }

            #[$test_attr]
            fn api_layer_norm_loads_named_parameters() {
                let backend = ($backend_ctor)();
                api_invariants::layer_norm_loads_named_parameters(&backend);
            }

            #[$test_attr]
            fn api_embedding_loads_named_weight() {
                let backend = ($backend_ctor)();
                api_invariants::embedding_loads_named_weight(&backend);
            }

            #[$test_attr]
            fn api_causal_self_attention_validates_projection_shapes() {
                let backend = ($backend_ctor)();
                api_invariants::causal_self_attention_validates_projection_shapes(&backend);
            }

            #[cfg(feature = "torch")]
            mod torch_parity_tests {
                use super::*;

                use $crate::torch_parity::{
                    arithmetic, attention, elementwise_fusion, embedding_layer, feed_forward_layer,
                    functional_ops, gated_feed_forward_layer, harness, layer_norm_layer, linear,
                    linear_attention, matmul, multi_head_attention_layer, rms_norm_layer,
                    rotary_ops, vision_ops,
                };

                macro_rules! run_parity {
                    ($backend:expr, $name:ident, $body:expr) => {{
                        harness::run_parity_test_with_modes(
                            ::std::sync::Arc::clone(&$backend),
                            stringify!($name),
                            $body,
                        );
                    }};
                }

                macro_rules! parity_test {
                    ($name:ident, $func:path) => {
                        #[$test_attr]
                        fn $name() {
                            let backend = ($backend_ctor)();
                            run_parity!(backend, $name, |backend| {
                                $func(backend);
                            });
                        }
                    };
                }

            parity_test!(torch_arithmetic_add_matches_torch_shape_2x3x4, arithmetic::add_matches_torch_shape_2x3x4);
            parity_test!(torch_arithmetic_add_matches_torch_shape_1x1x1, arithmetic::add_matches_torch_shape_1x1x1);
            parity_test!(torch_arithmetic_add_matches_torch_shape_2x7x13, arithmetic::add_matches_torch_shape_2x7x13);
            parity_test!(torch_arithmetic_add_matches_torch_shape_1x31x37, arithmetic::add_matches_torch_shape_1x31x37);
            parity_test!(torch_arithmetic_sub_matches_torch_shape_2x3x4, arithmetic::sub_matches_torch_shape_2x3x4);
            parity_test!(torch_arithmetic_sub_matches_torch_shape_3x5x9, arithmetic::sub_matches_torch_shape_3x5x9);
            parity_test!(torch_arithmetic_sub_matches_torch_shape_4x32, arithmetic::sub_matches_torch_shape_4x32);
            parity_test!(torch_arithmetic_mul_matches_torch_shape_2x3x4, arithmetic::mul_matches_torch_shape_2x3x4);
            parity_test!(torch_arithmetic_mul_matches_torch_shape_3x5x9, arithmetic::mul_matches_torch_shape_3x5x9);
            parity_test!(torch_arithmetic_mul_matches_torch_shape_2x3x1024, arithmetic::mul_matches_torch_shape_2x3x1024);
            parity_test!(torch_arithmetic_div_matches_torch_shape_2x3x4, arithmetic::div_matches_torch_shape_2x3x4);
            parity_test!(torch_arithmetic_div_matches_torch_shape_3x5x9, arithmetic::div_matches_torch_shape_3x5x9);
            parity_test!(torch_arithmetic_div_matches_torch_shape_2x3x1024, arithmetic::div_matches_torch_shape_2x3x1024);
            parity_test!(torch_arithmetic_neg_matches_torch_shape_2x3x4, arithmetic::neg_matches_torch_shape_2x3x4);
            parity_test!(torch_arithmetic_neg_matches_torch_shape_1x16, arithmetic::neg_matches_torch_shape_1x16);
            parity_test!(torch_arithmetic_abs_matches_torch_shape_2x3x4, arithmetic::abs_matches_torch_shape_2x3x4);
            parity_test!(torch_arithmetic_abs_matches_torch_shape_1x16, arithmetic::abs_matches_torch_shape_1x16);
            parity_test!(torch_arithmetic_max_matches_torch_shape_2x3x4, arithmetic::max_matches_torch_shape_2x3x4);
            parity_test!(torch_arithmetic_max_matches_torch_shape_2x7x13, arithmetic::max_matches_torch_shape_2x7x13);
            parity_test!(torch_arithmetic_min_matches_torch_shape_2x3x4, arithmetic::min_matches_torch_shape_2x3x4);
            parity_test!(torch_arithmetic_min_matches_torch_shape_2x7x13, arithmetic::min_matches_torch_shape_2x7x13);
            parity_test!(torch_arithmetic_clamp_matches_torch_min_max_shape_2x3x4, arithmetic::clamp_matches_torch_min_max_shape_2x3x4);
            parity_test!(torch_arithmetic_clamp_matches_torch_min_only_shape_2x3x4, arithmetic::clamp_matches_torch_min_only_shape_2x3x4);
            parity_test!(torch_arithmetic_clamp_matches_torch_max_only_shape_2x3x4, arithmetic::clamp_matches_torch_max_only_shape_2x3x4);
            parity_test!(torch_arithmetic_add_rejects_shape_mismatch, arithmetic::add_rejects_shape_mismatch);
            parity_test!(torch_arithmetic_div_rejects_shape_mismatch, arithmetic::div_rejects_shape_mismatch);
            parity_test!(torch_functional_softmax_last_dim_matches_torch, functional_ops::softmax_last_dim_matches_torch);
            parity_test!(torch_functional_softmax_last_dim_len1_matches_torch, functional_ops::softmax_last_dim_len1_matches_torch);
            parity_test!(torch_functional_softmax_last_dim_len2_matches_torch, functional_ops::softmax_last_dim_len2_matches_torch);
            parity_test!(torch_functional_softmax_last_dim_len7_matches_torch, functional_ops::softmax_last_dim_len7_matches_torch);
            parity_test!(torch_functional_softmax_last_dim_len128_matches_torch, functional_ops::softmax_last_dim_len128_matches_torch);
            parity_test!(torch_functional_softmax_last_dim_2x3x8_matches_torch, functional_ops::softmax_last_dim_2x3x8_matches_torch);
            parity_test!(torch_functional_softmax_last_dim_2x4x32_matches_torch, functional_ops::softmax_last_dim_2x4x32_matches_torch);
            parity_test!(torch_functional_softmax_last_dim_constant_logits_matches_torch, functional_ops::softmax_last_dim_constant_logits_matches_torch);
            parity_test!(torch_functional_softmax_last_dim_extreme_logits_matches_torch, functional_ops::softmax_last_dim_extreme_logits_matches_torch);
            parity_test!(torch_functional_softmax_last_dim_misaligned_matches_torch, functional_ops::softmax_last_dim_misaligned_matches_torch);
            parity_test!(torch_functional_gelu_matches_torch, functional_ops::gelu_matches_torch);
            parity_test!(torch_functional_silu_matches_torch, functional_ops::silu_matches_torch);
            parity_test!(torch_functional_silu_extreme_inputs_match_torch, functional_ops::silu_extreme_inputs_match_torch);
            parity_test!(torch_functional_swiglu_matches_torch, functional_ops::swiglu_matches_torch);
            parity_test!(torch_functional_swiglu_rejects_shape_mismatch, functional_ops::swiglu_rejects_shape_mismatch);
            parity_test!(torch_rotary_apply_matches_torch_full_rotary, rotary_ops::rope_apply_matches_torch_full_rotary);
            parity_test!(torch_rotary_apply_matches_torch_partial_rotary, rotary_ops::rope_apply_matches_torch_partial_rotary);
            parity_test!(torch_rotary_apply_rejects_sequence_mismatch, rotary_ops::rope_apply_rejects_sequence_mismatch);
            parity_test!(torch_rotary_cache_yarn_scaling_matches_formula, rotary_ops::rope_cache_yarn_scaling_matches_formula);
            parity_test!(torch_functional_gelu_matches_torch_1d_256, functional_ops::gelu_matches_torch_1d_256);
            parity_test!(torch_functional_gelu_matches_torch_2d_4x16, functional_ops::gelu_matches_torch_2d_4x16);
            parity_test!(torch_functional_gelu_matches_torch_3d_2x3x8, functional_ops::gelu_matches_torch_3d_2x3x8);
            parity_test!(torch_functional_gelu_extreme_inputs_match_torch, functional_ops::gelu_extreme_inputs_match_torch);
            parity_test!(torch_functional_gelu_tanh_matches_torch_3d_2x3x8, functional_ops::gelu_tanh_matches_torch_3d_2x3x8);
            parity_test!(torch_functional_gelu_tanh_extreme_inputs_match_torch, functional_ops::gelu_tanh_extreme_inputs_match_torch);
            parity_test!(torch_functional_add_bias_matches_torch, functional_ops::add_bias_matches_torch);
            parity_test!(torch_functional_add_bias_matches_torch_3d_2x5x8, functional_ops::add_bias_matches_torch_3d_2x5x8);
            parity_test!(torch_functional_add_bias_matches_torch_3d_1x16x64, functional_ops::add_bias_matches_torch_3d_1x16x64);
            parity_test!(torch_functional_add_bias_matches_torch_4d_2x3x4x5, functional_ops::add_bias_matches_torch_4d_2x3x4x5);
            parity_test!(torch_functional_add_bias_rejects_mismatched_dimension, functional_ops::add_bias_rejects_mismatched_dimension);
            parity_test!(torch_functional_add_bias_rejects_mismatched_dimension_3d, functional_ops::add_bias_rejects_mismatched_dimension_3d);
            parity_test!(torch_functional_layer_norm_matches_torch, functional_ops::layer_norm_matches_torch);
            parity_test!(torch_functional_rms_norm_matches_torch, functional_ops::rms_norm_matches_torch);
            parity_test!(torch_functional_rms_norm_rejects_gamma_mismatch, functional_ops::rms_norm_rejects_gamma_mismatch);
            parity_test!(torch_functional_layer_norm_matches_torch_embed_dim1, functional_ops::layer_norm_matches_torch_embed_dim1);
            parity_test!(torch_functional_layer_norm_matches_torch_prime_embed, functional_ops::layer_norm_matches_torch_prime_embed);
            parity_test!(torch_functional_layer_norm_matches_torch_large_embed, functional_ops::layer_norm_matches_torch_large_embed);
            parity_test!(torch_functional_layer_norm_matches_torch_batch1, functional_ops::layer_norm_matches_torch_batch1);
            parity_test!(torch_functional_layer_norm_matches_torch_constant_input, functional_ops::layer_norm_matches_torch_constant_input);
            parity_test!(torch_functional_layer_norm_matches_torch_eps_1e3, functional_ops::layer_norm_matches_torch_eps_1e3);
            parity_test!(torch_functional_layer_norm_rejects_gamma_mismatch, functional_ops::layer_norm_rejects_gamma_mismatch);
            parity_test!(torch_functional_linear_bf16_weight_rows1_matches_torch, functional_ops::linear_bf16_weight_rows1_matches_torch);
            parity_test!(torch_functional_linear_bf16_weight_rows64_in2100_matches_torch, functional_ops::linear_bf16_weight_rows64_in2100_matches_torch);
            parity_test!(torch_functional_linear_f32_weight_matches_torch, functional_ops::linear_f32_weight_matches_torch);
            parity_test!(torch_functional_linear_bf16_inputs_rows1_matches_torch, functional_ops::linear_bf16_inputs_rows1_matches_torch);
            parity_test!(torch_functional_linear_bf16_inputs_rows141_matches_torch, functional_ops::linear_bf16_inputs_rows141_matches_torch);
            parity_test!(torch_functional_linear_bf16_inputs_rows130_odd_in2049_matches_torch, functional_ops::linear_bf16_inputs_rows130_odd_in2049_matches_torch);
            parity_test!(torch_functional_cast_f32_bf16_roundtrip_matches_torch, functional_ops::cast_f32_bf16_roundtrip_matches_torch);
            parity_test!(torch_functional_sigmoid_matches_torch, functional_ops::sigmoid_matches_torch);
            parity_test!(torch_functional_softplus_matches_torch, functional_ops::softplus_matches_torch);
            parity_test!(torch_functional_exp_matches_torch, functional_ops::exp_matches_torch);
            parity_test!(torch_functional_mul_last_dim_matches_torch, functional_ops::mul_last_dim_matches_torch);
            parity_test!(torch_elementwise_fusion_two_offset_slices_feed_one_chain_matches_torch, elementwise_fusion::two_offset_slices_feed_one_chain_matches_torch);
            parity_test!(torch_elementwise_fusion_smaller_producers_broadcast_into_chain_matches_torch, elementwise_fusion::smaller_producers_broadcast_into_chain_matches_torch);
            parity_test!(torch_elementwise_fusion_chain_with_bf16_store_matches_torch, elementwise_fusion::chain_with_bf16_store_matches_torch);
            parity_test!(torch_matmul_matches_torch_1x1_1x1, matmul::matmul_matches_torch_1x1_1x1);
            parity_test!(torch_matmul_matches_torch_1x5_5x3, matmul::matmul_matches_torch_1x5_5x3);
            parity_test!(torch_matmul_matches_torch_4x5_5x1, matmul::matmul_matches_torch_4x5_5x1);
            parity_test!(torch_matmul_matches_torch_7x13_13x9, matmul::matmul_matches_torch_7x13_13x9);
            parity_test!(torch_matmul_matches_torch_33x65_65x31, matmul::matmul_matches_torch_33x65_65x31);
            parity_test!(torch_matmul_matches_torch_8x1_1x8, matmul::matmul_matches_torch_8x1_1x8);
            parity_test!(torch_matmul_matches_torch_1x16_16x17, matmul::matmul_matches_torch_1x16_16x17);
            parity_test!(torch_matmul_matches_torch_64x128_128x32, matmul::matmul_matches_torch_64x128_128x32);
            parity_test!(torch_batched_matmul_matches_torch_b1_4x5_5x2, matmul::batched_matmul_matches_torch_b1_4x5_5x2);
            parity_test!(torch_batched_matmul_matches_torch_b2_7x13_13x9, matmul::batched_matmul_matches_torch_b2_7x13_13x9);
            parity_test!(torch_batched_matmul_matches_torch_b8_4x8_8x4, matmul::batched_matmul_matches_torch_b8_4x8_8x4);
            parity_test!(torch_batched_matmul_matches_torch_b3_33x65_65x31, matmul::batched_matmul_matches_torch_b3_33x65_65x31);
            parity_test!(torch_batched_matmul_matches_torch_b2_6x40_40x150, matmul::batched_matmul_matches_torch_b2_6x40_40x150);
            parity_test!(torch_batched_dot_nt_matches_torch_b2_m5, matmul::batched_dot_nt_matches_torch_b2_m5);
            parity_test!(torch_batched_dot_nt_matches_torch_b2_m5_k1100, matmul::batched_dot_nt_matches_torch_b2_m5_k1100);
            parity_test!(torch_batched_dot_nt_matches_torch_b2_m70, matmul::batched_dot_nt_matches_torch_b2_m70);
            parity_test!(torch_batched_dot_tn_matches_torch_b2_m3, matmul::batched_dot_tn_matches_torch_b2_m3);
            parity_test!(torch_matmul_rejects_inner_dim_mismatch, matmul::matmul_rejects_inner_dim_mismatch);
            parity_test!(torch_batched_matmul_rejects_batch_mismatch, matmul::batched_matmul_rejects_batch_mismatch);
            parity_test!(torch_vision_conv2d_nhwc_matches_torch, vision_ops::conv2d_nhwc_matches_torch);
            parity_test!(torch_vision_conv2d_nhwc_kernel3_stride1_matches_torch, vision_ops::conv2d_nhwc_kernel3_stride1_matches_torch);
            parity_test!(torch_vision_conv2d_nhwc_kernel7_matches_torch, vision_ops::conv2d_nhwc_kernel7_matches_torch);
            parity_test!(torch_vision_conv2d_nhwc_kernel1_stride1_matches_torch, vision_ops::conv2d_nhwc_kernel1_stride1_matches_torch);
            parity_test!(torch_vision_conv2d_nhwc_kernel1_stride2_matches_torch, vision_ops::conv2d_nhwc_kernel1_stride2_matches_torch);
            parity_test!(torch_vision_conv2d_nhwc_kernel1_stride2_resnet_matches_torch, vision_ops::conv2d_nhwc_kernel1_stride2_resnet_matches_torch);
            parity_test!(torch_vision_conv2d_nhwc_kernel3_stride1_resnet_matches_torch, vision_ops::conv2d_nhwc_kernel3_stride1_resnet_matches_torch);
            parity_test!(torch_vision_conv2d_nhwc_kernel7_stride2_resnet_matches_torch, vision_ops::conv2d_nhwc_kernel7_stride2_resnet_matches_torch);
            parity_test!(torch_vision_depthwise_conv2d_nhwc_matches_torch, vision_ops::depthwise_conv2d_nhwc_matches_torch);
            parity_test!(torch_vision_depthwise_conv2d_nhwc_stride2_matches_torch, vision_ops::depthwise_conv2d_nhwc_stride2_matches_torch);
            parity_test!(torch_vision_max_pool2d_nhwc_matches_torch, vision_ops::max_pool2d_nhwc_matches_torch);
            parity_test!(torch_vision_relu6_matches_torch, vision_ops::relu6_matches_torch);
            parity_test!(torch_vision_global_avg_pool2d_matches_torch, vision_ops::global_avg_pool2d_matches_torch);
            parity_test!(torch_vision_conv2d_nhwc_k3_s1_p1_bias_n1_h11_w13_c3_cout8, vision_ops::conv2d_nhwc_k3_s1_p1_bias_n1_h11_w13_c3_cout8);
            parity_test!(torch_vision_conv2d_nhwc_k3_s2_p1_nobias_n2_h9_w10_c5_cout7, vision_ops::conv2d_nhwc_k3_s2_p1_nobias_n2_h9_w10_c5_cout7);
            parity_test!(torch_vision_conv2d_nhwc_k5_s1_p2_bias_n1_h15_w17_c4_cout6, vision_ops::conv2d_nhwc_k5_s1_p2_bias_n1_h15_w17_c4_cout6);
            parity_test!(torch_vision_conv2d_nhwc_k1_s1_p0_bias_n4_h7_w7_c8_cout8, vision_ops::conv2d_nhwc_k1_s1_p0_bias_n4_h7_w7_c8_cout8);
            parity_test!(torch_vision_conv2d_nhwc_k3x5_s2x1_p1x2_bias_n1_h11_w12_c4_cout6, vision_ops::conv2d_nhwc_k3x5_s2x1_p1x2_bias_n1_h11_w12_c4_cout6);
            parity_test!(torch_vision_conv2d_nhwc_k3_s1_p2_d2_bias_n1_h13_w13_c4_cout8, vision_ops::conv2d_nhwc_k3_s1_p2_d2_bias_n1_h13_w13_c4_cout8);
            parity_test!(torch_vision_group_conv2d_nhwc_g2_k3_s1_p1_bias_n1_h11_w11_c8_cout12, vision_ops::group_conv2d_nhwc_g2_k3_s1_p1_bias_n1_h11_w11_c8_cout12);
            parity_test!(torch_vision_depthwise_conv2d_nhwc_k5_s2_p2_n1_h15_w17_c8, vision_ops::depthwise_conv2d_nhwc_k5_s2_p2_n1_h15_w17_c8);
            parity_test!(torch_vision_max_pool2d_nhwc_w2_s2_p0_n1_h8_w8_c3, vision_ops::max_pool2d_nhwc_w2_s2_p0_n1_h8_w8_c3);
            parity_test!(torch_vision_relu6_edge_values_matches_torch, vision_ops::relu6_edge_values_matches_torch);
            parity_test!(torch_linear_matches_torch_with_bias, linear::linear_matches_torch_with_bias);
            parity_test!(torch_linear_matches_torch_without_bias, linear::linear_matches_torch_without_bias);
            parity_test!(torch_linear_matches_torch_batch1_in5_out3_bias, linear::linear_matches_torch_batch1_in5_out3_bias);
            parity_test!(torch_linear_matches_torch_batch64_in64_out64_bias, linear::linear_matches_torch_batch64_in64_out64_bias);
            parity_test!(torch_linear_matches_torch_batch7_in13_out17_bias, linear::linear_matches_torch_batch7_in13_out17_bias);
            parity_test!(torch_linear_matches_torch_batch4_in1_out1_bias, linear::linear_matches_torch_batch4_in1_out1_bias);
            parity_test!(torch_linear_matches_torch_batch4_in33_out65_no_bias, linear::linear_matches_torch_batch4_in33_out65_no_bias);
            parity_test!(torch_linear_rejects_input_dim_mismatch, linear::linear_rejects_input_dim_mismatch);
            parity_test!(torch_feed_forward_matches_torch_with_bias, feed_forward_layer::feed_forward_matches_torch_with_bias);
            parity_test!(torch_feed_forward_matches_torch_without_bias, feed_forward_layer::feed_forward_matches_torch_without_bias);
            parity_test!(torch_feed_forward_state_records_activation, feed_forward_layer::feed_forward_state_records_activation);
            parity_test!(torch_feed_forward_matches_torch_batch1_embed32_hidden128_bias, feed_forward_layer::feed_forward_matches_torch_batch1_embed32_hidden128_bias);
            parity_test!(torch_feed_forward_matches_torch_batch4_embed64_hidden256_bias, feed_forward_layer::feed_forward_matches_torch_batch4_embed64_hidden256_bias);
            parity_test!(torch_feed_forward_matches_torch_batch7_embed13_hidden31_no_bias, feed_forward_layer::feed_forward_matches_torch_batch7_embed13_hidden31_no_bias);
            parity_test!(torch_feed_forward_state_records_activation_batch4_embed32_hidden128, feed_forward_layer::feed_forward_state_records_activation_batch4_embed32_hidden128);
            parity_test!(torch_feed_forward_state_records_activation_extreme_inputs, feed_forward_layer::feed_forward_state_records_activation_extreme_inputs);
            parity_test!(torch_feed_forward_gelu_tanh_matches_torch, feed_forward_layer::feed_forward_gelu_tanh_matches_torch);
            parity_test!(torch_feed_forward_gelu_tanh_activation_extreme_inputs, feed_forward_layer::feed_forward_gelu_tanh_activation_extreme_inputs);
            parity_test!(torch_gated_feed_forward_matches_torch_with_bias, gated_feed_forward_layer::gated_feed_forward_matches_torch_with_bias);
            parity_test!(torch_gated_feed_forward_matches_torch_without_bias, gated_feed_forward_layer::gated_feed_forward_matches_torch_without_bias);
            parity_test!(torch_gated_feed_forward_matches_torch_batch4_embed64_hidden256_bias, gated_feed_forward_layer::gated_feed_forward_matches_torch_batch4_embed64_hidden256_bias);
            parity_test!(torch_gated_feed_forward_state_records_swiglu_hidden, gated_feed_forward_layer::gated_feed_forward_state_records_swiglu_hidden);
            parity_test!(torch_layer_norm_matches_torch_basic, layer_norm_layer::layer_norm_matches_torch_basic);
            parity_test!(torch_layer_norm_forward_with_state_matches_moments, layer_norm_layer::layer_norm_forward_with_state_matches_moments);
            parity_test!(torch_layer_norm_matches_torch_embed_dim1, layer_norm_layer::layer_norm_matches_torch_embed_dim1);
            parity_test!(torch_layer_norm_matches_torch_prime_embed, layer_norm_layer::layer_norm_matches_torch_prime_embed);
            parity_test!(torch_layer_norm_matches_torch_large_embed, layer_norm_layer::layer_norm_matches_torch_large_embed);
            parity_test!(torch_layer_norm_matches_torch_constant_input, layer_norm_layer::layer_norm_matches_torch_constant_input);
            parity_test!(torch_layer_norm_matches_torch_eps_1e3, layer_norm_layer::layer_norm_matches_torch_eps_1e3);
            parity_test!(torch_layer_norm_matches_torch_eps_1e1, layer_norm_layer::layer_norm_matches_torch_eps_1e1);
            parity_test!(torch_layer_norm_state_constant_input_matches_moments, layer_norm_layer::layer_norm_state_constant_input_matches_moments);
            parity_test!(torch_rms_norm_layer_matches_torch_basic, rms_norm_layer::rms_norm_layer_matches_torch_basic);
            parity_test!(torch_rms_norm_layer_matches_torch_seq_batch, rms_norm_layer::rms_norm_layer_matches_torch_seq_batch);
            parity_test!(torch_rms_norm_layer_matches_torch_constant_input, rms_norm_layer::rms_norm_layer_matches_torch_constant_input);
            parity_test!(torch_rms_norm_layer_unit_offset_matches_torch, rms_norm_layer::rms_norm_layer_unit_offset_matches_torch);
            parity_test!(torch_embedding_matches_torch_basic, embedding_layer::embedding_matches_torch_basic);
            parity_test!(torch_embedding_supports_duplicate_indices, embedding_layer::embedding_supports_duplicate_indices);
            parity_test!(torch_embedding_matches_torch_vocab64_embed32_seq16_rank1, embedding_layer::embedding_matches_torch_vocab64_embed32_seq16_rank1);
            parity_test!(torch_embedding_matches_torch_vocab32_embed8_seq5, embedding_layer::embedding_matches_torch_vocab32_embed8_seq5);
            parity_test!(torch_embedding_matches_torch_vocab32_embed128_seq8, embedding_layer::embedding_matches_torch_vocab32_embed128_seq8);
            parity_test!(torch_embedding_bf16_table_returns_f32_rows, embedding_layer::embedding_bf16_table_returns_f32_rows);
            parity_test!(torch_embedding_rejects_indices_rank2, embedding_layer::embedding_rejects_indices_rank2);
            parity_test!(torch_embedding_rejects_indices_rank3, embedding_layer::embedding_rejects_indices_rank3);
            parity_test!(torch_multi_head_attention_matches_torch_without_bias, multi_head_attention_layer::multi_head_attention_matches_torch_without_bias);
            parity_test!(torch_multi_head_attention_seq1_embed32_heads4_bias_matches_torch, multi_head_attention_layer::multi_head_attention_seq1_embed32_heads4_bias_matches_torch);
            parity_test!(torch_multi_head_attention_seq8_embed32_heads4_bias_matches_torch, multi_head_attention_layer::multi_head_attention_seq8_embed32_heads4_bias_matches_torch);
            parity_test!(torch_multi_head_attention_seq8_embed32_heads8_kv1_bias_matches_torch, multi_head_attention_layer::multi_head_attention_seq8_embed32_heads8_kv1_bias_matches_torch);
            parity_test!(torch_multi_head_attention_seq8_embed32_heads8_kv2_bias_matches_torch, multi_head_attention_layer::multi_head_attention_seq8_embed32_heads8_kv2_bias_matches_torch);
            parity_test!(torch_multi_head_attention_head_dim1_embed8_heads8_bias_matches_torch, multi_head_attention_layer::multi_head_attention_head_dim1_embed8_heads8_bias_matches_torch);
            parity_test!(torch_multi_head_attention_prefill4_decode3_matches_full_sequence, multi_head_attention_layer::multi_head_attention_prefill4_decode3_matches_full_sequence);
            parity_test!(torch_multi_head_attention_prefill4_decode3_grouped_kv2_matches_full_sequence, multi_head_attention_layer::multi_head_attention_prefill4_decode3_grouped_kv2_matches_full_sequence);
            parity_test!(torch_attention_full_rotary_grouped_kv2_prefill4_decode3_matches_torch, multi_head_attention_layer::attention_full_rotary_grouped_kv2_prefill4_decode3_matches_torch);
            parity_test!(torch_attention_qk_norm_matches_torch, multi_head_attention_layer::attention_qk_norm_matches_torch);
            parity_test!(torch_attention_output_gate_grouped_kv2_prefill5_decode2_matches_torch, multi_head_attention_layer::attention_output_gate_grouped_kv2_prefill5_decode2_matches_torch);
            parity_test!(torch_attention_partial_rotary_qk_norm_output_gate_prefill5_decode3_matches_torch, multi_head_attention_layer::attention_partial_rotary_qk_norm_output_gate_prefill5_decode3_matches_torch);
            parity_test!(torch_causal_conv1d_decode_step_matches_torch, linear_attention::causal_conv1d_decode_step_matches_torch);
            parity_test!(torch_causal_conv1d_prefill_matches_torch, linear_attention::causal_conv1d_prefill_matches_torch);
            parity_test!(torch_causal_conv1d_kernel2_matches_torch, linear_attention::causal_conv1d_kernel2_matches_torch);
            parity_test!(torch_gated_delta_rule_decode_step_matches_torch, linear_attention::gated_delta_rule_decode_step_matches_torch);
            parity_test!(torch_gated_delta_rule_prefill_zero_state_matches_torch, linear_attention::gated_delta_rule_prefill_zero_state_matches_torch);
            parity_test!(torch_gated_delta_rule_prefill_with_state_matches_torch, linear_attention::gated_delta_rule_prefill_with_state_matches_torch);
            parity_test!(torch_gated_delta_rule_multi_chunk_matches_torch, linear_attention::gated_delta_rule_multi_chunk_matches_torch);
            parity_test!(torch_gated_delta_rule_slow_decay_two_chunks_matches_torch, linear_attention::gated_delta_rule_slow_decay_two_chunks_matches_torch);
            parity_test!(torch_gated_delta_rule_repeated_keys_matches_torch, linear_attention::gated_delta_rule_repeated_keys_matches_torch);
            parity_test!(torch_attention_kv_cache_prefill_matches_torch, attention::attention_kv_cache_prefill_matches_torch);
            parity_test!(torch_attention_kv_cache_multi_query_full_prefill_matches_torch, attention::attention_kv_cache_multi_query_full_prefill_matches_torch);
            parity_test!(torch_attention_kv_cache_decode_step_matches_torch, attention::attention_kv_cache_decode_step_matches_torch);
            parity_test!(torch_attention_kv_cache_grouped_chunk_matches_torch, attention::attention_kv_cache_grouped_chunk_matches_torch);
            parity_test!(torch_attention_kv_cache_long_cache_matches_torch, attention::attention_kv_cache_long_cache_matches_torch);

            }
        }
    };
}
