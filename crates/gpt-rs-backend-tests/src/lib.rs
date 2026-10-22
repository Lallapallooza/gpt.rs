pub mod layers;
pub mod recording_backend;
#[cfg(feature = "torch")]
pub mod torch_parity;

use gpt_rs::gpt_rs_ir::ir::TensorType;
use gpt_rs::tensor::{DType, HostTensor};

/// The message of the panic that `action` raises.
pub fn panic_message<T>(action: impl FnOnce() -> T) -> String {
    let Err(payload) = std::panic::catch_unwind(std::panic::AssertUnwindSafe(action)) else {
        panic!("expected a panic");
    };
    match payload.downcast::<String>() {
        Ok(message) => *message,
        Err(other_payload) => other_payload
            .downcast::<&str>()
            .map(|message| message.to_string())
            .unwrap_or_default(),
    }
}

/// Host tensor holding `data` rounded to nearest-even in the floating `dtype`.
pub fn tensor_as(shape: &[usize], data: &[f32], dtype: DType) -> HostTensor {
    let bytes: Vec<u8> = match dtype {
        DType::F32 => data.iter().flat_map(|value| value.to_le_bytes()).collect(),
        DType::Bf16 => data
            .iter()
            .flat_map(|&value| half::bf16::from_f32(value).to_le_bytes())
            .collect(),
        DType::F16 => data
            .iter()
            .flat_map(|&value| half::f16::from_f32(value).to_le_bytes())
            .collect(),
        other => panic!("tensor_as rounds only to f32, bf16 and f16, not {other}"),
    };
    HostTensor::from_le_bytes(&TensorType::new(dtype, shape.iter().copied()), &bytes).unwrap()
}

#[macro_export]
macro_rules! define_backend_tests {
    ($module:ident, $backend_ctor:expr) => {
        $crate::define_backend_tests!($module, $backend_ctor, test);
    };
    ($module:ident, $backend_ctor:expr, $test_attr:meta) => {
        #[cfg(all(test, feature = "torch"))]
        mod $module {
            use std::sync::Arc;

            use $crate::torch_parity::{
                arithmetic, attention, conv1d_layer, elementwise_fusion, embedding_layer,
                empty_extents, functional_ops, gated_feed_forward_layer, harness, indexing,
                layer_norm_layer, linear, linear_attention, matmul, multi_head_attention_layer,
                narrow_weights, rms_norm_layer, rotary_ops, sampling, vision_layers, vision_ops,
            };

            macro_rules! run_parity {
                ($backend:expr, $name:ident, $body:expr) => {{
                    harness::run_parity_test(
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
            parity_test!(torch_arithmetic_add_matches_torch_shape_3x197x113, arithmetic::add_matches_torch_shape_3x197x113);
            parity_test!(torch_arithmetic_sub_matches_torch_shape_2x3x4, arithmetic::sub_matches_torch_shape_2x3x4);
            parity_test!(torch_arithmetic_mul_matches_torch_shape_2x3x4, arithmetic::mul_matches_torch_shape_2x3x4);
            parity_test!(torch_arithmetic_div_matches_torch_shape_2x3x4, arithmetic::div_matches_torch_shape_2x3x4);
            parity_test!(torch_arithmetic_neg_matches_torch_shape_2x3x4, arithmetic::neg_matches_torch_shape_2x3x4);
            parity_test!(torch_arithmetic_abs_matches_torch_shape_2x3x4, arithmetic::abs_matches_torch_shape_2x3x4);
            parity_test!(torch_arithmetic_clamp_matches_torch_min_only_shape_2x3x4, arithmetic::clamp_matches_torch_min_only_shape_2x3x4);
            parity_test!(torch_arithmetic_clamp_matches_torch_max_only_shape_2x3x4, arithmetic::clamp_matches_torch_max_only_shape_2x3x4);
            parity_test!(torch_arithmetic_max_matches_torch_special_values, arithmetic::max_matches_torch_special_values);
            parity_test!(torch_arithmetic_min_matches_torch_special_values, arithmetic::min_matches_torch_special_values);
            parity_test!(torch_arithmetic_clamp_matches_torch_special_values, arithmetic::clamp_matches_torch_special_values);
            parity_test!(torch_functional_softmax_last_dim_matches_torch, functional_ops::softmax_last_dim_matches_torch);
            parity_test!(torch_functional_softmax_last_dim_len1_matches_torch, functional_ops::softmax_last_dim_len1_matches_torch);
            parity_test!(torch_functional_softmax_last_dim_len7_matches_torch, functional_ops::softmax_last_dim_len7_matches_torch);
            parity_test!(torch_functional_softmax_last_dim_len128_matches_torch, functional_ops::softmax_last_dim_len128_matches_torch);
            parity_test!(torch_functional_softmax_last_dim_extreme_logits_matches_torch, functional_ops::softmax_last_dim_extreme_logits_matches_torch);
            parity_test!(torch_functional_softmax_last_dim_misaligned_matches_torch, functional_ops::softmax_last_dim_misaligned_matches_torch);
            parity_test!(torch_functional_softmax_middle_axis_matches_torch, functional_ops::softmax_middle_axis_matches_torch);
            parity_test!(torch_functional_gelu_matches_torch, functional_ops::gelu_matches_torch);
            parity_test!(torch_functional_silu_matches_torch, functional_ops::silu_matches_torch);
            parity_test!(torch_functional_silu_extreme_inputs_match_torch, functional_ops::silu_extreme_inputs_match_torch);
            parity_test!(torch_rotary_apply_matches_torch_full_rotary, rotary_ops::rope_apply_matches_torch_full_rotary);
            parity_test!(torch_rotary_apply_matches_torch_partial_rotary, rotary_ops::rope_apply_matches_torch_partial_rotary);
            parity_test!(torch_rotary_cache_yarn_scaling_matches_formula, rotary_ops::rope_cache_yarn_scaling_matches_formula);
            parity_test!(torch_functional_gelu_extreme_inputs_match_torch, functional_ops::gelu_extreme_inputs_match_torch);
            parity_test!(torch_functional_gelu_tanh_matches_torch_3d_2x3x8, functional_ops::gelu_tanh_matches_torch_3d_2x3x8);
            parity_test!(torch_functional_gelu_tanh_extreme_inputs_match_torch, functional_ops::gelu_tanh_extreme_inputs_match_torch);
            parity_test!(torch_functional_add_bias_matches_torch, functional_ops::add_bias_matches_torch);
            parity_test!(torch_functional_add_bias_matches_torch_3d_2x5x8, functional_ops::add_bias_matches_torch_3d_2x5x8);
            parity_test!(torch_functional_layer_norm_matches_torch, functional_ops::layer_norm_matches_torch);
            parity_test!(torch_functional_rms_norm_matches_torch, functional_ops::rms_norm_matches_torch);
            parity_test!(torch_functional_linear_bf16_weight_rows1_matches_torch, functional_ops::linear_bf16_weight_rows1_matches_torch);
            parity_test!(torch_functional_linear_bf16_weight_rows64_in2100_matches_torch, functional_ops::linear_bf16_weight_rows64_in2100_matches_torch);
            parity_test!(torch_functional_linear_bf16_inputs_rows1_matches_torch, functional_ops::linear_bf16_inputs_rows1_matches_torch);
            parity_test!(torch_functional_linear_bf16_inputs_rows141_matches_torch, functional_ops::linear_bf16_inputs_rows141_matches_torch);
            parity_test!(torch_functional_linear_bf16_inputs_rows130_odd_in2049_matches_torch, functional_ops::linear_bf16_inputs_rows130_odd_in2049_matches_torch);
            parity_test!(torch_functional_cast_f32_bf16_roundtrip_matches_torch, functional_ops::cast_f32_bf16_roundtrip_matches_torch);
            parity_test!(torch_functional_sigmoid_matches_torch, functional_ops::sigmoid_matches_torch);
            parity_test!(torch_functional_softplus_matches_torch, functional_ops::softplus_matches_torch);
            parity_test!(torch_functional_exp_matches_torch, functional_ops::exp_matches_torch);
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
            parity_test!(torch_matmul_inner0_matches_torch, empty_extents::matmul_inner0_matches_torch);
            parity_test!(torch_batched_matmul_inner0_matches_torch, empty_extents::batched_matmul_inner0_matches_torch);
            parity_test!(torch_linear_in0_matches_torch, empty_extents::linear_in0_matches_torch);
            parity_test!(torch_index_negative_bounds_and_steps_match_torch, indexing::negative_bounds_and_steps_match_torch);
            parity_test!(torch_argmax_matches_torch, sampling::argmax_matches_torch);
            parity_test!(torch_top_k_matches_torch, sampling::top_k_matches_torch);
            parity_test!(torch_sample_matches_torch_gumbel_max, sampling::sample_matches_torch_gumbel_max);
            parity_test!(torch_index_inclusive_and_empty_ranges_match_torch, indexing::inclusive_and_empty_ranges_match_torch);
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
            parity_test!(torch_vision_hard_sigmoid_matches_torch, vision_ops::hard_sigmoid_matches_torch);
            parity_test!(torch_vision_hard_swish_matches_torch, vision_ops::hard_swish_matches_torch);
            parity_test!(torch_vision_global_avg_pool2d_matches_torch, vision_ops::global_avg_pool2d_matches_torch);
            parity_test!(torch_vision_conv2d_nhwc_k3_s1_p1_bias_n1_h11_w13_c3_cout8, vision_ops::conv2d_nhwc_k3_s1_p1_bias_n1_h11_w13_c3_cout8);
            parity_test!(torch_vision_conv2d_nhwc_k3_s2_p1_nobias_n2_h9_w10_c5_cout7, vision_ops::conv2d_nhwc_k3_s2_p1_nobias_n2_h9_w10_c5_cout7);
            parity_test!(torch_vision_conv2d_nhwc_k5_s1_p2_bias_n1_h15_w17_c4_cout6, vision_ops::conv2d_nhwc_k5_s1_p2_bias_n1_h15_w17_c4_cout6);
            parity_test!(torch_vision_conv2d_nhwc_k1_s1_p0_bias_n4_h7_w7_c8_cout8, vision_ops::conv2d_nhwc_k1_s1_p0_bias_n4_h7_w7_c8_cout8);
            parity_test!(torch_vision_conv2d_nhwc_k3x5_s2x1_p1x2_bias_n1_h11_w12_c4_cout6, vision_ops::conv2d_nhwc_k3x5_s2x1_p1x2_bias_n1_h11_w12_c4_cout6);
            parity_test!(torch_vision_conv2d_nhwc_k3_s1_p2_d2_bias_n1_h13_w13_c4_cout8, vision_ops::conv2d_nhwc_k3_s1_p2_d2_bias_n1_h13_w13_c4_cout8);
            parity_test!(torch_vision_group_conv2d_nhwc_g2_k3_s1_p1_bias_n1_h11_w11_c8_cout12, vision_ops::group_conv2d_nhwc_g2_k3_s1_p1_bias_n1_h11_w11_c8_cout12);
            parity_test!(torch_vision_depthwise_conv2d_nhwc_k5_s2_p2_n1_h15_w17_c8, vision_ops::depthwise_conv2d_nhwc_k5_s2_p2_n1_h15_w17_c8);
            parity_test!(torch_vision_depthwise_conv2d_nhwc_k3_s1_p1_n2_h28_w28_c32, vision_ops::depthwise_conv2d_nhwc_k3_s1_p1_n2_h28_w28_c32);
            parity_test!(torch_vision_group_conv2d_nhwc_g4_k3_s1_p1_bias_n1_h9_w9_c8_cout4, vision_ops::group_conv2d_nhwc_g4_k3_s1_p1_bias_n1_h9_w9_c8_cout4);
            parity_test!(torch_vision_max_pool2d_nhwc_w2_s2_p0_n1_h8_w8_c3, vision_ops::max_pool2d_nhwc_w2_s2_p0_n1_h8_w8_c3);
            parity_test!(torch_vision_max_pool2d_nhwc_special_values_matches_torch, vision_ops::max_pool2d_nhwc_special_values_matches_torch);
            parity_test!(torch_vision_relu6_edge_values_matches_torch, vision_ops::relu6_edge_values_matches_torch);
            parity_test!(torch_vision_batch_norm2d_matches_torch, vision_layers::batch_norm2d_matches_torch);
            parity_test!(torch_vision_conv2d_batch_norm_relu_matches_torch, vision_layers::conv2d_batch_norm_relu_matches_torch);
            parity_test!(torch_vision_conv2d_tf_padding_stride2_matches_torch, vision_layers::conv2d_tf_padding_stride2_matches_torch);
            parity_test!(torch_vision_conv2d_tf_padding_dilated_matches_torch, vision_layers::conv2d_tf_padding_dilated_matches_torch);
            parity_test!(torch_linear_matches_torch_with_bias, linear::linear_matches_torch_with_bias);
            parity_test!(torch_linear_matches_torch_without_bias, linear::linear_matches_torch_without_bias);
            parity_test!(torch_linear_matches_torch_batch1_in5_out3_bias, linear::linear_matches_torch_batch1_in5_out3_bias);
            parity_test!(torch_linear_matches_torch_batch64_in64_out64_bias, linear::linear_matches_torch_batch64_in64_out64_bias);
            parity_test!(torch_linear_matches_torch_batch7_in13_out17_bias, linear::linear_matches_torch_batch7_in13_out17_bias);
            parity_test!(torch_linear_matches_torch_batch4_in1_out1_bias, linear::linear_matches_torch_batch4_in1_out1_bias);
            parity_test!(torch_linear_matches_torch_batch4_in33_out65_no_bias, linear::linear_matches_torch_batch4_in33_out65_no_bias);
            parity_test!(torch_linear_fp8_weight_rows3_matches_torch, narrow_weights::linear_fp8_weight_rows3_matches_torch);
            parity_test!(torch_linear_packed4_weight_rows1_matches_torch, narrow_weights::linear_packed4_weight_rows1_matches_torch);
            parity_test!(torch_linear_bf16_input_packed4_weight_rows64_matches_torch, narrow_weights::linear_bf16_input_packed4_weight_rows64_matches_torch);
            parity_test!(torch_linear_fp8_scaled_weight_rows1_out9000_matches_torch, narrow_weights::linear_fp8_scaled_weight_rows1_out9000_matches_torch);
            parity_test!(torch_linear_fp8_scaled_weight_rows64_matches_torch, narrow_weights::linear_fp8_scaled_weight_rows64_matches_torch);
            parity_test!(torch_linear_fp8_f32_scaled_weight_rows64_matches_torch, narrow_weights::linear_fp8_f32_scaled_weight_rows64_matches_torch);
            parity_test!(torch_linear_bf16_input_fp8_scaled_weight_rows3_matches_torch, narrow_weights::linear_bf16_input_fp8_scaled_weight_rows3_matches_torch);
            parity_test!(torch_linear_bf16_input_fp8_f32_scaled_weight_rows64_matches_torch, narrow_weights::linear_bf16_input_fp8_f32_scaled_weight_rows64_matches_torch);
            parity_test!(torch_linear_bf16_weight_rows2_out9000_matches_torch, narrow_weights::linear_bf16_weight_rows2_out9000_matches_torch);
            parity_test!(torch_linear_bf16_weight_rows4_matches_torch, narrow_weights::linear_bf16_weight_rows4_matches_torch);
            parity_test!(torch_linear_bf16_input_bf16_weight_rows3_matches_torch, narrow_weights::linear_bf16_input_bf16_weight_rows3_matches_torch);
            parity_test!(torch_conv1d_bf16_weight_rows5_matches_torch, narrow_weights::conv1d_bf16_weight_rows5_matches_torch);
            parity_test!(torch_conv1d_bf16_input_bf16_weight_rows5_matches_torch, narrow_weights::conv1d_bf16_input_bf16_weight_rows5_matches_torch);
            parity_test!(torch_conv1d_matches_torch_batch1_in32_out96, conv1d_layer::conv1d_matches_torch_batch1_in32_out96);
            parity_test!(torch_conv1d_matches_torch_batch1_in37_out3105, conv1d_layer::conv1d_matches_torch_batch1_in37_out3105);
            parity_test!(torch_conv1d_matches_torch_batch7_in13_out17, conv1d_layer::conv1d_matches_torch_batch7_in13_out17);
            parity_test!(torch_conv1d_matches_torch_square_batch64_in64_out64, conv1d_layer::conv1d_matches_torch_square_batch64_in64_out64);
            parity_test!(torch_gated_feed_forward_matches_torch_with_bias, gated_feed_forward_layer::gated_feed_forward_matches_torch_with_bias);
            parity_test!(torch_gated_feed_forward_matches_torch_without_bias, gated_feed_forward_layer::gated_feed_forward_matches_torch_without_bias);
            parity_test!(torch_gated_feed_forward_matches_torch_batch4_embed64_hidden256_bias, gated_feed_forward_layer::gated_feed_forward_matches_torch_batch4_embed64_hidden256_bias);
            parity_test!(torch_layer_norm_matches_torch_basic, layer_norm_layer::layer_norm_matches_torch_basic);
            parity_test!(torch_layer_norm_matches_torch_embed_dim1, layer_norm_layer::layer_norm_matches_torch_embed_dim1);
            parity_test!(torch_layer_norm_matches_torch_prime_embed, layer_norm_layer::layer_norm_matches_torch_prime_embed);
            parity_test!(torch_layer_norm_matches_torch_large_embed, layer_norm_layer::layer_norm_matches_torch_large_embed);
            parity_test!(torch_layer_norm_matches_torch_constant_input, layer_norm_layer::layer_norm_matches_torch_constant_input);
            parity_test!(torch_layer_norm_matches_torch_eps_1e3, layer_norm_layer::layer_norm_matches_torch_eps_1e3);
            parity_test!(torch_layer_norm_matches_torch_eps_1e1, layer_norm_layer::layer_norm_matches_torch_eps_1e1);
            parity_test!(torch_rms_norm_layer_matches_torch_basic, rms_norm_layer::rms_norm_layer_matches_torch_basic);
            parity_test!(torch_rms_norm_layer_matches_torch_seq_batch, rms_norm_layer::rms_norm_layer_matches_torch_seq_batch);
            parity_test!(torch_rms_norm_layer_matches_torch_constant_input, rms_norm_layer::rms_norm_layer_matches_torch_constant_input);
            parity_test!(torch_rms_norm_layer_unit_offset_matches_torch, rms_norm_layer::rms_norm_layer_unit_offset_matches_torch);
            parity_test!(torch_embedding_matches_torch_basic, embedding_layer::embedding_matches_torch_basic);
            parity_test!(torch_embedding_supports_duplicate_indices, embedding_layer::embedding_supports_duplicate_indices);
            parity_test!(torch_embedding_bf16_table_returns_f32_rows, embedding_layer::embedding_bf16_table_returns_f32_rows);
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
            parity_test!(torch_attention_fused_qkv_in_out_prefill4_decode3_matches_torch, multi_head_attention_layer::attention_fused_qkv_in_out_prefill4_decode3_matches_torch);
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
    };
}
