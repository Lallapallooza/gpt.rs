pub mod smoke;
#[cfg(feature = "torch")]
pub mod torch_parity;

#[macro_export]
macro_rules! define_backend_tests {
    ($module:ident, $backend_ctor:expr) => {
        #[cfg(test)]
        mod $module {
            use std::sync::Arc;

            use $crate::smoke;

            #[test]
            fn smoke_matmul_matches_expected() {
                let backend = ($backend_ctor)();
                smoke::matmul_matches_expected(&backend);
            }

            #[test]
            fn smoke_gpt_forward_shape() {
                let backend = ($backend_ctor)();
                smoke::gpt_forward_shape(&backend);
            }

            #[cfg(feature = "torch")]
            mod torch_parity_tests {
                use super::*;

                use $crate::torch_parity::{
                    arithmetic, attention, device_layers, embedding_layer, feed_forward_layer,
                    functional_ops, harness, layer_norm_layer, linear, matmul,
                    multi_head_attention_layer, vision_ops,
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
                        #[test]
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
            parity_test!(torch_functional_gelu_matches_torch_1d_256, functional_ops::gelu_matches_torch_1d_256);
            parity_test!(torch_functional_gelu_matches_torch_2d_4x16, functional_ops::gelu_matches_torch_2d_4x16);
            parity_test!(torch_functional_gelu_matches_torch_3d_2x3x8, functional_ops::gelu_matches_torch_3d_2x3x8);
            parity_test!(torch_functional_gelu_extreme_inputs_match_torch, functional_ops::gelu_extreme_inputs_match_torch);
            parity_test!(torch_functional_add_bias_matches_torch, functional_ops::add_bias_matches_torch);
            parity_test!(torch_functional_add_bias_matches_torch_3d_2x5x8, functional_ops::add_bias_matches_torch_3d_2x5x8);
            parity_test!(torch_functional_add_bias_matches_torch_3d_1x16x64, functional_ops::add_bias_matches_torch_3d_1x16x64);
            parity_test!(torch_functional_add_bias_matches_torch_4d_2x3x4x5, functional_ops::add_bias_matches_torch_4d_2x3x4x5);
            parity_test!(torch_functional_add_bias_rejects_mismatched_dimension, functional_ops::add_bias_rejects_mismatched_dimension);
            parity_test!(torch_functional_add_bias_rejects_mismatched_dimension_3d, functional_ops::add_bias_rejects_mismatched_dimension_3d);
            parity_test!(torch_functional_layer_norm_matches_torch, functional_ops::layer_norm_matches_torch);
            parity_test!(torch_functional_layer_norm_matches_torch_embed_dim1, functional_ops::layer_norm_matches_torch_embed_dim1);
            parity_test!(torch_functional_layer_norm_matches_torch_prime_embed, functional_ops::layer_norm_matches_torch_prime_embed);
            parity_test!(torch_functional_layer_norm_matches_torch_large_embed, functional_ops::layer_norm_matches_torch_large_embed);
            parity_test!(torch_functional_layer_norm_matches_torch_batch1, functional_ops::layer_norm_matches_torch_batch1);
            parity_test!(torch_functional_layer_norm_matches_torch_constant_input, functional_ops::layer_norm_matches_torch_constant_input);
            parity_test!(torch_functional_layer_norm_matches_torch_eps_1e3, functional_ops::layer_norm_matches_torch_eps_1e3);
            parity_test!(torch_functional_layer_norm_rejects_gamma_mismatch, functional_ops::layer_norm_rejects_gamma_mismatch);
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
            parity_test!(torch_attention_matches_torch, attention::attention_matches_torch);
            parity_test!(torch_attention_matches_torch_grouped, attention::attention_matches_torch_grouped);
            parity_test!(torch_attention_appends_cache, attention::attention_appends_cache);
            parity_test!(torch_attention_extends_cache_multiple_steps, attention::attention_extends_cache_multiple_steps);
            parity_test!(torch_attention_incremental_matches_full, attention::attention_incremental_matches_full);
            parity_test!(torch_attention_seq1_embed8_heads2_matches_torch, attention::attention_seq1_embed8_heads2_matches_torch);
            parity_test!(torch_attention_seq8_embed32_heads4_matches_torch, attention::attention_seq8_embed32_heads4_matches_torch);
            parity_test!(torch_attention_head_dim1_embed4_heads4_matches_torch, attention::attention_head_dim1_embed4_heads4_matches_torch);
            parity_test!(torch_attention_multi_query_kv1_matches_torch, attention::attention_multi_query_kv1_matches_torch);
            parity_test!(torch_attention_grouped_kv2_matches_torch, attention::attention_grouped_kv2_matches_torch);
            parity_test!(torch_attention_prefill4_decode3_matches_full_concat, attention::attention_prefill4_decode3_matches_full_concat);
            parity_test!(torch_attention_prefill4_decode3_grouped_kv2_matches_full_concat, attention::attention_prefill4_decode3_grouped_kv2_matches_full_concat);
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
            parity_test!(torch_layer_norm_matches_torch_basic, layer_norm_layer::layer_norm_matches_torch_basic);
            parity_test!(torch_layer_norm_forward_with_state_matches_moments, layer_norm_layer::layer_norm_forward_with_state_matches_moments);
            parity_test!(torch_layer_norm_matches_torch_embed_dim1, layer_norm_layer::layer_norm_matches_torch_embed_dim1);
            parity_test!(torch_layer_norm_matches_torch_prime_embed, layer_norm_layer::layer_norm_matches_torch_prime_embed);
            parity_test!(torch_layer_norm_matches_torch_large_embed, layer_norm_layer::layer_norm_matches_torch_large_embed);
            parity_test!(torch_layer_norm_matches_torch_constant_input, layer_norm_layer::layer_norm_matches_torch_constant_input);
            parity_test!(torch_layer_norm_matches_torch_eps_1e3, layer_norm_layer::layer_norm_matches_torch_eps_1e3);
            parity_test!(torch_layer_norm_matches_torch_eps_1e1, layer_norm_layer::layer_norm_matches_torch_eps_1e1);
            parity_test!(torch_layer_norm_state_constant_input_matches_moments, layer_norm_layer::layer_norm_state_constant_input_matches_moments);
            parity_test!(torch_embedding_matches_torch_basic, embedding_layer::embedding_matches_torch_basic);
            parity_test!(torch_embedding_supports_duplicate_indices, embedding_layer::embedding_supports_duplicate_indices);
            parity_test!(torch_embedding_matches_torch_vocab64_embed32_seq16_rank1, embedding_layer::embedding_matches_torch_vocab64_embed32_seq16_rank1);
            parity_test!(torch_embedding_matches_torch_vocab32_embed8_seq5, embedding_layer::embedding_matches_torch_vocab32_embed8_seq5);
            parity_test!(torch_embedding_matches_torch_vocab32_embed128_seq8, embedding_layer::embedding_matches_torch_vocab32_embed128_seq8);
            parity_test!(torch_embedding_rejects_indices_rank2, embedding_layer::embedding_rejects_indices_rank2);
            parity_test!(torch_embedding_rejects_indices_rank3, embedding_layer::embedding_rejects_indices_rank3);
            parity_test!(torch_device_linear_initializes_from_host_and_device_weights, device_layers::linear_initializes_from_host_and_device_weights);
            parity_test!(torch_device_layer_norm_initializes_from_host_and_device_tensors, device_layers::layer_norm_initializes_from_host_and_device_tensors);
            parity_test!(torch_device_embedding_initializes_from_device_tensor, device_layers::embedding_initializes_from_device_tensor);
            parity_test!(torch_device_feed_forward_accepts_mixed_tensor_inputs, device_layers::feed_forward_accepts_mixed_tensor_inputs);
            parity_test!(torch_device_multi_head_attention_accepts_device_parameters, device_layers::multi_head_attention_accepts_device_parameters);
            parity_test!(torch_multi_head_attention_matches_torch_with_bias, multi_head_attention_layer::multi_head_attention_matches_torch_with_bias);
            parity_test!(torch_multi_head_attention_matches_torch_grouped, multi_head_attention_layer::multi_head_attention_matches_torch_grouped);
            parity_test!(torch_multi_head_attention_matches_torch_without_bias, multi_head_attention_layer::multi_head_attention_matches_torch_without_bias);
            parity_test!(torch_multi_head_attention_state_records_context, multi_head_attention_layer::multi_head_attention_state_records_context);
            parity_test!(torch_multi_head_attention_seq1_embed32_heads4_bias_matches_torch, multi_head_attention_layer::multi_head_attention_seq1_embed32_heads4_bias_matches_torch);
            parity_test!(torch_multi_head_attention_seq8_embed32_heads4_bias_matches_torch, multi_head_attention_layer::multi_head_attention_seq8_embed32_heads4_bias_matches_torch);
            parity_test!(torch_multi_head_attention_seq8_embed32_heads8_kv1_bias_matches_torch, multi_head_attention_layer::multi_head_attention_seq8_embed32_heads8_kv1_bias_matches_torch);
            parity_test!(torch_multi_head_attention_seq8_embed32_heads8_kv2_bias_matches_torch, multi_head_attention_layer::multi_head_attention_seq8_embed32_heads8_kv2_bias_matches_torch);
            parity_test!(torch_multi_head_attention_head_dim1_embed8_heads8_bias_matches_torch, multi_head_attention_layer::multi_head_attention_head_dim1_embed8_heads8_bias_matches_torch);
            parity_test!(torch_multi_head_attention_prefill4_decode3_matches_full_concat, multi_head_attention_layer::multi_head_attention_prefill4_decode3_matches_full_concat);
            parity_test!(torch_multi_head_attention_prefill4_decode3_grouped_kv2_matches_full_concat, multi_head_attention_layer::multi_head_attention_prefill4_decode3_grouped_kv2_matches_full_concat);

            }
        }
    };
}
