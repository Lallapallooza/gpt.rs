use std::sync::Arc;

use gpt_rs::backend::spec::PortableBackend;
use gpt_rs::nn::layers::AttentionConfig;
use gpt_rs::ops::functional::{self, LayerNormResult};
use tch::{Kind, Tensor as TchTensor};

use super::common::*;

fn softmax_shape() -> [usize; 3] {
    [3, 2, 4]
}

fn vector_shape() -> [usize; 1] {
    [6]
}

fn matrix_shape() -> [usize; 2] {
    [3, 4]
}

fn layer_norm_shape() -> [usize; 3] {
    [2, 3, 4]
}

fn softmax_case<B: PortableBackend + 'static>(
    backend: &Arc<B>,
    shape: &[usize],
    data: &[f32],
    atol: Option<f64>,
    rtol: Option<f64>,
) {
    let expected = timed_torch(|| {
        let expected_tensor = tch_tensor_from_vec(shape, data).softmax(-1, Kind::Float);
        tensor_to_vec(&expected_tensor)
    });

    let actual = timed_gpt(|| {
        let device = device_tensor_from_data(backend, shape, data);
        let result = functional::softmax_last_dim(backend.as_ref(), &device).unwrap();
        to_host_vec(&result)
    });

    match (atol, rtol) {
        (Some(atol), Some(rtol)) => assert_close_tol(&expected, &actual, atol, rtol),
        _ => assert_close(&expected, &actual),
    }
}

fn gelu_case<B: PortableBackend + 'static>(backend: &Arc<B>, shape: &[usize], data: &[f32]) {
    let expected = timed_torch(|| {
        let expected_tensor = tch_tensor_from_vec(shape, data).gelu("none");
        tensor_to_vec(&expected_tensor)
    });

    let actual = timed_gpt(|| {
        let device = device_tensor_from_data(backend, shape, data);
        let result = functional::gelu(backend.as_ref(), &device).unwrap();
        to_host_vec(&result)
    });

    assert_close(&expected, &actual);
}

fn add_bias_case<B: PortableBackend + 'static>(
    backend: &Arc<B>,
    shape: &[usize],
    x_data: &[f32],
    bias: &[f32],
) {
    let bias_len = bias.len();

    let expected = timed_torch(|| {
        let x_t = tch_tensor_from_vec(shape, x_data);
        let mut bias_shape = vec![1i64; shape.len()];
        bias_shape[shape.len() - 1] = bias_len as i64;
        let dims: Vec<i64> = shape.iter().map(|d| *d as i64).collect();
        let bias_t = tch_tensor_from_vec(&[bias_len], bias).reshape(bias_shape);
        let bias_broadcast = bias_t.expand(dims, true);
        let expected_tensor = x_t + bias_broadcast;
        tensor_to_vec(&expected_tensor)
    });

    let actual = timed_gpt(|| {
        let x = device_tensor_from_data(backend, shape, x_data);
        let bias_dev = device_tensor_from_data(backend, &[bias_len], bias);
        let result = functional::add_bias(backend.as_ref(), &x, &bias_dev).unwrap();
        to_host_vec(&result)
    });

    assert_close(&expected, &actual);
}

fn layer_norm_case<B: PortableBackend + 'static>(
    backend: &Arc<B>,
    shape: &[usize],
    data: &[f32],
    gamma: &[f32],
    beta: &[f32],
    eps: f64,
) {
    let last_dim = *shape.last().unwrap();

    let (actual_output, actual_normalized, actual_mean, actual_inv_std) = timed_gpt(|| {
        let x = device_tensor_from_data(backend, shape, data);
        let gamma_dev = device_tensor_from_data(backend, &[last_dim], gamma);
        let beta_dev = device_tensor_from_data(backend, &[last_dim], beta);

        let LayerNormResult {
            output,
            normalized,
            mean,
            inv_std,
        } = functional::layer_norm(backend.as_ref(), &x, &gamma_dev, &beta_dev, eps as f32)
            .unwrap();

        (
            to_host_vec(&output),
            to_host_vec(&normalized),
            to_host_vec(&mean),
            to_host_vec(&inv_std),
        )
    });

    let (expected_output, expected_normalized, expected_mean, expected_inv_std) =
        timed_torch(|| {
            let x_t = tch_tensor_from_vec(shape, data);
            let gamma_t = tch_tensor_from_vec(&[last_dim], gamma);
            let beta_t = tch_tensor_from_vec(&[last_dim], beta);

            let dims: Vec<i64> = shape.iter().map(|&d| d as i64).collect();
            let last_axis = (shape.len() as i64) - 1;
            let mean_t = x_t.mean_dim(&[last_axis][..], true, Kind::Float);
            let var_t = x_t.var_dim_last(false, true);
            let inv_std_t = (&var_t + eps).rsqrt();
            let normalized_t = (x_t - &mean_t) * inv_std_t.expand(dims.clone(), true);
            let rank = shape.len();
            let mut broadcast_shape = vec![1i64; rank];
            broadcast_shape[rank - 1] = last_dim as i64;
            let gamma_broadcast = gamma_t
                .reshape(broadcast_shape.clone())
                .expand(dims.clone(), true);
            let beta_broadcast = beta_t.reshape(broadcast_shape).expand(dims.clone(), true);
            let output_t = &normalized_t * gamma_broadcast + beta_broadcast;

            (
                tensor_to_vec(&output_t),
                tensor_to_vec(&normalized_t),
                tensor_to_vec(&mean_t),
                tensor_to_vec(&inv_std_t),
            )
        });

    assert_close(&expected_mean, &actual_mean);
    assert_close(&expected_inv_std, &actual_inv_std);
    assert_close(&expected_normalized, &actual_normalized);
    assert_close(&expected_output, &actual_output);
}

/// Dropout parity relies on the PTIR inspection tests in
/// `crates/gpt-rs/tests/functional_softmax.rs::dropout_emits_rng_mask_sequence` because backend RNG
/// streams are not yet aligned with Torch for deterministic numerical comparisons.
pub fn softmax_last_dim_matches_torch<B: PortableBackend + 'static>(backend: &Arc<B>) {
    let mut rng = seeded_rng(7);
    let shape = softmax_shape();
    let len: usize = shape.iter().product();
    let data = random_vec(&mut rng, len);

    let expected = timed_torch(|| {
        let expected_tensor = tch_tensor_from_vec(&shape, &data).softmax(-1, Kind::Float);
        tensor_to_vec(&expected_tensor)
    });

    let actual = timed_gpt(|| {
        let device = device_tensor_from_data(backend, &shape, &data);
        let result = functional::softmax_last_dim(backend.as_ref(), &device).unwrap();
        to_host_vec(&result)
    });

    assert_close(&expected, &actual);
}

pub fn gelu_matches_torch<B: PortableBackend + 'static>(backend: &Arc<B>) {
    let mut rng = seeded_rng(11);
    let shape = vector_shape();
    let len: usize = shape.iter().product();
    let data = random_vec(&mut rng, len);

    let expected = timed_torch(|| {
        let expected_tensor = tch_tensor_from_vec(&shape, &data).gelu("none");
        tensor_to_vec(&expected_tensor)
    });

    let actual = timed_gpt(|| {
        let device = device_tensor_from_data(backend, &shape, &data);
        let result = functional::gelu(backend.as_ref(), &device).unwrap();
        to_host_vec(&result)
    });

    assert_close(&expected, &actual);
}

pub fn add_bias_matches_torch<B: PortableBackend + 'static>(backend: &Arc<B>) {
    let mut rng = seeded_rng(13);
    let shape = matrix_shape();
    let len: usize = shape.iter().product();
    let bias_len = shape[shape.len() - 1];

    let data = random_vec(&mut rng, len);
    let bias = random_vec(&mut rng, bias_len);

    let expected = timed_torch(|| {
        let x_t = tch_tensor_from_vec(&shape, &data);
        let mut bias_shape = vec![1i64; shape.len()];
        bias_shape[shape.len() - 1] = bias_len as i64;
        let dims: Vec<i64> = shape.iter().map(|d| *d as i64).collect();
        let bias_t = tch_tensor_from_vec(&[bias_len], &bias).reshape(bias_shape);
        let bias_broadcast = bias_t.expand(dims.clone(), true);
        let expected_tensor = x_t + bias_broadcast;
        tensor_to_vec(&expected_tensor)
    });

    let actual = timed_gpt(|| {
        let x = device_tensor_from_data(backend, &shape, &data);
        let bias_dev = device_tensor_from_data(backend, &[bias_len], &bias);

        let result = functional::add_bias(backend.as_ref(), &x, &bias_dev).unwrap();
        to_host_vec(&result)
    });

    assert_close(&expected, &actual);
}

pub fn add_bias_rejects_mismatched_dimension<B: PortableBackend + 'static>(backend: &Arc<B>) {
    let err = timed_gpt(|| {
        let x = device_tensor_from_data(backend, &matrix_shape(), &[0.0; 12]);
        let bias = device_tensor_from_data(backend, &[5], &[0.0; 5]);
        functional::add_bias(backend.as_ref(), &x, &bias).unwrap_err()
    });
    assert!(
        err.to_string().contains("last dimension"),
        "error should mention last dimension mismatch, got: {}",
        err
    );
}

pub fn layer_norm_matches_torch<B: PortableBackend + 'static>(backend: &Arc<B>) {
    let mut rng = seeded_rng(17);
    let shape = layer_norm_shape();
    let len: usize = shape.iter().product();
    let last_dim = *shape.last().unwrap();

    let data = random_vec(&mut rng, len);
    let gamma = random_vec(&mut rng, last_dim);
    let beta = random_vec(&mut rng, last_dim);
    let eps = 1e-5;

    let (actual_output, actual_normalized, actual_mean, actual_inv_std) = timed_gpt(|| {
        let x = device_tensor_from_data(backend, &shape, &data);
        let gamma_dev = device_tensor_from_data(backend, &[last_dim], &gamma);
        let beta_dev = device_tensor_from_data(backend, &[last_dim], &beta);

        let LayerNormResult {
            output,
            normalized,
            mean,
            inv_std,
        } = functional::layer_norm(backend.as_ref(), &x, &gamma_dev, &beta_dev, eps as f32)
            .unwrap();

        (
            to_host_vec(&output),
            to_host_vec(&normalized),
            to_host_vec(&mean),
            to_host_vec(&inv_std),
        )
    });

    let (expected_output, expected_normalized, expected_mean, expected_inv_std) =
        timed_torch(|| {
            let x_t = tch_tensor_from_vec(&shape, &data);
            let gamma_t = tch_tensor_from_vec(&[last_dim], &gamma);
            let beta_t = tch_tensor_from_vec(&[last_dim], &beta);

            let dims: Vec<i64> = shape.iter().map(|&d| d as i64).collect();
            let last_axis = (shape.len() as i64) - 1;
            let mean_t = x_t.mean_dim(&[last_axis][..], true, Kind::Float);
            let var_t = x_t.var_dim_last(false, true);
            let inv_std_t = (&var_t + eps).rsqrt();
            let normalized_t = (x_t - &mean_t) * inv_std_t.expand(dims.clone(), true);
            let rank = shape.len();
            let mut broadcast_shape = vec![1i64; rank];
            broadcast_shape[rank - 1] = last_dim as i64;
            let gamma_broadcast = gamma_t
                .reshape(broadcast_shape.clone())
                .expand(dims.clone(), true);
            let beta_broadcast = beta_t.reshape(broadcast_shape).expand(dims.clone(), true);
            let output_t = &normalized_t * gamma_broadcast + beta_broadcast;

            (
                tensor_to_vec(&output_t),
                tensor_to_vec(&normalized_t),
                tensor_to_vec(&mean_t),
                tensor_to_vec(&inv_std_t),
            )
        });

    assert_close(&expected_mean, &actual_mean);
    assert_close(&expected_inv_std, &actual_inv_std);
    assert_close(&expected_normalized, &actual_normalized);
    assert_close(&expected_output, &actual_output);
}

pub fn softmax_last_dim_len1_matches_torch<B: PortableBackend + 'static>(backend: &Arc<B>) {
    let mut rng = seeded_rng(31);
    let shape = [2usize, 1usize];
    let len: usize = shape.iter().product();
    let data = random_vec(&mut rng, len);
    softmax_case(backend, &shape, &data, None, None);
}

pub fn softmax_last_dim_len2_matches_torch<B: PortableBackend + 'static>(backend: &Arc<B>) {
    let mut rng = seeded_rng(32);
    let shape = [3usize, 2usize];
    let len: usize = shape.iter().product();
    let data = random_vec(&mut rng, len);
    softmax_case(backend, &shape, &data, None, None);
}

pub fn softmax_last_dim_len7_matches_torch<B: PortableBackend + 'static>(backend: &Arc<B>) {
    let mut rng = seeded_rng(33);
    let shape = [2usize, 7usize];
    let len: usize = shape.iter().product();
    let data = random_vec(&mut rng, len);
    softmax_case(backend, &shape, &data, None, None);
}

pub fn softmax_last_dim_len128_matches_torch<B: PortableBackend + 'static>(backend: &Arc<B>) {
    let mut rng = seeded_rng(34);
    let shape = [4usize, 128usize];
    let len: usize = shape.iter().product();
    let data = random_vec(&mut rng, len);
    softmax_case(backend, &shape, &data, None, None);
}

pub fn softmax_last_dim_2x3x8_matches_torch<B: PortableBackend + 'static>(backend: &Arc<B>) {
    let mut rng = seeded_rng(35);
    let shape = [2usize, 3usize, 8usize];
    let len: usize = shape.iter().product();
    let data = random_vec(&mut rng, len);
    softmax_case(backend, &shape, &data, None, None);
}

pub fn softmax_last_dim_2x4x32_matches_torch<B: PortableBackend + 'static>(backend: &Arc<B>) {
    let mut rng = seeded_rng(36);
    let shape = [2usize, 4usize, 32usize];
    let len: usize = shape.iter().product();
    let data = random_vec(&mut rng, len);
    softmax_case(backend, &shape, &data, None, None);
}

pub fn softmax_last_dim_constant_logits_matches_torch<B: PortableBackend + 'static>(
    backend: &Arc<B>,
) {
    let shape = [2usize, 3usize, 17usize];
    let len: usize = shape.iter().product();
    let data = const_vec(len, 0.25);
    softmax_case(backend, &shape, &data, None, None);
}

pub fn softmax_last_dim_extreme_logits_matches_torch<B: PortableBackend + 'static>(
    backend: &Arc<B>,
) {
    let mut rng = seeded_rng(38);
    let shape = [4usize, 16usize];
    let len: usize = shape.iter().product();
    let data = random_vec_range(&mut rng, len, -80.0, 80.0);
    softmax_case(backend, &shape, &data, Some(5e-3), Some(5e-3));
}

pub fn softmax_last_dim_misaligned_matches_torch<B: PortableBackend + 'static>(backend: &Arc<B>) {
    let mut rng = seeded_rng(39);
    let shape = [1usize, 3usize, 257usize];
    let len: usize = shape.iter().product();
    let data = random_vec(&mut rng, len);
    softmax_case(backend, &shape, &data, None, None);
}

pub fn gelu_matches_torch_1d_256<B: PortableBackend + 'static>(backend: &Arc<B>) {
    let mut rng = seeded_rng(41);
    let shape = [256usize];
    let len: usize = shape.iter().product();
    let data = random_vec(&mut rng, len);
    gelu_case(backend, &shape, &data);
}

pub fn gelu_matches_torch_2d_4x16<B: PortableBackend + 'static>(backend: &Arc<B>) {
    let mut rng = seeded_rng(42);
    let shape = [4usize, 16usize];
    let len: usize = shape.iter().product();
    let data = random_vec(&mut rng, len);
    gelu_case(backend, &shape, &data);
}

pub fn gelu_matches_torch_3d_2x3x8<B: PortableBackend + 'static>(backend: &Arc<B>) {
    let mut rng = seeded_rng(43);
    let shape = [2usize, 3usize, 8usize];
    let len: usize = shape.iter().product();
    let data = random_vec(&mut rng, len);
    gelu_case(backend, &shape, &data);
}

pub fn gelu_extreme_inputs_match_torch<B: PortableBackend + 'static>(backend: &Arc<B>) {
    let mut rng = seeded_rng(44);
    let shape = [256usize];
    let len: usize = shape.iter().product();
    let data = random_vec_range(&mut rng, len, -10.0, 10.0);
    gelu_case(backend, &shape, &data);
}

pub fn add_bias_matches_torch_3d_2x5x8<B: PortableBackend + 'static>(backend: &Arc<B>) {
    let mut rng = seeded_rng(45);
    let shape = [2usize, 5usize, 8usize];
    let len: usize = shape.iter().product();
    let data = random_vec(&mut rng, len);
    let bias = random_vec(&mut rng, shape[2]);
    add_bias_case(backend, &shape, &data, &bias);
}

pub fn add_bias_matches_torch_3d_1x16x64<B: PortableBackend + 'static>(backend: &Arc<B>) {
    let mut rng = seeded_rng(46);
    let shape = [1usize, 16usize, 64usize];
    let len: usize = shape.iter().product();
    let data = random_vec(&mut rng, len);
    let bias = random_vec(&mut rng, shape[2]);
    add_bias_case(backend, &shape, &data, &bias);
}

pub fn add_bias_matches_torch_4d_2x3x4x5<B: PortableBackend + 'static>(backend: &Arc<B>) {
    let mut rng = seeded_rng(47);
    let shape = [2usize, 3usize, 4usize, 5usize];
    let len: usize = shape.iter().product();
    let data = random_vec(&mut rng, len);
    let bias = random_vec(&mut rng, shape[3]);
    add_bias_case(backend, &shape, &data, &bias);
}

pub fn add_bias_rejects_mismatched_dimension_3d<B: PortableBackend + 'static>(backend: &Arc<B>) {
    let err = timed_gpt(|| {
        let x = device_tensor_from_data(backend, &[2, 5, 8], &[0.0; 80]);
        let bias = device_tensor_from_data(backend, &[7], &[0.0; 7]);
        functional::add_bias(backend.as_ref(), &x, &bias).unwrap_err()
    });
    assert!(err.to_string().contains("last dimension"));
}

pub fn layer_norm_matches_torch_embed_dim1<B: PortableBackend + 'static>(backend: &Arc<B>) {
    let mut rng = seeded_rng(48);
    let shape = [2usize, 3usize, 1usize];
    let len: usize = shape.iter().product();
    let data = random_vec(&mut rng, len);
    let gamma = random_vec(&mut rng, 1);
    let beta = random_vec(&mut rng, 1);
    layer_norm_case(backend, &shape, &data, &gamma, &beta, 1e-5);
}

pub fn layer_norm_matches_torch_prime_embed<B: PortableBackend + 'static>(backend: &Arc<B>) {
    let mut rng = seeded_rng(49);
    let shape = [2usize, 3usize, 13usize];
    let len: usize = shape.iter().product();
    let data = random_vec(&mut rng, len);
    let gamma = random_vec(&mut rng, 13);
    let beta = random_vec(&mut rng, 13);
    layer_norm_case(backend, &shape, &data, &gamma, &beta, 1e-5);
}

pub fn layer_norm_matches_torch_large_embed<B: PortableBackend + 'static>(backend: &Arc<B>) {
    let mut rng = seeded_rng(50);
    let shape = [2usize, 4usize, 256usize];
    let len: usize = shape.iter().product();
    let data = random_vec(&mut rng, len);
    let gamma = random_vec(&mut rng, 256);
    let beta = random_vec(&mut rng, 256);
    layer_norm_case(backend, &shape, &data, &gamma, &beta, 1e-5);
}

pub fn layer_norm_matches_torch_batch1<B: PortableBackend + 'static>(backend: &Arc<B>) {
    let mut rng = seeded_rng(51);
    let shape = [1usize, 8usize];
    let len: usize = shape.iter().product();
    let data = random_vec(&mut rng, len);
    let gamma = random_vec(&mut rng, 8);
    let beta = random_vec(&mut rng, 8);
    layer_norm_case(backend, &shape, &data, &gamma, &beta, 1e-5);
}

pub fn layer_norm_matches_torch_constant_input<B: PortableBackend + 'static>(backend: &Arc<B>) {
    let shape = [2usize, 3usize, 17usize];
    let len: usize = shape.iter().product();
    let data = const_vec(len, 0.5);
    let mut rng = seeded_rng(52);
    let gamma = random_vec(&mut rng, 17);
    let beta = random_vec(&mut rng, 17);
    layer_norm_case(backend, &shape, &data, &gamma, &beta, 1e-5);
}

pub fn layer_norm_matches_torch_eps_1e3<B: PortableBackend + 'static>(backend: &Arc<B>) {
    let mut rng = seeded_rng(53);
    let shape = [2usize, 3usize, 17usize];
    let len: usize = shape.iter().product();
    let data = random_vec(&mut rng, len);
    let gamma = random_vec(&mut rng, 17);
    let beta = random_vec(&mut rng, 17);
    layer_norm_case(backend, &shape, &data, &gamma, &beta, 1e-3);
}

pub fn layer_norm_rejects_gamma_mismatch<B: PortableBackend + 'static>(backend: &Arc<B>) {
    let err = timed_gpt(|| {
        let x = device_tensor_from_data(backend, &[2, 3, 8], &[0.0; 48]);
        let gamma = device_tensor_from_data(backend, &[7], &[0.0; 7]);
        let beta = device_tensor_from_data(backend, &[8], &[0.0; 8]);
        functional::layer_norm(backend.as_ref(), &x, &gamma, &beta, 1e-5)
    });
    assert!(err.is_err());
    if let Err(err) = err {
        assert!(err.to_string().contains("last dimension"));
    }
}

pub fn matmul_matches_torch<B: PortableBackend + 'static>(backend: &Arc<B>) {
    let mut rng = seeded_rng(19);
    let lhs_shape = [2, 3];
    let rhs_shape = [3, 4];
    let lhs_len: usize = lhs_shape.iter().product();
    let rhs_len: usize = rhs_shape.iter().product();
    let lhs_data = random_vec(&mut rng, lhs_len);
    let rhs_data = random_vec(&mut rng, rhs_len);

    let expected = timed_torch(|| {
        let lhs_t = tch_tensor_from_vec(&lhs_shape, &lhs_data);
        let rhs_t = tch_tensor_from_vec(&rhs_shape, &rhs_data);
        tensor_to_vec(&lhs_t.matmul(&rhs_t))
    });

    let actual = timed_gpt(|| {
        let lhs = device_tensor_from_data(backend, &lhs_shape, &lhs_data);
        let rhs = device_tensor_from_data(backend, &rhs_shape, &rhs_data);
        let result = functional::matmul(backend.as_ref(), &lhs, &rhs).unwrap();
        to_host_vec(&result)
    });

    assert_close(&expected, &actual);
}

pub fn attention_matches_torch<B: PortableBackend + 'static>(backend: &Arc<B>) {
    let mut rng = seeded_rng(23);
    let config = AttentionConfig::with_equal_heads(8, 2);
    let seq_len = 2usize;
    let shape = [seq_len, config.total_projection_dim()];
    let len: usize = shape.iter().product();
    let data = random_vec(&mut rng, len);

    let (actual_context, actual_keys, actual_values) = timed_gpt(|| {
        let qkv = device_tensor_from_data(backend, &shape, &data);
        let attention = functional::attention(backend.as_ref(), &config, &qkv, None).unwrap();
        let actual_context = to_host_vec(&attention.output);
        let actual_keys = to_host_vec(attention.cache.keys());
        let actual_values = to_host_vec(attention.cache.values());
        (actual_context, actual_keys, actual_values)
    });

    let (expected_context, expected_keys, expected_values) = timed_torch(|| {
        let qkv_tensor = tch_tensor_from_vec(&shape, &data);
        reference_attention(&config, &qkv_tensor)
    });

    assert_close(&expected_context, &actual_context);
    assert_close(&expected_keys, &actual_keys);
    assert_close(&expected_values, &actual_values);
}

trait VarDimLastExt {
    fn var_dim_last(&self, unbiased: bool, keepdim: bool) -> TchTensor;
}

impl VarDimLastExt for TchTensor {
    fn var_dim_last(&self, unbiased: bool, keepdim: bool) -> TchTensor {
        let last_axis = (self.dim() as i64) - 1;
        self.var_dim(&[last_axis][..], unbiased, keepdim)
    }
}

fn reference_attention(
    config: &AttentionConfig,
    qkv: &TchTensor,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let seq_len = qkv.size()[0];
    let q_proj = config.query_projection_dim() as i64;
    let kv_proj = config.key_value_projection_dim() as i64;
    let num_query_heads = config.num_query_heads as i64;
    let num_kv_heads = config.num_key_value_heads as i64;
    let head_dim = config.head_dim as i64;
    let kv_head_dim = config.kv_head_dim as i64;
    let kv_group_size = config.kv_group_size() as i64;

    let q_slice = qkv.narrow(1, 0, q_proj);
    let k_slice = qkv.narrow(1, q_proj, kv_proj);
    let v_slice = qkv.narrow(1, q_proj + kv_proj, kv_proj);

    let q_heads = q_slice
        .reshape([seq_len, num_query_heads, head_dim])
        .permute([1, 0, 2]);
    let k_cache = k_slice
        .reshape([seq_len, num_kv_heads, kv_head_dim])
        .permute([1, 0, 2]);
    let v_cache = v_slice
        .reshape([seq_len, num_kv_heads, kv_head_dim])
        .permute([1, 0, 2]);

    let k_grouped = k_cache
        .transpose(1, 2)
        .reshape([num_kv_heads, 1, kv_head_dim, seq_len])
        .expand([num_kv_heads, kv_group_size, kv_head_dim, seq_len], true)
        .reshape([num_query_heads, kv_head_dim, seq_len]);
    let v_grouped = v_cache
        .reshape([num_kv_heads, 1, seq_len, kv_head_dim])
        .expand([num_kv_heads, kv_group_size, seq_len, kv_head_dim], true)
        .reshape([num_query_heads, seq_len, kv_head_dim]);

    let scale = (config.head_dim as f64).sqrt();
    let scores = (q_heads.bmm(&k_grouped)) / scale;

    let allowed = TchTensor::ones([seq_len, seq_len], (Kind::Float, qkv.device())).tril(0);
    let disallowed = TchTensor::ones_like(&allowed) - &allowed;
    let mask = disallowed
        .unsqueeze(0)
        .expand([num_query_heads, seq_len, seq_len], true)
        * -1e9f64;
    let masked = scores + mask;

    let (max_scores, _) = masked.max_dim(-1, true);
    let stabilized = masked - max_scores;
    let exp_scores = stabilized.exp();
    let sum_axes = [-1i64];
    let sum_scores = exp_scores.sum_dim_intlist(&sum_axes[..], true, Kind::Float);
    let softmax = exp_scores / sum_scores;

    let context = softmax.bmm(&v_grouped);
    let context_out = context
        .permute([1, 0, 2])
        .reshape([seq_len, config.embed_dim as i64]);

    (
        tensor_to_vec(&context_out),
        tensor_to_vec(&k_cache),
        tensor_to_vec(&v_cache),
    )
}
