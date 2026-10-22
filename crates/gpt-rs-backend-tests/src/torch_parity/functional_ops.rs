use std::sync::Arc;

use gpt_rs::backend::PortableBackend;
use gpt_rs::module::Layer;
use gpt_rs::nn::layers::Linear;
use gpt_rs::ops::functional;
use gpt_rs::tensor::DType;
use tch::Kind;

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

/// Softmax along the last axis.
fn softmax_case<B: PortableBackend + 'static>(
    backend: &Arc<B>,
    shape: &[usize],
    data: &[f32],
    atol: Option<f64>,
    rtol: Option<f64>,
) {
    softmax_axis_case(backend, shape, data, -1, atol, rtol);
}

fn softmax_axis_case<B: PortableBackend + 'static>(
    backend: &Arc<B>,
    shape: &[usize],
    data: &[f32],
    axis: i64,
    atol: Option<f64>,
    rtol: Option<f64>,
) {
    let expected = timed_torch(|| {
        let expected_tensor = tch_tensor_from_vec(shape, data).softmax(axis, Kind::Float);
        tensor_to_vec(&expected_tensor)
    });

    let actual = timed_gpt(|| {
        let device = device_tensor_from_data(backend, shape, data);
        let result = functional::softmax(&device, axis);
        to_host_vec(&result)
    });

    match (atol, rtol) {
        (Some(abs_tolerance), Some(rel_tolerance)) => {
            assert_close_tol(&expected, &actual, abs_tolerance, rel_tolerance);
        }
        _ => assert_close(&expected, &actual),
    }
}

fn gelu_case<B: PortableBackend + 'static>(backend: &Arc<B>, shape: &[usize], data: &[f32]) {
    gelu_approximate_case(backend, shape, data, "none");
}

fn gelu_approximate_case<B: PortableBackend + 'static>(
    backend: &Arc<B>,
    shape: &[usize],
    data: &[f32],
    approximate: &str,
) {
    let expected = timed_torch(|| {
        let expected_tensor = tch_tensor_from_vec(shape, data).gelu(approximate);
        tensor_to_vec(&expected_tensor)
    });

    let actual = timed_gpt(|| {
        let device = device_tensor_from_data(backend, shape, data);
        let result = match approximate {
            "none" => functional::gelu(&device),
            "tanh" => functional::gelu_tanh(&device),
            other => panic!("unsupported gelu approximation {other}"),
        };
        to_host_vec(&result)
    });

    assert_close(&expected, &actual);
}

fn silu_case<B: PortableBackend + 'static>(backend: &Arc<B>, shape: &[usize], data: &[f32]) {
    let expected = timed_torch(|| {
        let expected_tensor = tch_tensor_from_vec(shape, data).silu();
        tensor_to_vec(&expected_tensor)
    });

    let actual = timed_gpt(|| {
        let device = device_tensor_from_data(backend, shape, data);
        let result = functional::silu(&device);
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
        let dims: Vec<i64> = shape.iter().map(|&extent| extent as i64).collect();
        let bias_t = tch_tensor_from_vec(&[bias_len], bias).reshape(bias_shape);
        let bias_broadcast = bias_t.expand(dims, true);
        let expected_tensor = x_t + bias_broadcast;
        tensor_to_vec(&expected_tensor)
    });

    let actual = timed_gpt(|| {
        let x = device_tensor_from_data(backend, shape, x_data);
        let bias_dev = device_tensor_from_data(backend, &[bias_len], bias);
        let result = x + bias_dev;
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

    let actual = timed_gpt(|| {
        let x = device_tensor_from_data(backend, shape, data);
        let gamma_dev = device_tensor_from_data(backend, &[last_dim], gamma);
        let beta_dev = device_tensor_from_data(backend, &[last_dim], beta);
        to_host_vec(&functional::layer_norm(
            &x, &gamma_dev, &beta_dev, eps as f32,
        ))
    });

    let expected = timed_torch(|| {
        let x_t = tch_tensor_from_vec(shape, data);
        let gamma_t = tch_tensor_from_vec(&[last_dim], gamma);
        let beta_t = tch_tensor_from_vec(&[last_dim], beta);
        let output = x_t.layer_norm([last_dim as i64], Some(&gamma_t), Some(&beta_t), eps, false);
        tensor_to_vec(&output)
    });

    assert_close(&expected, &actual);
}

fn rms_norm_case<B: PortableBackend + 'static>(
    backend: &Arc<B>,
    shape: &[usize],
    data: &[f32],
    gamma: &[f32],
    eps: f64,
) {
    let last_dim = *shape.last().unwrap();

    let actual = timed_gpt(|| {
        let x = device_tensor_from_data(backend, shape, data);
        let gamma_dev = device_tensor_from_data(backend, &[last_dim], gamma);
        to_host_vec(&functional::rms_norm(&x, &gamma_dev, eps as f32))
    });

    let expected = timed_torch(|| {
        let x_t = tch_tensor_from_vec(shape, data);
        let gamma_t = tch_tensor_from_vec(&[last_dim], gamma);
        let last_axis = (shape.len() as i64) - 1;
        let mean_square = x_t.square().mean_dim(&[last_axis][..], true, Kind::Float);
        tensor_to_vec(&(&x_t * (mean_square + eps).rsqrt() * gamma_t))
    });

    assert_close(&expected, &actual);
}

pub fn softmax_last_dim_matches_torch<B: PortableBackend + 'static>(backend: &Arc<B>) {
    let mut rng = seeded_rng(7);
    let shape = softmax_shape();
    let len: usize = shape.iter().product();
    let data = random_vec(&mut rng, len);
    softmax_case(backend, &shape, &data, None, None);
}

/// A middle axis, which the reduction and the broadcast back must both keep in place.
pub fn softmax_middle_axis_matches_torch<B: PortableBackend + 'static>(backend: &Arc<B>) {
    let mut rng = seeded_rng(40);
    let shape = [2usize, 5usize, 3usize];
    let len: usize = shape.iter().product();
    let data = random_vec(&mut rng, len);
    softmax_axis_case(backend, &shape, &data, -2, None, None);
}

pub fn gelu_matches_torch<B: PortableBackend + 'static>(backend: &Arc<B>) {
    let mut rng = seeded_rng(11);
    let shape = vector_shape();
    let len: usize = shape.iter().product();
    let data = random_vec(&mut rng, len);
    gelu_case(backend, &shape, &data);
}

pub fn gelu_tanh_matches_torch_3d_2x3x8<B: PortableBackend + 'static>(backend: &Arc<B>) {
    let mut rng = seeded_rng(47);
    let shape = [2usize, 3usize, 8usize];
    let data = random_vec(&mut rng, shape.iter().product());
    gelu_approximate_case(backend, &shape, &data, "tanh");
}

pub fn gelu_tanh_extreme_inputs_match_torch<B: PortableBackend + 'static>(backend: &Arc<B>) {
    let mut rng = seeded_rng(48);
    let data = random_vec_range(&mut rng, 256, -10.0, 10.0);
    gelu_approximate_case(backend, &[256], &data, "tanh");
}

pub fn silu_matches_torch<B: PortableBackend + 'static>(backend: &Arc<B>) {
    let mut rng = seeded_rng(0x5100_u64);
    let shape = vector_shape();
    let len: usize = shape.iter().product();
    let data = random_vec(&mut rng, len);
    silu_case(backend, &shape, &data);
}

pub fn silu_extreme_inputs_match_torch<B: PortableBackend + 'static>(backend: &Arc<B>) {
    let mut rng = seeded_rng(0x5101_u64);
    let shape = [3usize, 17usize];
    let len: usize = shape.iter().product();
    let data = random_vec_range(&mut rng, len, -40.0, 40.0);
    silu_case(backend, &shape, &data);
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
        let dims: Vec<i64> = shape.iter().map(|&extent| extent as i64).collect();
        let bias_t = tch_tensor_from_vec(&[bias_len], &bias).reshape(bias_shape);
        let bias_broadcast = bias_t.expand(dims.clone(), true);
        let expected_tensor = x_t + bias_broadcast;
        tensor_to_vec(&expected_tensor)
    });

    let actual = timed_gpt(|| {
        let x = device_tensor_from_data(backend, &shape, &data);
        let bias_dev = device_tensor_from_data(backend, &[bias_len], &bias);

        let result = x + bias_dev;
        to_host_vec(&result)
    });

    assert_close(&expected, &actual);
}

pub fn layer_norm_matches_torch<B: PortableBackend + 'static>(backend: &Arc<B>) {
    let mut rng = seeded_rng(17);
    let shape = layer_norm_shape();
    let len: usize = shape.iter().product();
    let last_dim = *shape.last().unwrap();
    let data = random_vec(&mut rng, len);
    let gamma = random_vec(&mut rng, last_dim);
    let beta = random_vec(&mut rng, last_dim);
    layer_norm_case(backend, &shape, &data, &gamma, &beta, 1e-5);
}

pub fn rms_norm_matches_torch<B: PortableBackend + 'static>(backend: &Arc<B>) {
    let mut rng = seeded_rng(0xA504_u64);
    let shape = layer_norm_shape();
    let len: usize = shape.iter().product();
    let last_dim = *shape.last().unwrap();
    let data = random_vec(&mut rng, len);
    let gamma = random_vec(&mut rng, last_dim);
    rms_norm_case(backend, &shape, &data, &gamma, 1e-5);
}

pub fn softmax_last_dim_len1_matches_torch<B: PortableBackend + 'static>(backend: &Arc<B>) {
    let mut rng = seeded_rng(31);
    let shape = [2usize, 1usize];
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

pub fn softmax_last_dim_extreme_logits_matches_torch<B: PortableBackend + 'static>(
    backend: &Arc<B>,
) {
    let mut rng = seeded_rng(38);
    let shape = [4usize, 16usize];
    let len: usize = shape.iter().product();
    let data = random_vec_range(&mut rng, len, -80.0, 80.0);
    softmax_case(backend, &shape, &data, None, None);
}

pub fn softmax_last_dim_misaligned_matches_torch<B: PortableBackend + 'static>(backend: &Arc<B>) {
    let mut rng = seeded_rng(39);
    let shape = [1usize, 3usize, 257usize];
    let len: usize = shape.iter().product();
    let data = random_vec(&mut rng, len);
    softmax_case(backend, &shape, &data, None, None);
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

fn bf16_round(values: &[f32]) -> Vec<f32> {
    let rounded = tch_tensor_from_vec(&[values.len()], values)
        .to_kind(Kind::BFloat16)
        .to_kind(Kind::Float);
    tensor_to_vec(&rounded)
}

// ---------------------------------------------------------------------------------------------
// linear and cast
// ---------------------------------------------------------------------------------------------

/// `nn::Linear` against `x @ w^T`, with `w` stored as `weight_dtype`. With `bf16_inputs`, the
/// layer rounds its input to bf16 and still accumulates in f32, and the reference does the same.
fn run_linear_case<B: PortableBackend + 'static>(
    backend: &Arc<B>,
    (rows, out_features, in_features): (usize, usize, usize),
    weight_dtype: DType,
    bf16_inputs: bool,
    seed: u64,
) {
    let mut rng = seeded_rng(seed);
    let x = random_vec(&mut rng, rows * in_features);
    let raw_w = random_vec(&mut rng, out_features * in_features);
    // The reference sees exactly the weight values gpt-rs stores.
    let weight_values = match weight_dtype {
        DType::Bf16 => bf16_round(&raw_w),
        _ => raw_w,
    };

    let expected = timed_torch(|| {
        let x_ref = if bf16_inputs {
            bf16_round(&x)
        } else {
            x.clone()
        };
        let x_t = tch_tensor_from_vec(&[rows, in_features], &x_ref);
        let w_t = tch_tensor_from_vec(&[out_features, in_features], &weight_values);
        tensor_to_vec(&x_t.matmul(&w_t.transpose(0, 1)))
    });

    let actual = timed_gpt(|| {
        let x_d = device_tensor_from_data(backend, &[rows, in_features], &x);
        let weight_host =
            crate::tensor_as(&[out_features, in_features], &weight_values, weight_dtype);
        let mut layer = load_layer(backend, [("proj.weight".to_string(), weight_host)], |p| {
            p.scoped("proj", Linear::new)
        })
        .unwrap();
        layer.input_dtype = bf16_inputs.then_some(DType::Bf16);
        let output = layer.call(&x_d);
        assert_eq!(output.shape().dims(), [rows, out_features]);
        assert_eq!(output.dtype(), DType::F32);
        to_host_vec(&output)
    });

    assert_close(&expected, &actual);
}

/// One row, odd `out_features` and `in_features`.
pub fn linear_bf16_weight_rows1_matches_torch<B: PortableBackend + 'static>(backend: &Arc<B>) {
    run_linear_case(backend, (1, 33, 65), DType::Bf16, false, 0x1101);
}

/// K spans several blocks and the activations are large.
pub fn linear_bf16_weight_rows64_in2100_matches_torch<B: PortableBackend + 'static>(
    backend: &Arc<B>,
) {
    run_linear_case(backend, (64, 20, 2100), DType::Bf16, false, 0x1118);
}

/// One row, odd `out_features` and `in_features`.
pub fn linear_bf16_inputs_rows1_matches_torch<B: PortableBackend + 'static>(backend: &Arc<B>) {
    run_linear_case(backend, (1, 33, 65), DType::Bf16, true, 0x1114);
}

/// Prefill-sized, with no dimension a multiple of 8 and K spanning several blocks.
pub fn linear_bf16_inputs_rows141_matches_torch<B: PortableBackend + 'static>(backend: &Arc<B>) {
    run_linear_case(backend, (141, 72, 1100), DType::Bf16, true, 0x1117);
}

/// Prefill-sized, with large activations and an odd K that spans several blocks and leaves no
/// bf16 pairs.
pub fn linear_bf16_inputs_rows130_odd_in2049_matches_torch<B: PortableBackend + 'static>(
    backend: &Arc<B>,
) {
    run_linear_case(backend, (130, 40, 2049), DType::Bf16, true, 0x1119);
}

pub fn cast_f32_bf16_roundtrip_matches_torch<B: PortableBackend + 'static>(backend: &Arc<B>) {
    let mut rng = seeded_rng(0x1105);
    let mut values = random_vec_range(&mut rng, 61, -300.0, 300.0);
    // Exercise round-to-nearest-even ties and the extremes of the bf16 range.
    values.extend_from_slice(&[
        1.0 + 1.0 / 256.0,
        1.0 + 3.0 / 256.0,
        -(1.0 + 1.0 / 256.0),
        3.389e38,
        -3.389e38,
        1e-40,
        0.0,
    ]);
    let expected = timed_torch(|| bf16_round(&values));
    let actual = timed_gpt(|| {
        let x = device_tensor_from_data(backend, &[values.len()], &values);
        let narrowed = x.cast(DType::Bf16);
        assert_eq!(narrowed.dtype(), DType::Bf16);
        let widened = narrowed.cast(DType::F32);
        to_host_vec(&widened)
    });
    assert_close_tol(&expected, &actual, 0.0, 0.0);
}

// ---------------------------------------------------------------------------------------------
// elementwise
// ---------------------------------------------------------------------------------------------

pub fn sigmoid_matches_torch<B: PortableBackend + 'static>(backend: &Arc<B>) {
    let mut rng = seeded_rng(0x1106);
    let mut values = random_vec_range(&mut rng, 4 * 29, -12.0, 12.0);
    values[0] = -80.0;
    values[1] = 80.0;
    let expected = timed_torch(|| tensor_to_vec(&tch_tensor_from_vec(&[4, 29], &values).sigmoid()));
    let actual = timed_gpt(|| {
        let x = device_tensor_from_data(backend, &[4, 29], &values);
        to_host_vec(&functional::sigmoid(&x))
    });
    assert_close(&expected, &actual);
}

pub fn softplus_matches_torch<B: PortableBackend + 'static>(backend: &Arc<B>) {
    let mut rng = seeded_rng(0x1107);
    let mut values = random_vec_range(&mut rng, 3 * 40, -30.0, 30.0);
    values[..6].copy_from_slice(&[-90.0, -20.5, 0.0, 19.9, 20.1, 95.0]);
    let expected =
        timed_torch(|| tensor_to_vec(&tch_tensor_from_vec(&[3, 40], &values).softplus()));
    let actual = timed_gpt(|| {
        let x = device_tensor_from_data(backend, &[3, 40], &values);
        to_host_vec(&functional::softplus(&x))
    });
    assert_close(&expected, &actual);
}

pub fn exp_matches_torch<B: PortableBackend + 'static>(backend: &Arc<B>) {
    let mut rng = seeded_rng(0x1109);
    let mut values = random_vec_range(&mut rng, 2 * 33, -20.0, 20.0);
    values[..3].copy_from_slice(&[-100.0, 0.0, 80.0]);
    let expected = timed_torch(|| tensor_to_vec(&tch_tensor_from_vec(&[2, 33], &values).exp()));
    let actual = timed_gpt(|| {
        let x = device_tensor_from_data(backend, &[2, 33], &values);
        to_host_vec(&x.exp())
    });
    assert_close(&expected, &actual);
}
