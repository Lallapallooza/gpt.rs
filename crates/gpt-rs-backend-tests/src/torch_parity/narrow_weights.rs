//! Projections of f32 or bf16 activations by weights stored in bf16, FP8 or packed 4-bit codes,
//! against Torch on the values the stored weights hold.

use std::sync::Arc;

use gpt_rs::backend::PortableBackend;
use gpt_rs::gpt_rs_ir::ir::TensorType;
use gpt_rs::module::Layer;
use gpt_rs::nn::layers::Linear;
use gpt_rs::ops::functional::WeightLayout;
use gpt_rs::tensor::{DType, HostTensor};
use rand::rngs::StdRng;
use rand::Rng;
use tch::{Kind, Tensor as TchTensor};

use super::common::*;
use crate::tensor_as;

#[derive(Clone, Copy)]
enum Stored {
    Bf16,
    Fp8,
    ScaledFp8 {
        scale: f32,
        dtype: DType,
    },
    /// compressed-tensors' `pack-quantized` 4-bit codes, with a bf16 scale for each group of
    /// `group` codes along `in_features`.
    Packed4 {
        group: usize,
    },
}

/// The value of the e4m3 byte `byte` as the OCP 8-bit floating-point format defines it: a sign
/// bit, 4 exponent bits with bias 7 and 3 mantissa bits.
fn e4m3_value(byte: u8) -> f32 {
    let sign = if byte & 0x80 == 0 { 1.0 } else { -1.0 };
    let fraction = f32::from(byte & 0x07) / 8.0;
    let magnitude = match i32::from((byte >> 3) & 0x0f) {
        0 => fraction * 2f32.powi(-6),
        exponent => (1.0 + fraction) * 2f32.powi(exponent - 7),
    };
    sign * magnitude
}

fn kind(dtype: DType) -> Kind {
    match dtype {
        DType::Bf16 => Kind::BFloat16,
        DType::F32 => Kind::Float,
        other => panic!("no case uses {other:?}"),
    }
}

/// A weight of extents `dims` stored as `stored`, and the values it holds before any scale.
fn stored_weight(rng: &mut StdRng, dims: &[usize], stored: Stored) -> (HostTensor, Vec<f32>) {
    let len = dims.iter().product();
    if let Stored::Bf16 = stored {
        let raw = random_vec(rng, len);
        let rounded = tch_tensor_from_vec(&[len], &raw).to_kind(kind(DType::Bf16));
        let values = tensor_to_vec(&rounded.to_kind(Kind::Float));
        return (tensor_as(dims, &raw, DType::Bf16), values);
    }
    // Magnitudes up to 3.75, subnormals included, and no NaN.
    let bytes: Vec<u8> = (0..len)
        .map(|_| rng.gen_range(0..0x48u8) | (rng.gen::<u8>() & 0x80))
        .collect();
    let values = bytes.iter().map(|&byte| e4m3_value(byte)).collect();
    let ty = TensorType::new(DType::F8E4M3, dims.iter().copied());
    let tensor = HostTensor::from_le_bytes(&ty, &bytes).unwrap();
    (tensor, values)
}

/// The tensors of an `[out_features, in_features]` weight in compressed-tensors' `pack-quantized`
/// format, and the weight that `transformers` dequantizes from them, `(code - 8) * scale` in bf16.
/// The format packs eight 4-bit codes into each int32 word along `in_features`, from the least
/// significant bits up. Each code is a signed value plus 8. Each group of `group` codes has a bf16
/// scale.
fn packed_weight(
    rng: &mut StdRng,
    [out_features, in_features]: [usize; 2],
    group: usize,
) -> (Vec<(String, HostTensor)>, TchTensor) {
    let codes: Vec<u8> = (0..out_features * in_features)
        .map(|_| rng.gen_range(0..16u8))
        .collect();
    let mut words = Vec::with_capacity(codes.len() / 2);
    for word_codes in codes.chunks(8) {
        let mut word = 0u32;
        for (position, &code) in word_codes.iter().enumerate() {
            word |= u32::from(code) << (4 * position);
        }
        words.extend(word.to_le_bytes());
    }
    let packed_type = TensorType::new(DType::Si32, [out_features, in_features / 8]);
    let packed = HostTensor::from_le_bytes(&packed_type, &words).unwrap();
    let groups = in_features / group;
    let scales = random_vec_range(rng, out_features * groups, 0.001, 0.1);
    let scale = tensor_as(&[out_features, groups], &scales, DType::Bf16);
    let mut signed = Vec::with_capacity(codes.len());
    for &code in &codes {
        signed.push(f32::from(code) - 8.0);
    }
    let codes_ref =
        tch_tensor_from_vec(&[out_features, in_features], &signed).to_kind(Kind::BFloat16);
    let scales_ref = tch_tensor_from_vec(&[out_features, groups], &scales)
        .to_kind(Kind::BFloat16)
        .repeat_interleave_self_int(group as i64, 1, None);
    let tensors = vec![
        ("proj.weight_packed".to_string(), packed),
        ("proj.weight_scale".to_string(), scale),
    ];
    (tensors, codes_ref * scales_ref)
}

/// `nn::Linear` with an `[out_features, in_features]` weight stored as `stored`, and its inputs
/// cast to `input` when one is given, against `x @ w^T` with f32 sums.
fn run_linear_case<B: PortableBackend + 'static>(
    backend: &Arc<B>,
    (rows, out_features, in_features): (usize, usize, usize),
    stored: Stored,
    input: Option<DType>,
    seed: u64,
) {
    let mut rng = seeded_rng(seed);
    let x = random_vec(&mut rng, rows * in_features);
    let dims = [out_features, in_features];
    let (mut tensors, mut w_ref) = match stored {
        Stored::Packed4 { group } => packed_weight(&mut rng, dims, group),
        _ => {
            let (weight, values) = stored_weight(&mut rng, &dims, stored);
            let tensors = vec![("proj.weight".to_string(), weight)];
            (tensors, tch_tensor_from_vec(&dims, &values))
        }
    };
    if let Stored::ScaledFp8 { scale, dtype } = stored {
        // `transformers` dequantizes in the dtype of the scale.
        w_ref = w_ref.to_kind(kind(dtype)) * TchTensor::from(scale).to_kind(kind(dtype));
        let scale_tensor = tensor_as(&[], &[scale], dtype);
        tensors.push(("proj.weight_scale_inv".to_string(), scale_tensor));
    }
    let mut x_ref = tch_tensor_from_vec(&[rows, in_features], &x);
    if let Some(dtype) = input {
        x_ref = x_ref.to_kind(kind(dtype));
        w_ref = w_ref.to_kind(kind(dtype));
    }

    let expected = timed_torch(|| {
        let (input_f32, weight_f32) = (x_ref.to_kind(Kind::Float), w_ref.to_kind(Kind::Float));
        tensor_to_vec(&input_f32.matmul(&weight_f32.transpose(0, 1)))
    });

    let actual = timed_gpt(|| {
        let mut layer = load_layer(backend, tensors, |p| p.scoped("proj", Linear::new)).unwrap();
        layer.input_dtype = input;
        let input_device = device_tensor_from_data(backend, &[rows, in_features], &x);
        to_host_vec(&layer.call(&input_device))
    });

    assert_close(&expected, &actual);
}

/// A [`Linear`] with a bf16 `[in_features, out_features]` weight, and its inputs cast to `input`
/// when one is given, against `x @ w + b`.
fn run_conv1d_bf16_case<B: PortableBackend + 'static>(
    backend: &Arc<B>,
    (rows, in_features, out_features): (usize, usize, usize),
    input: Option<DType>,
    seed: u64,
) {
    let mut rng = seeded_rng(seed);
    let x = random_vec(&mut rng, rows * in_features);
    let dims = [in_features, out_features];
    let (weight, values) = stored_weight(&mut rng, &dims, Stored::Bf16);
    let bias = random_vec(&mut rng, out_features);
    let mut x_ref = tch_tensor_from_vec(&[rows, in_features], &x);
    if let Some(dtype) = input {
        x_ref = x_ref.to_kind(kind(dtype));
    }

    let expected = timed_torch(|| {
        let input_f32 = x_ref.to_kind(Kind::Float);
        let output = input_f32.matmul(&tch_tensor_from_vec(&dims, &values))
            + tch_tensor_from_vec(&[out_features], &bias);
        tensor_to_vec(&output)
    });

    let actual = timed_gpt(|| {
        let bias_host = tensor_as(&[out_features], &bias, DType::F32);
        let tensors = [
            ("proj.weight".to_string(), weight),
            ("proj.bias".to_string(), bias_host),
        ];
        let mut layer = load_layer(backend, tensors, |params| {
            params.scoped("proj", |p| Linear::with_layout(p, WeightLayout::InOut))
        })
        .unwrap();
        layer.input_dtype = input;
        let input_device = device_tensor_from_data(backend, &[rows, in_features], &x);
        to_host_vec(&layer.call(&input_device))
    });

    assert_close(&expected, &actual);
}

/// Three rows, and an inner extent that is no multiple of a vector width.
pub fn linear_fp8_weight_rows3_matches_torch<B: PortableBackend + 'static>(backend: &Arc<B>) {
    run_linear_case(backend, (3, 67, 257), Stored::Fp8, None, 0x8F01);
}

/// One row, and enough output features to split across threads.
pub fn linear_fp8_scaled_weight_rows1_out9000_matches_torch<B: PortableBackend + 'static>(
    backend: &Arc<B>,
) {
    let stored = Stored::ScaledFp8 {
        scale: 0.0137,
        dtype: DType::Bf16,
    };
    run_linear_case(backend, (1, 9000, 257), stored, None, 0x8F02);
}

/// Many rows, and an inner extent of several blocks.
pub fn linear_fp8_scaled_weight_rows64_matches_torch<B: PortableBackend + 'static>(
    backend: &Arc<B>,
) {
    let stored = Stored::ScaledFp8 {
        scale: 3.1,
        dtype: DType::Bf16,
    };
    run_linear_case(backend, (64, 300, 1000), stored, None, 0x8F03);
}

/// An f32 scale, whose products with the weight round to f32.
pub fn linear_fp8_f32_scaled_weight_rows64_matches_torch<B: PortableBackend + 'static>(
    backend: &Arc<B>,
) {
    let stored = Stored::ScaledFp8 {
        scale: 3.1,
        dtype: DType::F32,
    };
    run_linear_case(backend, (64, 300, 1000), stored, None, 0x8F07);
}

/// bf16 inputs and a bf16 scale.
pub fn linear_bf16_input_fp8_scaled_weight_rows3_matches_torch<B: PortableBackend + 'static>(
    backend: &Arc<B>,
) {
    let stored = Stored::ScaledFp8 {
        scale: 0.0137,
        dtype: DType::Bf16,
    };
    run_linear_case(backend, (3, 67, 257), stored, Some(DType::Bf16), 0x8F08);
}

/// bf16 inputs and an f32 scale, whose products with the weight round to f32 and then to bf16.
pub fn linear_bf16_input_fp8_f32_scaled_weight_rows64_matches_torch<
    B: PortableBackend + 'static,
>(
    backend: &Arc<B>,
) {
    let stored = Stored::ScaledFp8 {
        scale: 3.1,
        dtype: DType::F32,
    };
    run_linear_case(backend, (64, 300, 1000), stored, Some(DType::Bf16), 0x8F09);
}

/// One row, a decode step, with groups of 128 codes.
pub fn linear_packed4_weight_rows1_matches_torch<B: PortableBackend + 'static>(backend: &Arc<B>) {
    let stored = Stored::Packed4 { group: 128 };
    run_linear_case(backend, (1, 300, 1024), stored, None, 0x8F0C);
}

/// Many rows of bf16 inputs, and groups of 32 codes.
pub fn linear_bf16_input_packed4_weight_rows64_matches_torch<B: PortableBackend + 'static>(
    backend: &Arc<B>,
) {
    let stored = Stored::Packed4 { group: 32 };
    run_linear_case(backend, (64, 96, 512), stored, Some(DType::Bf16), 0x8F0D);
}

/// Two rows, and enough output features to split across threads.
pub fn linear_bf16_weight_rows2_out9000_matches_torch<B: PortableBackend + 'static>(
    backend: &Arc<B>,
) {
    run_linear_case(backend, (2, 9000, 257), Stored::Bf16, None, 0x8F04);
}

/// Four rows and few output features.
pub fn linear_bf16_weight_rows4_matches_torch<B: PortableBackend + 'static>(backend: &Arc<B>) {
    run_linear_case(backend, (4, 64, 130), Stored::Bf16, None, 0x8F05);
}

/// bf16 inputs, which multiply the bf16 weight without a cast.
pub fn linear_bf16_input_bf16_weight_rows3_matches_torch<B: PortableBackend + 'static>(
    backend: &Arc<B>,
) {
    let input = Some(DType::Bf16);
    run_linear_case(backend, (3, 67, 257), Stored::Bf16, input, 0x8F0A);
}

/// The `[in_features, out_features]` layout of a GPT-2 checkpoint.
pub fn conv1d_bf16_weight_rows5_matches_torch<B: PortableBackend + 'static>(backend: &Arc<B>) {
    run_conv1d_bf16_case(backend, (5, 37, 3105), None, 0x8F06);
}

/// bf16 inputs, which multiply the bf16 `[in_features, out_features]` weight without a cast.
pub fn conv1d_bf16_input_bf16_weight_rows5_matches_torch<B: PortableBackend + 'static>(
    backend: &Arc<B>,
) {
    run_conv1d_bf16_case(backend, (5, 37, 3105), Some(DType::Bf16), 0x8F0B);
}
