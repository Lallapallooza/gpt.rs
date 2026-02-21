use std::sync::Arc;

use gpt_rs::backend::spec::PortableBackend;
use gpt_rs::ops::functional::{self, RopeConfig, RopeScaling};
use gpt_rs::tensor::DeviceTensor;
use tch::{Device, Kind, Tensor as TchTensor};

use super::common::*;

fn rope_reference(input: &TchTensor, cos: &TchTensor, sin: &TchTensor) -> TchTensor {
    let input_shape = input.size();
    let num_heads = input_shape[0];
    let seq_len = input_shape[1];
    let head_dim = input_shape[2];
    let half_rotary = cos.size()[1];
    let rotary_dim = half_rotary * 2;

    let x_rot = input.narrow(2, 0, rotary_dim);
    let x_first = x_rot.narrow(2, 0, half_rotary);
    let x_second = x_rot.narrow(2, half_rotary, half_rotary);

    let cos_b = cos
        .reshape([1, seq_len, half_rotary])
        .expand([num_heads, seq_len, half_rotary], true);
    let sin_b = sin
        .reshape([1, seq_len, half_rotary])
        .expand([num_heads, seq_len, half_rotary], true);

    let rotated_first = &x_first * &cos_b - &x_second * &sin_b;
    let rotated_second = &x_second * &cos_b + &x_first * &sin_b;
    let rotated = TchTensor::cat(&[rotated_first, rotated_second], 2);

    if rotary_dim == head_dim {
        rotated
    } else {
        let passthrough = input.narrow(2, rotary_dim, head_dim - rotary_dim);
        TchTensor::cat(&[rotated, passthrough], 2)
    }
}

fn run_rope_case<B: PortableBackend + 'static>(
    backend: &Arc<B>,
    num_heads: usize,
    seq_len: usize,
    head_dim: usize,
    rotary_dim: usize,
    seed: u64,
) {
    let mut rng = seeded_rng(seed);
    let x_len = num_heads * seq_len * head_dim;
    let x_data = random_vec(&mut rng, x_len);

    let cache = functional::rotary_cos_sin_cache(
        seq_len,
        RopeConfig {
            rotary_dim,
            theta: 10_000.0,
            scaling: RopeScaling::None,
        },
    )
    .unwrap();
    let half = rotary_dim / 2;

    let expected = timed_torch(|| {
        let x_t = tch_tensor_from_vec(&[num_heads, seq_len, head_dim], &x_data);
        let cos_t = tch_tensor_from_vec(&[seq_len, half], cache.cos.data());
        let sin_t = tch_tensor_from_vec(&[seq_len, half], cache.sin.data());
        tensor_to_vec(&rope_reference(&x_t, &cos_t, &sin_t))
    });

    let actual = timed_gpt(|| {
        let x = device_tensor_from_data(backend, &[num_heads, seq_len, head_dim], &x_data);
        let cos = DeviceTensor::from_host(Arc::clone(backend), cache.cos.clone()).unwrap();
        let sin = DeviceTensor::from_host(Arc::clone(backend), cache.sin.clone()).unwrap();
        let out = functional::apply_rope(backend.as_ref(), &x, &cos, &sin).unwrap();
        to_host_vec(&out)
    });

    assert_close(&expected, &actual);
}

pub fn rope_apply_matches_torch_full_rotary<B: PortableBackend + 'static>(backend: &Arc<B>) {
    run_rope_case(backend, 4, 9, 16, 16, 0x7010_u64);
}

pub fn rope_apply_matches_torch_partial_rotary<B: PortableBackend + 'static>(backend: &Arc<B>) {
    run_rope_case(backend, 3, 7, 24, 16, 0x7011_u64);
}

pub fn rope_apply_rejects_sequence_mismatch<B: PortableBackend + 'static>(backend: &Arc<B>) {
    let err = timed_gpt(|| {
        let x = device_tensor_from_data(backend, &[2, 8, 16], &[0.0; 2 * 8 * 16]);
        let cos = device_tensor_from_data(backend, &[7, 8], &[0.0; 7 * 8]);
        let sin = device_tensor_from_data(backend, &[7, 8], &[0.0; 7 * 8]);
        match functional::apply_rope(backend.as_ref(), &x, &cos, &sin) {
            Ok(_) => panic!("apply_rope should reject sequence mismatch"),
            Err(err) => err,
        }
    });
    assert!(
        err.to_string().contains("sequence length"),
        "error should mention sequence mismatch, got: {}",
        err
    );
}

pub fn rope_cache_yarn_scaling_matches_formula<B: PortableBackend + 'static>(_backend: &Arc<B>) {
    let seq_len = 12usize;
    let rotary_dim = 16usize;
    let half = rotary_dim / 2;
    let theta = 10_000.0f32;
    let factor = 8.0f32;
    let mscale = 1.35f32;
    let mscale_all_dim = 0.75f32;
    let beta_fast = 32.0f32;
    let beta_slow = 1.0f32;
    let original_max_position_embeddings = 2_048usize;

    let cache = functional::rotary_cos_sin_cache(
        seq_len,
        RopeConfig {
            rotary_dim,
            theta,
            scaling: RopeScaling::Yarn {
                factor,
                mscale: Some(mscale),
                mscale_all_dim: Some(mscale_all_dim),
                beta_fast: Some(beta_fast),
                beta_slow: Some(beta_slow),
                original_max_position_embeddings: Some(original_max_position_embeddings),
                truncate: true,
            },
        },
    )
    .unwrap();

    let (expected_cos, expected_sin) = timed_torch(|| {
        let get_mscale = |scale: f64, mscale: f64| {
            if scale <= 1.0 {
                1.0
            } else {
                0.1 * mscale * scale.ln() + 1.0
            }
        };
        let attention_factor = get_mscale(factor as f64, mscale as f64)
            / get_mscale(factor as f64, mscale_all_dim as f64);

        let find_correction_dim = |num_rotations: f64| {
            (rotary_dim as f64
                * ((original_max_position_embeddings as f64)
                    / (num_rotations * 2.0 * std::f64::consts::PI))
                    .ln())
                / (2.0 * (theta as f64).ln())
        };

        let mut low = find_correction_dim(beta_fast as f64).floor();
        let mut high = find_correction_dim(beta_slow as f64).ceil();
        low = low.max(0.0);
        high = high.min((rotary_dim.saturating_sub(1)) as f64);
        let mut ramp_max = high;
        if (low - ramp_max).abs() < f64::EPSILON {
            ramp_max += 0.001;
        }

        let idx = TchTensor::arange(half as i64, (Kind::Float, Device::Cpu));
        let exponent = (&idx * 2.0) / rotary_dim as f64;
        let pos_freq = ((theta as f64).ln() * &exponent).exp();
        let inv_freq_extrap = pos_freq.reciprocal();
        let inv_freq_interp = &inv_freq_extrap / factor as f64;
        let ramp = ((&idx - low) / (ramp_max - low)).clamp(0.0, 1.0);
        let extrapolation_factor = 1.0 - &ramp;
        let inv_freq: TchTensor = &inv_freq_interp * (1.0 - &extrapolation_factor)
            + &inv_freq_extrap * extrapolation_factor;

        let positions = TchTensor::arange(seq_len as i64, (Kind::Float, Device::Cpu));
        let angles = positions.unsqueeze(1) * inv_freq.unsqueeze(0);
        let cos = angles.cos() * attention_factor;
        let sin = angles.sin() * attention_factor;
        (tensor_to_vec(&cos), tensor_to_vec(&sin))
    });

    assert_close(&expected_cos, cache.cos.data());
    assert_close(&expected_sin, cache.sin.data());
}
