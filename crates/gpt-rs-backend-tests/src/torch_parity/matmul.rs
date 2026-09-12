use std::sync::Arc;

use gpt_rs::backend::spec::PortableBackend;
use gpt_rs::capture;
use gpt_rs::ops::ptir::{axes_iter, DotAttrs, DotDims};
use gpt_rs::tensor::DeviceTensorOps;

use super::common::*;

fn run_matmul_case<B: PortableBackend + 'static>(
    backend: &Arc<B>,
    lhs_shape: &[usize],
    rhs_shape: &[usize],
    seed: u64,
) {
    let mut rng = seeded_rng(seed);
    let lhs_len: usize = lhs_shape.iter().product();
    let rhs_len: usize = rhs_shape.iter().product();
    let lhs_data = random_vec(&mut rng, lhs_len);
    let rhs_data = random_vec(&mut rng, rhs_len);

    let expected = timed_torch(|| {
        let lhs_t = tch_tensor_from_vec(lhs_shape, &lhs_data);
        let rhs_t = tch_tensor_from_vec(rhs_shape, &rhs_data);
        tensor_to_vec(&lhs_t.matmul(&rhs_t))
    });

    let actual = timed_gpt(|| {
        let lhs_dev = device_tensor_from_data(backend, lhs_shape, &lhs_data);
        let rhs_dev = device_tensor_from_data(backend, rhs_shape, &rhs_data);
        to_host_vec(&lhs_dev.matmul(&rhs_dev).unwrap())
    });

    assert_close(&expected, &actual);
}

fn run_batched_matmul_case<B: PortableBackend + 'static>(
    backend: &Arc<B>,
    lhs_shape: &[usize],
    rhs_shape: &[usize],
    seed: u64,
) {
    let mut rng = seeded_rng(seed);
    let lhs_len: usize = lhs_shape.iter().product();
    let rhs_len: usize = rhs_shape.iter().product();
    let lhs_data = random_vec(&mut rng, lhs_len);
    let rhs_data = random_vec(&mut rng, rhs_len);

    let expected = timed_torch(|| {
        let lhs_t = tch_tensor_from_vec(lhs_shape, &lhs_data);
        let rhs_t = tch_tensor_from_vec(rhs_shape, &rhs_data);
        tensor_to_vec(&lhs_t.bmm(&rhs_t))
    });

    let actual = timed_gpt(|| {
        let lhs_dev = device_tensor_from_data(backend, lhs_shape, &lhs_data);
        let rhs_dev = device_tensor_from_data(backend, rhs_shape, &rhs_data);
        to_host_vec(&lhs_dev.matmul(&rhs_dev).unwrap())
    });

    assert_close(&expected, &actual);
}

macro_rules! matmul_case {
    ($name:ident, $lhs:expr, $rhs:expr, $seed:expr) => {
        pub fn $name<B: PortableBackend + 'static>(backend: &Arc<B>) {
            run_matmul_case(backend, $lhs, $rhs, $seed);
        }
    };
}

macro_rules! bmm_case {
    ($name:ident, $lhs:expr, $rhs:expr, $seed:expr) => {
        pub fn $name<B: PortableBackend + 'static>(backend: &Arc<B>) {
            run_batched_matmul_case(backend, $lhs, $rhs, $seed);
        }
    };
}

matmul_case!(matmul_matches_torch_1x1_1x1, &[1, 1], &[1, 1], 10);
matmul_case!(matmul_matches_torch_1x5_5x3, &[1, 5], &[5, 3], 11);
matmul_case!(matmul_matches_torch_4x5_5x1, &[4, 5], &[5, 1], 12);
matmul_case!(matmul_matches_torch_7x13_13x9, &[7, 13], &[13, 9], 13);
matmul_case!(matmul_matches_torch_33x65_65x31, &[33, 65], &[65, 31], 14);
matmul_case!(matmul_matches_torch_8x1_1x8, &[8, 1], &[1, 8], 15);
matmul_case!(matmul_matches_torch_1x16_16x17, &[1, 16], &[16, 17], 16);
matmul_case!(
    matmul_matches_torch_64x128_128x32,
    &[64, 128],
    &[128, 32],
    17
);

bmm_case!(
    batched_matmul_matches_torch_b1_4x5_5x2,
    &[1, 4, 5],
    &[1, 5, 2],
    18
);
bmm_case!(
    batched_matmul_matches_torch_b2_7x13_13x9,
    &[2, 7, 13],
    &[2, 13, 9],
    19
);
bmm_case!(
    batched_matmul_matches_torch_b8_4x8_8x4,
    &[8, 4, 8],
    &[8, 8, 4],
    20
);
bmm_case!(
    batched_matmul_matches_torch_b3_33x65_65x31,
    &[3, 33, 65],
    &[3, 65, 31],
    21
);
bmm_case!(
    batched_matmul_matches_torch_b2_6x40_40x150,
    &[2, 6, 40],
    &[2, 40, 150],
    23
);

/// Batched `dot_general` of `[batch, *, *]` operands with batch axis 0 and the contraction axes
/// `(contract_lhs, contract_rhs)`.
fn run_batched_dot_case<B: PortableBackend + 'static>(
    backend: &Arc<B>,
    lhs_shape: &[usize],
    rhs_shape: &[usize],
    contract: (usize, usize),
    seed: u64,
) {
    let mut rng = seeded_rng(seed);
    let lhs_data = random_vec(&mut rng, lhs_shape.iter().product());
    let rhs_data = random_vec(&mut rng, rhs_shape.iter().product());

    let expected = timed_torch(|| {
        // Move each contraction axis next to the other operand's free axis, then bmm.
        let lhs_t = tch_tensor_from_vec(lhs_shape, &lhs_data);
        let rhs_t = tch_tensor_from_vec(rhs_shape, &rhs_data);
        let lhs_t = if contract.0 == 1 {
            lhs_t.transpose(1, 2)
        } else {
            lhs_t
        };
        let rhs_t = if contract.1 == 2 {
            rhs_t.transpose(1, 2)
        } else {
            rhs_t
        };
        tensor_to_vec(&lhs_t.bmm(&rhs_t).contiguous())
    });

    let actual = timed_gpt(|| -> anyhow::Result<Vec<f32>> {
        let lhs_dev = device_tensor_from_data(backend, lhs_shape, &lhs_data);
        let rhs_dev = device_tensor_from_data(backend, rhs_shape, &rhs_data);
        let dims = DotDims::new(
            axes_iter([0]),
            axes_iter([contract.0]),
            axes_iter([contract.1]),
        );
        let out = capture!(|lhs_dev, rhs_dev| lhs_dev.dot_general(
            &rhs_dev,
            &dims,
            &DotAttrs::default()
        ))?;
        Ok(to_host_vec(&out))
    })
    .unwrap();

    assert_close(&expected, &actual);
}

macro_rules! batched_dot_case {
    ($name:ident, $lhs:expr, $rhs:expr, $contract:expr, $seed:expr) => {
        pub fn $name<B: PortableBackend + 'static>(backend: &Arc<B>) {
            run_batched_dot_case(backend, $lhs, $rhs, $contract, $seed);
        }
    };
}

// `q . k^T` form, contracting the last axis of both: odd row, column and contraction sizes, a
// contraction long enough to span several blocks, and enough rows for a blocked GEMM.
batched_dot_case!(
    batched_dot_nt_matches_torch_b2_m5,
    &[2, 5, 37],
    &[2, 13, 37],
    (2, 2),
    32
);
batched_dot_case!(
    batched_dot_nt_matches_torch_b2_m5_k1100,
    &[2, 5, 1100],
    &[2, 7, 1100],
    (2, 2),
    33
);
batched_dot_case!(
    batched_dot_nt_matches_torch_b2_m70,
    &[2, 70, 37],
    &[2, 13, 37],
    (2, 2),
    35
);
// `k^T . v` form, contracting the middle axis of both.
batched_dot_case!(
    batched_dot_tn_matches_torch_b2_m3,
    &[2, 19, 3],
    &[2, 19, 21],
    (1, 1),
    34
);

pub fn matmul_rejects_inner_dim_mismatch<B: PortableBackend + 'static>(backend: &Arc<B>) {
    let err = timed_gpt(|| {
        let lhs = device_tensor_from_data(backend, &[4, 5], &[0.0; 20]);
        let rhs = device_tensor_from_data(backend, &[4, 3], &[0.0; 12]);
        lhs.matmul(&rhs).unwrap_err()
    });
    assert!(err.to_string().contains("dimension mismatch"));
}

pub fn batched_matmul_rejects_batch_mismatch<B: PortableBackend + 'static>(backend: &Arc<B>) {
    let err = timed_gpt(|| {
        let lhs = device_tensor_from_data(backend, &[2, 4, 5], &[0.0; 40]);
        let rhs = device_tensor_from_data(backend, &[3, 5, 2], &[0.0; 30]);
        lhs.matmul(&rhs).unwrap_err()
    });
    assert!(err.to_string().contains("batch dimension mismatch"));
}
