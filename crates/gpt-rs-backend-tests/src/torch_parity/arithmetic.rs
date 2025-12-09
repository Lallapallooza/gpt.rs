use std::sync::Arc;

use gpt_rs::backend::spec::PortableBackend;
use gpt_rs::tensor::DeviceTensorOps;

use super::common::*;

enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Max,
    Min,
}

enum UnaryOp {
    Neg,
    Abs,
}

fn run_binary_case<B: PortableBackend + 'static>(
    backend: &Arc<B>,
    shape: &[usize],
    seed: u64,
    op: BinaryOp,
    nonzero_rhs: bool,
) {
    let mut rng = seeded_rng(seed);
    let len: usize = shape.iter().product();
    let a_data = random_vec(&mut rng, len);
    let b_data = if nonzero_rhs {
        random_vec_nonzero(&mut rng, len, 0.5)
    } else {
        random_vec(&mut rng, len)
    };

    let expected = timed_torch(|| {
        let a_t = tch_tensor_from_vec(shape, &a_data);
        let b_t = tch_tensor_from_vec(shape, &b_data);
        let out = match op {
            BinaryOp::Add => a_t.f_add(&b_t).unwrap(),
            BinaryOp::Sub => a_t.f_sub(&b_t).unwrap(),
            BinaryOp::Mul => a_t.f_mul(&b_t).unwrap(),
            BinaryOp::Div => a_t.f_div(&b_t).unwrap(),
            BinaryOp::Max => a_t.f_maximum(&b_t).unwrap(),
            BinaryOp::Min => a_t.f_minimum(&b_t).unwrap(),
        };
        tensor_to_vec(&out)
    });

    let actual = timed_gpt(|| {
        let a_dev = device_tensor_from_data(backend, shape, &a_data);
        let b_dev = device_tensor_from_data(backend, shape, &b_data);
        let out = match op {
            BinaryOp::Add => a_dev.add(&b_dev).unwrap(),
            BinaryOp::Sub => a_dev.sub(&b_dev).unwrap(),
            BinaryOp::Mul => a_dev.mul(&b_dev).unwrap(),
            BinaryOp::Div => a_dev.div(&b_dev).unwrap(),
            BinaryOp::Max => a_dev.maximum(&b_dev).unwrap(),
            BinaryOp::Min => a_dev.minimum(&b_dev).unwrap(),
        };
        to_host_vec(&out)
    });

    assert_close(&expected, &actual);
}

fn run_unary_case<B: PortableBackend + 'static>(
    backend: &Arc<B>,
    shape: &[usize],
    seed: u64,
    op: UnaryOp,
) {
    let mut rng = seeded_rng(seed);
    let len: usize = shape.iter().product();
    let data = random_vec(&mut rng, len);

    let expected = timed_torch(|| {
        let tensor = tch_tensor_from_vec(shape, &data);
        let out = match op {
            UnaryOp::Neg => tensor.f_neg().unwrap(),
            UnaryOp::Abs => tensor.abs(),
        };
        tensor_to_vec(&out)
    });

    let actual = timed_gpt(|| {
        let dev = device_tensor_from_data(backend, shape, &data);
        let out = match op {
            UnaryOp::Neg => dev.neg().unwrap(),
            UnaryOp::Abs => dev.abs().unwrap(),
        };
        to_host_vec(&out)
    });

    assert_close(&expected, &actual);
}

fn run_clamp_case<B: PortableBackend + 'static>(
    backend: &Arc<B>,
    shape: &[usize],
    seed: u64,
    min_only: bool,
    max_only: bool,
) {
    let mut rng = seeded_rng(seed);
    let len: usize = shape.iter().product();

    let base_data = random_vec(&mut rng, len);
    let min_data: Vec<f32> = base_data.iter().map(|v| v - 0.5).collect();
    let max_data: Vec<f32> = base_data.iter().map(|v| v + 0.5).collect();

    let expected = timed_torch(|| {
        let base_t = tch_tensor_from_vec(shape, &base_data);
        let min_t = tch_tensor_from_vec(shape, &min_data);
        let max_t = tch_tensor_from_vec(shape, &max_data);
        let out = if min_only {
            base_t.maximum(&min_t)
        } else if max_only {
            base_t.minimum(&max_t)
        } else {
            base_t.maximum(&min_t).minimum(&max_t)
        };
        tensor_to_vec(&out)
    });

    let actual = timed_gpt(|| {
        let base_dev = device_tensor_from_data(backend, shape, &base_data);
        let min_dev = device_tensor_from_data(backend, shape, &min_data);
        let max_dev = device_tensor_from_data(backend, shape, &max_data);
        let out = if min_only {
            base_dev.clamp(Some(&min_dev), None).unwrap()
        } else if max_only {
            base_dev.clamp(None, Some(&max_dev)).unwrap()
        } else {
            base_dev.clamp(Some(&min_dev), Some(&max_dev)).unwrap()
        };
        to_host_vec(&out)
    });

    assert_close(&expected, &actual);
}

macro_rules! binary_case {
    ($name:ident, $op:ident, $shape:expr, $seed:expr, $nonzero:expr) => {
        pub fn $name<B: PortableBackend + 'static>(backend: &Arc<B>) {
            run_binary_case(backend, $shape, $seed, BinaryOp::$op, $nonzero);
        }
    };
}

macro_rules! unary_case {
    ($name:ident, $op:ident, $shape:expr, $seed:expr) => {
        pub fn $name<B: PortableBackend + 'static>(backend: &Arc<B>) {
            run_unary_case(backend, $shape, $seed, UnaryOp::$op);
        }
    };
}

binary_case!(add_matches_torch_shape_2x3x4, Add, &[2, 3, 4], 0, false);
binary_case!(add_matches_torch_shape_1x1x1, Add, &[1, 1, 1], 1, false);
binary_case!(add_matches_torch_shape_2x7x13, Add, &[2, 7, 13], 2, false);
binary_case!(add_matches_torch_shape_1x31x37, Add, &[1, 31, 37], 3, false);

binary_case!(sub_matches_torch_shape_2x3x4, Sub, &[2, 3, 4], 4, false);
binary_case!(sub_matches_torch_shape_3x5x9, Sub, &[3, 5, 9], 5, false);
binary_case!(sub_matches_torch_shape_4x32, Sub, &[4, 32], 6, false);

binary_case!(mul_matches_torch_shape_2x3x4, Mul, &[2, 3, 4], 7, false);
binary_case!(mul_matches_torch_shape_3x5x9, Mul, &[3, 5, 9], 8, false);
binary_case!(
    mul_matches_torch_shape_2x3x1024,
    Mul,
    &[2, 3, 1024],
    9,
    false
);

binary_case!(div_matches_torch_shape_2x3x4, Div, &[2, 3, 4], 10, true);
binary_case!(div_matches_torch_shape_3x5x9, Div, &[3, 5, 9], 11, true);
binary_case!(
    div_matches_torch_shape_2x3x1024,
    Div,
    &[2, 3, 1024],
    12,
    true
);

unary_case!(neg_matches_torch_shape_2x3x4, Neg, &[2, 3, 4], 13);
unary_case!(neg_matches_torch_shape_1x16, Neg, &[1, 16], 14);

unary_case!(abs_matches_torch_shape_2x3x4, Abs, &[2, 3, 4], 15);
unary_case!(abs_matches_torch_shape_1x16, Abs, &[1, 16], 16);

binary_case!(max_matches_torch_shape_2x3x4, Max, &[2, 3, 4], 17, false);
binary_case!(max_matches_torch_shape_2x7x13, Max, &[2, 7, 13], 18, false);

binary_case!(min_matches_torch_shape_2x3x4, Min, &[2, 3, 4], 19, false);
binary_case!(min_matches_torch_shape_2x7x13, Min, &[2, 7, 13], 20, false);

pub fn clamp_matches_torch_min_max_shape_2x3x4<B: PortableBackend + 'static>(backend: &Arc<B>) {
    run_clamp_case(backend, &[2, 3, 4], 21, false, false);
}

pub fn clamp_matches_torch_min_only_shape_2x3x4<B: PortableBackend + 'static>(backend: &Arc<B>) {
    run_clamp_case(backend, &[2, 3, 4], 22, true, false);
}

pub fn clamp_matches_torch_max_only_shape_2x3x4<B: PortableBackend + 'static>(backend: &Arc<B>) {
    run_clamp_case(backend, &[2, 3, 4], 23, false, true);
}

pub fn add_rejects_shape_mismatch<B: PortableBackend + 'static>(backend: &Arc<B>) {
    let err = timed_gpt(|| {
        let a = device_tensor_from_data(backend, &[2, 3], &[0.0; 6]);
        let b = device_tensor_from_data(backend, &[2, 4], &[0.0; 8]);
        a.add(&b).unwrap_err()
    });
    assert!(err.to_string().contains("share shape"));
}

pub fn div_rejects_shape_mismatch<B: PortableBackend + 'static>(backend: &Arc<B>) {
    let err = timed_gpt(|| {
        let a = device_tensor_from_data(backend, &[2, 3, 4], &[0.0; 24]);
        let b = device_tensor_from_data(backend, &[2, 3, 5], &[1.0; 30]);
        a.div(&b).unwrap_err()
    });
    assert!(err.to_string().contains("share shape"));
}
