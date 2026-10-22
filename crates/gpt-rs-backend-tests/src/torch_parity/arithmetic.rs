use std::sync::Arc;

use gpt_rs::backend::PortableBackend;
use gpt_rs::ops::functional;

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
    check_binary(backend, shape, &a_data, &b_data, op);
}

fn run_binary_special_values_case<B: PortableBackend + 'static>(backend: &Arc<B>, op: BinaryOp) {
    let special_count = SPECIAL_VALUES.len();
    let a_data: Vec<f32> = (0..special_count * special_count)
        .map(|index| SPECIAL_VALUES[index / special_count])
        .collect();
    let b_data: Vec<f32> = (0..special_count * special_count)
        .map(|index| SPECIAL_VALUES[index % special_count])
        .collect();
    check_binary(
        backend,
        &[special_count, special_count],
        &a_data,
        &b_data,
        op,
    );
}

fn check_binary<B: PortableBackend + 'static>(
    backend: &Arc<B>,
    shape: &[usize],
    a_data: &[f32],
    b_data: &[f32],
    op: BinaryOp,
) {
    let expected = timed_torch(|| {
        let a_t = tch_tensor_from_vec(shape, a_data);
        let b_t = tch_tensor_from_vec(shape, b_data);
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
        let a_dev = device_tensor_from_data(backend, shape, a_data);
        let b_dev = device_tensor_from_data(backend, shape, b_data);
        let out = match op {
            BinaryOp::Add => a_dev.add(&b_dev),
            BinaryOp::Sub => a_dev.sub(&b_dev),
            BinaryOp::Mul => a_dev.mul(&b_dev),
            BinaryOp::Div => a_dev.div(&b_dev),
            BinaryOp::Max => a_dev.maximum(&b_dev),
            BinaryOp::Min => a_dev.minimum(&b_dev),
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
            UnaryOp::Neg => dev.neg(),
            UnaryOp::Abs => dev.abs(),
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
    let lower_bounds: Vec<f32> = base_data.iter().map(|value| value - 0.5).collect();
    let upper_bounds: Vec<f32> = base_data.iter().map(|value| value + 0.5).collect();
    let min_data = (!max_only).then_some(lower_bounds.as_slice());
    let max_data = (!min_only).then_some(upper_bounds.as_slice());
    check_clamp(backend, shape, &base_data, min_data, max_data);
}

fn check_clamp<B: PortableBackend + 'static>(
    backend: &Arc<B>,
    shape: &[usize],
    base_data: &[f32],
    min_data: Option<&[f32]>,
    max_data: Option<&[f32]>,
) {
    let expected = timed_torch(|| {
        let mut out = tch_tensor_from_vec(shape, base_data);
        if let Some(lower_bounds) = min_data {
            out = out.maximum(&tch_tensor_from_vec(shape, lower_bounds));
        }
        if let Some(upper_bounds) = max_data {
            out = out.minimum(&tch_tensor_from_vec(shape, upper_bounds));
        }
        tensor_to_vec(&out)
    });

    let actual = timed_gpt(|| {
        let base_dev = device_tensor_from_data(backend, shape, base_data);
        let min_dev = min_data.map(|data| device_tensor_from_data(backend, shape, data));
        let max_dev = max_data.map(|data| device_tensor_from_data(backend, shape, data));
        let out = functional::clamp(&base_dev, min_dev.as_ref(), max_dev.as_ref());
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
// At `PARALLEL_MIN_WORK` elements or more, the C backend runs elementwise loops threaded.
binary_case!(
    add_matches_torch_shape_3x197x113,
    Add,
    &[3, 197, 113],
    1,
    false
);
binary_case!(sub_matches_torch_shape_2x3x4, Sub, &[2, 3, 4], 4, false);
binary_case!(mul_matches_torch_shape_2x3x4, Mul, &[2, 3, 4], 7, false);
binary_case!(div_matches_torch_shape_2x3x4, Div, &[2, 3, 4], 10, true);
unary_case!(neg_matches_torch_shape_2x3x4, Neg, &[2, 3, 4], 13);
unary_case!(abs_matches_torch_shape_2x3x4, Abs, &[2, 3, 4], 15);

pub fn clamp_matches_torch_min_only_shape_2x3x4<B: PortableBackend + 'static>(backend: &Arc<B>) {
    run_clamp_case(backend, &[2, 3, 4], 22, true, false);
}

pub fn clamp_matches_torch_max_only_shape_2x3x4<B: PortableBackend + 'static>(backend: &Arc<B>) {
    run_clamp_case(backend, &[2, 3, 4], 23, false, true);
}

pub fn max_matches_torch_special_values<B: PortableBackend + 'static>(backend: &Arc<B>) {
    run_binary_special_values_case(backend, BinaryOp::Max);
}

pub fn min_matches_torch_special_values<B: PortableBackend + 'static>(backend: &Arc<B>) {
    run_binary_special_values_case(backend, BinaryOp::Min);
}

pub fn clamp_matches_torch_special_values<B: PortableBackend + 'static>(backend: &Arc<B>) {
    let special_count = SPECIAL_VALUES.len();
    let len = special_count * special_count * special_count;
    let base: Vec<f32> = (0..len)
        .map(|index| SPECIAL_VALUES[index / (special_count * special_count)])
        .collect();
    let min: Vec<f32> = (0..len)
        .map(|index| SPECIAL_VALUES[index / special_count % special_count])
        .collect();
    let max: Vec<f32> = (0..len)
        .map(|index| SPECIAL_VALUES[index % special_count])
        .collect();
    check_clamp(
        backend,
        &[special_count, special_count, special_count],
        &base,
        Some(&min),
        Some(&max),
    );
}
