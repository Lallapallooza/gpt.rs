//! The panics of functionals and tensor ops on inputs that do not fit, with their exact messages
//! (backend independent).

mod support;

use std::sync::Arc;

use gpt_rs::ops::functional::{self, AttentionPositions, DecodeKvCache, WeightLayout};
use gpt_rs::tensor::{DType, DeviceTensor, HostTensor};
use gpt_rs_backend_ref_cpu::CpuPortableBackend;
use gpt_rs_backend_tests::{panic_message, tensor_as};

use self::support::filled;

#[test]
fn gated_delta_rule_rejects_value_heads_not_divisible_by_key_heads() {
    let backend = Arc::new(CpuPortableBackend::new());
    let d = |shape: &[usize]| filled(&backend, shape, 0.1);
    let message = panic_message(|| {
        functional::gated_delta_rule(
            &d(&[2, 3, 4]),
            &d(&[2, 3, 4]),
            &d(&[2, 4, 5]),
            &d(&[2, 4]),
            &d(&[2, 4]),
            &d(&[4, 4, 5]),
        )
    });
    assert_eq!(
        message,
        "gated_delta_rule: value heads (4) must be a multiple of key heads (3)"
    );
}

#[test]
fn attention_kv_cache_rejects_writes_past_capacity() {
    let backend = Arc::new(CpuPortableBackend::new());
    let zeros = |shape: &[usize]| filled(&backend, shape, 0.0);
    let cache = DecodeKvCache::new(zeros(&[2, 4, 8]), zeros(&[2, 4, 8]), 3);
    let positions = DeviceTensor::from_host(
        Arc::clone(&backend),
        HostTensor::si32([2], vec![3, 4]).unwrap(),
    )
    .unwrap();
    let positions = AttentionPositions::new(positions, None);
    let message = panic_message(|| {
        functional::attention_kv_cache(
            &zeros(&[2, 2, 8]),
            &zeros(&[2, 2, 8]),
            &zeros(&[2, 2, 8]),
            &cache,
            &positions,
        )
    });
    assert_eq!(
        message,
        "attention_kv_cache: update would exceed capacity (len 3 + seq_len 2 > 4)"
    );
}

#[test]
fn linear_in_out_contracts_the_first_weight_axis() {
    let backend = Arc::new(CpuPortableBackend::new());
    let tensor = |dims: [usize; 2], data: Vec<f32>| {
        let host = HostTensor::f32(dims, data).unwrap();
        DeviceTensor::from_host(Arc::clone(&backend), host).unwrap()
    };
    let x = tensor([2, 4], (1..=8).map(|v| v as f32).collect());
    let w = tensor([4, 3], (1..=12).map(|v| v as f32).collect());
    let y = functional::linear(&x, &w, WeightLayout::InOut)
        .to_host()
        .unwrap();
    assert_eq!(y.dims[..], [2, 3]);
    // x @ w with w row-major [in, out], computed by hand.
    assert_eq!(y.as_f32().unwrap(), [70.0, 80.0, 90.0, 158.0, 184.0, 210.0]);
    assert_eq!(
        panic_message(|| functional::linear(&x, &w, WeightLayout::OutIn)),
        "linear: weight in_features 3 must match x in_features 4"
    );
}

#[test]
fn validation_messages_name_the_functional_and_the_argument() {
    let backend = Arc::new(CpuPortableBackend::new());
    let f32 = |shape: &[usize]| filled(&backend, shape, 0.5);
    let indices = HostTensor::si32([2, 4], vec![0; 8]).unwrap();
    let indices = DeviceTensor::from_host(Arc::clone(&backend), indices).unwrap();

    assert_eq!(
        panic_message(|| functional::linear(&f32(&[4]), &f32(&[3, 4]), WeightLayout::OutIn)),
        "linear: x must have rank 2, got [4]"
    );
    assert_eq!(
        panic_message(|| functional::linear(&indices, &f32(&[3, 4]), WeightLayout::OutIn)),
        "linear: x must have dtype f32 | bf16, got si32"
    );
    assert_eq!(
        panic_message(|| functional::apply_rope(&f32(&[1, 2, 4]), &f32(&[2, 2]), &f32(&[2, 1]))),
        "apply_rope: sin shape [2, 1] must match cos shape [2, 2]"
    );
    assert_eq!(
        panic_message(|| functional::batch_norm(
            &f32(&[1, 2, 2, 4]),
            &f32(&[4]),
            &f32(&[4]),
            &f32(&[4]),
            &f32(&[3]),
            1e-5,
        )),
        "batch_norm: running_var must have last dimension 4, got [3]"
    );
}

#[test]
fn shape_checks_name_the_extents_that_do_not_match() {
    let backend = Arc::new(CpuPortableBackend::new());
    let f32 = |shape: &[usize]| filled(&backend, shape, 0.5);
    let indices = HostTensor::si32([3, 2], vec![0; 6]).unwrap();
    let indices = DeviceTensor::from_host(Arc::clone(&backend), indices).unwrap();

    assert_eq!(
        panic_message(|| functional::rms_norm(&f32(&[2, 3, 4]), &f32(&[3]), 1e-5)),
        "rms_norm: gamma must have last dimension 4, got [3]"
    );
    assert_eq!(
        panic_message(|| functional::layer_norm(&f32(&[2, 3, 8]), &f32(&[7]), &f32(&[8]), 1e-5)),
        "layer_norm: gamma must have last dimension 8, got [7]"
    );
    assert_eq!(
        panic_message(|| functional::embedding_lookup(&f32(&[8, 4]), &indices)),
        "embedding_lookup: indices must have rank 1, got [3, 2]"
    );
    assert_eq!(
        panic_message(|| functional::matmul(&f32(&[4, 5]), &f32(&[4, 3]))),
        "matmul: the last dimension 5 of a must match dimension 4 of b"
    );
    assert_eq!(
        panic_message(|| functional::matmul(&f32(&[2, 4, 5]), &f32(&[3, 5, 2]))),
        "matmul: batch 2 of a must match batch 3 of b"
    );
    assert_eq!(
        panic_message(|| functional::apply_rope(&f32(&[2, 8, 16]), &f32(&[7, 8]), &f32(&[7, 8]))),
        "apply_rope: cos sequence length 7 must match x sequence length 8"
    );
}

#[test]
fn softmax_rejects_an_axis_outside_the_rank_of_x() {
    let backend = Arc::new(CpuPortableBackend::new());
    let x = filled(&backend, &[2, 3], 0.5);
    assert_eq!(
        panic_message(|| functional::softmax(&x, 2)),
        "softmax: axis 2 is out of range for x of rank 2"
    );
    assert_eq!(
        panic_message(|| functional::softmax(&x, -3)),
        "softmax: axis -3 is out of range for x of rank 2"
    );
}

#[test]
fn binary_ops_broadcast_as_torch_and_reject_other_shapes() {
    let backend = Arc::new(CpuPortableBackend::new());
    let rows = |data: Vec<f32>| {
        let host = HostTensor::f32([2, 3], data).unwrap();
        DeviceTensor::from_host(Arc::clone(&backend), host).unwrap()
    };
    let column = DeviceTensor::from_host(
        Arc::clone(&backend),
        HostTensor::f32([2, 1], vec![10.0, 20.0]).unwrap(),
    )
    .unwrap();
    let x = rows(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let sum = (&x + &column - 1.0).to_host().unwrap();
    assert_eq!(sum.dims[..], [2, 3]);
    assert_eq!(sum.as_f32().unwrap(), [10.0, 11.0, 12.0, 23.0, 24.0, 25.0]);
    let bias = filled(&backend, &[4], 0.0);
    assert_eq!(
        panic_message(|| &x + &bias),
        "shapes [2, 3] and [4] do not broadcast"
    );
    let bf16 = tensor_as(&[3], &[0.0; 3], DType::Bf16);
    let bf16 = DeviceTensor::from_host(Arc::clone(&backend), bf16).unwrap();
    assert_eq!(
        panic_message(|| &x * &bf16),
        "tensor.mul: operands must have equal types, got tensor<2x3xf32> and tensor<2x3xbf16>"
    );
}

#[test]
fn ops_reject_tensors_of_two_backend_instances() {
    let (one, two) = (
        Arc::new(CpuPortableBackend::new()),
        Arc::new(CpuPortableBackend::new()),
    );
    let (a, b) = (filled(&one, &[2], 1.0), filled(&two, &[2], 1.0));
    assert_eq!(
        panic_message(|| &a + &b),
        "a ref-cpu tensor of another backend instance cannot join this graph"
    );
}
