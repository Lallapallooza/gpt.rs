//! Validation errors of functionals, raised before any capture (backend independent).

use std::sync::Arc;

use gpt_rs::ops::functional::{self, DecodeKvCache};
use gpt_rs::tensor::{DType, DeviceTensor, Shape, Tensor};
use gpt_rs_backend_ref_cpu::CpuPortableBackend;
use gpt_rs_backend_tests::tensor_as;

fn filled(
    backend: &Arc<CpuPortableBackend>,
    shape: &[usize],
    value: f32,
) -> DeviceTensor<CpuPortableBackend> {
    let host = Tensor::from_vec(
        Shape::new(shape.to_vec()),
        vec![value; shape.iter().product()],
    )
    .unwrap();
    DeviceTensor::from_host(Arc::clone(backend), host).unwrap()
}

#[test]
fn gated_delta_rule_rejects_value_heads_not_divisible_by_key_heads() {
    let backend = Arc::new(CpuPortableBackend::new());
    let d = |shape: &[usize]| filled(&backend, shape, 0.1);
    let err = functional::gated_delta_rule(
        &d(&[2, 3, 4]),
        &d(&[2, 3, 4]),
        &d(&[2, 4, 5]),
        &d(&[2, 4]),
        &d(&[2, 4]),
        &d(&[4, 4, 5]),
    )
    .err()
    .expect("value heads not divisible by key heads must be rejected");
    assert!(err.to_string().contains("multiple of key heads"), "{err}");
}

#[test]
fn attention_kv_cache_rejects_writes_past_capacity() {
    let backend = Arc::new(CpuPortableBackend::new());
    let zeros = |shape: &[usize]| filled(&backend, shape, 0.0);
    let cache = DecodeKvCache::new(zeros(&[2, 4, 8]), zeros(&[2, 4, 8]), 3).unwrap();
    let update_starts = DeviceTensor::from_host(
        Arc::clone(&backend),
        Tensor::from_i32(Shape::new([3]), vec![0, 3, 0]).unwrap(),
    )
    .unwrap();
    let err = functional::attention_kv_cache(
        &zeros(&[2, 2, 8]),
        &zeros(&[2, 2, 8]),
        &zeros(&[2, 2, 8]),
        &cache,
        &update_starts,
    )
    .err()
    .expect("writing past the cache capacity must be rejected");
    assert!(err.to_string().contains("exceed capacity"), "{err}");
}

#[test]
fn linear_rejects_in_features_mismatch() {
    let backend = Arc::new(CpuPortableBackend::new());
    let x = filled(&backend, &[2, 4], 0.0);
    let host = tensor_as(&[3, 5], &[0.0; 15], DType::BF16);
    let w = DeviceTensor::from_host(Arc::clone(&backend), host).unwrap();
    let err = functional::linear(&x, &w).expect_err("mismatched in_features must be rejected");
    assert!(err.to_string().contains("in_features"), "{err}");
}

fn error_of<T>(result: anyhow::Result<T>) -> String {
    match result {
        Ok(_) => panic!("invalid inputs must be rejected"),
        Err(err) => err.to_string(),
    }
}

#[test]
fn validation_messages_name_the_functional_and_the_argument() {
    let backend = Arc::new(CpuPortableBackend::new());
    let f32 = |shape: &[usize]| filled(&backend, shape, 0.5);
    let typed = |shape: &[usize], dtype| {
        let host = tensor_as(shape, &vec![0.0; shape.iter().product()], dtype);
        DeviceTensor::from_host(Arc::clone(&backend), host).unwrap()
    };
    let indices = Tensor::from_i32(Shape::new([2, 4]), vec![0; 8]).unwrap();
    let indices = DeviceTensor::from_host(Arc::clone(&backend), indices).unwrap();

    assert_eq!(
        error_of(functional::linear(&f32(&[4]), &f32(&[3, 4]))),
        "linear: x must have rank 2, got [4]"
    );
    assert_eq!(
        error_of(functional::linear(&indices, &f32(&[3, 4]))),
        "linear: x must have dtype F32 | BF16, got I32"
    );
    assert_eq!(
        error_of(functional::swiglu(&f32(&[2, 3]), &f32(&[2, 4]))),
        "swiglu: up shape [2, 4] must match gate shape [2, 3]"
    );
    assert_eq!(
        error_of(functional::add_bias(
            &f32(&[2, 4]),
            &typed(&[4], DType::BF16)
        )),
        "add_bias: bias dtype BF16 must match x dtype F32"
    );
    assert_eq!(
        error_of(functional::add_bias(&f32(&[2, 4]), &f32(&[5]))),
        "add_bias: bias must have last dimension 4, got [5]"
    );
}
