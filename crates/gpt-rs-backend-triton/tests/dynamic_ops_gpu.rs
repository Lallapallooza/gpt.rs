use std::sync::Arc;

use anyhow::Result;
use gpt_rs::ops::functional::CaptureIntoDeviceTensor;
use gpt_rs::tensor::{DeviceTensor, Shape, Tensor};
use gpt_rs_backend_triton::TritonBackend;
use gpt_rs_macros::capture_ptir;

fn capture_dynamic_slice(
    input: &DeviceTensor<TritonBackend>,
    starts: &DeviceTensor<TritonBackend>,
    sizes: Vec<usize>,
) -> Result<DeviceTensor<TritonBackend>> {
    capture_ptir!({ input, starts }, |_session| {
        Ok(input.dynamic_slice(&starts, sizes.clone()).id())
    })?
    .into_device_tensor()
}

fn capture_dynamic_update_slice(
    base: &DeviceTensor<TritonBackend>,
    update: &DeviceTensor<TritonBackend>,
    starts: &DeviceTensor<TritonBackend>,
    sizes: Vec<usize>,
) -> Result<DeviceTensor<TritonBackend>> {
    capture_ptir!({ base, update, starts }, |_session| {
        Ok(base
            .dynamic_update_slice(&update, &starts, sizes.clone())
            .id())
    })?
    .into_device_tensor()
}

#[test]
fn dynamic_slice_f32_clamps_start_indices_on_gpu() -> Result<()> {
    let backend = Arc::new(TritonBackend::new());
    let input = DeviceTensor::from_host(
        Arc::clone(&backend),
        Tensor::from_vec(Shape::new([5]), vec![1.0f32, 2.0, 3.0, 4.0, 5.0])?,
    )?;

    let starts_neg = DeviceTensor::from_host(
        Arc::clone(&backend),
        Tensor::from_i32(Shape::new([1]), vec![-3])?,
    )?;
    let out_neg = capture_dynamic_slice(&input, &starts_neg, vec![2])?;
    assert_eq!(out_neg.to_host()?.data(), &[1.0, 2.0]);

    let starts_hi = DeviceTensor::from_host(
        Arc::clone(&backend),
        Tensor::from_i32(Shape::new([1]), vec![9])?,
    )?;
    let out_hi = capture_dynamic_slice(&input, &starts_hi, vec![2])?;
    assert_eq!(out_hi.to_host()?.data(), &[4.0, 5.0]);

    Ok(())
}

#[test]
fn dynamic_slice_si32_rank1_clamps_start_indices_on_gpu() -> Result<()> {
    let backend = Arc::new(TritonBackend::new());
    let input = DeviceTensor::from_host(
        Arc::clone(&backend),
        Tensor::from_i32(Shape::new([5]), vec![10, 20, 30, 40, 50])?,
    )?;

    let starts = DeviceTensor::from_host(
        Arc::clone(&backend),
        Tensor::from_i32(Shape::new([1]), vec![7])?,
    )?;
    let out = capture_dynamic_slice(&input, &starts, vec![2])?;
    assert_eq!(out.to_host()?.data_i32(), &[40, 50]);

    Ok(())
}

#[test]
fn dynamic_update_slice_f32_clamps_start_indices_on_gpu() -> Result<()> {
    let backend = Arc::new(TritonBackend::new());
    let base = DeviceTensor::from_host(
        Arc::clone(&backend),
        Tensor::from_vec(Shape::new([5]), vec![10.0f32, 11.0, 12.0, 13.0, 14.0])?,
    )?;
    let update = DeviceTensor::from_host(
        Arc::clone(&backend),
        Tensor::from_vec(Shape::new([2]), vec![99.0f32, 88.0])?,
    )?;

    let starts_neg = DeviceTensor::from_host(
        Arc::clone(&backend),
        Tensor::from_i32(Shape::new([1]), vec![-2])?,
    )?;
    let out_neg = capture_dynamic_update_slice(&base, &update, &starts_neg, vec![2])?;
    assert_eq!(out_neg.to_host()?.data(), &[99.0, 88.0, 12.0, 13.0, 14.0]);

    let starts_hi = DeviceTensor::from_host(
        Arc::clone(&backend),
        Tensor::from_i32(Shape::new([1]), vec![8])?,
    )?;
    let out_hi = capture_dynamic_update_slice(&base, &update, &starts_hi, vec![2])?;
    assert_eq!(out_hi.to_host()?.data(), &[10.0, 11.0, 12.0, 99.0, 88.0]);

    Ok(())
}
