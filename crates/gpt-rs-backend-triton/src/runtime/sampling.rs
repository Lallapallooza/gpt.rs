use std::ffi::c_void;
use std::sync::{Arc, OnceLock};

use gpt_rs::backend::shape_helpers::static_dims_or_error;
use gpt_rs::backend::spec::{BackendError, BackendResult, DType, DecodeSampleRequest, TensorSpec};
use gpt_rs::profiling::{self, WorkStats};

use crate::device::{self, CudaDriver};
use crate::kernels::{
    sample_argmax_last_axis_kernel_spec, sample_temperature_last_axis_kernel_spec, KernelSpec,
};
use crate::tensor::TritonTensor;

use super::ops::launch_program_grid;
use super::TritonExecutor;

const F32_BYTES: usize = 4;

fn argmax_kernel_spec() -> &'static KernelSpec {
    static SPEC: OnceLock<KernelSpec> = OnceLock::new();
    SPEC.get_or_init(sample_argmax_last_axis_kernel_spec)
}

fn temperature_kernel_spec() -> &'static KernelSpec {
    static SPEC: OnceLock<KernelSpec> = OnceLock::new();
    SPEC.get_or_init(sample_temperature_last_axis_kernel_spec)
}

impl TritonExecutor {
    pub fn sample_decode_token(
        &self,
        logits: &TritonTensor,
        logits_spec: &TensorSpec,
        request: DecodeSampleRequest,
    ) -> BackendResult<Option<usize>> {
        if request.top_k.is_some() {
            return Ok(None);
        }
        if logits.spec != *logits_spec {
            return Err(BackendError::execution(
                "triton decode sampling logits/spec mismatch",
            ));
        }
        if logits_spec.dtype != DType::F32 {
            return Ok(None);
        }

        let dims = static_dims_or_error(&logits_spec.shape, |_| {
            BackendError::execution("triton decode sampling requires static logits shape")
        })?;
        if dims.is_empty() {
            return Ok(None);
        }
        let vocab = *dims
            .last()
            .ok_or_else(|| BackendError::execution("invalid logits shape"))?;
        if vocab == 0 {
            return Ok(None);
        }

        let rows = if dims.len() == 1 {
            1usize
        } else {
            dims[..dims.len() - 1]
                .iter()
                .try_fold(1usize, |acc, dim| acc.checked_mul(*dim))
                .ok_or_else(|| BackendError::execution("decode logits row count overflow"))?
        };
        if rows == 0 {
            return Ok(None);
        }

        let row_start_elem = rows
            .checked_sub(1)
            .and_then(|row| row.checked_mul(vocab))
            .ok_or_else(|| BackendError::execution("decode logits row offset overflow"))?;
        let row_start_bytes = row_start_elem
            .checked_mul(F32_BYTES)
            .ok_or_else(|| BackendError::execution("decode logits byte offset overflow"))?;
        let row_ptr =
            logits
                .buffer
                .device_ptr()
                .checked_add(u64::try_from(row_start_bytes).map_err(|_| {
                    BackendError::execution("decode logits byte offset exceeds u64")
                })?)
                .ok_or_else(|| BackendError::execution("decode logits device pointer overflow"))?;

        let driver = device::driver()?;
        let start = std::time::Instant::now();
        let token = if request.temperature <= 0.0 {
            self.sample_argmax_row(&driver, row_ptr, vocab)?
        } else {
            let random_u = match request.random_u {
                Some(value) if value.is_finite() => value,
                _ => return Ok(None),
            };
            let seed = random_u
                .to_bits()
                .wrapping_mul(747_796_405)
                .wrapping_add(2_891_336_453);
            self.sample_temperature_row(&driver, row_ptr, vocab, request.temperature, seed)?
        };

        profiling::record_backend_aggregate(
            "backend.triton.sample_decode_token",
            1,
            start.elapsed(),
            WorkStats {
                elements: vocab as u64,
                ..WorkStats::default()
            },
        );

        if token < vocab {
            return Ok(Some(token));
        }
        Ok(Some(self.sample_argmax_row(&driver, row_ptr, vocab)?))
    }

    fn sample_argmax_row(
        &self,
        driver: &Arc<CudaDriver>,
        row_ptr: u64,
        vocab: usize,
    ) -> BackendResult<usize> {
        let kernel = self.load_kernel(driver, argmax_kernel_spec())?;
        let out = driver.alloc_zeroed(std::mem::size_of::<i32>())?;
        let mut logits_ptr = row_ptr;
        let mut out_ptr = out.device_ptr();
        let mut rows_i32 = 1i32;
        let mut cols_i32 = i32::try_from(vocab)
            .map_err(|_| BackendError::execution("decode vocab exceeds i32 range"))?;
        let mut opaque_ptr = 0u64;
        let mut params = [
            (&mut logits_ptr as *mut u64).cast::<c_void>(),
            (&mut out_ptr as *mut u64).cast::<c_void>(),
            (&mut rows_i32 as *mut i32).cast::<c_void>(),
            (&mut cols_i32 as *mut i32).cast::<c_void>(),
            (&mut opaque_ptr as *mut u64).cast::<c_void>(),
        ];
        launch_program_grid(driver, &kernel, 1, 256, 1, &mut params)?;

        let bytes = driver.download_with_metric(
            out.device_ptr(),
            std::mem::size_of::<i32>(),
            "backend.triton.memcpy.d2h.token",
        )?;
        let idx = i32::from_ne_bytes(bytes.as_slice().try_into().map_err(|_| {
            BackendError::execution("invalid token sample byte length from argmax kernel")
        })?);
        usize::try_from(idx).map_err(|_| BackendError::execution("negative token id sampled"))
    }

    fn sample_temperature_row(
        &self,
        driver: &Arc<CudaDriver>,
        row_ptr: u64,
        vocab: usize,
        temperature: f32,
        seed: u32,
    ) -> BackendResult<usize> {
        let kernel = self.load_kernel(driver, temperature_kernel_spec())?;
        let out = driver.alloc_zeroed(std::mem::size_of::<i32>())?;
        let mut logits_ptr = row_ptr;
        let mut out_ptr = out.device_ptr();
        let mut rows_i32 = 1i32;
        let mut cols_i32 = i32::try_from(vocab)
            .map_err(|_| BackendError::execution("decode vocab exceeds i32 range"))?;
        let mut temperature_f32 = temperature;
        let mut seed_u32 = seed;
        let mut opaque_ptr = 0u64;
        let mut params = [
            (&mut logits_ptr as *mut u64).cast::<c_void>(),
            (&mut out_ptr as *mut u64).cast::<c_void>(),
            (&mut rows_i32 as *mut i32).cast::<c_void>(),
            (&mut cols_i32 as *mut i32).cast::<c_void>(),
            (&mut temperature_f32 as *mut f32).cast::<c_void>(),
            (&mut seed_u32 as *mut u32).cast::<c_void>(),
            (&mut opaque_ptr as *mut u64).cast::<c_void>(),
        ];
        launch_program_grid(driver, &kernel, 1, 256, 1, &mut params)?;

        let bytes = driver.download_with_metric(
            out.device_ptr(),
            std::mem::size_of::<i32>(),
            "backend.triton.memcpy.d2h.token",
        )?;
        let idx = i32::from_ne_bytes(bytes.as_slice().try_into().map_err(|_| {
            BackendError::execution("invalid token sample byte length from temperature kernel")
        })?);
        usize::try_from(idx).map_err(|_| BackendError::execution("negative token id sampled"))
    }
}
