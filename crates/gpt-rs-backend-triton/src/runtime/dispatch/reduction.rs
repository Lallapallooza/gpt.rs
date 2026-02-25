use super::*;

impl TritonExecutor {
    pub(super) fn execute_reduce(
        &self,
        driver: &Arc<CudaDriver>,
        kernels: &HashMap<&str, &KernelSpec>,
        input_spec: &TensorSpec,
        spec: &gpt_rs::backend::spec::ReduceSpec,
        input: &TritonTensor,
        out: OutputBinding<'_>,
    ) -> BackendResult<TritonTensor> {
        match spec.kind {
            ReduceKind::Sum => {
                let kernel = kernels.get(REDUCE_SUM_LAST_AXIS_KERNEL_ID).ok_or_else(|| {
                    BackendError::execution(
                        "missing reduce_sum_last_axis kernel in triton artifact",
                    )
                })?;
                self.execute_reduce_sum_last_axis(
                    driver,
                    kernel,
                    input_spec,
                    spec,
                    input,
                    out.clone(),
                )
            }
            ReduceKind::Max => {
                let kernel = kernels.get(REDUCE_MAX_LAST_AXIS_KERNEL_ID).ok_or_else(|| {
                    BackendError::execution(
                        "missing reduce_max_last_axis kernel in triton artifact",
                    )
                })?;
                self.execute_reduce_max_last_axis(driver, kernel, input_spec, spec, input, out)
            }
            other => Err(BackendError::execution(format!(
                "reduce runtime unsupported kind {:?}",
                other
            ))),
        }
    }

    pub(super) fn execute_reduce_max_last_axis(
        &self,
        driver: &Arc<CudaDriver>,
        kernel: &KernelSpec,
        input_spec: &TensorSpec,
        spec: &gpt_rs::backend::spec::ReduceSpec,
        input: &TritonTensor,
        out: OutputBinding<'_>,
    ) -> BackendResult<TritonTensor> {
        let out_spec = out.spec;
        let output = out.tensor;
        if !matches!(kernel.kind, KernelKind::ReduceMaxLastAxisF32) {
            return Err(BackendError::execution(format!(
                "unexpected reduce_max kernel kind: {:?}",
                kernel.kind
            )));
        }
        if input.spec != *input_spec {
            return Err(BackendError::execution("reduce_max tensor/spec mismatch"));
        }
        if input_spec.dtype != DType::F32 || out_spec.dtype != DType::F32 {
            return Err(BackendError::execution(
                "reduce_max currently supports F32 only",
            ));
        }
        let input_dims = static_dims_or_error(&input_spec.shape, |_| {
            BackendError::execution("dynamic dimensions are not supported by triton runtime")
        })?;
        if input_dims.is_empty() {
            return Err(BackendError::execution(
                "reduce_max does not support scalar inputs",
            ));
        }
        let last_axis = input_dims.len() - 1;
        if spec.axes.as_slice() != [last_axis] {
            return Err(BackendError::execution(
                "reduce_max supports reducing the last axis only",
            ));
        }
        let rows = input_dims[..last_axis]
            .iter()
            .try_fold(1usize, |acc: usize, dim| acc.checked_mul(*dim))
            .ok_or_else(|| BackendError::execution("reduce_max row dimension overflow"))?;
        let cols = input_dims[last_axis];
        let out = allocate_output_tensor(driver, out_spec, output)?;
        if rows == 0 || cols == 0 {
            return Ok(out);
        }
        let loaded = self.load_kernel(driver, kernel)?;
        let mut in_ptr = input.buffer.device_ptr();
        let mut out_ptr = out.buffer.device_ptr();
        let mut rows_i32 = i32::try_from(rows)
            .map_err(|_| BackendError::execution("reduce_max rows exceeds i32 range"))?;
        let mut cols_i32 = i32::try_from(cols)
            .map_err(|_| BackendError::execution("reduce_max cols exceeds i32 range"))?;
        let mut opaque_ptr = 0u64;
        let mut params = [
            (&mut in_ptr as *mut u64).cast::<c_void>(),
            (&mut out_ptr as *mut u64).cast::<c_void>(),
            (&mut rows_i32 as *mut i32).cast::<c_void>(),
            (&mut cols_i32 as *mut i32).cast::<c_void>(),
            (&mut opaque_ptr as *mut u64).cast::<c_void>(),
        ];
        let rows_u32 = u32::try_from(rows)
            .map_err(|_| BackendError::execution("reduce_max rows exceeds u32 range"))?;
        launch_program_grid(driver, &loaded, rows_u32, 256, rows_u32, &mut params)?;
        Ok(out)
    }

    pub(super) fn execute_reduce_sum_last_axis(
        &self,
        driver: &Arc<CudaDriver>,
        kernel: &KernelSpec,
        input_spec: &TensorSpec,
        spec: &gpt_rs::backend::spec::ReduceSpec,
        input: &TritonTensor,
        out: OutputBinding<'_>,
    ) -> BackendResult<TritonTensor> {
        let out_spec = out.spec;
        let output = out.tensor;
        if input.spec != *input_spec {
            return Err(BackendError::execution("reduce tensor/spec mismatch"));
        }
        if input_spec.dtype != DType::F32 || out_spec.dtype != DType::F32 {
            return Err(BackendError::execution(
                "reduce runtime currently supports F32 only",
            ));
        }
        if spec.kind != ReduceKind::Sum {
            return Err(BackendError::execution(
                "reduce runtime currently supports sum only",
            ));
        }
        if !matches!(kernel.kind, KernelKind::ReduceSumLastAxisF32) {
            return Err(BackendError::execution(format!(
                "unexpected reduce_sum kernel kind: {:?}",
                kernel.kind
            )));
        }

        let input_dims = static_dims_or_error(&input_spec.shape, |_| {
            BackendError::execution("dynamic dimensions are not supported by triton runtime")
        })?;
        if input_dims.is_empty() {
            return Err(BackendError::execution(
                "reduce runtime does not support scalar inputs",
            ));
        }
        let last_axis = input_dims.len() - 1;
        if spec.axes.as_slice() != [last_axis] {
            return Err(BackendError::execution(
                "reduce runtime supports reducing the last axis only",
            ));
        }

        let rows = input_dims[..last_axis]
            .iter()
            .try_fold(1usize, |acc: usize, dim| acc.checked_mul(*dim))
            .ok_or_else(|| BackendError::execution("reduce row dimension overflow"))?;
        let cols = input_dims[last_axis];

        let out = allocate_output_tensor(driver, out_spec, output)?;
        if rows == 0 || cols == 0 {
            return Ok(out);
        }
        let loaded = self.load_kernel(driver, kernel)?;
        let mut in_ptr = input.buffer.device_ptr();
        let mut out_ptr = out.buffer.device_ptr();
        let mut rows_i32 = i32::try_from(rows)
            .map_err(|_| BackendError::execution("reduce_sum rows exceeds i32 range"))?;
        let mut cols_i32 = i32::try_from(cols)
            .map_err(|_| BackendError::execution("reduce_sum cols exceeds i32 range"))?;
        let mut opaque_ptr = 0u64;
        let mut params = [
            (&mut in_ptr as *mut u64).cast::<c_void>(),
            (&mut out_ptr as *mut u64).cast::<c_void>(),
            (&mut rows_i32 as *mut i32).cast::<c_void>(),
            (&mut cols_i32 as *mut i32).cast::<c_void>(),
            (&mut opaque_ptr as *mut u64).cast::<c_void>(),
        ];
        let rows_u32 = u32::try_from(rows)
            .map_err(|_| BackendError::execution("reduce_sum rows exceeds u32 range"))?;
        launch_program_grid(driver, &loaded, rows_u32, 256, rows_u32, &mut params)?;
        Ok(out)
    }
}
