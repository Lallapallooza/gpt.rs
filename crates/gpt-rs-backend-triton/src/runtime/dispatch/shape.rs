use super::*;

impl TritonExecutor {
    pub(super) fn execute_broadcast(
        &self,
        driver: &Arc<CudaDriver>,
        kernels: &HashMap<&str, &KernelSpec>,
        input: &TritonTensor,
        out_spec: &TensorSpec,
        output: Option<TritonTensor>,
    ) -> BackendResult<TritonTensor> {
        if input.spec.dtype != out_spec.dtype {
            return Err(BackendError::execution(
                "broadcast input/output dtype mismatch",
            ));
        }
        if !matches!(out_spec.dtype, DType::F32 | DType::Si32) {
            return Err(BackendError::execution(format!(
                "broadcast runtime supports F32/Si32 only, got {:?}",
                out_spec.dtype
            )));
        }

        let out = allocate_output_tensor(driver, out_spec, output)?;
        let element_count = static_element_count(&out_spec.shape)?;
        if element_count == 0 {
            return Ok(out);
        }

        let kernel_id = match out_spec.dtype {
            DType::F32 => BROADCAST_KERNEL_ID,
            DType::Si32 => BROADCAST_SI32_KERNEL_ID,
            _ => {
                return Err(BackendError::execution(format!(
                    "unsupported broadcast dtype {:?}",
                    out_spec.dtype
                )))
            }
        };
        let kernel = kernels.get(kernel_id).ok_or_else(|| {
            BackendError::execution(format!("missing broadcast kernel {kernel_id} in artifact"))
        })?;
        let expected_kind = match out_spec.dtype {
            DType::F32 => KernelKind::BroadcastF32Rank4,
            DType::Si32 => KernelKind::BroadcastSi32Rank4,
            _ => unreachable!(),
        };
        if kernel.kind != expected_kind {
            return Err(BackendError::execution(format!(
                "unexpected broadcast kernel kind: {:?}",
                kernel.kind
            )));
        }

        let (out_dims, in_strides) = broadcast_rank4_layout(&input.spec.shape, &out_spec.shape)?;
        let loaded = self.load_kernel(driver, kernel)?;

        let mut in_ptr = input.buffer.device_ptr();
        let mut out_ptr = out.buffer.device_ptr();
        let mut n = u32::try_from(element_count)
            .map_err(|_| BackendError::execution("broadcast element count exceeds u32 range"))?;
        let mut od0 = out_dims[0];
        let mut od1 = out_dims[1];
        let mut od2 = out_dims[2];
        let mut od3 = out_dims[3];
        let mut is0 = in_strides[0];
        let mut is1 = in_strides[1];
        let mut is2 = in_strides[2];
        let mut is3 = in_strides[3];
        let mut opaque_ptr = 0u64;
        let mut params = [
            (&mut in_ptr as *mut u64).cast::<c_void>(),
            (&mut out_ptr as *mut u64).cast::<c_void>(),
            (&mut n as *mut u32).cast::<c_void>(),
            (&mut od0 as *mut i32).cast::<c_void>(),
            (&mut od1 as *mut i32).cast::<c_void>(),
            (&mut od2 as *mut i32).cast::<c_void>(),
            (&mut od3 as *mut i32).cast::<c_void>(),
            (&mut is0 as *mut i32).cast::<c_void>(),
            (&mut is1 as *mut i32).cast::<c_void>(),
            (&mut is2 as *mut i32).cast::<c_void>(),
            (&mut is3 as *mut i32).cast::<c_void>(),
            (&mut opaque_ptr as *mut u64).cast::<c_void>(),
        ];
        launch_1d(driver, &loaded, n, 256, &mut params)?;
        Ok(out)
    }

    pub(super) fn execute_slice(
        &self,
        driver: &Arc<CudaDriver>,
        kernels: &HashMap<&str, &KernelSpec>,
        input: &TritonTensor,
        spec: &gpt_rs::backend::spec::SliceSpec,
        out_spec: &TensorSpec,
        output: Option<TritonTensor>,
    ) -> BackendResult<TritonTensor> {
        if input.spec.dtype != out_spec.dtype {
            return Err(BackendError::execution("slice input/output dtype mismatch"));
        }
        if spec.starts.len() != spec.sizes.len() {
            return Err(BackendError::execution(
                "slice starts and sizes length mismatch",
            ));
        }
        if let Some(copy) = contiguous_slice_copy_plan(&input.spec, out_spec, &spec.starts)? {
            let out = allocate_output_tensor(driver, out_spec, output)?;
            if copy.byte_len == 0 {
                return Ok(out);
            }
            let src_ptr = input
                .buffer
                .device_ptr()
                .checked_add(copy.byte_offset)
                .ok_or_else(|| BackendError::execution("slice contiguous pointer overflow"))?;
            driver.copy_device_to_device(out.buffer.device_ptr(), src_ptr, copy.byte_len)?;
            profiling::cache_event("triton_backend.slice_d2d");
            return Ok(out);
        }

        match out_spec.dtype {
            DType::F32 => {
                self.execute_slice_f32(driver, kernels, input, &spec.starts, out_spec, output)
            }
            DType::Si32 => {
                // Minimal i32 slice support for attention decode path (rank-1).
                let in_dims = static_dims_or_error(&input.spec.shape, |_| {
                    BackendError::execution(
                        "dynamic dimensions are not supported by triton runtime",
                    )
                })?;
                let out_dims = static_dims_or_error(&out_spec.shape, |_| {
                    BackendError::execution(
                        "dynamic dimensions are not supported by triton runtime",
                    )
                })?;
                if in_dims.len() != 1 || out_dims.len() != 1 || spec.starts.len() != 1 {
                    return Err(BackendError::execution(
                        "slice Si32 path currently supports rank-1 only",
                    ));
                }
                let start = spec.starts[0];
                let len = out_dims[0];
                if match start.checked_add(len) {
                    Some(end) => end > in_dims[0],
                    None => true,
                } {
                    return Err(BackendError::execution("slice Si32 bounds check failed"));
                }
                let out = allocate_output_tensor(driver, out_spec, output)?;
                let offset_bytes = start
                    .checked_mul(4)
                    .ok_or_else(|| BackendError::execution("slice Si32 offset overflow"))?;
                let src_ptr = input
                    .buffer
                    .device_ptr()
                    .checked_add(u64::try_from(offset_bytes).map_err(|_| {
                        BackendError::execution("slice Si32 offset conversion overflow")
                    })?)
                    .ok_or_else(|| BackendError::execution("slice Si32 pointer overflow"))?;
                driver.copy_device_to_device(out.buffer.device_ptr(), src_ptr, len * 4)?;
                Ok(out)
            }
            _ => Err(BackendError::execution(format!(
                "slice runtime supports F32/Si32 only, got {:?}",
                out_spec.dtype
            ))),
        }
    }

    pub(super) fn execute_slice_f32(
        &self,
        driver: &Arc<CudaDriver>,
        kernels: &HashMap<&str, &KernelSpec>,
        input: &TritonTensor,
        starts: &[usize],
        out_spec: &TensorSpec,
        output: Option<TritonTensor>,
    ) -> BackendResult<TritonTensor> {
        let kernel = kernels
            .get(SLICE_KERNEL_ID)
            .ok_or_else(|| BackendError::execution("missing slice kernel in triton artifact"))?;
        if !matches!(kernel.kind, KernelKind::SliceF32Rank4) {
            return Err(BackendError::execution(format!(
                "unexpected slice kernel kind: {:?}",
                kernel.kind
            )));
        }
        let (out_dims, in_strides, starts4) =
            slice_rank4_layout(&input.spec.shape, &out_spec.shape, starts)?;

        let out = allocate_output_tensor(driver, out_spec, output)?;
        let element_count = static_element_count(&out_spec.shape)?;
        if element_count == 0 {
            return Ok(out);
        }
        let loaded = self.load_kernel(driver, kernel)?;

        let mut in_ptr = input.buffer.device_ptr();
        let mut out_ptr = out.buffer.device_ptr();
        let mut n = u32::try_from(element_count)
            .map_err(|_| BackendError::execution("slice element count exceeds u32 range"))?;
        let mut od0 = out_dims[0];
        let mut od1 = out_dims[1];
        let mut od2 = out_dims[2];
        let mut od3 = out_dims[3];
        let mut is0 = in_strides[0];
        let mut is1 = in_strides[1];
        let mut is2 = in_strides[2];
        let mut is3 = in_strides[3];
        let mut st0 = starts4[0];
        let mut st1 = starts4[1];
        let mut st2 = starts4[2];
        let mut st3 = starts4[3];
        let mut opaque_ptr = 0u64;
        let mut params = [
            (&mut in_ptr as *mut u64).cast::<c_void>(),
            (&mut out_ptr as *mut u64).cast::<c_void>(),
            (&mut n as *mut u32).cast::<c_void>(),
            (&mut od0 as *mut i32).cast::<c_void>(),
            (&mut od1 as *mut i32).cast::<c_void>(),
            (&mut od2 as *mut i32).cast::<c_void>(),
            (&mut od3 as *mut i32).cast::<c_void>(),
            (&mut is0 as *mut i32).cast::<c_void>(),
            (&mut is1 as *mut i32).cast::<c_void>(),
            (&mut is2 as *mut i32).cast::<c_void>(),
            (&mut is3 as *mut i32).cast::<c_void>(),
            (&mut st0 as *mut i32).cast::<c_void>(),
            (&mut st1 as *mut i32).cast::<c_void>(),
            (&mut st2 as *mut i32).cast::<c_void>(),
            (&mut st3 as *mut i32).cast::<c_void>(),
            (&mut opaque_ptr as *mut u64).cast::<c_void>(),
        ];
        launch_1d(driver, &loaded, n, 256, &mut params)?;
        Ok(out)
    }

    pub(super) fn execute_dynamic_slice(
        &self,
        driver: &Arc<CudaDriver>,
        kernels: &HashMap<&str, &KernelSpec>,
        input: &TritonTensor,
        starts: &TritonTensor,
        spec: &gpt_rs::backend::spec::DynamicSliceSpec,
        out: OutputBinding<'_>,
    ) -> BackendResult<TritonTensor> {
        let out_spec = out.spec;
        let output = out.tensor;
        if starts.spec.dtype != DType::Si32 {
            return Err(BackendError::execution("dynamic_slice starts must be Si32"));
        }
        let rank = input.spec.shape.rank();
        if spec.sizes.len() != rank {
            return Err(BackendError::execution(
                "dynamic_slice sizes length must match input rank",
            ));
        }
        let starts_values = read_i32_tensor(starts)?;
        if starts_values.len() != rank {
            return Err(BackendError::execution(
                "dynamic_slice starts length must match input rank",
            ));
        }
        let mut static_starts = Vec::with_capacity(rank);
        for value in starts_values {
            if value < 0 {
                return Err(BackendError::execution(
                    "dynamic_slice starts must be non-negative",
                ));
            }
            static_starts.push(value as usize);
        }

        match out_spec.dtype {
            DType::F32 => {
                self.execute_slice_f32(driver, kernels, input, &static_starts, out_spec, output)
            }
            DType::Si32 => {
                // Rank-1 Si32 dynamic slice is required by decode attention query position path.
                let in_dims = static_dims_or_error(&input.spec.shape, |_| {
                    BackendError::execution(
                        "dynamic dimensions are not supported by triton runtime",
                    )
                })?;
                if in_dims.len() != 1 || static_starts.len() != 1 || spec.sizes.len() != 1 {
                    return Err(BackendError::execution(
                        "dynamic_slice Si32 path currently supports rank-1 only",
                    ));
                }
                let start = static_starts[0];
                let len = spec.sizes[0];
                if match start.checked_add(len) {
                    Some(end) => end > in_dims[0],
                    None => true,
                } {
                    return Err(BackendError::execution(
                        "dynamic_slice Si32 bounds check failed",
                    ));
                }
                let out = allocate_output_tensor(driver, out_spec, output)?;
                let offset_bytes = start
                    .checked_mul(4)
                    .ok_or_else(|| BackendError::execution("dynamic_slice Si32 offset overflow"))?;
                let src_ptr = input
                    .buffer
                    .device_ptr()
                    .checked_add(u64::try_from(offset_bytes).map_err(|_| {
                        BackendError::execution("dynamic_slice Si32 offset conversion overflow")
                    })?)
                    .ok_or_else(|| {
                        BackendError::execution("dynamic_slice Si32 pointer overflow")
                    })?;
                driver.copy_device_to_device(out.buffer.device_ptr(), src_ptr, len * 4)?;
                Ok(out)
            }
            _ => Err(BackendError::execution(format!(
                "dynamic_slice unsupported dtype {:?}",
                out_spec.dtype
            ))),
        }
    }

    pub(super) fn execute_transpose(
        &self,
        driver: &Arc<CudaDriver>,
        kernels: &HashMap<&str, &KernelSpec>,
        input: &TritonTensor,
        spec: &gpt_rs::backend::spec::TransposeSpec,
        out_spec: &TensorSpec,
        output: Option<TritonTensor>,
    ) -> BackendResult<TritonTensor> {
        if input.spec.dtype != DType::F32 || out_spec.dtype != DType::F32 {
            return Err(BackendError::execution(
                "transpose runtime currently supports F32 only",
            ));
        }
        let kernel = kernels.get(TRANSPOSE_KERNEL_ID).ok_or_else(|| {
            BackendError::execution("missing transpose kernel in triton artifact")
        })?;
        if !matches!(kernel.kind, KernelKind::TransposeF32Rank5) {
            return Err(BackendError::execution(format!(
                "unexpected transpose kernel kind: {:?}",
                kernel.kind
            )));
        }
        let (out_dims, in_strides) =
            transpose_rank5_layout(&input.spec.shape, &out_spec.shape, &spec.perm)?;
        let out = allocate_output_tensor(driver, out_spec, output)?;
        let element_count = static_element_count(&out_spec.shape)?;
        if element_count == 0 {
            return Ok(out);
        }
        let loaded = self.load_kernel(driver, kernel)?;

        let mut in_ptr = input.buffer.device_ptr();
        let mut out_ptr = out.buffer.device_ptr();
        let mut n = u32::try_from(element_count)
            .map_err(|_| BackendError::execution("transpose element count exceeds u32 range"))?;
        let mut od0 = out_dims[0];
        let mut od1 = out_dims[1];
        let mut od2 = out_dims[2];
        let mut od3 = out_dims[3];
        let mut od4 = out_dims[4];
        let mut is0 = in_strides[0];
        let mut is1 = in_strides[1];
        let mut is2 = in_strides[2];
        let mut is3 = in_strides[3];
        let mut is4 = in_strides[4];
        let mut opaque_ptr = 0u64;
        let mut params = [
            (&mut in_ptr as *mut u64).cast::<c_void>(),
            (&mut out_ptr as *mut u64).cast::<c_void>(),
            (&mut n as *mut u32).cast::<c_void>(),
            (&mut od0 as *mut i32).cast::<c_void>(),
            (&mut od1 as *mut i32).cast::<c_void>(),
            (&mut od2 as *mut i32).cast::<c_void>(),
            (&mut od3 as *mut i32).cast::<c_void>(),
            (&mut od4 as *mut i32).cast::<c_void>(),
            (&mut is0 as *mut i32).cast::<c_void>(),
            (&mut is1 as *mut i32).cast::<c_void>(),
            (&mut is2 as *mut i32).cast::<c_void>(),
            (&mut is3 as *mut i32).cast::<c_void>(),
            (&mut is4 as *mut i32).cast::<c_void>(),
            (&mut opaque_ptr as *mut u64).cast::<c_void>(),
        ];
        launch_1d(driver, &loaded, n, 256, &mut params)?;
        Ok(out)
    }

    pub(super) fn execute_concat(
        &self,
        driver: &Arc<CudaDriver>,
        kernels: &HashMap<&str, &KernelSpec>,
        lhs: &TritonTensor,
        rhs: &TritonTensor,
        spec: &gpt_rs::backend::spec::ConcatSpec,
        out: OutputBinding<'_>,
    ) -> BackendResult<TritonTensor> {
        let out_spec = out.spec;
        let output = out.tensor;
        if lhs.spec.dtype != DType::F32
            || rhs.spec.dtype != DType::F32
            || out_spec.dtype != DType::F32
        {
            return Err(BackendError::execution(
                "concat runtime currently supports F32 only",
            ));
        }
        let kernel = kernels
            .get(CONCAT_KERNEL_ID)
            .ok_or_else(|| BackendError::execution("missing concat kernel in triton artifact"))?;
        if !matches!(kernel.kind, KernelKind::ConcatF32Rank4) {
            return Err(BackendError::execution(format!(
                "unexpected concat kernel kind: {:?}",
                kernel.kind
            )));
        }
        let (out_dims, lhs_strides, rhs_strides, axis, split) =
            concat_rank4_layout(&lhs.spec.shape, &rhs.spec.shape, &out_spec.shape, spec.axis)?;
        let out = allocate_output_tensor(driver, out_spec, output)?;
        let element_count = static_element_count(&out_spec.shape)?;
        if element_count == 0 {
            return Ok(out);
        }
        let loaded = self.load_kernel(driver, kernel)?;

        let mut lhs_ptr = lhs.buffer.device_ptr();
        let mut rhs_ptr = rhs.buffer.device_ptr();
        let mut out_ptr = out.buffer.device_ptr();
        let mut n = u32::try_from(element_count)
            .map_err(|_| BackendError::execution("concat element count exceeds u32 range"))?;
        let mut od0 = out_dims[0];
        let mut od1 = out_dims[1];
        let mut od2 = out_dims[2];
        let mut od3 = out_dims[3];
        let mut axis_i32 = axis;
        let mut split_i32 = split;
        let mut ls0 = lhs_strides[0];
        let mut ls1 = lhs_strides[1];
        let mut ls2 = lhs_strides[2];
        let mut ls3 = lhs_strides[3];
        let mut rs0 = rhs_strides[0];
        let mut rs1 = rhs_strides[1];
        let mut rs2 = rhs_strides[2];
        let mut rs3 = rhs_strides[3];
        let mut opaque_ptr = 0u64;
        let mut params = [
            (&mut lhs_ptr as *mut u64).cast::<c_void>(),
            (&mut rhs_ptr as *mut u64).cast::<c_void>(),
            (&mut out_ptr as *mut u64).cast::<c_void>(),
            (&mut n as *mut u32).cast::<c_void>(),
            (&mut od0 as *mut i32).cast::<c_void>(),
            (&mut od1 as *mut i32).cast::<c_void>(),
            (&mut od2 as *mut i32).cast::<c_void>(),
            (&mut od3 as *mut i32).cast::<c_void>(),
            (&mut axis_i32 as *mut i32).cast::<c_void>(),
            (&mut split_i32 as *mut i32).cast::<c_void>(),
            (&mut ls0 as *mut i32).cast::<c_void>(),
            (&mut ls1 as *mut i32).cast::<c_void>(),
            (&mut ls2 as *mut i32).cast::<c_void>(),
            (&mut ls3 as *mut i32).cast::<c_void>(),
            (&mut rs0 as *mut i32).cast::<c_void>(),
            (&mut rs1 as *mut i32).cast::<c_void>(),
            (&mut rs2 as *mut i32).cast::<c_void>(),
            (&mut rs3 as *mut i32).cast::<c_void>(),
            (&mut opaque_ptr as *mut u64).cast::<c_void>(),
        ];
        launch_1d(driver, &loaded, n, 256, &mut params)?;
        Ok(out)
    }

    pub(super) fn execute_dynamic_update_slice(
        &self,
        driver: &Arc<CudaDriver>,
        kernel: &KernelSpec,
        base: &TritonTensor,
        update: &TritonTensor,
        starts: &TritonTensor,
        out: OutputBinding<'_>,
    ) -> BackendResult<TritonTensor> {
        let out_spec = out.spec;
        let output = out.tensor;
        if base.spec.dtype != DType::F32
            || update.spec.dtype != DType::F32
            || starts.spec.dtype != DType::Si32
            || out_spec.dtype != DType::F32
        {
            return Err(BackendError::execution(
                "dynamic_update_slice runtime currently supports F32 base/update with Si32 starts",
            ));
        }
        if !matches!(kernel.kind, KernelKind::DynamicUpdateSliceF32Rank4) {
            return Err(BackendError::execution(format!(
                "unexpected dynamic_update_slice kernel kind: {:?}",
                kernel.kind
            )));
        }
        let base_dims = static_dims_or_error(&base.spec.shape, |_| {
            BackendError::execution("dynamic dimensions are not supported by triton runtime")
        })?;
        let update_dims = static_dims_or_error(&update.spec.shape, |_| {
            BackendError::execution("dynamic dimensions are not supported by triton runtime")
        })?;
        if base_dims.len() > 4 {
            return Err(BackendError::execution(
                "dynamic_update_slice runtime supports rank <= 4 only",
            ));
        }
        let starts_values = read_i32_tensor(starts)?;
        if starts_values.len() != base_dims.len() {
            return Err(BackendError::execution(
                "dynamic_update_slice starts length mismatch",
            ));
        }
        let mut starts_usize = Vec::with_capacity(starts_values.len());
        for value in starts_values {
            if value < 0 {
                return Err(BackendError::execution(
                    "dynamic_update_slice starts must be non-negative",
                ));
            }
            starts_usize.push(value as usize);
        }
        for axis in 0..base_dims.len() {
            let end = starts_usize[axis]
                .checked_add(update_dims[axis])
                .ok_or_else(|| BackendError::execution("dynamic_update_slice bounds overflow"))?;
            if end > base_dims[axis] {
                return Err(BackendError::execution(
                    "dynamic_update_slice update exceeds base bounds",
                ));
            }
        }

        let out = allocate_output_tensor(driver, out_spec, output)?;
        driver.copy_device_to_device(
            out.buffer.device_ptr(),
            base.buffer.device_ptr(),
            byte_len(out_spec)?,
        )?;
        let update_elems = static_element_count(&update.spec.shape)?;
        if update_elems == 0 {
            return Ok(out);
        }
        let loaded = self.load_kernel(driver, kernel)?;
        let (update_dims4, out_strides4, starts4) =
            dynamic_update_rank4_layout(&update_dims, &base_dims, &starts_usize)?;

        let mut update_ptr = update.buffer.device_ptr();
        let mut out_ptr = out.buffer.device_ptr();
        let mut n = u32::try_from(update_elems).map_err(|_| {
            BackendError::execution("dynamic_update_slice update elements exceeds u32 range")
        })?;
        let mut ud0 = update_dims4[0];
        let mut ud1 = update_dims4[1];
        let mut ud2 = update_dims4[2];
        let mut ud3 = update_dims4[3];
        let mut os0 = out_strides4[0];
        let mut os1 = out_strides4[1];
        let mut os2 = out_strides4[2];
        let mut os3 = out_strides4[3];
        let mut st0 = starts4[0];
        let mut st1 = starts4[1];
        let mut st2 = starts4[2];
        let mut st3 = starts4[3];
        let mut opaque_ptr = 0u64;
        let mut params = [
            (&mut update_ptr as *mut u64).cast::<c_void>(),
            (&mut out_ptr as *mut u64).cast::<c_void>(),
            (&mut n as *mut u32).cast::<c_void>(),
            (&mut ud0 as *mut i32).cast::<c_void>(),
            (&mut ud1 as *mut i32).cast::<c_void>(),
            (&mut ud2 as *mut i32).cast::<c_void>(),
            (&mut ud3 as *mut i32).cast::<c_void>(),
            (&mut os0 as *mut i32).cast::<c_void>(),
            (&mut os1 as *mut i32).cast::<c_void>(),
            (&mut os2 as *mut i32).cast::<c_void>(),
            (&mut os3 as *mut i32).cast::<c_void>(),
            (&mut st0 as *mut i32).cast::<c_void>(),
            (&mut st1 as *mut i32).cast::<c_void>(),
            (&mut st2 as *mut i32).cast::<c_void>(),
            (&mut st3 as *mut i32).cast::<c_void>(),
            (&mut opaque_ptr as *mut u64).cast::<c_void>(),
        ];
        launch_1d(driver, &loaded, n, 256, &mut params)?;
        Ok(out)
    }
}
