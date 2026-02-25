use super::*;

impl TritonExecutor {
    pub(super) fn execute_extract_patches(
        &self,
        driver: &Arc<CudaDriver>,
        kernel: &KernelSpec,
        input: &TritonTensor,
        spec: &gpt_rs::backend::spec::ExtractPatchesSpec,
        out_spec: &TensorSpec,
        output: Option<TritonTensor>,
    ) -> BackendResult<TritonTensor> {
        if !matches!(kernel.kind, KernelKind::ExtractPatchesNhwcF32) {
            return Err(BackendError::execution(format!(
                "unexpected extract_patches kernel kind: {:?}",
                kernel.kind
            )));
        }
        if input.spec.dtype != DType::F32 || out_spec.dtype != DType::F32 {
            return Err(BackendError::execution(
                "extract_patches runtime currently supports F32 only",
            ));
        }
        let in_dims = static_dims_or_error(&input.spec.shape, |_| {
            BackendError::execution("dynamic dimensions are not supported by triton runtime")
        })?;
        let out_dims = static_dims_or_error(&out_spec.shape, |_| {
            BackendError::execution("dynamic dimensions are not supported by triton runtime")
        })?;
        if in_dims.len() != 4 || out_dims.len() != 4 {
            return Err(BackendError::execution(
                "extract_patches runtime expects rank-4 NHWC input/output",
            ));
        }
        if spec.window.len() != 2
            || spec.strides.len() != 2
            || spec.dilation.len() != 2
            || spec.padding.len() != 2
        {
            return Err(BackendError::execution(
                "extract_patches runtime currently supports 2D window only",
            ));
        }
        let out = allocate_output_tensor(driver, out_spec, output)?;
        let elem_count = static_element_count(&out_spec.shape)?;
        if elem_count == 0 {
            return Ok(out);
        }
        let patch_dim = out_dims[3];
        let loaded = self.load_kernel(driver, kernel)?;

        let mut in_ptr = input.buffer.device_ptr();
        let mut out_ptr = out.buffer.device_ptr();
        let mut n = u32::try_from(elem_count).map_err(|_| {
            BackendError::execution("extract_patches element count exceeds u32 range")
        })?;
        let mut in_h = i32::try_from(in_dims[1])
            .map_err(|_| BackendError::execution("extract_patches in_h exceeds i32 range"))?;
        let mut in_w = i32::try_from(in_dims[2])
            .map_err(|_| BackendError::execution("extract_patches in_w exceeds i32 range"))?;
        let mut in_c = i32::try_from(in_dims[3])
            .map_err(|_| BackendError::execution("extract_patches in_c exceeds i32 range"))?;
        let mut out_h = i32::try_from(out_dims[1])
            .map_err(|_| BackendError::execution("extract_patches out_h exceeds i32 range"))?;
        let mut out_w = i32::try_from(out_dims[2])
            .map_err(|_| BackendError::execution("extract_patches out_w exceeds i32 range"))?;
        let mut k_h = i32::try_from(spec.window[0])
            .map_err(|_| BackendError::execution("extract_patches k_h exceeds i32 range"))?;
        let mut k_w = i32::try_from(spec.window[1])
            .map_err(|_| BackendError::execution("extract_patches k_w exceeds i32 range"))?;
        let mut s_h = i32::try_from(spec.strides[0])
            .map_err(|_| BackendError::execution("extract_patches s_h exceeds i32 range"))?;
        let mut s_w = i32::try_from(spec.strides[1])
            .map_err(|_| BackendError::execution("extract_patches s_w exceeds i32 range"))?;
        let mut d_h = i32::try_from(spec.dilation[0])
            .map_err(|_| BackendError::execution("extract_patches d_h exceeds i32 range"))?;
        let mut d_w = i32::try_from(spec.dilation[1])
            .map_err(|_| BackendError::execution("extract_patches d_w exceeds i32 range"))?;
        let mut pad_top = i32::try_from(spec.padding[0].0)
            .map_err(|_| BackendError::execution("extract_patches pad_top exceeds i32 range"))?;
        let mut pad_left = i32::try_from(spec.padding[1].0)
            .map_err(|_| BackendError::execution("extract_patches pad_left exceeds i32 range"))?;
        let mut patch_dim_i32 = i32::try_from(patch_dim)
            .map_err(|_| BackendError::execution("extract_patches patch_dim exceeds i32 range"))?;
        let mut opaque_ptr = 0u64;
        let mut params = [
            (&mut in_ptr as *mut u64).cast::<c_void>(),
            (&mut out_ptr as *mut u64).cast::<c_void>(),
            (&mut n as *mut u32).cast::<c_void>(),
            (&mut in_h as *mut i32).cast::<c_void>(),
            (&mut in_w as *mut i32).cast::<c_void>(),
            (&mut in_c as *mut i32).cast::<c_void>(),
            (&mut out_h as *mut i32).cast::<c_void>(),
            (&mut out_w as *mut i32).cast::<c_void>(),
            (&mut k_h as *mut i32).cast::<c_void>(),
            (&mut k_w as *mut i32).cast::<c_void>(),
            (&mut s_h as *mut i32).cast::<c_void>(),
            (&mut s_w as *mut i32).cast::<c_void>(),
            (&mut d_h as *mut i32).cast::<c_void>(),
            (&mut d_w as *mut i32).cast::<c_void>(),
            (&mut pad_top as *mut i32).cast::<c_void>(),
            (&mut pad_left as *mut i32).cast::<c_void>(),
            (&mut patch_dim_i32 as *mut i32).cast::<c_void>(),
            (&mut opaque_ptr as *mut u64).cast::<c_void>(),
        ];
        launch_1d(driver, &loaded, n, 256, &mut params)?;
        Ok(out)
    }

    pub(super) fn execute_reduce_window(
        &self,
        driver: &Arc<CudaDriver>,
        kernel: &KernelSpec,
        input: &TritonTensor,
        spec: &gpt_rs::backend::spec::ReduceWindowSpec,
        out_spec: &TensorSpec,
        output: Option<TritonTensor>,
    ) -> BackendResult<TritonTensor> {
        if !matches!(kernel.kind, KernelKind::ReduceWindowMaxNhwcF32) {
            return Err(BackendError::execution(format!(
                "unexpected reduce_window kernel kind: {:?}",
                kernel.kind
            )));
        }
        if spec.reduce != ReduceKind::Max {
            return Err(BackendError::execution(
                "reduce_window runtime currently supports max only",
            ));
        }
        if input.spec.dtype != DType::F32 || out_spec.dtype != DType::F32 {
            return Err(BackendError::execution(
                "reduce_window runtime currently supports F32 only",
            ));
        }
        let in_dims = static_dims_or_error(&input.spec.shape, |_| {
            BackendError::execution("dynamic dimensions are not supported by triton runtime")
        })?;
        let out_dims = static_dims_or_error(&out_spec.shape, |_| {
            BackendError::execution("dynamic dimensions are not supported by triton runtime")
        })?;
        if in_dims.len() != 4 || out_dims.len() != 4 {
            return Err(BackendError::execution(
                "reduce_window runtime expects rank-4 NHWC input/output",
            ));
        }
        if spec.window_dims.len() != 4
            || spec.strides.len() != 4
            || spec.padding.len() != 4
            || spec.base_dilation.len() != 4
            || spec.window_dilation.len() != 4
        {
            return Err(BackendError::execution(
                "reduce_window runtime currently supports rank-4 specs only",
            ));
        }
        if spec.window_dims[0] != 1 || spec.window_dims[3] != 1 {
            return Err(BackendError::execution(
                "reduce_window runtime expects NHWC window with N/C window = 1",
            ));
        }
        if spec.base_dilation != vec![1, 1, 1, 1] {
            return Err(BackendError::execution(
                "reduce_window runtime currently supports base_dilation=[1,1,1,1] only",
            ));
        }

        let out = allocate_output_tensor(driver, out_spec, output)?;
        let elem_count = static_element_count(&out_spec.shape)?;
        if elem_count == 0 {
            return Ok(out);
        }
        let loaded = self.load_kernel(driver, kernel)?;

        let mut in_ptr = input.buffer.device_ptr();
        let mut out_ptr = out.buffer.device_ptr();
        let mut n = u32::try_from(elem_count).map_err(|_| {
            BackendError::execution("reduce_window element count exceeds u32 range")
        })?;
        let mut in_h = i32::try_from(in_dims[1])
            .map_err(|_| BackendError::execution("reduce_window in_h exceeds i32 range"))?;
        let mut in_w = i32::try_from(in_dims[2])
            .map_err(|_| BackendError::execution("reduce_window in_w exceeds i32 range"))?;
        let mut in_c = i32::try_from(in_dims[3])
            .map_err(|_| BackendError::execution("reduce_window in_c exceeds i32 range"))?;
        let mut out_h = i32::try_from(out_dims[1])
            .map_err(|_| BackendError::execution("reduce_window out_h exceeds i32 range"))?;
        let mut out_w = i32::try_from(out_dims[2])
            .map_err(|_| BackendError::execution("reduce_window out_w exceeds i32 range"))?;
        let mut k_h = i32::try_from(spec.window_dims[1])
            .map_err(|_| BackendError::execution("reduce_window k_h exceeds i32 range"))?;
        let mut k_w = i32::try_from(spec.window_dims[2])
            .map_err(|_| BackendError::execution("reduce_window k_w exceeds i32 range"))?;
        let mut s_h = i32::try_from(spec.strides[1])
            .map_err(|_| BackendError::execution("reduce_window s_h exceeds i32 range"))?;
        let mut s_w = i32::try_from(spec.strides[2])
            .map_err(|_| BackendError::execution("reduce_window s_w exceeds i32 range"))?;
        let mut d_h = i32::try_from(spec.window_dilation[1])
            .map_err(|_| BackendError::execution("reduce_window d_h exceeds i32 range"))?;
        let mut d_w = i32::try_from(spec.window_dilation[2])
            .map_err(|_| BackendError::execution("reduce_window d_w exceeds i32 range"))?;
        let mut pad_top = i32::try_from(spec.padding[1].0)
            .map_err(|_| BackendError::execution("reduce_window pad_top exceeds i32 range"))?;
        let mut pad_left = i32::try_from(spec.padding[2].0)
            .map_err(|_| BackendError::execution("reduce_window pad_left exceeds i32 range"))?;
        let mut opaque_ptr = 0u64;
        let mut params = [
            (&mut in_ptr as *mut u64).cast::<c_void>(),
            (&mut out_ptr as *mut u64).cast::<c_void>(),
            (&mut n as *mut u32).cast::<c_void>(),
            (&mut in_h as *mut i32).cast::<c_void>(),
            (&mut in_w as *mut i32).cast::<c_void>(),
            (&mut in_c as *mut i32).cast::<c_void>(),
            (&mut out_h as *mut i32).cast::<c_void>(),
            (&mut out_w as *mut i32).cast::<c_void>(),
            (&mut k_h as *mut i32).cast::<c_void>(),
            (&mut k_w as *mut i32).cast::<c_void>(),
            (&mut s_h as *mut i32).cast::<c_void>(),
            (&mut s_w as *mut i32).cast::<c_void>(),
            (&mut d_h as *mut i32).cast::<c_void>(),
            (&mut d_w as *mut i32).cast::<c_void>(),
            (&mut pad_top as *mut i32).cast::<c_void>(),
            (&mut pad_left as *mut i32).cast::<c_void>(),
            (&mut opaque_ptr as *mut u64).cast::<c_void>(),
        ];
        launch_1d(driver, &loaded, n, 128, &mut params)?;
        Ok(out)
    }
}
