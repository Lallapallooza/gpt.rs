use super::*;

impl TritonExecutor {
    pub(super) fn execute_elementwise_binary(
        &self,
        driver: &Arc<CudaDriver>,
        kernel: &KernelSpec,
        op: ElementwiseBinaryOp,
        lhs: &TritonTensor,
        rhs: &TritonTensor,
        out: OutputBinding<'_>,
    ) -> BackendResult<TritonTensor> {
        let spec = out.spec;
        let output = out.tensor;
        if lhs.spec != *spec || rhs.spec != *spec {
            return Err(BackendError::execution(
                "elementwise operands/spec mismatch in triton runtime",
            ));
        }
        if spec.dtype != DType::F32 {
            return Err(BackendError::execution(format!(
                "elementwise runtime currently supports F32 only, got {:?}",
                spec.dtype
            )));
        }
        if !matches!(kernel.kind, KernelKind::ElementwiseBinaryF32) {
            return Err(BackendError::execution(format!(
                "unexpected kernel kind for elementwise binary: {:?}",
                kernel.kind
            )));
        }

        let element_count = static_element_count(&spec.shape)?;
        let out = allocate_output_tensor(driver, spec, output)?;

        let loaded = self.load_kernel(driver, kernel)?;
        let opcode = binary_opcode(op);
        let count_u32 = u32::try_from(element_count).map_err(|_| {
            BackendError::execution("elementwise tensor too large for u32 launch size")
        })?;

        let mut lhs_ptr = lhs.buffer.device_ptr();
        let mut rhs_ptr = rhs.buffer.device_ptr();
        let mut out_ptr = out.buffer.device_ptr();
        let mut n = count_u32;
        let mut op_u32 = opcode;
        // Python Triton-compiled kernels currently expose one trailing opaque
        // pointer parameter; built-in kernels ignore this extra argument.
        let mut opaque_ptr = 0u64;
        let mut params = [
            (&mut lhs_ptr as *mut u64).cast::<c_void>(),
            (&mut rhs_ptr as *mut u64).cast::<c_void>(),
            (&mut out_ptr as *mut u64).cast::<c_void>(),
            (&mut n as *mut u32).cast::<c_void>(),
            (&mut op_u32 as *mut u32).cast::<c_void>(),
            (&mut opaque_ptr as *mut u64).cast::<c_void>(),
        ];
        launch_1d(driver, &loaded, count_u32, 256, &mut params)?;

        Ok(out)
    }

    pub(super) fn execute_elementwise_unary(
        &self,
        driver: &Arc<CudaDriver>,
        kernel: &KernelSpec,
        op: ElementwiseUnaryOp,
        input: &TritonTensor,
        spec: &TensorSpec,
        output: Option<TritonTensor>,
    ) -> BackendResult<TritonTensor> {
        if input.spec != *spec {
            return Err(BackendError::execution(
                "elementwise unary operand/spec mismatch in triton runtime",
            ));
        }
        if spec.dtype != DType::F32 {
            return Err(BackendError::execution(format!(
                "elementwise unary runtime currently supports F32 only, got {:?}",
                spec.dtype
            )));
        }
        if !matches!(kernel.kind, KernelKind::ElementwiseUnaryF32) {
            return Err(BackendError::execution(format!(
                "unexpected kernel kind for elementwise unary: {:?}",
                kernel.kind
            )));
        }

        let element_count = static_element_count(&spec.shape)?;
        let out = allocate_output_tensor(driver, spec, output)?;

        let loaded = self.load_kernel(driver, kernel)?;
        let opcode = unary_opcode(op)?;
        let count_u32 = u32::try_from(element_count).map_err(|_| {
            BackendError::execution("elementwise unary tensor too large for u32 launch size")
        })?;

        let mut in_ptr = input.buffer.device_ptr();
        let mut out_ptr = out.buffer.device_ptr();
        let mut n = count_u32;
        let mut op_u32 = opcode;
        // Python Triton-compiled kernels currently expose one trailing opaque
        // pointer parameter; built-in kernels ignore this extra argument.
        let mut opaque_ptr = 0u64;
        let mut params = [
            (&mut in_ptr as *mut u64).cast::<c_void>(),
            (&mut out_ptr as *mut u64).cast::<c_void>(),
            (&mut n as *mut u32).cast::<c_void>(),
            (&mut op_u32 as *mut u32).cast::<c_void>(),
            (&mut opaque_ptr as *mut u64).cast::<c_void>(),
        ];
        launch_1d(driver, &loaded, count_u32, 256, &mut params)?;

        Ok(out)
    }

    pub(super) fn execute_iota(
        &self,
        driver: &Arc<CudaDriver>,
        kernel: &KernelSpec,
        spec: &gpt_rs::backend::spec::IotaSpec,
        out_spec: &TensorSpec,
        output: Option<TritonTensor>,
    ) -> BackendResult<TritonTensor> {
        if out_spec.dtype != DType::Si32 {
            return Err(BackendError::execution(
                "iota runtime currently supports Si32 output only",
            ));
        }
        if !matches!(kernel.kind, KernelKind::IotaSi32Rank4) {
            return Err(BackendError::execution(format!(
                "unexpected iota kernel kind: {:?}",
                kernel.kind
            )));
        }
        let out_dims = static_dims_or_error(&out_spec.shape, |_| {
            BackendError::execution("dynamic dimensions are not supported by triton runtime")
        })?;
        if spec.axis >= out_dims.len() {
            return Err(BackendError::execution("iota axis out of bounds"));
        }
        if out_dims.len() > 4 {
            return Err(BackendError::execution(
                "iota runtime supports rank <= 4 only",
            ));
        }
        let out = allocate_output_tensor(driver, out_spec, output)?;
        let element_count = static_element_count(&out_spec.shape)?;
        if element_count == 0 {
            return Ok(out);
        }
        let loaded = self.load_kernel(driver, kernel)?;
        let (od0, od1, od2, od3) = align_dims4(&out_dims)?;
        let aligned_axis = (4usize - out_dims.len())
            .checked_add(spec.axis)
            .ok_or_else(|| BackendError::execution("iota axis alignment overflow"))?;
        let axis = i32::try_from(aligned_axis)
            .map_err(|_| BackendError::execution("iota axis exceeds i32 range"))?;

        let mut out_ptr = out.buffer.device_ptr();
        let mut n = u32::try_from(element_count)
            .map_err(|_| BackendError::execution("iota element count exceeds u32 range"))?;
        let mut od0 = od0;
        let mut od1 = od1;
        let mut od2 = od2;
        let mut od3 = od3;
        let mut axis_i32 = axis;
        let mut opaque_ptr = 0u64;
        let mut params = [
            (&mut out_ptr as *mut u64).cast::<c_void>(),
            (&mut n as *mut u32).cast::<c_void>(),
            (&mut od0 as *mut i32).cast::<c_void>(),
            (&mut od1 as *mut i32).cast::<c_void>(),
            (&mut od2 as *mut i32).cast::<c_void>(),
            (&mut od3 as *mut i32).cast::<c_void>(),
            (&mut axis_i32 as *mut i32).cast::<c_void>(),
            (&mut opaque_ptr as *mut u64).cast::<c_void>(),
        ];
        launch_1d(driver, &loaded, n, 256, &mut params)?;
        Ok(out)
    }

    pub(super) fn execute_compare(
        &self,
        driver: &Arc<CudaDriver>,
        kernel: &KernelSpec,
        op: ComparisonOp,
        lhs: &TritonTensor,
        rhs: &TritonTensor,
        out: OutputBinding<'_>,
    ) -> BackendResult<TritonTensor> {
        let out_spec = out.spec;
        let output = out.tensor;
        if lhs.spec != rhs.spec {
            return Err(BackendError::execution("compare operand spec mismatch"));
        }
        if lhs.spec.dtype != DType::Si32 || out_spec.dtype != DType::I1 {
            return Err(BackendError::execution(
                "compare runtime currently supports Si32 -> I1 only",
            ));
        }
        if !matches!(kernel.kind, KernelKind::CompareSi32I1) {
            return Err(BackendError::execution(format!(
                "unexpected compare kernel kind: {:?}",
                kernel.kind
            )));
        }
        let out = allocate_output_tensor(driver, out_spec, output)?;
        let element_count = static_element_count(&out_spec.shape)?;
        if element_count == 0 {
            return Ok(out);
        }
        let loaded = self.load_kernel(driver, kernel)?;
        let opcode = compare_opcode(op);
        let mut lhs_ptr = lhs.buffer.device_ptr();
        let mut rhs_ptr = rhs.buffer.device_ptr();
        let mut out_ptr = out.buffer.device_ptr();
        let mut n = u32::try_from(element_count)
            .map_err(|_| BackendError::execution("compare element count exceeds u32 range"))?;
        let mut op_u32 = opcode;
        let mut opaque_ptr = 0u64;
        let mut params = [
            (&mut lhs_ptr as *mut u64).cast::<c_void>(),
            (&mut rhs_ptr as *mut u64).cast::<c_void>(),
            (&mut out_ptr as *mut u64).cast::<c_void>(),
            (&mut n as *mut u32).cast::<c_void>(),
            (&mut op_u32 as *mut u32).cast::<c_void>(),
            (&mut opaque_ptr as *mut u64).cast::<c_void>(),
        ];
        launch_1d(driver, &loaded, n, 256, &mut params)?;
        Ok(out)
    }

    pub(super) fn execute_select(
        &self,
        driver: &Arc<CudaDriver>,
        kernel: &KernelSpec,
        predicate: &TritonTensor,
        when_true: &TritonTensor,
        when_false: &TritonTensor,
        out: OutputBinding<'_>,
    ) -> BackendResult<TritonTensor> {
        let out_spec = out.spec;
        let output = out.tensor;
        if predicate.spec.dtype != DType::I1 {
            return Err(BackendError::execution("select predicate must be I1"));
        }
        if when_true.spec != when_false.spec || when_true.spec != *out_spec {
            return Err(BackendError::execution(
                "select branch/output spec mismatch",
            ));
        }
        if out_spec.dtype != DType::F32 {
            return Err(BackendError::execution(
                "select runtime currently supports F32 branches only",
            ));
        }
        if !matches!(kernel.kind, KernelKind::SelectI1F32) {
            return Err(BackendError::execution(format!(
                "unexpected select kernel kind: {:?}",
                kernel.kind
            )));
        }
        let out = allocate_output_tensor(driver, out_spec, output)?;
        let element_count = static_element_count(&out_spec.shape)?;
        if element_count == 0 {
            return Ok(out);
        }
        let loaded = self.load_kernel(driver, kernel)?;
        let mut pred_ptr = predicate.buffer.device_ptr();
        let mut true_ptr = when_true.buffer.device_ptr();
        let mut false_ptr = when_false.buffer.device_ptr();
        let mut out_ptr = out.buffer.device_ptr();
        let mut n = u32::try_from(element_count)
            .map_err(|_| BackendError::execution("select element count exceeds u32 range"))?;
        let mut opaque_ptr = 0u64;
        let mut params = [
            (&mut pred_ptr as *mut u64).cast::<c_void>(),
            (&mut true_ptr as *mut u64).cast::<c_void>(),
            (&mut false_ptr as *mut u64).cast::<c_void>(),
            (&mut out_ptr as *mut u64).cast::<c_void>(),
            (&mut n as *mut u32).cast::<c_void>(),
            (&mut opaque_ptr as *mut u64).cast::<c_void>(),
        ];
        launch_1d(driver, &loaded, n, 256, &mut params)?;
        Ok(out)
    }

    pub(super) fn execute_take(
        &self,
        driver: &Arc<CudaDriver>,
        kernel: &KernelSpec,
        params: &TritonTensor,
        indices: &TritonTensor,
        out_spec: &TensorSpec,
        output: Option<TritonTensor>,
    ) -> BackendResult<TritonTensor> {
        if params.spec.dtype != DType::F32
            || indices.spec.dtype != DType::Si32
            || out_spec.dtype != DType::F32
        {
            return Err(BackendError::execution(
                "take runtime expects F32 params, Si32 indices, F32 output",
            ));
        }
        if !matches!(kernel.kind, KernelKind::TakeF32I32) {
            return Err(BackendError::execution(format!(
                "unexpected take kernel kind: {:?}",
                kernel.kind
            )));
        }
        let params_dims = static_dims_or_error(&params.spec.shape, |_| {
            BackendError::execution("dynamic dimensions are not supported by triton runtime")
        })?;
        if params_dims.len() != 2 {
            return Err(BackendError::execution(
                "take runtime currently supports rank-2 params only",
            ));
        }
        let index_count = static_element_count(&indices.spec.shape)?;
        let embed_dim = params_dims[1];
        let vocab = params_dims[0];
        let expected_out = index_count
            .checked_mul(embed_dim)
            .ok_or_else(|| BackendError::execution("take output element count overflow"))?;
        let out_count = static_element_count(&out_spec.shape)?;
        if expected_out != out_count {
            return Err(BackendError::execution(format!(
                "take output shape mismatch: expected {} elements, got {}",
                expected_out, out_count
            )));
        }

        let out = allocate_output_tensor(driver, out_spec, output)?;
        if out_count == 0 {
            return Ok(out);
        }
        let loaded = self.load_kernel(driver, kernel)?;
        let mut weight_ptr = params.buffer.device_ptr();
        let mut indices_ptr = indices.buffer.device_ptr();
        let mut out_ptr = out.buffer.device_ptr();
        let mut n = u32::try_from(out_count)
            .map_err(|_| BackendError::execution("take element count exceeds u32 range"))?;
        let mut embed = i32::try_from(embed_dim)
            .map_err(|_| BackendError::execution("take embed_dim exceeds i32 range"))?;
        let mut vocab = i32::try_from(vocab)
            .map_err(|_| BackendError::execution("take vocab exceeds i32 range"))?;
        let mut opaque_ptr = 0u64;
        let mut params_arr = [
            (&mut weight_ptr as *mut u64).cast::<c_void>(),
            (&mut indices_ptr as *mut u64).cast::<c_void>(),
            (&mut out_ptr as *mut u64).cast::<c_void>(),
            (&mut n as *mut u32).cast::<c_void>(),
            (&mut embed as *mut i32).cast::<c_void>(),
            (&mut vocab as *mut i32).cast::<c_void>(),
            (&mut opaque_ptr as *mut u64).cast::<c_void>(),
        ];
        launch_1d(driver, &loaded, n, 256, &mut params_arr)?;
        Ok(out)
    }
}
