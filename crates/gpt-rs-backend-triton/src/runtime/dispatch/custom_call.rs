use super::*;

impl TritonExecutor {
    pub(super) fn execute_fused_elementwise_custom_call(
        &self,
        driver: &Arc<CudaDriver>,
        values: &DenseValueStore,
        instruction: &gpt_rs::backend::spec::Instruction,
        spec: &CustomCallSpec,
        out_spec: &TensorSpec,
        output: Option<TritonTensor>,
    ) -> BackendResult<TritonTensor> {
        let plan = fused_elementwise::FusedElementwisePlan::parse(spec)?;
        let input_tensors = instruction
            .operands
            .iter()
            .map(|operand| self.resolve_operand_tensor(driver, values, Some(operand)))
            .collect::<BackendResult<Vec<_>>>()?;
        self.execute_fused_elementwise_plan(
            driver,
            &plan,
            input_tensors.as_slice(),
            out_spec,
            output,
        )
    }

    pub(super) fn execute_fused_elementwise_plan(
        &self,
        driver: &Arc<CudaDriver>,
        plan: &fused_elementwise::FusedElementwisePlan,
        input_tensors: &[TritonTensor],
        out_spec: &TensorSpec,
        output: Option<TritonTensor>,
    ) -> BackendResult<TritonTensor> {
        let input_specs = input_tensors
            .iter()
            .map(|tensor| tensor.spec.clone())
            .collect::<Vec<_>>();
        let kernel_key = plan.cache_fingerprint(out_spec, input_specs.as_slice())?;
        let kernel = {
            let mut cache = lock_named(&self.fused_kernel_specs, "fused kernel-spec cache")?;
            if let Some(cached) = cache.get(&kernel_key).cloned() {
                profiling::cache_event("triton_backend.fused_kernel_spec_hit");
                cached
            } else {
                profiling::cache_event("triton_backend.fused_kernel_spec_miss");
                let built = plan.build_kernel_spec(out_spec, input_specs.as_slice())?;
                cache.insert(kernel_key, built.clone());
                built
            }
        };
        let out = allocate_output_tensor(driver, out_spec, output)?;
        let loaded = self.load_kernel(driver, &kernel)?;
        let n = static_element_count(&out_spec.shape)?;
        if n == 0 {
            return Ok(out);
        }
        let n_u32 = u32::try_from(n).map_err(|_| {
            BackendError::execution("fused elementwise element count exceeds u32 range")
        })?;
        let mut params: Vec<*mut c_void> = Vec::with_capacity(input_tensors.len() + 3);
        let mut ptr_args: Vec<u64> = input_tensors
            .iter()
            .map(|tensor| tensor.buffer.device_ptr())
            .collect::<Vec<_>>();
        for ptr in &mut ptr_args {
            params.push((ptr as *mut u64).cast::<c_void>());
        }
        let mut out_ptr = out.buffer.device_ptr();
        params.push((&mut out_ptr as *mut u64).cast::<c_void>());
        let mut n_kernel = n_u32;
        params.push((&mut n_kernel as *mut u32).cast::<c_void>());
        let mut opaque_ptr = 0u64;
        params.push((&mut opaque_ptr as *mut u64).cast::<c_void>());
        launch_1d(driver, &loaded, n_u32, 256, params.as_mut_slice())?;
        Ok(out)
    }

    pub(super) fn execute_fused_dot_bias_custom_call(
        &self,
        driver: &Arc<CudaDriver>,
        kernels: &HashMap<&str, &KernelSpec>,
        values: &DenseValueStore,
        instruction: &gpt_rs::backend::spec::Instruction,
        spec: &CustomCallSpec,
        out: OutputBinding<'_>,
    ) -> BackendResult<TritonTensor> {
        let out_spec = out.spec;
        let output = out.tensor;
        let _scope = profiling::backend_scope("backend.triton.fused.dot_epilogue");
        if instruction.operands.len() < 3 {
            return Err(BackendError::execution(
                "fused dot+bias custom_call requires at least three operands",
            ));
        }
        let plan = fused_dot_epilogue::FusedDotBiasPlan::parse(spec, instruction.operands.len())?;
        let lhs = self.resolve_operand_tensor(driver, values, instruction.operands.first())?;
        let rhs = self.resolve_operand_tensor(driver, values, instruction.operands.get(1))?;
        let bias =
            self.resolve_operand_tensor(driver, values, instruction.operands.get(plan.add_input))?;

        if prefer_custom_dot_bias_kernel(&lhs.spec, &rhs.spec, out_spec) {
            if let Some(kernel) = kernels.get(DOT_BIAS_RANK2_KERNEL_ID) {
                let rank2_args = DotBiasRank2Args {
                    driver,
                    kernel,
                    plan: &plan,
                    lhs: &lhs,
                    rhs: &rhs,
                    bias: &bias,
                    out_spec,
                    output: output.clone(),
                };
                if let Some(out) = self.try_execute_dot_bias_rank2(rank2_args)? {
                    return Ok(out);
                }
            }
        }

        ensure_static_broadcastable(&out_spec.shape, &bias.spec.shape, "fused dot+bias bias")?;

        let dot = self.execute_dot_general(
            driver,
            DotGeneralArgs {
                spec: &plan.dot,
                lhs_spec: &lhs.spec,
                rhs_spec: &rhs.spec,
                out_spec,
            },
            &lhs,
            &rhs,
            None,
        )?;
        let epilogue_plan = fused_elementwise::FusedElementwisePlan::from_components(
            vec![1],
            vec![0],
            vec![0],
            vec![1],
        )?;
        let epilogue_inputs = vec![dot, bias];
        self.execute_fused_elementwise_plan(
            driver,
            &epilogue_plan,
            epilogue_inputs.as_slice(),
            out_spec,
            output,
        )
    }

    pub(super) fn try_execute_dot_bias_rank2(
        &self,
        args: DotBiasRank2Args<'_>,
    ) -> BackendResult<Option<TritonTensor>> {
        let DotBiasRank2Args {
            driver,
            kernel,
            plan,
            lhs,
            rhs,
            bias,
            out_spec,
            output,
        } = args;
        if !matches!(kernel.kind, KernelKind::DotBiasRank2F32) {
            return Err(BackendError::execution(format!(
                "unexpected dot+bias kernel kind: {:?}",
                kernel.kind
            )));
        }

        if !plan.dot.batch_lhs.is_empty()
            || !plan.dot.batch_rhs.is_empty()
            || plan.dot.contract_lhs.as_slice() != [1]
            || plan.dot.contract_rhs.as_slice() != [0]
        {
            return Ok(None);
        }

        if lhs.spec.dtype != DType::F32
            || rhs.spec.dtype != DType::F32
            || bias.spec.dtype != DType::F32
            || out_spec.dtype != DType::F32
        {
            return Ok(None);
        }

        let lhs_dims = static_dims_or_error(&lhs.spec.shape, |_| {
            BackendError::execution("dynamic dimensions are not supported by triton runtime")
        })?;
        let rhs_dims = static_dims_or_error(&rhs.spec.shape, |_| {
            BackendError::execution("dynamic dimensions are not supported by triton runtime")
        })?;
        let bias_dims = static_dims_or_error(&bias.spec.shape, |_| {
            BackendError::execution("dynamic dimensions are not supported by triton runtime")
        })?;
        let out_dims = static_dims_or_error(&out_spec.shape, |_| {
            BackendError::execution("dynamic dimensions are not supported by triton runtime")
        })?;

        if lhs_dims.len() != 2 || rhs_dims.len() != 2 || out_dims.len() != 2 {
            return Ok(None);
        }

        let m = lhs_dims[0];
        let k = lhs_dims[1];
        let rhs_k = rhs_dims[0];
        let n = rhs_dims[1];
        if k != rhs_k || out_dims[0] != m || out_dims[1] != n {
            return Ok(None);
        }
        let bias_rank1 = bias_dims.len() == 1 && bias_dims[0] == n;
        let bias_row = bias_dims.len() == 2 && m == 1 && bias_dims[0] == 1 && bias_dims[1] == n;
        if !(bias_rank1 || bias_row) {
            return Ok(None);
        }

        if m == 0 || n == 0 || k == 0 {
            let out = allocate_output_tensor(driver, out_spec, output)?;
            return Ok(Some(out));
        }

        let mut m_i32 = i32::try_from(m)
            .map_err(|_| BackendError::execution("dot+bias m exceeds i32 range"))?;
        let mut n_i32 = i32::try_from(n)
            .map_err(|_| BackendError::execution("dot+bias n exceeds i32 range"))?;
        let mut k_i32 = i32::try_from(k)
            .map_err(|_| BackendError::execution("dot+bias k exceeds i32 range"))?;

        let out = allocate_output_tensor(driver, out_spec, output)?;
        let loaded = self.load_kernel(driver, kernel)?;

        let mut lhs_ptr = lhs.buffer.device_ptr();
        let mut rhs_ptr = rhs.buffer.device_ptr();
        let mut bias_ptr = bias.buffer.device_ptr();
        let mut out_ptr = out.buffer.device_ptr();
        let mut opaque_ptr = 0u64;

        let mut params = [
            (&mut lhs_ptr as *mut u64).cast::<c_void>(),
            (&mut rhs_ptr as *mut u64).cast::<c_void>(),
            (&mut bias_ptr as *mut u64).cast::<c_void>(),
            (&mut out_ptr as *mut u64).cast::<c_void>(),
            (&mut m_i32 as *mut i32).cast::<c_void>(),
            (&mut n_i32 as *mut i32).cast::<c_void>(),
            (&mut k_i32 as *mut i32).cast::<c_void>(),
            (&mut opaque_ptr as *mut u64).cast::<c_void>(),
        ];

        let block_n = 128u32;
        let m_u32 = u32::try_from(m)
            .map_err(|_| BackendError::execution("dot+bias m exceeds u32 range"))?;
        let n_u32 = u32::try_from(n)
            .map_err(|_| BackendError::execution("dot+bias n exceeds u32 range"))?;
        let n_blocks = n_u32.div_ceil(block_n);
        let work = m_u32
            .checked_mul(n_u32)
            .ok_or_else(|| BackendError::execution("dot+bias work element overflow"))?;
        launch_program_grid_2d(driver, &loaded, m_u32, n_blocks, block_n, work, &mut params)?;
        Ok(Some(out))
    }
}
