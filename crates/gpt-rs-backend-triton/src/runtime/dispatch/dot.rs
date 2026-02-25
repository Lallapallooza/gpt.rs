use super::*;

impl TritonExecutor {
    pub(super) fn execute_dot_general(
        &self,
        driver: &Arc<CudaDriver>,
        args: DotGeneralArgs<'_>,
        lhs: &TritonTensor,
        rhs: &TritonTensor,
        output: Option<TritonTensor>,
    ) -> BackendResult<TritonTensor> {
        let DotGeneralArgs {
            spec,
            lhs_spec,
            rhs_spec,
            out_spec,
        } = args;
        if lhs.spec != *lhs_spec || rhs.spec != *rhs_spec {
            return Err(BackendError::execution("dot_general tensor/spec mismatch"));
        }
        if lhs_spec.dtype != DType::F32
            || rhs_spec.dtype != DType::F32
            || out_spec.dtype != DType::F32
        {
            return Err(BackendError::execution(
                "dot_general runtime currently supports F32 only",
            ));
        }

        let lhs_dims = static_dims_or_error(&lhs_spec.shape, |_| {
            BackendError::execution("dynamic dimensions are not supported by triton runtime")
        })?;
        let rhs_dims = static_dims_or_error(&rhs_spec.shape, |_| {
            BackendError::execution("dynamic dimensions are not supported by triton runtime")
        })?;
        let out_dims = static_dims_or_error(&out_spec.shape, |_| {
            BackendError::execution("dynamic dimensions are not supported by triton runtime")
        })?;
        let cublas = self.cublas(driver)?;

        // Rank-2 matrix multiplication: [M,K] · [K,N] => [M,N].
        if spec.batch_lhs.is_empty()
            && spec.batch_rhs.is_empty()
            && spec.contract_lhs.as_slice() == [1]
            && spec.contract_rhs.as_slice() == [0]
        {
            if lhs_dims.len() != 2 || rhs_dims.len() != 2 || out_dims.len() != 2 {
                return Err(BackendError::execution(
                    "dot_general rank-2 path expects rank-2 tensors",
                ));
            }

            let m = lhs_dims[0];
            let k = lhs_dims[1];
            let k_rhs = rhs_dims[0];
            let n = rhs_dims[1];
            if k != k_rhs || out_dims[0] != m || out_dims[1] != n {
                return Err(BackendError::execution(
                    "dot_general shape mismatch for matrix multiplication",
                ));
            }

            let out = allocate_output_tensor(driver, out_spec, output.clone())?;
            cublas.sgemm_row_major(&lhs.buffer, &rhs.buffer, &out.buffer, m, n, k)?;
            return Ok(out);
        }

        // Batched rank-3 matrix multiplication: [B,M,K] · [B,K,N] => [B,M,N].
        if spec.batch_lhs.as_slice() == [0]
            && spec.batch_rhs.as_slice() == [0]
            && spec.contract_lhs.as_slice() == [2]
            && spec.contract_rhs.as_slice() == [1]
        {
            if lhs_dims.len() != 3 || rhs_dims.len() != 3 || out_dims.len() != 3 {
                return Err(BackendError::execution(
                    "dot_general batched path expects rank-3 tensors",
                ));
            }

            let batches = lhs_dims[0];
            let m = lhs_dims[1];
            let k = lhs_dims[2];
            let rhs_batches = rhs_dims[0];
            let k_rhs = rhs_dims[1];
            let n = rhs_dims[2];

            if batches != rhs_batches
                || out_dims[0] != batches
                || out_dims[1] != m
                || out_dims[2] != n
                || k != k_rhs
            {
                return Err(BackendError::execution(
                    "dot_general shape mismatch for batched matrix multiplication",
                ));
            }

            let out = allocate_output_tensor(driver, out_spec, output.clone())?;
            let lhs_stride = m
                .checked_mul(k)
                .ok_or_else(|| BackendError::execution("batched lhs stride overflow"))?;
            let rhs_stride = k
                .checked_mul(n)
                .ok_or_else(|| BackendError::execution("batched rhs stride overflow"))?;
            let out_stride = m
                .checked_mul(n)
                .ok_or_else(|| BackendError::execution("batched out stride overflow"))?;
            let cfg = StridedBatchedGemmConfig {
                m,
                n,
                k,
                lhs_stride,
                rhs_stride,
                out_stride,
                batches,
            };
            cublas.sgemm_row_major_strided_batched(&lhs.buffer, &rhs.buffer, &out.buffer, cfg)?;
            return Ok(out);
        }

        // Batched rank-3 matrix multiplication: [B,M,K] · [B,N,K] => [B,M,N].
        if spec.batch_lhs.as_slice() == [0]
            && spec.batch_rhs.as_slice() == [0]
            && spec.contract_lhs.as_slice() == [2]
            && spec.contract_rhs.as_slice() == [2]
        {
            if lhs_dims.len() != 3 || rhs_dims.len() != 3 || out_dims.len() != 3 {
                return Err(BackendError::execution(
                    "dot_general batched path expects rank-3 tensors",
                ));
            }

            let batches = lhs_dims[0];
            let m = lhs_dims[1];
            let k = lhs_dims[2];
            let rhs_batches = rhs_dims[0];
            let n = rhs_dims[1];
            let k_rhs = rhs_dims[2];

            if batches != rhs_batches
                || out_dims[0] != batches
                || out_dims[1] != m
                || out_dims[2] != n
                || k != k_rhs
            {
                return Err(BackendError::execution(
                    "dot_general shape mismatch for batched matrix multiplication [B,M,K] · [B,N,K]",
                ));
            }

            let out = allocate_output_tensor(driver, out_spec, output)?;
            let lhs_stride = m
                .checked_mul(k)
                .ok_or_else(|| BackendError::execution("batched lhs stride overflow"))?;
            let rhs_stride = n
                .checked_mul(k)
                .ok_or_else(|| BackendError::execution("batched rhs stride overflow"))?;
            let out_stride = m
                .checked_mul(n)
                .ok_or_else(|| BackendError::execution("batched out stride overflow"))?;
            let cfg = StridedBatchedGemmConfig {
                m,
                n,
                k,
                lhs_stride,
                rhs_stride,
                out_stride,
                batches,
            };
            cublas.sgemm_row_major_strided_batched_rhs_transposed(
                &lhs.buffer,
                &rhs.buffer,
                &out.buffer,
                cfg,
            )?;
            return Ok(out);
        }

        Err(BackendError::execution(
            "dot_general runtime supports rank-2 MxK·KxN and selected rank-3 batched variants",
        ))
    }
}
