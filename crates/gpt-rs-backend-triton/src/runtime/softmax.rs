use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::Arc;

use gpt_rs::backend::shape_helpers::static_dims_or_error;
use gpt_rs::backend::spec::{BackendError, BackendResult, CustomCallAttr, CustomCallSpec, DType};

use crate::device::CudaDriver;
use crate::kernels::{KernelKind, KernelSpec, SOFTMAX_LAST_AXIS_KERNEL_ID};
use crate::tensor::TritonTensor;

use super::{
    allocate_output_tensor, launch_program_grid, DenseValueStore, OutputBinding, TritonExecutor,
};

impl TritonExecutor {
    pub(super) fn execute_softmax_last_axis_custom_call(
        &self,
        driver: &Arc<CudaDriver>,
        kernels: &HashMap<&str, &KernelSpec>,
        values: &DenseValueStore,
        instruction: &gpt_rs::backend::spec::Instruction,
        spec: &CustomCallSpec,
        out: OutputBinding<'_>,
    ) -> BackendResult<TritonTensor> {
        let _scope = gpt_rs::profiling::backend_scope("backend.triton.fused.softmax_last_axis");
        if instruction.operands.len() != 1 {
            return Err(BackendError::execution(
                "fused softmax custom_call requires exactly one operand",
            ));
        }
        let axis = match spec.attrs.get("axis") {
            Some(CustomCallAttr::I64(value)) => *value,
            _ => {
                return Err(BackendError::execution(
                    "fused softmax custom_call missing i64 attr 'axis'",
                ));
            }
        };

        let input = self.resolve_operand_tensor(driver, values, instruction.operands.first())?;
        self.execute_softmax_last_axis_f32(driver, kernels, &input, axis, out)
    }

    fn execute_softmax_last_axis_f32(
        &self,
        driver: &Arc<CudaDriver>,
        kernels: &HashMap<&str, &KernelSpec>,
        input: &TritonTensor,
        axis: i64,
        out: OutputBinding<'_>,
    ) -> BackendResult<TritonTensor> {
        let out_spec = out.spec;
        let output = out.tensor;
        if input.spec.dtype != DType::F32 || out_spec.dtype != DType::F32 {
            return Err(BackendError::execution(
                "fused softmax currently supports F32 only",
            ));
        }
        if input.spec.shape != out_spec.shape {
            return Err(BackendError::execution(
                "fused softmax input/output shape mismatch",
            ));
        }

        let dims = static_dims_or_error(&input.spec.shape, |_| {
            BackendError::execution("dynamic dimensions are not supported by fused softmax")
        })?;
        if dims.is_empty() {
            return Err(BackendError::execution(
                "fused softmax requires rank >= 1 input tensor",
            ));
        }

        let expected_axis = i64::try_from(dims.len() - 1)
            .map_err(|_| BackendError::execution("fused softmax rank exceeds i64 range"))?;
        if axis != expected_axis {
            return Err(BackendError::execution(
                "fused softmax currently supports last-axis only",
            ));
        }

        let cols = dims[dims.len() - 1];
        let rows = dims[..dims.len() - 1]
            .iter()
            .try_fold(1usize, |acc, dim| acc.checked_mul(*dim))
            .ok_or_else(|| BackendError::execution("fused softmax row dimension overflow"))?;

        let out = allocate_output_tensor(driver, out_spec, output)?;
        if rows == 0 || cols == 0 {
            return Ok(out);
        }

        let kernel = kernels.get(SOFTMAX_LAST_AXIS_KERNEL_ID).ok_or_else(|| {
            BackendError::execution("missing softmax_last_axis kernel in triton artifact")
        })?;
        if !matches!(kernel.kind, KernelKind::SoftmaxLastAxisF32) {
            return Err(BackendError::execution(format!(
                "unexpected softmax kernel kind: {:?}",
                kernel.kind
            )));
        }
        let loaded = self.load_kernel(driver, kernel)?;

        let mut in_ptr = input.buffer.device_ptr();
        let mut out_ptr = out.buffer.device_ptr();
        let mut rows_i32 = i32::try_from(rows)
            .map_err(|_| BackendError::execution("fused softmax rows exceeds i32 range"))?;
        let mut cols_i32 = i32::try_from(cols)
            .map_err(|_| BackendError::execution("fused softmax cols exceeds i32 range"))?;
        let mut opaque_ptr = 0u64;
        let mut params = [
            (&mut in_ptr as *mut u64).cast::<c_void>(),
            (&mut out_ptr as *mut u64).cast::<c_void>(),
            (&mut rows_i32 as *mut i32).cast::<c_void>(),
            (&mut cols_i32 as *mut i32).cast::<c_void>(),
            (&mut opaque_ptr as *mut u64).cast::<c_void>(),
        ];

        let rows_u32 = u32::try_from(rows)
            .map_err(|_| BackendError::execution("fused softmax rows exceeds u32 range"))?;
        let work_elements = rows
            .checked_mul(cols)
            .ok_or_else(|| BackendError::execution("fused softmax work overflow"))?;
        let work_elements = u32::try_from(work_elements)
            .map_err(|_| BackendError::execution("fused softmax work exceeds u32 range"))?;
        launch_program_grid(driver, &loaded, rows_u32, 256, work_elements, &mut params)?;
        Ok(out)
    }
}
