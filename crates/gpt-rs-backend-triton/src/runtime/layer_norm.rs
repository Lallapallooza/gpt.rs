use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::Arc;

use gpt_rs::backend::shape_helpers::static_dims_or_error;
use gpt_rs::backend::spec::{BackendError, BackendResult, CustomCallAttr, CustomCallSpec, DType};

use crate::device::CudaDriver;
use crate::kernels::{KernelKind, KernelSpec, LAYER_NORM_F32_KERNEL_ID};
use crate::tensor::TritonTensor;

use super::{
    allocate_output_tensor, launch_program_grid, DenseValueStore, OutputBinding, TritonExecutor,
};

impl TritonExecutor {
    pub(super) fn execute_layer_norm_custom_call(
        &self,
        driver: &Arc<CudaDriver>,
        kernels: &HashMap<&str, &KernelSpec>,
        values: &DenseValueStore,
        instruction: &gpt_rs::backend::spec::Instruction,
        spec: &CustomCallSpec,
        out: OutputBinding<'_>,
    ) -> BackendResult<TritonTensor> {
        let _scope = gpt_rs::profiling::backend_scope("backend.triton.fused.layer_norm");
        if instruction.operands.len() != 3 {
            return Err(BackendError::execution(
                "fused layer_norm custom_call requires exactly three operands",
            ));
        }

        let axis = match spec.attrs.get("axis") {
            Some(CustomCallAttr::I64(value)) => *value,
            _ => {
                return Err(BackendError::execution(
                    "fused layer_norm custom_call missing i64 attr 'axis'",
                ));
            }
        };
        let eps = match spec.attrs.get("eps") {
            Some(CustomCallAttr::F64(value)) => *value as f32,
            _ => {
                return Err(BackendError::execution(
                    "fused layer_norm custom_call missing f64 attr 'eps'",
                ));
            }
        };
        if !eps.is_finite() || eps <= 0.0 {
            return Err(BackendError::execution(
                "fused layer_norm custom_call requires finite positive eps",
            ));
        }

        let input = self.resolve_operand_tensor(driver, values, instruction.operands.first())?;
        let gamma = self.resolve_operand_tensor(driver, values, instruction.operands.get(1))?;
        let beta = self.resolve_operand_tensor(driver, values, instruction.operands.get(2))?;
        self.execute_layer_norm_f32(driver, kernels, &input, &gamma, &beta, axis, eps, out)
    }

    fn execute_layer_norm_f32(
        &self,
        driver: &Arc<CudaDriver>,
        kernels: &HashMap<&str, &KernelSpec>,
        input: &TritonTensor,
        gamma: &TritonTensor,
        beta: &TritonTensor,
        axis: i64,
        eps: f32,
        out: OutputBinding<'_>,
    ) -> BackendResult<TritonTensor> {
        let out_spec = out.spec;
        let output = out.tensor;
        if input.spec.dtype != DType::F32
            || gamma.spec.dtype != DType::F32
            || beta.spec.dtype != DType::F32
            || out_spec.dtype != DType::F32
        {
            return Err(BackendError::execution(
                "fused layer_norm currently supports F32 only",
            ));
        }
        if input.spec.shape != out_spec.shape {
            return Err(BackendError::execution(
                "fused layer_norm input/output shape mismatch",
            ));
        }

        let input_dims = static_dims_or_error(&input.spec.shape, |_| {
            BackendError::execution("dynamic dimensions are not supported by fused layer_norm")
        })?;
        if input_dims.is_empty() {
            return Err(BackendError::execution(
                "fused layer_norm requires rank >= 1 input tensor",
            ));
        }
        let expected_axis = i64::try_from(input_dims.len() - 1)
            .map_err(|_| BackendError::execution("fused layer_norm rank exceeds i64 range"))?;
        if axis != expected_axis {
            return Err(BackendError::execution(
                "fused layer_norm currently supports last-axis only",
            ));
        }

        let cols = input_dims[input_dims.len() - 1];
        let gamma_dims = static_dims_or_error(&gamma.spec.shape, |_| {
            BackendError::execution("dynamic dimensions are not supported by fused layer_norm")
        })?;
        let beta_dims = static_dims_or_error(&beta.spec.shape, |_| {
            BackendError::execution("dynamic dimensions are not supported by fused layer_norm")
        })?;
        if gamma_dims.as_slice() != [cols] || beta_dims.as_slice() != [cols] {
            return Err(BackendError::execution(
                "fused layer_norm expects gamma/beta shape [hidden_size]",
            ));
        }

        let rows = input_dims[..input_dims.len() - 1]
            .iter()
            .try_fold(1usize, |acc, dim| acc.checked_mul(*dim))
            .ok_or_else(|| BackendError::execution("fused layer_norm row dimension overflow"))?;
        let out = allocate_output_tensor(driver, out_spec, output)?;
        if rows == 0 || cols == 0 {
            return Ok(out);
        }

        let kernel = kernels.get(LAYER_NORM_F32_KERNEL_ID).ok_or_else(|| {
            BackendError::execution("missing layer_norm kernel in triton artifact")
        })?;
        if !matches!(kernel.kind, KernelKind::LayerNormF32) {
            return Err(BackendError::execution(format!(
                "unexpected layer_norm kernel kind: {:?}",
                kernel.kind
            )));
        }
        let loaded = self.load_kernel(driver, kernel)?;

        let mut in_ptr = input.buffer.device_ptr();
        let mut gamma_ptr = gamma.buffer.device_ptr();
        let mut beta_ptr = beta.buffer.device_ptr();
        let mut out_ptr = out.buffer.device_ptr();
        let mut rows_i32 = i32::try_from(rows)
            .map_err(|_| BackendError::execution("fused layer_norm rows exceeds i32 range"))?;
        let mut cols_i32 = i32::try_from(cols)
            .map_err(|_| BackendError::execution("fused layer_norm cols exceeds i32 range"))?;
        let mut eps_f32 = eps;
        let mut opaque_ptr = 0u64;
        let mut params = [
            (&mut in_ptr as *mut u64).cast::<c_void>(),
            (&mut gamma_ptr as *mut u64).cast::<c_void>(),
            (&mut beta_ptr as *mut u64).cast::<c_void>(),
            (&mut out_ptr as *mut u64).cast::<c_void>(),
            (&mut rows_i32 as *mut i32).cast::<c_void>(),
            (&mut cols_i32 as *mut i32).cast::<c_void>(),
            (&mut eps_f32 as *mut f32).cast::<c_void>(),
            (&mut opaque_ptr as *mut u64).cast::<c_void>(),
        ];
        let rows_u32 = u32::try_from(rows)
            .map_err(|_| BackendError::execution("fused layer_norm rows exceeds u32 range"))?;
        let work_elements = rows
            .checked_mul(cols)
            .ok_or_else(|| BackendError::execution("fused layer_norm work overflow"))?;
        let work_elements = u32::try_from(work_elements)
            .map_err(|_| BackendError::execution("fused layer_norm work exceeds u32 range"))?;
        launch_program_grid(driver, &loaded, rows_u32, 256, work_elements, &mut params)?;
        Ok(out)
    }
}
