use std::collections::HashMap;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Duration;

use gpt_rs::backend::spec::{BackendError, BackendResult, Function, Instruction, ValueId};
use gpt_rs::profiling::{self, WorkStats};

use crate::artifact::TritonArtifact;
use crate::device::{self, CudaDriver};
use crate::kernels::KernelSpec;
use crate::tensor::TritonTensor;

use super::{
    compute_value_last_use, first_missing_operand_value, initialize_values,
    validate_function_topology, DenseValueStore, ExecutionKernelGuard, LoadedKernel, SlotAllocator,
    TritonExecutor, KERNEL_LAUNCH_COUNT,
};

impl TritonExecutor {
    pub fn execute_artifact(
        &self,
        artifact: &TritonArtifact,
        entry_inputs: &[TritonTensor],
    ) -> BackendResult<Vec<TritonTensor>> {
        let function = artifact
            .program
            .functions
            .iter()
            .find(|function| function.name == artifact.program.entry)
            .ok_or_else(|| BackendError::execution("triton artifact entry function not found"))?;
        validate_function_topology(function)?;

        if function.parameter_ids.len() != entry_inputs.len() {
            return Err(BackendError::execution(format!(
                "triton entry input arity mismatch: expected {}, got {}",
                function.parameter_ids.len(),
                entry_inputs.len()
            )));
        }
        let function_slot_plan = artifact.buffer_plan.functions.get(&function.name);
        let value_last_use = compute_value_last_use(function, function_slot_plan);

        let driver = device::driver()?;
        let kernels = artifact
            .kernels
            .iter()
            .map(|spec| (spec.id.as_str(), spec))
            .collect::<HashMap<_, _>>();
        let preloaded = self.preload_execution_kernels(&driver, &artifact.kernels)?;
        let _execution_kernel_guard = ExecutionKernelGuard::push(preloaded);

        let mut values = initialize_values(function, entry_inputs);
        if function_slot_plan.is_some() {
            profiling::cache_event("triton_backend.slot_plan_available");
        } else {
            profiling::cache_event("triton_backend.slot_plan_missing");
        }
        let mut slot_allocator = SlotAllocator::new(function_slot_plan, value_last_use);
        let launch_start = KERNEL_LAUNCH_COUNT.load(Ordering::Relaxed);
        let dispatch_start = std::time::Instant::now();
        if let Some((missing, instruction)) = self.execute_body_single_pass(
            &driver,
            &kernels,
            &mut values,
            &mut slot_allocator,
            function,
        )? {
            return Err(BackendError::execution(format!(
                "operand value {} missing before instruction {} ({:?}); artifact is not topologically executable",
                missing.0, instruction.id.0, instruction.op
            )));
        }

        let mut results = Vec::with_capacity(function.result_ids.len());
        for result_id in &function.result_ids {
            let value = values.get_cloned(*result_id).ok_or_else(|| {
                BackendError::execution(format!(
                    "missing result value {} in triton runtime",
                    result_id.0
                ))
            })?;
            results.push(value);
        }
        let dispatch_elapsed = dispatch_start.elapsed();
        profiling::record_backend_aggregate(
            "backend.triton.exec.dispatch",
            1,
            dispatch_elapsed,
            WorkStats {
                elements: function.body.len() as u64,
                ..WorkStats::default()
            },
        );
        let launch_end = KERNEL_LAUNCH_COUNT.load(Ordering::Relaxed);
        let launched = launch_end.saturating_sub(launch_start);
        if launched != 0 {
            profiling::record_backend_aggregate(
                "backend.triton.exec.launches",
                launched,
                Duration::ZERO,
                WorkStats::default(),
            );
        }
        Ok(results)
    }

    fn execute_body_single_pass(
        &self,
        driver: &Arc<CudaDriver>,
        kernels: &HashMap<&str, &KernelSpec>,
        values: &mut DenseValueStore,
        slot_allocator: &mut SlotAllocator,
        function: &Function,
    ) -> BackendResult<Option<(ValueId, Instruction)>> {
        for (idx, instruction) in function.body.iter().enumerate() {
            if let Some(missing) = first_missing_operand_value(instruction, values) {
                return Ok(Some((missing, instruction.clone())));
            }
            let instruction_pos = idx + 1;
            let output = self.execute_instruction(
                driver,
                kernels,
                values,
                slot_allocator,
                instruction,
                instruction_pos,
            )?;
            values.insert(instruction.id, output);
        }
        Ok(None)
    }

    fn preload_execution_kernels(
        &self,
        driver: &Arc<CudaDriver>,
        specs: &[KernelSpec],
    ) -> BackendResult<HashMap<String, Arc<LoadedKernel>>> {
        let mut out = HashMap::with_capacity(specs.len());
        for spec in specs {
            let loaded = self.load_kernel(driver, spec)?;
            out.insert(spec.id.clone(), loaded);
        }
        Ok(out)
    }
}
