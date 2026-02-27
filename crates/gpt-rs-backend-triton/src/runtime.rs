mod blas;
mod dispatch;
mod executor;
mod fused_dot_epilogue;
mod fused_elementwise;
mod kernel_cache;
mod layer_norm;
mod literals;
mod ops;
mod sampling;
mod slots;
mod softmax;
mod values;

use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};
use std::time::Duration;

use gpt_rs::backend::shape_helpers::{
    checked_element_count_or_error, contiguous_strides_or_error, static_dims_or_error,
};
use gpt_rs::backend::spec::{
    BackendError, BackendResult, ComparisonOp, CustomCallSpec, DType, ElementwiseBinaryOp,
    ElementwiseUnaryOp, Operand, Operation, ReduceKind, TensorLiteral, TensorSpec, ValueId,
    ValueType,
};
use gpt_rs::backend::topology;
use gpt_rs::profiling::{self, ScopeMeta, WorkStats};

use crate::artifact::TritonFunctionBufferPlan;
use crate::compiler::KernelCompiler;
use crate::device::{CudaDriver, DeviceBuffer};
use crate::kernels::{
    KernelKind, KernelSpec, BROADCAST_KERNEL_ID, BROADCAST_SI32_KERNEL_ID,
    COMPARE_SI32_I1_KERNEL_ID, CONCAT_KERNEL_ID, DOT_BIAS_RANK2_KERNEL_ID,
    DYNAMIC_SLICE_F32_KERNEL_ID, DYNAMIC_SLICE_SI32_RANK1_KERNEL_ID,
    DYNAMIC_UPDATE_SLICE_F32_KERNEL_ID, EWISE_BINARY_KERNEL_ID, EWISE_UNARY_KERNEL_ID,
    EXTRACT_PATCHES_NHWC_KERNEL_ID, IOTA_SI32_KERNEL_ID, REDUCE_MAX_LAST_AXIS_KERNEL_ID,
    REDUCE_SUM_LAST_AXIS_KERNEL_ID, REDUCE_WINDOW_MAX_NHWC_KERNEL_ID, SELECT_I1_F32_KERNEL_ID,
    SLICE_KERNEL_ID, TAKE_F32_I32_KERNEL_ID, TRANSPOSE_KERNEL_ID,
};
use crate::targets::{
    TARGET_DOT_BIAS_FUSED_F32_V1, TARGET_ELEMENTWISE_FUSED_F32_V1, TARGET_LAYER_NORM_FUSED_F32_V1,
    TARGET_SOFTMAX_LAST_AXIS_FUSED_F32_V1,
};
use crate::tensor::TritonTensor;
use blas::{
    CublasContext, DotBiasRank2Args, DotGeneralArgs, OutputBinding, StridedBatchedGemmConfig,
};
use kernel_cache::{ExecutionKernelGuard, LoadedKernel};
use ops::*;
use slots::SlotAllocator;
use values::DenseValueStore;

static GPU_EVENT_TIMING_ENABLED: AtomicBool = AtomicBool::new(false);
static KERNEL_LAUNCH_COUNT: AtomicU64 = AtomicU64::new(0);

pub(crate) fn set_gpu_event_timing_enabled(enabled: bool) {
    GPU_EVENT_TIMING_ENABLED.store(enabled, Ordering::Relaxed);
}

pub struct TritonExecutor {
    compiler: KernelCompiler,
    loaded_kernels: Mutex<HashMap<u64, Arc<LoadedKernel>>>,
    fused_kernel_specs: Mutex<HashMap<u64, KernelSpec>>,
    literal_tensors: Mutex<HashMap<u64, TritonTensor>>,
    cublas: OnceLock<Result<Arc<CublasContext>, String>>,
}

impl TritonExecutor {
    pub fn new() -> Self {
        Self {
            compiler: KernelCompiler::new(),
            loaded_kernels: Mutex::new(HashMap::new()),
            fused_kernel_specs: Mutex::new(HashMap::new()),
            literal_tensors: Mutex::new(HashMap::new()),
            cublas: OnceLock::new(),
        }
    }
}

fn first_missing_operand_value(
    instruction: &gpt_rs::backend::spec::Instruction,
    values: &DenseValueStore,
) -> Option<ValueId> {
    for operand in &instruction.operands {
        match operand {
            Operand::Value(id) if !values.contains(*id) => return Some(*id),
            Operand::TupleElement { tuple, .. } if !values.contains(*tuple) => return Some(*tuple),
            Operand::Literal(_) | Operand::Value(_) | Operand::TupleElement { .. } => {}
        }
    }
    None
}

fn validate_function_topology(function: &gpt_rs::backend::spec::Function) -> BackendResult<()> {
    topology::validate_function_topology(function).map_err(|err| {
        BackendError::execution(format!(
            "triton runtime function topology validation failed: {err}"
        ))
    })
}

fn compute_value_last_use(
    function: &gpt_rs::backend::spec::Function,
    slot_plan: Option<&TritonFunctionBufferPlan>,
) -> HashMap<ValueId, usize> {
    let mut last_use = HashMap::new();
    for (idx, instruction) in function.body.iter().enumerate() {
        let pos = idx + 1;
        last_use.entry(instruction.id).or_insert(pos);
        for operand in &instruction.operands {
            match operand {
                Operand::Value(id) => {
                    last_use
                        .entry(*id)
                        .and_modify(|existing| *existing = (*existing).max(pos))
                        .or_insert(pos);
                }
                Operand::TupleElement { tuple, .. } => {
                    last_use
                        .entry(*tuple)
                        .and_modify(|existing| *existing = (*existing).max(pos))
                        .or_insert(pos);
                }
                Operand::Literal(_) => {}
            }
        }
    }
    let result_pos = function.body.len() + 1;
    for value in &function.result_ids {
        last_use
            .entry(*value)
            .and_modify(|existing| *existing = (*existing).max(result_pos))
            .or_insert(result_pos);
    }
    if let Some(slot_plan) = slot_plan {
        for slot_binding in slot_plan
            .buffers
            .iter()
            .filter(|buffer| buffer.slot.is_some() && buffer.path.is_empty())
        {
            last_use
                .entry(slot_binding.value)
                .and_modify(|existing| *existing = (*existing).max(slot_binding.live_range.end))
                .or_insert(slot_binding.live_range.end);
        }
    }
    last_use
}

fn initialize_values(
    function: &gpt_rs::backend::spec::Function,
    entry_inputs: &[TritonTensor],
) -> DenseValueStore {
    let mut values = DenseValueStore::new(function);
    for (value_id, input) in function.parameter_ids.iter().zip(entry_inputs.iter()) {
        values.insert(*value_id, input.clone());
    }
    values
}

fn lock_named<'a, T>(mutex: &'a Mutex<T>, name: &str) -> BackendResult<MutexGuard<'a, T>> {
    mutex
        .lock()
        .map_err(|_| BackendError::execution(format!("triton {name} mutex poisoned")))
}

impl Default for TritonExecutor {
    fn default() -> Self {
        Self::new()
    }
}
