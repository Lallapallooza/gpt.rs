use std::collections::HashMap;
use std::sync::Arc;

use gpt_rs::backend::spec::{BackendError, BackendResult, TensorSpec, ValueId};

use crate::artifact::TritonFunctionSlotPlan;
use crate::device::{CudaDriver, DeviceBuffer};
use crate::tensor::TritonTensor;

/// Runtime allocator backed by bufferization slot assignments.
pub(crate) struct SlotAllocator {
    value_to_slot: HashMap<ValueId, usize>,
    slot_byte_len: Vec<Option<usize>>,
    slot_buffers: Vec<Option<Arc<DeviceBuffer>>>,
}

impl SlotAllocator {
    pub(crate) fn new(plan: Option<&TritonFunctionSlotPlan>) -> Self {
        let Some(plan) = plan else {
            return Self {
                value_to_slot: HashMap::new(),
                slot_byte_len: Vec::new(),
                slot_buffers: Vec::new(),
            };
        };
        let mut value_to_slot = HashMap::with_capacity(plan.value_slots.len());
        for binding in &plan.value_slots {
            value_to_slot.insert(binding.value, binding.slot);
        }
        Self {
            value_to_slot,
            slot_byte_len: plan.slots.iter().map(|slot| slot.byte_len).collect(),
            slot_buffers: vec![None; plan.slots.len()],
        }
    }

    pub(crate) fn output_for_value(
        &mut self,
        driver: &Arc<CudaDriver>,
        value: ValueId,
        spec: &TensorSpec,
    ) -> BackendResult<Option<TritonTensor>> {
        let Some(slot_id) = self.value_to_slot.get(&value).copied() else {
            return Ok(None);
        };
        let expected = self.slot_byte_len.get(slot_id).copied().flatten();
        let actual = spec.byte_len().ok_or_else(|| {
            BackendError::execution(format!(
                "cannot compute byte length for slot-backed value {} with spec {:?}",
                value.0, spec
            ))
        })?;
        if let Some(expected_bytes) = expected {
            if expected_bytes != actual {
                return Err(BackendError::execution(format!(
                    "slot byte size mismatch for value {}: slot={} spec={}",
                    value.0, expected_bytes, actual
                )));
            }
        }
        let maybe_existing = self.slot_buffers.get(slot_id).ok_or_else(|| {
            BackendError::execution(format!(
                "slot {} for value {} is outside runtime allocator bounds",
                slot_id, value.0
            ))
        })?;
        let buffer = match maybe_existing {
            Some(buffer) => Arc::clone(buffer),
            None => {
                let allocated = driver.alloc_zeroed(actual)?;
                let slot_ref = self.slot_buffers.get_mut(slot_id).ok_or_else(|| {
                    BackendError::execution(format!(
                        "slot {} for value {} disappeared during allocation",
                        slot_id, value.0
                    ))
                })?;
                *slot_ref = Some(Arc::clone(&allocated));
                allocated
            }
        };
        Ok(Some(TritonTensor::new(spec.clone(), buffer)))
    }
}
