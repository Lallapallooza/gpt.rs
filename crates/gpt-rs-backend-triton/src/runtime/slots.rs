use std::collections::HashMap;
use std::sync::Arc;

use gpt_rs::backend::spec::{BackendError, BackendResult, TensorSpec, ValueId};

use crate::artifact::TritonFunctionSlotPlan;
use crate::device::{CudaDriver, DeviceBuffer};
use crate::tensor::TritonTensor;

/// Runtime allocator backed by bufferization slot assignments.
pub(crate) struct SlotAllocator {
    value_to_slot: HashMap<ValueId, SlotBinding>,
    value_last_use: HashMap<ValueId, usize>,
    slot_byte_len: Vec<Option<usize>>,
    slot_buffers: Vec<Option<Arc<DeviceBuffer>>>,
    slot_release_pos: Vec<usize>,
}

impl SlotAllocator {
    pub(crate) fn new(
        plan: Option<&TritonFunctionSlotPlan>,
        value_last_use: HashMap<ValueId, usize>,
    ) -> Self {
        let Some(plan) = plan else {
            return Self {
                value_to_slot: HashMap::new(),
                value_last_use,
                slot_byte_len: Vec::new(),
                slot_buffers: Vec::new(),
                slot_release_pos: Vec::new(),
            };
        };
        let mut value_to_slot = HashMap::with_capacity(plan.value_slots.len());
        for binding in &plan.value_slots {
            value_to_slot.insert(binding.value, SlotBinding { slot: binding.slot });
        }
        Self {
            value_to_slot,
            value_last_use,
            slot_byte_len: plan.slots.iter().map(|slot| slot.byte_len).collect(),
            slot_buffers: vec![None; plan.slots.len()],
            slot_release_pos: vec![0; plan.slots.len()],
        }
    }

    pub(crate) fn output_for_value(
        &mut self,
        driver: &Arc<CudaDriver>,
        value: ValueId,
        spec: &TensorSpec,
        instruction_pos: usize,
    ) -> BackendResult<Option<TritonTensor>> {
        let Some(binding) = self.value_to_slot.get(&value).copied() else {
            return Ok(None);
        };
        let slot_id = binding.slot;
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
        let slot_release_pos = self.slot_release_pos.get(slot_id).copied().ok_or_else(|| {
            BackendError::execution(format!(
                "slot {} for value {} is outside runtime release bounds",
                slot_id, value.0
            ))
        })?;
        if slot_release_pos >= instruction_pos {
            // Runtime order indicates this slot is still live; skip slot reuse for safety.
            return Ok(None);
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
        let value_live_end = self.value_last_use.get(&value).copied().ok_or_else(|| {
            BackendError::execution(format!(
                "missing last-use position for slot-backed value {}",
                value.0
            ))
        })?;
        let release_ref = self.slot_release_pos.get_mut(slot_id).ok_or_else(|| {
            BackendError::execution(format!(
                "slot {} for value {} disappeared during release update",
                slot_id, value.0
            ))
        })?;
        *release_ref = value_live_end;
        Ok(Some(TritonTensor::new(spec.clone(), buffer)))
    }

    pub(crate) fn propagate_alias(&mut self, source: ValueId, alias: ValueId) {
        let Some(source_binding) = self.value_to_slot.get(&source).copied() else {
            return;
        };
        let Some(alias_binding) = self.value_to_slot.get(&alias).copied() else {
            return;
        };
        if source_binding.slot != alias_binding.slot {
            return;
        }
        let alias_live_end = self
            .value_last_use
            .get(&alias)
            .copied()
            .unwrap_or(usize::MAX);
        if let Some(release_ref) = self.slot_release_pos.get_mut(source_binding.slot) {
            *release_ref = (*release_ref).max(alias_live_end);
        }
    }
}

#[derive(Copy, Clone)]
struct SlotBinding {
    slot: usize,
}
