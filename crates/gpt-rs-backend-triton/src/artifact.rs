use std::collections::HashMap;

use gpt_rs::backend::conversion::BufferPlan;
use gpt_rs::backend::spec::{Program, ValueId};
use serde::{Deserialize, Serialize};

use crate::kernels::KernelSpec;

pub const TRITON_ARTIFACT_VERSION: u32 = 2;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TritonArtifact {
    pub artifact_version: u32,
    pub entrypoint: String,
    pub program: Program,
    pub buffer_plan: TritonArtifactBufferPlan,
    pub kernels: Vec<KernelSpec>,
}

impl TritonArtifact {
    pub fn new(
        entrypoint: String,
        program: Program,
        buffer_plan: BufferPlan,
        kernels: Vec<KernelSpec>,
    ) -> Self {
        Self {
            artifact_version: TRITON_ARTIFACT_VERSION,
            entrypoint,
            program,
            buffer_plan: buffer_plan.into(),
            kernels,
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TritonArtifactBufferPlan {
    pub functions: HashMap<String, TritonFunctionSlotPlan>,
}

impl From<BufferPlan> for TritonArtifactBufferPlan {
    fn from(value: BufferPlan) -> Self {
        let functions = value
            .functions
            .iter()
            .map(|(name, plan)| (name.clone(), TritonFunctionSlotPlan::from(plan)))
            .collect();
        Self { functions }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TritonFunctionSlotPlan {
    pub slots: Vec<TritonSlotSpec>,
    pub value_slots: Vec<TritonValueSlot>,
}

impl From<&gpt_rs::backend::conversion::FunctionBufferPlan> for TritonFunctionSlotPlan {
    fn from(value: &gpt_rs::backend::conversion::FunctionBufferPlan) -> Self {
        let slots = value
            .slots
            .iter()
            .map(|slot| TritonSlotSpec {
                id: slot.id,
                byte_len: slot.byte_len,
            })
            .collect();
        let value_slots = value
            .buffers
            .iter()
            .filter_map(|buffer| {
                buffer.slot.map(|slot| TritonValueSlot {
                    value: buffer.value,
                    slot,
                })
            })
            .collect();
        Self { slots, value_slots }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TritonSlotSpec {
    pub id: usize,
    pub byte_len: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TritonValueSlot {
    pub value: ValueId,
    pub slot: usize,
}
