use std::collections::HashMap;

use gpt_rs::backend::conversion::{BufferPlan, BufferSlot, BufferSpec};
use gpt_rs::backend::spec::{Program, RegionId};
use serde::{Deserialize, Serialize};

use crate::kernels::KernelSpec;

pub const TRITON_ARTIFACT_VERSION: u32 = 3;

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
    pub functions: HashMap<String, TritonFunctionBufferPlan>,
    #[serde(default)]
    pub regions: Vec<TritonRegionBufferPlan>,
}

impl From<BufferPlan> for TritonArtifactBufferPlan {
    fn from(value: BufferPlan) -> Self {
        let mut regions = value
            .regions
            .into_iter()
            .map(|(id, plan)| TritonRegionBufferPlan {
                id,
                plan: plan.into(),
            })
            .collect::<Vec<_>>();
        regions.sort_by_key(|entry| entry.id.0);
        Self {
            functions: value
                .functions
                .into_iter()
                .map(|(name, plan)| (name, plan.into()))
                .collect(),
            regions,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TritonRegionBufferPlan {
    pub id: RegionId,
    pub plan: TritonFunctionBufferPlan,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TritonFunctionBufferPlan {
    pub buffers: Vec<BufferSpec>,
    pub slots: Vec<BufferSlot>,
}

impl From<gpt_rs::backend::conversion::FunctionBufferPlan> for TritonFunctionBufferPlan {
    fn from(value: gpt_rs::backend::conversion::FunctionBufferPlan) -> Self {
        let mut buffers = value.buffers;
        canonicalize_alias_groups(buffers.as_mut_slice());
        Self {
            buffers,
            slots: value.slots,
        }
    }
}

fn canonicalize_alias_groups(buffers: &mut [BufferSpec]) {
    let mut map = HashMap::<usize, usize>::new();
    let mut next = 0usize;
    for buffer in buffers {
        let canonical = match map.get(&buffer.alias_group) {
            Some(value) => *value,
            None => {
                let value = next;
                next = next.saturating_add(1);
                map.insert(buffer.alias_group, value);
                value
            }
        };
        buffer.alias_group = canonical;
    }
}
