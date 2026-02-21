use gpt_rs::backend::spec::Program;
use serde::{Deserialize, Serialize};

use crate::kernels::KernelSpec;

pub const TRITON_ARTIFACT_VERSION: u32 = 1;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TritonArtifact {
    pub artifact_version: u32,
    pub entrypoint: String,
    pub program: Program,
    pub kernels: Vec<KernelSpec>,
}

impl TritonArtifact {
    pub fn new(entrypoint: String, program: Program, kernels: Vec<KernelSpec>) -> Self {
        Self {
            artifact_version: TRITON_ARTIFACT_VERSION,
            entrypoint,
            program,
            kernels,
        }
    }
}
