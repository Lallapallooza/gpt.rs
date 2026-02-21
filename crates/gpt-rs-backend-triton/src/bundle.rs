use gpt_rs::backend::spec::{DotGeneralSpec, ElementwiseBinaryOp, ReduceSpec, TensorSpec, ValueId};
use serde::{Deserialize, Serialize};

pub const TRITON_BUNDLE_VERSION: u32 = 1;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TritonBundle {
    pub bundle_version: u32,
    pub entrypoint: String,
    pub parameter_ids: Vec<u32>,
    pub result_ids: Vec<u32>,
    pub kernels: Vec<KernelSpec>,
    pub steps: Vec<BundleStep>,
}

impl TritonBundle {
    pub fn new(entrypoint: String, parameter_ids: Vec<ValueId>, result_ids: Vec<ValueId>) -> Self {
        Self {
            bundle_version: TRITON_BUNDLE_VERSION,
            entrypoint,
            parameter_ids: parameter_ids.into_iter().map(|id| id.0).collect(),
            result_ids: result_ids.into_iter().map(|id| id.0).collect(),
            kernels: Vec::new(),
            steps: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KernelSpec {
    pub id: String,
    pub kind: KernelKind,
    pub source: String,
    pub symbol: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum KernelKind {
    ElementwiseBinaryF32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BundleStep {
    Constant {
        value_id: u32,
        literal: SerializableLiteral,
    },
    Alias {
        value_id: u32,
        source_id: u32,
        spec: TensorSpec,
    },
    ElementwiseBinary {
        value_id: u32,
        lhs_id: u32,
        rhs_id: u32,
        op: ElementwiseBinaryOp,
        spec: TensorSpec,
        kernel_id: String,
    },
    DotGeneral {
        value_id: u32,
        lhs_id: u32,
        rhs_id: u32,
        lhs_spec: TensorSpec,
        rhs_spec: TensorSpec,
        out_spec: TensorSpec,
        spec: DotGeneralSpec,
    },
    Reduce {
        value_id: u32,
        input_id: u32,
        input_spec: TensorSpec,
        out_spec: TensorSpec,
        spec: ReduceSpec,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializableLiteral {
    pub spec: TensorSpec,
    pub bytes: Vec<u8>,
}
