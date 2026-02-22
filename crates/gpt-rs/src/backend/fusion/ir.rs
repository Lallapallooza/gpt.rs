use std::collections::BTreeMap;

use crate::backend::fusion::{FUSION_KIND_DOT_EPILOGUE_V1, FUSION_KIND_ELEMENTWISE_DAG_V1};
use crate::backend::spec::{CustomCallAttr, DotGeneralSpec, HintKind, HintPolicy, ValueId};

#[derive(Debug, Clone, PartialEq)]
pub enum FusionDetail {
    ElementwiseDag(ElementwiseDagPayload),
    DotEpilogue(DotEpiloguePayload),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ElementwiseDagPayload {
    pub ops_kind: Vec<i64>,
    pub ops_code: Vec<i64>,
    pub lhs: Vec<i64>,
    pub rhs: Vec<i64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DotEpiloguePayload {
    pub dot: DotGeneralSpec,
    pub add_input: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FusionCandidate {
    pub id: u32,
    pub kind: HintKind,
    pub anchor: ValueId,
    pub inst_values: Vec<ValueId>,
    pub inputs: Vec<ValueId>,
    pub exports: Vec<ValueId>,
    pub detail: FusionDetail,
}

impl FusionCandidate {
    pub fn attrs(&self) -> BTreeMap<String, CustomCallAttr> {
        let mut attrs = BTreeMap::new();
        match &self.detail {
            FusionDetail::ElementwiseDag(payload) => {
                attrs.insert("fusion_version".into(), CustomCallAttr::I64(1));
                attrs.insert(
                    "fusion_kind".into(),
                    CustomCallAttr::String(FUSION_KIND_ELEMENTWISE_DAG_V1.to_string()),
                );
                attrs.insert(
                    "ops_kind".into(),
                    CustomCallAttr::I64Array(payload.ops_kind.clone()),
                );
                attrs.insert(
                    "ops_code".into(),
                    CustomCallAttr::I64Array(payload.ops_code.clone()),
                );
                attrs.insert("lhs".into(), CustomCallAttr::I64Array(payload.lhs.clone()));
                attrs.insert("rhs".into(), CustomCallAttr::I64Array(payload.rhs.clone()));
            }
            FusionDetail::DotEpilogue(payload) => {
                attrs.insert("fusion_version".into(), CustomCallAttr::I64(1));
                attrs.insert(
                    "fusion_kind".into(),
                    CustomCallAttr::String(FUSION_KIND_DOT_EPILOGUE_V1.to_string()),
                );
                attrs.insert(
                    "dot_batch_lhs".into(),
                    CustomCallAttr::I64Array(
                        payload.dot.batch_lhs.iter().map(|&v| v as i64).collect(),
                    ),
                );
                attrs.insert(
                    "dot_batch_rhs".into(),
                    CustomCallAttr::I64Array(
                        payload.dot.batch_rhs.iter().map(|&v| v as i64).collect(),
                    ),
                );
                attrs.insert(
                    "dot_contract_lhs".into(),
                    CustomCallAttr::I64Array(
                        payload.dot.contract_lhs.iter().map(|&v| v as i64).collect(),
                    ),
                );
                attrs.insert(
                    "dot_contract_rhs".into(),
                    CustomCallAttr::I64Array(
                        payload.dot.contract_rhs.iter().map(|&v| v as i64).collect(),
                    ),
                );
                attrs.insert(
                    "dot_add_input".into(),
                    CustomCallAttr::I64(payload.add_input as i64),
                );
            }
        }
        attrs
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct SelectedFusion {
    pub candidate: FusionCandidate,
    pub policy: HintPolicy,
    pub score: i64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FusionRejectReason {
    pub reason: String,
}

impl FusionRejectReason {
    pub fn new(reason: impl Into<String>) -> Self {
        Self {
            reason: reason.into(),
        }
    }
}
