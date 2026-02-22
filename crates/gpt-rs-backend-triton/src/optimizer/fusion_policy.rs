use std::collections::{HashMap, HashSet};

use gpt_rs::backend::fusion::{
    FusionCandidate, FusionDetail, FusionRejectReason, HintCostModel, HintLegalizer,
};
use gpt_rs::backend::optimizer::OptimizeContext;
use gpt_rs::backend::spec::{DType, Function, HintKind, HintPolicy, Operand, ValueType};

pub struct TritonHintLegalizer;

impl HintLegalizer<crate::TritonBackend> for TritonHintLegalizer {
    fn can_fuse(
        &self,
        function: &Function,
        candidate: &FusionCandidate,
        _cx: &OptimizeContext<crate::TritonBackend>,
    ) -> Result<HintPolicy, FusionRejectReason> {
        match candidate.kind {
            HintKind::ElementwiseDag | HintKind::DotEpilogue => {}
            HintKind::ReductionChain => {
                return Err(FusionRejectReason::new(
                    "reduction-chain fusion is not implemented for triton",
                ));
            }
        }

        if candidate.exports.len() != 1 {
            return Err(FusionRejectReason::new(
                "triton fusion currently requires exactly one export",
            ));
        }
        if candidate.inst_values.len() < 2 {
            return Err(FusionRejectReason::new(
                "triton fusion requires at least two instructions",
            ));
        }

        let specs = build_tensor_specs(function);
        for value in candidate.inputs.iter().chain(candidate.exports.iter()) {
            let spec = specs
                .get(value)
                .ok_or_else(|| FusionRejectReason::new("missing tensor type for fusion value"))?;
            if spec.dtype != DType::F32 {
                return Err(FusionRejectReason::new(
                    "triton fusion currently supports F32 tensors only",
                ));
            }
            if spec
                .shape
                .dims()
                .iter()
                .any(|dim| dim.as_static().is_none())
            {
                return Err(FusionRejectReason::new(
                    "triton fusion currently requires static shapes",
                ));
            }
        }

        let export_set = candidate.exports.iter().copied().collect::<HashSet<_>>();
        let inst_set = candidate
            .inst_values
            .iter()
            .copied()
            .collect::<HashSet<_>>();
        let users = collect_value_users(function);
        for value in &candidate.inst_values {
            if export_set.contains(value) {
                continue;
            }
            let external_uses = users
                .get(value)
                .map(|list| list.iter().filter(|user| !inst_set.contains(user)).count())
                .unwrap_or(0);
            if external_uses > 0 {
                return Err(FusionRejectReason::new(
                    "fusion internal value is used outside candidate",
                ));
            }
        }

        match &candidate.detail {
            FusionDetail::ElementwiseDag(payload) => {
                if payload.ops_kind.len() < 2 {
                    return Err(FusionRejectReason::new(
                        "elementwise fusion requires at least two nodes",
                    ));
                }
            }
            FusionDetail::DotEpilogue(payload) => {
                if candidate.inputs.len() < 3 {
                    return Err(FusionRejectReason::new(
                        "dot-epilogue fusion requires lhs/rhs/bias inputs",
                    ));
                }
                if payload.add_input >= candidate.inputs.len() {
                    return Err(FusionRejectReason::new(
                        "dot-epilogue add_input index is out of bounds",
                    ));
                }
            }
        }

        Ok(HintPolicy::Preferred)
    }
}

pub struct TritonHintCostModel;

impl HintCostModel<crate::TritonBackend> for TritonHintCostModel {
    fn score(
        &self,
        _function: &Function,
        candidate: &FusionCandidate,
        _cx: &OptimizeContext<crate::TritonBackend>,
    ) -> i64 {
        let launch_reduction = candidate.inst_values.len().saturating_sub(1) as i64;
        let temp_write_reduction = candidate
            .inst_values
            .len()
            .saturating_sub(candidate.exports.len()) as i64;
        let kind_bonus = match candidate.kind {
            HintKind::DotEpilogue => 60,
            HintKind::ElementwiseDag => 20,
            HintKind::ReductionChain => 0,
        };
        let size_penalty = if candidate.inst_values.len() > 8 {
            ((candidate.inst_values.len() - 8) as i64) * 10
        } else {
            0
        };
        launch_reduction * 100 + temp_write_reduction * 25 + kind_bonus - size_penalty
    }
}

fn build_tensor_specs(
    function: &Function,
) -> HashMap<gpt_rs::backend::spec::ValueId, gpt_rs::backend::spec::TensorSpec> {
    let mut out = HashMap::new();
    for (value_id, value_type) in function
        .parameter_ids
        .iter()
        .zip(function.parameters.iter())
    {
        if let ValueType::Tensor(spec) = value_type {
            out.insert(*value_id, spec.clone());
        }
    }
    for instruction in &function.body {
        if let ValueType::Tensor(spec) = &instruction.output {
            out.insert(instruction.id, spec.clone());
        }
    }
    out
}

fn collect_value_users(
    function: &Function,
) -> HashMap<gpt_rs::backend::spec::ValueId, Vec<gpt_rs::backend::spec::ValueId>> {
    let mut out = HashMap::<_, Vec<_>>::new();
    for instruction in &function.body {
        for operand in &instruction.operands {
            if let Operand::Value(value) = operand {
                out.entry(*value).or_default().push(instruction.id);
            }
        }
    }
    out
}

trait StaticDimExt {
    fn as_static(&self) -> Option<usize>;
}

impl StaticDimExt for gpt_rs::backend::spec::Dimension {
    fn as_static(&self) -> Option<usize> {
        match self {
            gpt_rs::backend::spec::Dimension::Static(v) => Some(*v),
            gpt_rs::backend::spec::Dimension::Dynamic(_) => None,
        }
    }
}
