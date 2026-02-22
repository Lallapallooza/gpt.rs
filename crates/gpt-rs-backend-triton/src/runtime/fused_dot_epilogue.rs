use gpt_rs::backend::fusion::{FUSION_ATTR_KIND, FUSION_ATTR_VERSION, FUSION_KIND_DOT_EPILOGUE_V1};
use gpt_rs::backend::spec::{
    BackendError, BackendResult, CustomCallAttr, CustomCallSpec, DotGeneralSpec,
};

#[derive(Debug, Clone)]
pub struct FusedDotBiasPlan {
    pub dot: DotGeneralSpec,
    pub add_input: usize,
}

impl FusedDotBiasPlan {
    pub fn parse(spec: &CustomCallSpec, operand_count: usize) -> BackendResult<Self> {
        let version = custom_call_i64(spec, FUSION_ATTR_VERSION)?;
        if version != 1 {
            return Err(BackendError::execution(format!(
                "unsupported fused dot-epilogue payload version {version}"
            )));
        }
        let kind = custom_call_string(spec, FUSION_ATTR_KIND)?;
        if kind != FUSION_KIND_DOT_EPILOGUE_V1 {
            return Err(BackendError::execution(format!(
                "unsupported fused dot-epilogue payload kind '{kind}'"
            )));
        }

        let batch_lhs = i64_array_to_usize(custom_call_i64_array(spec, "dot_batch_lhs")?)?;
        let batch_rhs = i64_array_to_usize(custom_call_i64_array(spec, "dot_batch_rhs")?)?;
        let contract_lhs = i64_array_to_usize(custom_call_i64_array(spec, "dot_contract_lhs")?)?;
        let contract_rhs = i64_array_to_usize(custom_call_i64_array(spec, "dot_contract_rhs")?)?;
        let add_input_i64 = custom_call_i64(spec, "dot_add_input")?;
        let add_input = usize::try_from(add_input_i64)
            .map_err(|_| BackendError::execution("dot_add_input must be a non-negative integer"))?;
        if add_input >= operand_count {
            return Err(BackendError::execution(
                "dot_add_input index is out of range for fused dot+bias operands",
            ));
        }

        Ok(Self {
            dot: DotGeneralSpec {
                batch_lhs,
                batch_rhs,
                contract_lhs,
                contract_rhs,
                accum_dtype: None,
                out_dtype: None,
            },
            add_input,
        })
    }
}

fn custom_call_i64(spec: &CustomCallSpec, key: &str) -> BackendResult<i64> {
    match spec.attrs.get(key) {
        Some(CustomCallAttr::I64(value)) => Ok(*value),
        _ => Err(BackendError::execution(format!(
            "triton custom_call missing i64 attr '{key}'"
        ))),
    }
}

fn custom_call_string<'a>(spec: &'a CustomCallSpec, key: &str) -> BackendResult<&'a str> {
    match spec.attrs.get(key) {
        Some(CustomCallAttr::String(value)) => Ok(value.as_str()),
        _ => Err(BackendError::execution(format!(
            "triton custom_call missing string attr '{key}'"
        ))),
    }
}

fn custom_call_i64_array<'a>(spec: &'a CustomCallSpec, key: &str) -> BackendResult<&'a Vec<i64>> {
    match spec.attrs.get(key) {
        Some(CustomCallAttr::I64Array(values)) => Ok(values),
        _ => Err(BackendError::execution(format!(
            "triton custom_call missing i64 array attr '{key}'"
        ))),
    }
}

fn i64_array_to_usize(values: &[i64]) -> BackendResult<Vec<usize>> {
    values
        .iter()
        .map(|value| {
            usize::try_from(*value).map_err(|_| {
                BackendError::execution("dot-epilogue axis attributes must be non-negative")
            })
        })
        .collect()
}
