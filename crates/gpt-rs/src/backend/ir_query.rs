use std::fmt;

use crate::backend::spec::{Function, Operand, TensorSpec, ValueType};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OperandTensorSpecError {
    MissingOperand,
    TupleElementUnsupported,
    ValueNotFound(u32),
    ValueNotTensor(u32),
}

impl fmt::Display for OperandTensorSpecError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingOperand => write!(f, "missing operand"),
            Self::TupleElementUnsupported => write!(f, "tuple element operands are unsupported"),
            Self::ValueNotFound(id) => write!(f, "value {id} not found in function"),
            Self::ValueNotTensor(id) => write!(f, "value {id} is not a tensor"),
        }
    }
}

pub fn tensor_spec_from_value_type(value: &ValueType) -> Option<TensorSpec> {
    match value {
        ValueType::Tensor(spec) => Some(spec.clone()),
        ValueType::Tuple(_) => None,
    }
}

pub fn tensor_spec_for_value(function: &Function, value_id: u32) -> Option<TensorSpec> {
    if let Some((idx, _)) = function
        .parameter_ids
        .iter()
        .enumerate()
        .find(|(_, id)| id.0 == value_id)
    {
        return function
            .parameters
            .get(idx)
            .and_then(tensor_spec_from_value_type);
    }

    function.body.iter().find_map(|instruction| {
        if instruction.id.0 != value_id {
            return None;
        }
        tensor_spec_from_value_type(&instruction.output)
    })
}

pub fn tensor_spec_for_operand(
    function: &Function,
    operand: Option<&Operand>,
) -> Result<TensorSpec, OperandTensorSpecError> {
    match operand {
        Some(Operand::Value(id)) => {
            tensor_spec_for_value(function, id.0).ok_or(OperandTensorSpecError::ValueNotFound(id.0))
        }
        Some(Operand::Literal(literal)) => Ok(literal.spec.clone()),
        Some(Operand::TupleElement { .. }) => Err(OperandTensorSpecError::TupleElementUnsupported),
        None => Err(OperandTensorSpecError::MissingOperand),
    }
}
