use gpt_rs::backend::conversion::{ConversionError, ConversionResult};
use gpt_rs::backend::spec::{DType, Function, Operand, Operation, Program, ReduceKind, ValueType};

use crate::artifact::TritonArtifact;
use crate::kernels::elementwise_binary_kernel_spec;

pub fn lower_program_to_artifact(
    program: &Program,
    entrypoint_symbol: &str,
) -> ConversionResult<TritonArtifact> {
    let function = entry_function(program)?;

    let mut has_elementwise_binary = false;
    for instruction in &function.body {
        validate_instruction(function, instruction)?;
        if matches!(instruction.op, Operation::ElementwiseBinary(_)) {
            has_elementwise_binary = true;
        }
    }

    let mut kernels = Vec::new();
    if has_elementwise_binary {
        kernels.push(elementwise_binary_kernel_spec());
    }

    Ok(TritonArtifact::new(
        entrypoint_symbol.to_string(),
        program.clone(),
        kernels,
    ))
}

fn validate_instruction(
    function: &Function,
    instruction: &gpt_rs::backend::spec::Instruction,
) -> ConversionResult<()> {
    match &instruction.op {
        Operation::Constant(literal) => {
            if literal.spec.dtype != DType::F32 {
                return Err(ConversionError::new(format!(
                    "triton constant lowering supports F32 only, got {:?}",
                    literal.spec.dtype
                )));
            }
            ensure_tensor_output(&instruction.output)?;
            Ok(())
        }
        Operation::StopGradient | Operation::Reshape(_) => {
            let _ = operand_value_id(instruction.operands.first(), "alias source")?;
            ensure_tensor_output(&instruction.output)?;
            Ok(())
        }
        Operation::ElementwiseBinary(_) => {
            let lhs_id = operand_value_id(instruction.operands.first(), "elementwise lhs")?;
            let rhs_id = operand_value_id(instruction.operands.get(1), "elementwise rhs")?;
            let lhs_spec = operand_tensor_spec(function, lhs_id)
                .ok_or_else(|| ConversionError::new("failed to resolve elementwise lhs spec"))?;
            let rhs_spec = operand_tensor_spec(function, rhs_id)
                .ok_or_else(|| ConversionError::new("failed to resolve elementwise rhs spec"))?;
            let out_spec = ensure_tensor_output(&instruction.output)?;
            if lhs_spec.dtype != DType::F32
                || rhs_spec.dtype != DType::F32
                || out_spec.dtype != DType::F32
            {
                return Err(ConversionError::new(
                    "triton elementwise lowering supports F32 only",
                ));
            }
            if lhs_spec != out_spec || rhs_spec != out_spec {
                return Err(ConversionError::new(
                    "triton elementwise lowering requires equal tensor specs for lhs/rhs/out",
                ));
            }
            Ok(())
        }
        Operation::DotGeneral(spec) => {
            let lhs_id = operand_value_id(instruction.operands.first(), "dot lhs")?;
            let rhs_id = operand_value_id(instruction.operands.get(1), "dot rhs")?;
            let lhs_spec = operand_tensor_spec(function, lhs_id)
                .ok_or_else(|| ConversionError::new("failed to resolve dot lhs spec"))?;
            let rhs_spec = operand_tensor_spec(function, rhs_id)
                .ok_or_else(|| ConversionError::new("failed to resolve dot rhs spec"))?;
            let out_spec = ensure_tensor_output(&instruction.output)?;

            if lhs_spec.dtype != DType::F32
                || rhs_spec.dtype != DType::F32
                || out_spec.dtype != DType::F32
            {
                return Err(ConversionError::new(
                    "triton dot lowering supports F32 only",
                ));
            }
            if !spec.batch_lhs.is_empty()
                || !spec.batch_rhs.is_empty()
                || spec.contract_lhs.as_slice() != [1]
                || spec.contract_rhs.as_slice() != [0]
            {
                return Err(ConversionError::new(
                    "triton dot lowering supports rank-2 MxK · KxN only",
                ));
            }
            Ok(())
        }
        Operation::Reduce(spec) => {
            let input_id = operand_value_id(instruction.operands.first(), "reduce input")?;
            let input_spec = operand_tensor_spec(function, input_id)
                .ok_or_else(|| ConversionError::new("failed to resolve reduce input spec"))?;
            let out_spec = ensure_tensor_output(&instruction.output)?;

            if spec.kind != ReduceKind::Sum {
                return Err(ConversionError::new(
                    "triton reduce lowering supports sum only",
                ));
            }
            if input_spec.dtype != DType::F32 || out_spec.dtype != DType::F32 {
                return Err(ConversionError::new(
                    "triton reduce lowering supports F32 only",
                ));
            }
            Ok(())
        }
        other => Err(ConversionError::new(format!(
            "triton lowering does not support operation: {:?}",
            other
        ))),
    }
}

fn entry_function(program: &Program) -> ConversionResult<&Function> {
    program
        .functions
        .iter()
        .find(|function| function.name == program.entry)
        .ok_or_else(|| ConversionError::new("entry function not found"))
}

fn ensure_tensor_output(output: &ValueType) -> ConversionResult<gpt_rs::backend::spec::TensorSpec> {
    match output {
        ValueType::Tensor(spec) => Ok(spec.clone()),
        ValueType::Tuple(_) => Err(ConversionError::new(
            "tuple outputs are not supported by triton lowering",
        )),
    }
}

fn operand_value_id(operand: Option<&Operand>, what: &str) -> ConversionResult<u32> {
    match operand {
        Some(Operand::Value(id)) => Ok(id.0),
        Some(Operand::TupleElement { .. }) => Err(ConversionError::new(format!(
            "{what} uses tuple element operands, unsupported in triton lowering"
        ))),
        Some(Operand::Literal(_)) => Err(ConversionError::new(format!(
            "{what} uses inline literal operands, unsupported in triton lowering"
        ))),
        None => Err(ConversionError::new(format!("missing operand for {what}"))),
    }
}

fn operand_tensor_spec(
    function: &Function,
    value_id: u32,
) -> Option<gpt_rs::backend::spec::TensorSpec> {
    if let Some((idx, _)) = function
        .parameter_ids
        .iter()
        .enumerate()
        .find(|(_, id)| id.0 == value_id)
    {
        return match function.parameters.get(idx) {
            Some(ValueType::Tensor(spec)) => Some(spec.clone()),
            _ => None,
        };
    }

    function.body.iter().find_map(|instruction| {
        if instruction.id.0 != value_id {
            return None;
        }
        match &instruction.output {
            ValueType::Tensor(spec) => Some(spec.clone()),
            ValueType::Tuple(_) => None,
        }
    })
}
