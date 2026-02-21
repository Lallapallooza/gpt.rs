use gpt_rs::backend::conversion::{ConversionError, ConversionResult};
use gpt_rs::backend::spec::{
    DType, Function, Operand, Operation, Program, ReduceKind, TensorSpec, ValueType,
};

use crate::bundle::{BundleStep, SerializableLiteral, TritonBundle};
use crate::kernels::elementwise_binary_kernel_spec;

pub fn lower_program_to_bundle(
    program: &Program,
    entrypoint_symbol: &str,
) -> ConversionResult<TritonBundle> {
    let function = entry_function(program)?;
    let mut bundle = TritonBundle::new(
        entrypoint_symbol.to_string(),
        function.parameter_ids.clone(),
        function.result_ids.clone(),
    );

    let mut elementwise_kernel_emitted = false;

    for instruction in &function.body {
        match &instruction.op {
            Operation::Constant(literal) => {
                let serialized = SerializableLiteral {
                    spec: literal.spec.clone(),
                    bytes: literal.bytes.as_ref().to_vec(),
                };
                bundle.steps.push(BundleStep::Constant {
                    value_id: instruction.id.0,
                    literal: serialized,
                });
            }
            Operation::StopGradient | Operation::Reshape(_) => {
                let source_id = operand_value_id(instruction.operands.first(), "alias source")?;
                let spec = output_tensor_spec(&instruction.output)?;
                bundle.steps.push(BundleStep::Alias {
                    value_id: instruction.id.0,
                    source_id,
                    spec,
                });
            }
            Operation::ElementwiseBinary(op) => {
                let lhs_id = operand_value_id(instruction.operands.first(), "elementwise lhs")?;
                let rhs_id = operand_value_id(instruction.operands.get(1), "elementwise rhs")?;
                let spec = output_tensor_spec(&instruction.output)?;
                if spec.dtype != DType::F32 {
                    return Err(ConversionError::new(format!(
                        "triton elementwise lowering supports F32 only, got {:?}",
                        spec.dtype
                    )));
                }
                if !elementwise_kernel_emitted {
                    bundle.kernels.push(elementwise_binary_kernel_spec());
                    elementwise_kernel_emitted = true;
                }
                bundle.steps.push(BundleStep::ElementwiseBinary {
                    value_id: instruction.id.0,
                    lhs_id,
                    rhs_id,
                    op: *op,
                    spec,
                    kernel_id: crate::kernels::EWISE_BINARY_KERNEL_ID.to_string(),
                });
            }
            Operation::DotGeneral(spec) => {
                let lhs_id = operand_value_id(instruction.operands.first(), "dot lhs")?;
                let rhs_id = operand_value_id(instruction.operands.get(1), "dot rhs")?;
                let lhs_spec = operand_tensor_spec(function, lhs_id)
                    .ok_or_else(|| ConversionError::new("failed to resolve dot lhs spec"))?;
                let rhs_spec = operand_tensor_spec(function, rhs_id)
                    .ok_or_else(|| ConversionError::new("failed to resolve dot rhs spec"))?;
                let out_spec = output_tensor_spec(&instruction.output)?;
                if lhs_spec.dtype != DType::F32
                    || rhs_spec.dtype != DType::F32
                    || out_spec.dtype != DType::F32
                {
                    return Err(ConversionError::new(
                        "triton dot lowering supports F32 only",
                    ));
                }
                bundle.steps.push(BundleStep::DotGeneral {
                    value_id: instruction.id.0,
                    lhs_id,
                    rhs_id,
                    lhs_spec,
                    rhs_spec,
                    out_spec,
                    spec: spec.clone(),
                });
            }
            Operation::Reduce(spec) => {
                let input_id = operand_value_id(instruction.operands.first(), "reduce input")?;
                let input_spec = operand_tensor_spec(function, input_id)
                    .ok_or_else(|| ConversionError::new("failed to resolve reduce input spec"))?;
                let out_spec = output_tensor_spec(&instruction.output)?;
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
                bundle.steps.push(BundleStep::Reduce {
                    value_id: instruction.id.0,
                    input_id,
                    input_spec,
                    out_spec,
                    spec: spec.clone(),
                });
            }
            other => {
                return Err(ConversionError::new(format!(
                    "triton lowering does not support operation: {:?}",
                    other
                )));
            }
        }
    }

    Ok(bundle)
}

fn entry_function(program: &Program) -> ConversionResult<&Function> {
    program
        .functions
        .iter()
        .find(|function| function.name == program.entry)
        .ok_or_else(|| ConversionError::new("entry function not found"))
}

fn output_tensor_spec(output: &ValueType) -> ConversionResult<TensorSpec> {
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

fn operand_tensor_spec(function: &Function, value_id: u32) -> Option<TensorSpec> {
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
