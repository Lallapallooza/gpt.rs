use std::sync::Arc;

use gpt_rs::backend::ir_query::{self, OperandTensorSpecError};
use gpt_rs::backend::spec::{
    DType, Dimension, ElementwiseUnaryOp, Function, Instruction, Operand, Operation, Shape,
    TensorLiteral, TensorSpec, ValueId, ValueType,
};
use gpt_rs::backend::topology;

fn f32_vector_spec(len: usize) -> TensorSpec {
    TensorSpec::new(DType::F32, Shape::new(vec![Dimension::from_usize(len)]))
}

fn simple_function() -> Function {
    let spec = f32_vector_spec(4);
    Function {
        name: "main".to_string(),
        parameters: vec![ValueType::Tensor(spec.clone())],
        parameter_ids: vec![ValueId(0)],
        results: vec![ValueType::Tensor(spec.clone())],
        body: vec![Instruction {
            id: ValueId(1),
            op: Operation::ElementwiseUnary(ElementwiseUnaryOp::Neg),
            operands: vec![Operand::Value(ValueId(0))],
            output: ValueType::Tensor(spec),
        }],
        hints: Vec::new(),
        result_ids: vec![ValueId(1)],
    }
}

#[test]
fn ir_query_resolves_operand_tensor_specs() {
    let function = simple_function();
    let spec = f32_vector_spec(4);

    let from_value =
        ir_query::tensor_spec_for_operand(&function, Some(&Operand::Value(ValueId(0))));
    assert_eq!(from_value.expect("value operand spec"), spec);

    let literal = TensorLiteral::new(spec.clone(), Arc::from(vec![0u8; 16]));
    let from_literal =
        ir_query::tensor_spec_for_operand(&function, Some(&Operand::Literal(literal)));
    assert_eq!(from_literal.expect("literal operand spec"), spec);
}

#[test]
fn ir_query_rejects_tuple_operands() {
    let function = simple_function();
    let err = ir_query::tensor_spec_for_operand(
        &function,
        Some(&Operand::TupleElement {
            tuple: ValueId(0),
            index: 0,
        }),
    )
    .expect_err("tuple element should be rejected");
    assert_eq!(err, OperandTensorSpecError::TupleElementUnsupported);
}

#[test]
fn topology_validator_accepts_topological_function() {
    let function = simple_function();
    topology::validate_function_topology(&function).expect("topological function");
}

#[test]
fn topology_validator_rejects_missing_dependency() {
    let spec = f32_vector_spec(4);
    let function = Function {
        name: "main".to_string(),
        parameters: vec![ValueType::Tensor(spec.clone())],
        parameter_ids: vec![ValueId(0)],
        results: vec![ValueType::Tensor(spec.clone())],
        body: vec![
            Instruction {
                id: ValueId(1),
                op: Operation::ElementwiseUnary(ElementwiseUnaryOp::Neg),
                operands: vec![Operand::Value(ValueId(2))],
                output: ValueType::Tensor(spec.clone()),
            },
            Instruction {
                id: ValueId(2),
                op: Operation::ElementwiseUnary(ElementwiseUnaryOp::Abs),
                operands: vec![Operand::Value(ValueId(0))],
                output: ValueType::Tensor(spec.clone()),
            },
        ],
        hints: Vec::new(),
        result_ids: vec![ValueId(1)],
    };

    let err = topology::validate_function_topology(&function).expect_err("dependency failure");
    assert_eq!(err.missing_value, 2);
    assert_eq!(err.instruction_id, 1);
}
