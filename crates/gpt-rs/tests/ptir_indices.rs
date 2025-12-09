use gpt_rs::{
    backend::{
        index::{FunctionIndexError, FunctionIndices, InstId},
        ptir_utils::{tensor_literal_f32_zeros, tensor_spec_static, value_type_tensor},
        spec::{DType, Function, Instruction, Operand, Operation, ValueId, ValueType},
    },
    ptir_program,
};

fn tensor_value_type(dtype: DType, dims: &[usize]) -> ValueType {
    value_type_tensor(tensor_spec_static(dtype, dims))
}

#[test]
fn function_indices_track_definitions_and_users() {
    const PROGRAM: &str = r#"
func @chain(%x: tensor<f32, 2x2>) -> tensor<f32, 2x2> {
  %sg0 = stop_gradient %x -> tensor<f32, 2x2>
  %sg1 = stop_gradient %sg0 -> tensor<f32, 2x2>
  return %sg1
}
"#;
    let function = ptir_program!(PROGRAM)
        .functions
        .into_iter()
        .next()
        .expect("program must define function");
    let tensor_ty = function.parameters[0].clone();
    let param = function.parameter_ids[0];
    let stop_grad = function.body[0].id;
    let second = function.body[1].id;

    let indices = FunctionIndices::build(&function).expect("indices build");

    assert_eq!(indices.position(InstId(0)), Some(0));
    assert_eq!(indices.position(InstId(1)), Some(1));
    assert_eq!(indices.value_of(InstId(0)), Some(stop_grad));
    assert_eq!(indices.value_of(InstId(1)), Some(second));
    assert_eq!(indices.inst_of(stop_grad), Some(InstId(0)));
    assert_eq!(indices.inst_of(second), Some(InstId(1)));
    assert_eq!(indices.users_of(param), &[InstId(0)]);
    assert_eq!(indices.users_of(stop_grad), &[InstId(1)]);
    assert!(indices.users_of(second).is_empty());
    assert_eq!(
        indices.type_of(param),
        Some(&tensor_ty),
        "parameter type lookup works"
    );
    assert_eq!(
        indices.type_of(stop_grad),
        Some(&tensor_ty),
        "instruction result type lookup works"
    );
    assert_eq!(indices.next_inst, 2);
    assert_eq!(indices.next_value, second.0 + 1);
}

#[test]
fn function_indices_enforce_duplicate_definition_error() {
    let tensor_ty = tensor_value_type(DType::F32, &[1, 1]);
    let instruction = Instruction {
        id: ValueId(0),
        op: Operation::StopGradient,
        operands: vec![],
        output: tensor_ty.clone(),
    };
    let function = Function {
        name: "dup".to_string(),
        parameters: vec![tensor_ty.clone()],
        parameter_ids: vec![ValueId(0)],
        results: vec![tensor_ty.clone()],
        body: vec![instruction.clone()],
        result_ids: vec![ValueId(0)],
    };

    let err = FunctionIndices::build(&function).expect_err("duplicate value should error");
    assert_eq!(
        err,
        FunctionIndexError::DuplicateValue { value: ValueId(0) }
    );
}

#[test]
fn function_indices_error_on_missing_operand_definition() {
    let tensor_ty = tensor_value_type(DType::F32, &[1, 1]);
    let instruction = Instruction {
        id: ValueId(1),
        op: Operation::StopGradient,
        operands: vec![Operand::Value(ValueId(42))],
        output: tensor_ty.clone(),
    };
    let function = Function {
        name: "missing_operand".to_string(),
        parameters: vec![tensor_ty.clone()],
        parameter_ids: vec![ValueId(0)],
        results: vec![tensor_ty],
        body: vec![instruction],
        result_ids: vec![ValueId(1)],
    };

    let err =
        FunctionIndices::build(&function).expect_err("missing operand definition should error");
    assert_eq!(
        err,
        FunctionIndexError::MissingValueDefinition { value: ValueId(42) }
    );
}

#[test]
fn function_indices_handle_tuple_operands_and_literals() {
    let tensor_ty = tensor_value_type(DType::F32, &[4]);
    let tuple_result = ValueId(1);
    let final_result = ValueId(2);

    let tuple_inst = Instruction {
        id: tuple_result,
        op: Operation::StopGradient,
        operands: vec![Operand::Value(ValueId(0))],
        output: tensor_ty.clone(),
    };

    let literal = tensor_literal_f32_zeros(&[1]);

    let consumer = Instruction {
        id: final_result,
        op: Operation::StopGradient,
        operands: vec![
            Operand::TupleElement {
                tuple: tuple_result,
                index: 0,
            },
            Operand::Literal(literal),
        ],
        output: tensor_ty.clone(),
    };

    let function = Function {
        name: "tuple_user".to_string(),
        parameters: vec![tensor_ty.clone()],
        parameter_ids: vec![ValueId(0)],
        results: vec![tensor_ty],
        body: vec![tuple_inst, consumer],
        result_ids: vec![final_result],
    };

    let indices = FunctionIndices::build(&function).expect("indices build");
    assert_eq!(indices.users_of(tuple_result), &[InstId(1)]);
    assert!(indices.users_of(final_result).is_empty());
}

#[test]
fn function_indices_error_when_result_missing_definition() {
    let tensor_ty = tensor_value_type(DType::F32, &[1]);
    let function = Function {
        name: "missing_result".to_string(),
        parameters: vec![tensor_ty.clone()],
        parameter_ids: vec![ValueId(0)],
        results: vec![tensor_ty],
        body: Vec::new(),
        result_ids: vec![ValueId(1)],
    };

    let err =
        FunctionIndices::build(&function).expect_err("missing result definition should error");
    assert_eq!(
        err,
        FunctionIndexError::MissingValueDefinition { value: ValueId(1) }
    );
}
