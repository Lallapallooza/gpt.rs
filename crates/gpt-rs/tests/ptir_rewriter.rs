use gpt_rs::{
    backend::{
        index::InstId,
        ptir_utils::{tensor_literal_zeros, tensor_spec_static, value_type_tensor},
        rewriter::ProgramRewriter,
        spec::{DType, Function, Operand, Operation},
    },
    ptir_program,
};

fn chain_function() -> Function {
    const PROGRAM: &str = r#"
func @chain(%x: tensor<f32, 2>) -> tensor<f32, 2> {
  %sg0 = stop_gradient %x -> tensor<f32, 2>
  %sg1 = stop_gradient %sg0 -> tensor<f32, 2>
  return %sg1
}
"#;
    ptir_program!(PROGRAM)
        .functions
        .into_iter()
        .next()
        .expect("program must define function")
}

#[test]
fn rewriter_replace_and_erase_instruction() {
    let mut function = chain_function();
    let param_id = function.parameter_ids[0];
    let first_value = function.body[0].id;
    let mut rewriter = ProgramRewriter::new(&mut function).expect("build indices");

    rewriter.replace_all_uses(first_value, param_id);
    assert!(rewriter.users_of(first_value).is_empty());
    let param_users = rewriter.users_of(param_id);
    assert!(
        param_users.contains(&InstId(1)),
        "param should feed second instruction"
    );
    assert_eq!(rewriter.version(InstId(1)), Some(1));

    rewriter
        .erase_inst(InstId(0))
        .expect("erase should succeed");
    assert_eq!(rewriter.func.body.len(), 1);
    assert_eq!(rewriter.users_of(param_id), &[InstId(1)]);
    assert_eq!(rewriter.version(InstId(1)), Some(1));

    assert_eq!(rewriter.func.body.len(), 1);
    assert_eq!(
        rewriter
            .func
            .body
            .first()
            .expect("instruction remains")
            .operands,
        vec![Operand::Value(param_id)]
    );
    assert!(rewriter.verify());
}

#[test]
fn rewriter_insert_and_materialize_constant() {
    let mut function = chain_function();
    let mut rewriter = ProgramRewriter::new(&mut function).expect("build indices");
    let const_spec = tensor_spec_static(DType::F32, &[1]);
    let const_ty = value_type_tensor(const_spec.clone());
    let literal = tensor_literal_zeros(const_spec);
    let literal_clone = literal.clone();

    let (const_inst, const_value) = rewriter
        .materialize_constant(InstId(0), literal, const_ty.clone())
        .expect("materialize constant");

    assert_eq!(
        rewriter.op(const_inst),
        &Operation::Constant(literal_clone),
        "constant op inserted"
    );
    assert_eq!(rewriter.value_of(const_inst), const_value);
    assert!(rewriter.users_of(const_value).is_empty());

    let (inserted_inst, inserted_value) = rewriter
        .insert_before(
            InstId(0),
            Operation::StopGradient,
            vec![Operand::Value(const_value)],
            const_ty.clone(),
        )
        .expect("insert rewrite");

    assert_eq!(rewriter.value_of(inserted_inst), inserted_value);
    assert_eq!(rewriter.users_of(const_value), &[inserted_inst]);
    assert!(rewriter.verify());
}
