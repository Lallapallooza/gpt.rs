use gpt_rs::backend::{
    index::InstId,
    pattern::{OperationKey, PatternTemplate, TemplateOperand, TemplateValueRef},
    rewriter::ProgramRewriter,
    spec::{
        DType, Dimension, ElementwiseBinaryOp, ElementwiseUnaryOp, Operand, Operation,
        ProgramBuilder, ReshapeDim, ReshapeSpec, Shape, TensorSpec, ValueId, ValueType,
    },
};
use gpt_rs::ptir_program;

fn rewriter_from_program(src: &str) -> ProgramRewriter<'static> {
    let program = ptir_program!(src);
    let function = program
        .functions
        .into_iter()
        .next()
        .expect("program must define a function");
    let function_ref: &'static mut _ = Box::leak(Box::new(function));
    ProgramRewriter::new(function_ref).expect("build rewriter")
}

fn rewriter_from_function(function: gpt_rs::backend::spec::Function) -> ProgramRewriter<'static> {
    let function_ref: &'static mut _ = Box::leak(Box::new(function));
    ProgramRewriter::new(function_ref).expect("build rewriter")
}

#[test]
fn template_match_binds_repeated_ops_and_inputs() {
    const PROGRAM: &str = r#"
func @mul_mul_add(%a: tensor<f32, 2>, %b: tensor<f32, 2>, %c: tensor<f32, 2>) -> tensor<f32, 2> {
  %mul1 = mul %a, %b -> tensor<f32, 2>
  %mul2 = mul %mul1, %c -> tensor<f32, 2>
  %add = add %mul2, %mul1 -> tensor<f32, 2>
  return %add
}
"#;

    let rewriter = rewriter_from_program(PROGRAM);

    let mul_key =
        OperationKey::from_operation(&Operation::ElementwiseBinary(ElementwiseBinaryOp::Mul));
    let add_key =
        OperationKey::from_operation(&Operation::ElementwiseBinary(ElementwiseBinaryOp::Add));

    let mut builder = PatternTemplate::builder();
    let mul1 = builder.node(
        mul_key,
        vec![
            TemplateOperand::Value(TemplateValueRef::Input(0)),
            TemplateOperand::Value(TemplateValueRef::Input(1)),
        ],
    );
    let mul2 = builder.node(
        mul_key,
        vec![
            TemplateOperand::Value(TemplateValueRef::Node(mul1)),
            TemplateOperand::Value(TemplateValueRef::Input(2)),
        ],
    );
    let add = builder.node(
        add_key,
        vec![
            TemplateOperand::Value(TemplateValueRef::Node(mul2)),
            TemplateOperand::Value(TemplateValueRef::Node(mul1)),
        ],
    );
    let template = builder.finish(mul1, TemplateValueRef::Node(add));

    let matched = template
        .match_from_anchor(InstId(0), &rewriter)
        .expect("expected template to match");

    assert_eq!(matched.inst(mul1), Some(InstId(0)));
    assert_eq!(matched.inst(mul2), Some(InstId(1)));
    assert_eq!(matched.inst(add), Some(InstId(2)));
    assert_eq!(matched.output, rewriter.value_of(InstId(2)));

    assert_eq!(matched.input(0), Some(ValueId(0)));
    assert_eq!(matched.input(1), Some(ValueId(1)));
    assert_eq!(matched.input(2), Some(ValueId(2)));
}

#[test]
fn template_match_commutative_add_accepts_swapped_operands() {
    const PROGRAM: &str = r#"
func @mul_mul_add_swapped(%a: tensor<f32, 2>, %b: tensor<f32, 2>, %c: tensor<f32, 2>) -> tensor<f32, 2> {
  %mul1 = mul %a, %b -> tensor<f32, 2>
  %mul2 = mul %mul1, %c -> tensor<f32, 2>
  %add = add %mul1, %mul2 -> tensor<f32, 2>
  return %add
}
"#;

    let rewriter = rewriter_from_program(PROGRAM);

    let mul_key =
        OperationKey::from_operation(&Operation::ElementwiseBinary(ElementwiseBinaryOp::Mul));
    let add_key =
        OperationKey::from_operation(&Operation::ElementwiseBinary(ElementwiseBinaryOp::Add));

    let mut builder = PatternTemplate::builder();
    let mul1 = builder.node(
        mul_key,
        vec![
            TemplateOperand::Value(TemplateValueRef::Input(0)),
            TemplateOperand::Value(TemplateValueRef::Input(1)),
        ],
    );
    let mul2 = builder.node(
        mul_key,
        vec![
            TemplateOperand::Value(TemplateValueRef::Node(mul1)),
            TemplateOperand::Value(TemplateValueRef::Input(2)),
        ],
    );
    // Template expects `add(mul2, mul1)`, but program is `add(mul1, mul2)`.
    let add = builder.node(
        add_key,
        vec![
            TemplateOperand::Value(TemplateValueRef::Node(mul2)),
            TemplateOperand::Value(TemplateValueRef::Node(mul1)),
        ],
    );
    let template = builder.finish(mul1, TemplateValueRef::Node(add));

    let matched = template
        .match_from_anchor(InstId(0), &rewriter)
        .expect("expected template to match despite operand swap");

    assert_eq!(matched.inst(mul1), Some(InstId(0)));
    assert_eq!(matched.inst(mul2), Some(InstId(1)));
    assert_eq!(matched.inst(add), Some(InstId(2)));
}

#[test]
fn template_match_skips_through_adapter_ops() {
    let spec = TensorSpec::new(
        DType::F32,
        Shape::new([Dimension::Static(2), Dimension::Static(2)]),
    );
    let ty = ValueType::Tensor(spec.clone());

    let mut builder = ProgramBuilder::new();
    let a = builder.add_parameter(ty.clone());
    let b = builder.add_parameter(ty.clone());
    let c = builder.add_parameter(ty.clone());

    let mul1 = builder.emit_single(
        Operation::ElementwiseBinary(ElementwiseBinaryOp::Mul),
        vec![Operand::Value(a), Operand::Value(b)],
        ty.clone(),
    );
    let reshape = builder.emit_single(
        Operation::Reshape(ReshapeSpec {
            new_shape: vec![
                ReshapeDim::Explicit(Dimension::Static(2)),
                ReshapeDim::Explicit(Dimension::Static(2)),
            ],
        }),
        vec![Operand::Value(mul1)],
        ty.clone(),
    );
    let add = builder.emit_single(
        Operation::ElementwiseBinary(ElementwiseBinaryOp::Add),
        vec![Operand::Value(reshape), Operand::Value(c)],
        ty.clone(),
    );

    let function = builder.finish("mul_reshape_add", vec![add]);
    let rewriter = rewriter_from_function(function);

    let mul_key =
        OperationKey::from_operation(&Operation::ElementwiseBinary(ElementwiseBinaryOp::Mul));
    let add_key =
        OperationKey::from_operation(&Operation::ElementwiseBinary(ElementwiseBinaryOp::Add));

    let mut template_builder = PatternTemplate::builder();
    let mul_node = template_builder.node(
        mul_key,
        vec![
            TemplateOperand::Value(TemplateValueRef::Input(0)),
            TemplateOperand::Value(TemplateValueRef::Input(1)),
        ],
    );
    let add_node = template_builder.node(
        add_key,
        vec![
            TemplateOperand::Value(TemplateValueRef::Node(mul_node)),
            TemplateOperand::Value(TemplateValueRef::Input(2)),
        ],
    );
    let template = template_builder.finish(mul_node, TemplateValueRef::Node(add_node));

    assert!(
        template.match_from_anchor(InstId(0), &rewriter).is_none(),
        "expected match to fail without skip-through"
    );

    let matched = template
        .match_from_anchor_with_config(
            InstId(0),
            &rewriter,
            gpt_rs::backend::pattern::MatchConfig {
                allow_commutative_binops: true,
                max_skip_through_depth: 1,
            },
        )
        .expect("expected match to succeed with skip-through");

    assert_eq!(matched.inst(mul_node), Some(InstId(0)));
    assert_eq!(matched.inst(add_node), Some(InstId(2)));
}

#[test]
fn template_mismatch_on_op_kind() {
    const PROGRAM: &str = r#"
func @mul_mul_add(%a: tensor<f32, 2>, %b: tensor<f32, 2>, %c: tensor<f32, 2>) -> tensor<f32, 2> {
  %mul1 = mul %a, %b -> tensor<f32, 2>
  %mul2 = mul %mul1, %c -> tensor<f32, 2>
  %add = add %mul2, %mul1 -> tensor<f32, 2>
  return %add
}
"#;

    let rewriter = rewriter_from_program(PROGRAM);

    let mul_key =
        OperationKey::from_operation(&Operation::ElementwiseBinary(ElementwiseBinaryOp::Mul));
    let add_key =
        OperationKey::from_operation(&Operation::ElementwiseBinary(ElementwiseBinaryOp::Add));

    let mut builder = PatternTemplate::builder();
    let mul1 = builder.node(
        mul_key,
        vec![
            TemplateOperand::Value(TemplateValueRef::Input(0)),
            TemplateOperand::Value(TemplateValueRef::Input(1)),
        ],
    );
    // Wrong: expect add here, but program has mul.
    let bad = builder.node(
        add_key,
        vec![
            TemplateOperand::Value(TemplateValueRef::Node(mul1)),
            TemplateOperand::Value(TemplateValueRef::Input(2)),
        ],
    );
    let template = builder.finish(mul1, TemplateValueRef::Node(bad));

    assert!(
        template.match_from_anchor(InstId(0), &rewriter).is_none(),
        "expected mismatch when op kind differs"
    );
}

#[test]
fn closure_report_finds_external_users_of_internal_nodes() {
    const PROGRAM: &str = r#"
func @mul_mul_add_external(%a: tensor<f32, 2>, %b: tensor<f32, 2>, %c: tensor<f32, 2>) -> tensor<f32, 2> {
  %mul1 = mul %a, %b -> tensor<f32, 2>
  %mul2 = mul %mul1, %c -> tensor<f32, 2>
  %add = add %mul2, %mul1 -> tensor<f32, 2>
  %other = exp %mul1 -> tensor<f32, 2>
  return %add
}
"#;

    let rewriter = rewriter_from_program(PROGRAM);

    let mul_key =
        OperationKey::from_operation(&Operation::ElementwiseBinary(ElementwiseBinaryOp::Mul));
    let add_key =
        OperationKey::from_operation(&Operation::ElementwiseBinary(ElementwiseBinaryOp::Add));

    let mut builder = PatternTemplate::builder();
    let mul1 = builder.node(
        mul_key,
        vec![
            TemplateOperand::Value(TemplateValueRef::Input(0)),
            TemplateOperand::Value(TemplateValueRef::Input(1)),
        ],
    );
    let mul2 = builder.node(
        mul_key,
        vec![
            TemplateOperand::Value(TemplateValueRef::Node(mul1)),
            TemplateOperand::Value(TemplateValueRef::Input(2)),
        ],
    );
    let add = builder.node(
        add_key,
        vec![
            TemplateOperand::Value(TemplateValueRef::Node(mul2)),
            TemplateOperand::Value(TemplateValueRef::Node(mul1)),
        ],
    );
    let template = builder.finish(mul1, TemplateValueRef::Node(add));

    let matched = template
        .match_from_anchor(InstId(0), &rewriter)
        .expect("expected template to match");

    let report = matched.closure_report(&rewriter);
    assert!(
        !report.is_closed(),
        "expected match to be non-closed due to external exp user"
    );
    assert_eq!(report.external_uses.len(), 1);
    assert_eq!(report.external_uses[0].value, rewriter.value_of(InstId(0)));
    assert_eq!(report.external_uses[0].users, vec![InstId(3)]);

    assert!(matches!(
        rewriter.op(InstId(3)),
        Operation::ElementwiseUnary(ElementwiseUnaryOp::Exp)
    ));
}
