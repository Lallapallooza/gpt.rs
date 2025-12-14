use gpt_rs::backend::conversion::{ConversionOptions, ConversionTarget};
use gpt_rs::backend::ptir_utils::{tensor_spec_static, value_type_tensor};
use gpt_rs::backend::spec::{
    CompareSpec, ComparisonOp, DType, Operand, Operation, Program, ProgramBuilder,
};
use gpt_rs::ptir_program;
use gpt_rs_backend_c::CConversionTarget;

#[test]
fn c_codegen_emits_broadcast_loops() {
    let program = ptir_program!(
        r#"
func @main(%x: tensor<f32, 1x3>) -> tensor<f32, 2x3> {
  %b = broadcast_to(%x) shape[2, 3] -> tensor<f32, 2x3>
  return %b
}
"#
    );
    let target = CConversionTarget::new();
    let ir = target
        .convert(&program, &ConversionOptions::default())
        .expect("convert succeeds");
    assert!(ir.module.contains("i0 < 2"));
    assert!(ir.module.contains("i1 < 3"));
}

#[test]
fn c_codegen_emits_reduce_sum() {
    let program = ptir_program!(
        r#"
func @main(%x: tensor<f32, 2x3>) -> tensor<f32, 1x3> {
  %r = reduce_sum(%x) axes[0] keepdims[true] -> tensor<f32, 1x3>
  return %r
}
"#
    );
    let target = CConversionTarget::new();
    let ir = target
        .convert(&program, &ConversionOptions::default())
        .expect("convert succeeds");
    assert!(ir.module.contains("gpt_rs_c_matmul_f32") || ir.module.contains("acc +="));
}

#[test]
fn c_codegen_emits_compare() {
    let spec = tensor_spec_static(DType::F32, &[4]);
    let mut builder = ProgramBuilder::new();
    let lhs = builder.add_parameter(value_type_tensor(spec.clone()));
    let rhs = builder.add_parameter(value_type_tensor(spec.clone()));
    let out = builder.emit_single(
        Operation::Compare(CompareSpec {
            op: ComparisonOp::Less,
        }),
        vec![Operand::Value(lhs), Operand::Value(rhs)],
        value_type_tensor(tensor_spec_static(DType::I1, &[4])),
    );
    let function = builder.finish("main", vec![out]);
    let program = Program::new("main").with_functions(vec![function]);

    let target = CConversionTarget::new();
    let ir = target
        .convert(&program, &ConversionOptions::default())
        .expect("convert succeeds");
    assert!(ir.module.contains("uint8_t* out"));
    assert!(ir.module.contains("lhs[i] < rhs[i]"));
}

#[test]
fn c_codegen_emits_dot_general() {
    let program = ptir_program!(
        r#"
func @main(%a: tensor<f32, 2x3>, %b: tensor<f32, 3x4>) -> tensor<f32, 2x4> {
  %out = dot_general(%a, %b) contract_lhs[1] contract_rhs[0] -> tensor<f32, 2x4>
  return %out
}
"#
    );
    let target = CConversionTarget::new();
    let ir = target
        .convert(&program, &ConversionOptions::default())
        .expect("convert succeeds");
    assert!(ir.module.contains("gpt_rs_c_matmul_f32"));
}
