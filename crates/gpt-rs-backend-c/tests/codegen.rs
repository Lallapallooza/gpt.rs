use gpt_rs::backend::conversion::{ConversionOptions, ConversionTarget};
use gpt_rs::backend::optimizer::{
    default_optimizer, EntryParam, EntrySignature, OptimizeConfig, OptimizeContext,
    OptimizeServices,
};
use gpt_rs::backend::ptir_utils::{tensor_spec_static, value_type_tensor};
use gpt_rs::backend::spec::{
    CastSpec, CompareSpec, ComparisonOp, DType, DotGeneralSpec, ElementwiseBinaryOp,
    ElementwiseUnaryOp, Function, Operand, Operation, PortableBackend, Program, ProgramBuilder,
    SliceSpec, TensorInit, TensorLiteral,
};
use gpt_rs::ptir_program;
use gpt_rs::tensor::InputRole;
use gpt_rs_backend_c::{CBackend, CConversionTarget};

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
    let module = convert(&ptir_program!(
        r#"
func @main(%a: tensor<f32, 2x3>, %b: tensor<f32, 3x4>) -> tensor<f32, 2x4> {
  %out = dot_general(%a, %b) contract_lhs[1] contract_rhs[0] -> tensor<f32, 2x4>
  return %out
}
"#
    ));
    // A weight that is an entry input keeps its packed copy across calls.
    assert!(
        module.contains("gpt_rs_c_matmul_f32((const float*)"),
        "{module}"
    );
    assert!(module.contains("&gpt_rs_bcache_"), "{module}");
}

fn convert(program: &Program) -> String {
    CConversionTarget::new()
        .convert(program, &ConversionOptions::default())
        .expect("convert succeeds")
        .module
}

/// Returns `program`'s entry function after the C backend optimizer pipeline. `param_inputs` bind
/// as model parameters with the given literals. The other inputs bind as call arguments.
fn optimize(program: &Program, param_inputs: &[(usize, TensorLiteral)]) -> Function {
    let backend = CBackend::new();
    let params = backend.param_resolver().expect("C backend resolves params");
    let mut function = program.functions[0].clone();
    let entry_params = function
        .parameter_ids
        .iter()
        .zip(&function.parameters)
        .enumerate()
        .map(|(index, (id, ty))| {
            let literal = param_inputs
                .iter()
                .find(|(param, _)| *param == index)
                .map(|(_, literal)| literal);
            let stable_id = literal.map(|literal| {
                let stable_id = 1 + index as u128;
                let handle = backend
                    .materialize(TensorInit::Literal(literal.clone()))
                    .expect("materialize param");
                params.set(stable_id, handle);
                stable_id
            });
            EntryParam {
                id: *id,
                ty: ty.clone(),
                role: if literal.is_some() {
                    InputRole::Param
                } else {
                    InputRole::Arg
                },
                stable_id,
            }
        })
        .collect();
    let services = OptimizeServices {
        params: Some(params.as_ref()),
    };
    let mut cx = OptimizeContext::new(
        &backend,
        services,
        EntrySignature::new(entry_params),
        OptimizeConfig::default(),
    );
    default_optimizer(backend.pipeline()).optimize(&mut function, &mut cx);
    function
}

fn custom_call_targets(function: &Function) -> Vec<&str> {
    function
        .body
        .iter()
        .filter_map(|inst| match &inst.op {
            Operation::CustomCall(spec) => Some(spec.target.as_str()),
            _ => None,
        })
        .collect()
}

#[test]
fn c_fusion_reads_slices_in_place_and_fuses_the_whole_chain() {
    // SwiGLU-style: out = gate * exp(gate) * up, with gate = x[:, :4] read twice and up = x[:, 4:].
    let full = tensor_spec_static(DType::F32, &[4, 8]);
    let half = value_type_tensor(tensor_spec_static(DType::F32, &[4, 4]));
    let mut builder = ProgramBuilder::new();
    let x = builder.add_parameter(value_type_tensor(full));
    let slice = |builder: &mut ProgramBuilder, start: usize| {
        builder.emit_single(
            Operation::Slice(SliceSpec {
                starts: vec![0, start],
                sizes: vec![4, 4],
            }),
            vec![Operand::Value(x)],
            half.clone(),
        )
    };
    let gate = slice(&mut builder, 0);
    let up = slice(&mut builder, 4);
    let exp = builder.emit_single(
        Operation::ElementwiseUnary(ElementwiseUnaryOp::Exp),
        vec![Operand::Value(gate)],
        half.clone(),
    );
    let activated = builder.emit_single(
        Operation::ElementwiseBinary(ElementwiseBinaryOp::Mul),
        vec![Operand::Value(gate), Operand::Value(exp)],
        half.clone(),
    );
    let out = builder.emit_single(
        Operation::ElementwiseBinary(ElementwiseBinaryOp::Mul),
        vec![Operand::Value(activated), Operand::Value(up)],
        half.clone(),
    );
    let function = builder.finish("main", vec![out]);
    let program = Program::new("main").with_functions(vec![function]);

    let optimized = optimize(&program, &[]);
    assert_eq!(
        custom_call_targets(&optimized),
        ["gpt_rs.c.fused_elementwise.v2"],
        "{optimized:#?}"
    );
    assert!(
        !optimized.body.iter().any(|inst| matches!(
            inst.op,
            Operation::Slice(_) | Operation::ElementwiseUnary(_) | Operation::ElementwiseBinary(_)
        )),
        "slices, exp and muls should all be read or computed by the fused kernel: {optimized:#?}"
    );
}

/// The loop a standalone bf16 -> f32 `cast` lowers to.
const WIDENING_CAST_LOOP: &str = "gpt_rs_bf16_to_f32(in[i])";
/// A call of the bf16-weight linear kernel. The kernel definition itself does not match it.
const LINEAR_BF16_CALL: &str = "gpt_rs_c_linear_nt_f32_bf16((const float*)";

/// `x: f32 [2, 4] . cast(w: bf16 -> f32)`, contracting `x` axis 1 with `w` axis `contract_rhs`.
/// With `return_cast`, the program also returns the widened weight.
fn f32_dot_of_widened_bf16_weight(
    w_dims: &[usize],
    contract_rhs: usize,
    return_cast: bool,
) -> Program {
    let mut builder = ProgramBuilder::new();
    let x = builder.add_parameter(value_type_tensor(tensor_spec_static(DType::F32, &[2, 4])));
    let w = builder.add_parameter(value_type_tensor(tensor_spec_static(DType::Bf16, w_dims)));
    let widened = builder.emit_single(
        Operation::Cast(CastSpec { dtype: DType::F32 }),
        vec![Operand::Value(w)],
        value_type_tensor(tensor_spec_static(DType::F32, w_dims)),
    );
    let n = w_dims[1 - contract_rhs];
    let out = builder.emit_single(
        Operation::DotGeneral(DotGeneralSpec {
            batch_lhs: vec![],
            batch_rhs: vec![],
            contract_lhs: vec![1],
            contract_rhs: vec![contract_rhs],
            accum_dtype: Some(DType::F32),
            out_dtype: Some(DType::F32),
        }),
        vec![Operand::Value(x), Operand::Value(widened)],
        value_type_tensor(tensor_spec_static(DType::F32, &[2, n])),
    );
    let results = if return_cast {
        vec![out, widened]
    } else {
        vec![out]
    };
    Program::new("main").with_functions(vec![builder.finish("main", results)])
}

#[test]
fn c_linear_reads_bf16_weights_without_widening_them() {
    let module = convert(&f32_dot_of_widened_bf16_weight(&[3, 4], 1, false));
    assert!(module.contains(LINEAR_BF16_CALL), "{module}");
    assert!(!module.contains(WIDENING_CAST_LOOP), "{module}");
}

#[test]
fn c_linear_reads_bf16_param_weights_before_param_folding() {
    // A model weight is a Param input. If the pipeline folded the widening cast into a derived
    // f32 Param first, the dot would lose its bf16 weight.
    let w_spec = tensor_spec_static(DType::Bf16, &[3, 4]);
    let w = TensorLiteral::new(w_spec, vec![0u8; 3 * 4 * 2].into());
    let optimized = optimize(
        &f32_dot_of_widened_bf16_weight(&[3, 4], 1, false),
        &[(1, w)],
    );
    assert_eq!(
        custom_call_targets(&optimized),
        ["gpt_rs.c.linear_nt.f32_bf16.v1"],
        "{optimized:#?}"
    );
}

#[test]
fn c_linear_keeps_a_widening_cast_that_has_other_users() {
    let module = convert(&f32_dot_of_widened_bf16_weight(&[3, 4], 1, true));
    assert!(module.contains(LINEAR_BF16_CALL), "{module}");
    assert!(module.contains(WIDENING_CAST_LOOP), "{module}");
}

#[test]
fn c_non_linear_dot_of_a_bf16_weight_keeps_the_widening_cast() {
    let module = convert(&f32_dot_of_widened_bf16_weight(&[4, 3], 0, false));
    assert!(module.contains(WIDENING_CAST_LOOP), "{module}");
    assert!(
        module.contains("gpt_rs_c_matmul_f32((const float*)"),
        "{module}"
    );
    assert!(!module.contains(LINEAR_BF16_CALL), "{module}");
}

#[test]
fn c_codegen_picks_dot_kernels_from_operand_strides() {
    let kernel_for = |lhs: &str, rhs: &str, attrs: &str, out: &str| {
        convert(&ptir_program!(&format!(
            r#"
func @main(%a: tensor<f32, {lhs}>, %b: tensor<f32, {rhs}>) -> tensor<f32, {out}> {{
  %out = dot_general(%a, %b) {attrs} -> tensor<f32, {out}>
  return %out
}}
"#
        )))
    };
    let linear = kernel_for("2x3", "4x3", "contract_lhs[1] contract_rhs[1]", "2x4");
    assert!(
        linear.contains("gpt_rs_c_linear_nt_f32((const float*)"),
        "{linear}"
    );
    // With enough rows, a linear layout uses the GEMM, which keeps its packed weight.
    let tall_linear = kernel_for("64x3", "4x3", "contract_lhs[1] contract_rhs[1]", "64x4");
    assert!(
        tall_linear.contains("gpt_rs_c_matmul_f32((const float*)"),
        "{tall_linear}"
    );
    assert!(tall_linear.contains("&gpt_rs_bcache_"), "{tall_linear}");
    // A single row multiplies b in place, so it gets no packed copy.
    let row = kernel_for("1x3", "3x4", "contract_lhs[1] contract_rhs[0]", "1x4");
    assert!(row.contains("gpt_rs_c_matmul_f32((const float*)"), "{row}");
    assert!(!row.contains("&gpt_rs_bcache_"), "{row}");
    let batched_linear = kernel_for(
        "2x3x4",
        "2x5x4",
        "batch[0] contract_lhs[2] contract_rhs[2]",
        "2x3x5",
    );
    assert!(
        batched_linear.contains("gpt_rs_c_linear_nt_f32((const float*)"),
        "{batched_linear}"
    );
    let batched = kernel_for(
        "2x4x3",
        "2x4x5",
        "batch[0] contract_lhs[1] contract_rhs[1]",
        "2x3x5",
    );
    assert!(
        batched.contains("gpt_rs_c_matmul_f32((const float*)"),
        "{batched}"
    );
}

#[test]
fn c_codegen_rejects_bf16_dots_without_a_kernel() {
    let program = ptir_program!(
        r#"
func @main(%a: tensor<bf16, 2x3x4>, %b: tensor<bf16, 2x4x5>) -> tensor<f32, 2x3x5> {
  %out = dot_general(%a, %b) batch[0] contract_lhs[2] contract_rhs[1] accum_dtype[f32] out_dtype[f32] -> tensor<f32, 2x3x5>
  return %out
}
"#
    );
    let err = CConversionTarget::new()
        .convert(&program, &ConversionOptions::default())
        .expect_err("no bf16 GEMM kernel");
    assert!(err.to_string().contains("no C kernel"), "{err}");
}

#[test]
fn c_codegen_rejects_out_of_range_dot_axes() {
    let program = ptir_program!(
        r#"
func @main(%a: tensor<f32, 2x3>, %b: tensor<f32, 3x4>) -> tensor<f32, 2x4> {
  %out = dot_general(%a, %b) contract_lhs[2] contract_rhs[0] -> tensor<f32, 2x4>
  return %out
}
"#
    );
    let err = CConversionTarget::new()
        .convert(&program, &ConversionOptions::default())
        .expect_err("contract axis 2 is out of range");
    assert!(err.to_string().contains("invalid for rank 2"), "{err}");
}

#[test]
fn c_codegen_shares_one_function_between_ops_that_differ_only_in_buffers() {
    let module = convert(&ptir_program!(
        r#"
func @main(%x: tensor<f32, 8x64>, %y: tensor<f32, 8x64>) -> tensor<f32, 8x1> {
  %a = reduce_sum(%x) axes[1] keepdims[true] -> tensor<f32, 8x1>
  %b = reduce_sum(%y) axes[1] keepdims[true] -> tensor<f32, 8x1>
  %c = add(%a, %b) -> tensor<f32, 8x1>
  return %c
}
"#
    ));
    let reduce_functions: Vec<&str> = module
        .split("static GPTRS_NOINLINE int ")
        .skip(1)
        .filter(|definition| definition.contains("acc +="))
        .map(|definition| &definition[..definition.find('(').expect("parameter list")])
        .collect();
    let [reduce] = reduce_functions.as_slice() else {
        panic!("expected one shared reduce function: {module}");
    };
    // One definition and a call per reduce.
    assert_eq!(module.matches(&format!("{reduce}(")).count(), 3, "{module}");
}

/// Runs `functional::conv2d` (1x3x3x2 NHWC input, 4 output channels, 3x3 kernel) on a recording
/// backend and returns the captured entry function after the C optimizer pipeline.
fn optimized_conv2d(bias: bool) -> Function {
    use gpt_rs::ops::functional::{self, Conv2dParams2d};
    use gpt_rs::tensor::{DeviceTensor, Shape, Tensor};
    use gpt_rs_backend_tests::recording_backend::RecordingBackend;
    use std::sync::Arc;

    let backend = Arc::new(RecordingBackend::default());
    let upload = |dims: &[usize]| {
        let len = dims.iter().product();
        DeviceTensor::from_host(
            Arc::clone(&backend),
            Tensor::from_vec(Shape::new(dims.to_vec()), vec![0.5; len]).expect("host tensor"),
        )
        .expect("upload")
    };
    let x = upload(&[1, 3, 3, 2]);
    let weight = upload(&[4, 2, 3, 3]);
    let bias_tensor = upload(&[4]);
    let out = functional::conv2d(
        &x,
        &weight,
        bias.then_some(&bias_tensor),
        Conv2dParams2d::square(3, 1, 1),
    )
    .expect("conv2d");
    out.materialize().expect("materialize conv2d");
    optimize(&backend.recorded_program_or_panic(), &[])
}

#[test]
fn c_conv2d_lowers_to_the_conv2d_kernel() {
    for bias in [false, true] {
        let optimized = optimized_conv2d(bias);
        assert_eq!(
            custom_call_targets(&optimized),
            ["gpt_rs.c.conv2d.nhwc.f32.v1"],
            "bias={bias}: {optimized:#?}"
        );
    }
}
