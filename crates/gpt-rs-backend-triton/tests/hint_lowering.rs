use std::collections::BTreeMap;

use gpt_rs::backend::conversion::{ConversionOptions, ConversionTarget};
use gpt_rs::backend::fusion::{FUSION_ATTR_KIND, FUSION_ATTR_VERSION, FUSION_KIND_DOT_EPILOGUE_V1};
use gpt_rs::backend::spec::{
    CustomCallAttr, DType, Dimension, DotGeneralSpec, Function, HintKind, HintPolicy, HintRegion,
    Instruction, Operand, Operation, Program, Shape, TensorSpec, ValueId, ValueType,
};

const DOT_BIAS_TARGET: &str = "gpt_rs.triton.fused_dot_bias.f32.v1";

#[test]
fn convert_lowers_dot_epilogue_hint_into_custom_call() {
    let spec_in = TensorSpec::new(
        DType::F32,
        Shape::new(vec![Dimension::from_usize(2), Dimension::from_usize(3)]),
    );
    let spec_w = TensorSpec::new(
        DType::F32,
        Shape::new(vec![Dimension::from_usize(3), Dimension::from_usize(4)]),
    );
    let spec_out = TensorSpec::new(
        DType::F32,
        Shape::new(vec![Dimension::from_usize(2), Dimension::from_usize(4)]),
    );

    let dot = Instruction {
        id: ValueId(3),
        op: Operation::DotGeneral(DotGeneralSpec {
            batch_lhs: vec![],
            batch_rhs: vec![],
            contract_lhs: vec![1],
            contract_rhs: vec![0],
            accum_dtype: None,
            out_dtype: None,
        }),
        operands: vec![Operand::Value(ValueId(0)), Operand::Value(ValueId(1))],
        output: ValueType::Tensor(spec_out.clone()),
    };
    let add = Instruction {
        id: ValueId(4),
        op: Operation::ElementwiseBinary(gpt_rs::backend::spec::ElementwiseBinaryOp::Add),
        operands: vec![Operand::Value(ValueId(3)), Operand::Value(ValueId(2))],
        output: ValueType::Tensor(spec_out.clone()),
    };
    let mut attrs = BTreeMap::new();
    attrs.insert(FUSION_ATTR_VERSION.to_string(), CustomCallAttr::I64(1));
    attrs.insert(
        FUSION_ATTR_KIND.to_string(),
        CustomCallAttr::String(FUSION_KIND_DOT_EPILOGUE_V1.to_string()),
    );
    attrs.insert(
        "dot_batch_lhs".to_string(),
        CustomCallAttr::I64Array(vec![]),
    );
    attrs.insert(
        "dot_batch_rhs".to_string(),
        CustomCallAttr::I64Array(vec![]),
    );
    attrs.insert(
        "dot_contract_lhs".to_string(),
        CustomCallAttr::I64Array(vec![1]),
    );
    attrs.insert(
        "dot_contract_rhs".to_string(),
        CustomCallAttr::I64Array(vec![0]),
    );
    attrs.insert("dot_add_input".to_string(), CustomCallAttr::I64(2));
    let function = Function {
        name: "main".to_string(),
        parameters: vec![
            ValueType::Tensor(spec_in),
            ValueType::Tensor(spec_w),
            ValueType::Tensor(spec_out.clone()),
        ],
        parameter_ids: vec![ValueId(0), ValueId(1), ValueId(2)],
        results: vec![ValueType::Tensor(spec_out.clone())],
        body: vec![dot.clone(), add.clone()],
        hints: vec![HintRegion {
            id: 1,
            kind: HintKind::DotEpilogue,
            policy: HintPolicy::Preferred,
            inputs: vec![ValueId(0), ValueId(1), ValueId(2)],
            exports: vec![ValueId(4)],
            body: vec![dot, add],
            attrs,
        }],
        result_ids: vec![ValueId(4)],
    };
    let program = Program::new("main").with_functions(vec![function]);

    let target = gpt_rs_backend_triton::TritonConversionTarget::new();
    let converted = target
        .convert(&program, &ConversionOptions::default())
        .expect("triton conversion should succeed");
    let artifact_json: serde_json::Value =
        serde_json::from_str(&converted.module).expect("artifact json");
    let lowered_program: Program =
        serde_json::from_value(artifact_json["program"].clone()).expect("artifact program");
    let entry_fn = lowered_program
        .functions
        .iter()
        .find(|func| func.name == lowered_program.entry)
        .expect("entry function");
    let has_dot_bias = entry_fn.body.iter().any(|inst| match &inst.op {
        Operation::CustomCall(spec) => spec.target == DOT_BIAS_TARGET,
        _ => false,
    });
    assert!(
        has_dot_bias,
        "expected dot+bias fused custom_call in lowered body"
    );
}

#[test]
fn required_unsupported_hint_fails_conversion() {
    let function = Function {
        name: "main".to_string(),
        parameters: vec![],
        parameter_ids: vec![],
        results: vec![],
        body: vec![],
        hints: vec![HintRegion {
            id: 7,
            kind: HintKind::ReductionChain,
            policy: HintPolicy::Required,
            inputs: vec![],
            exports: vec![],
            body: vec![],
            attrs: BTreeMap::new(),
        }],
        result_ids: vec![],
    };
    let program = Program::new("main").with_functions(vec![function]);

    let target = gpt_rs_backend_triton::TritonConversionTarget::new();
    let err = target
        .convert(&program, &ConversionOptions::default())
        .expect_err("required hint should fail conversion");
    assert!(
        err.to_string().contains("required hint"),
        "unexpected error: {err}"
    );
}
