use gpt_rs::backend::conversion::ConversionOptions;
use gpt_rs::backend::spec::{
    DType, Dimension, Function, Program, Shape, TensorSpec, ValueId, ValueType,
};
use gpt_rs_backend_triton::TritonBackend;

fn tensor_value_type() -> ValueType {
    ValueType::Tensor(TensorSpec::new(
        DType::F32,
        Shape::new(vec![Dimension::Static(4)]),
    ))
}

#[test]
fn triton_artifact_preserves_full_buffer_plan_paths() {
    let tuple_type = ValueType::Tuple(vec![tensor_value_type(), tensor_value_type()]);
    let function = Function {
        name: "main".to_string(),
        parameters: vec![tuple_type.clone()],
        parameter_ids: vec![ValueId(0)],
        results: vec![tuple_type],
        body: Vec::new(),
        hints: Vec::new(),
        result_ids: vec![ValueId(0)],
    };
    let program = Program::new("main").with_functions(vec![function]);

    let backend = TritonBackend::new();
    let converted = backend
        .convert_for_execution(&program, &ConversionOptions::default())
        .expect("triton conversion should succeed for tuple-typed plan coverage");

    let module_json: serde_json::Value =
        serde_json::from_str(&converted.module).expect("converted artifact must be valid JSON");
    let buffers = module_json
        .pointer("/buffer_plan/functions/main/buffers")
        .and_then(serde_json::Value::as_array)
        .expect("buffer_plan must include serialized function buffers");

    let has_tuple_path = buffers.iter().any(|buffer| {
        buffer
            .get("path")
            .and_then(serde_json::Value::as_array)
            .is_some_and(|path| !path.is_empty())
    });
    assert!(
        has_tuple_path,
        "expected tuple-element buffer paths to be preserved in triton artifact"
    );
}
