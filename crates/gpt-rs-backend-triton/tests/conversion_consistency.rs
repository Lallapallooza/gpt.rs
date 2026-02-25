use std::sync::Arc;

use gpt_rs::backend::conversion::{ConversionOptions, ConversionTarget};
use gpt_rs::ops::functional;
use gpt_rs::tensor::{DeviceTensor, Shape, Tensor};
use gpt_rs_backend_tests::recording_backend::RecordingBackend;
use gpt_rs_backend_triton::{TritonBackend, TritonConversionTarget};

#[test]
fn cli_and_execution_conversion_paths_match() {
    let recorder = Arc::new(RecordingBackend::default());
    let input = DeviceTensor::from_host(
        Arc::clone(&recorder),
        Tensor::from_vec(Shape::new([2, 4]), vec![1.0f32; 8]).expect("input tensor"),
    )
    .expect("input upload");
    let out = functional::softmax_last_dim(recorder.as_ref(), &input).expect("softmax");
    out.materialize().expect("materialize");
    let recorded = recorder.recorded_program_or_panic();

    let options = ConversionOptions::default();
    let target = TritonConversionTarget::new();
    let cli_converted = target
        .convert(&recorded, &options)
        .expect("target conversion should succeed");
    let backend = TritonBackend::new();
    let exec_converted = backend
        .convert_for_execution(&recorded, &options)
        .expect("backend conversion should succeed");

    let cli_json: serde_json::Value =
        serde_json::from_str(&cli_converted.module).expect("cli artifact json");
    let exec_json: serde_json::Value =
        serde_json::from_str(&exec_converted.module).expect("exec artifact json");
    assert_eq!(
        cli_json, exec_json,
        "triton conversion artifacts diverged between CLI and execution paths"
    );
}
