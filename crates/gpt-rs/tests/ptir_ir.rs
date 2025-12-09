use std::{
    env, fs,
    path::PathBuf,
    time::{SystemTime, UNIX_EPOCH},
};

use gpt_rs::backend::spec::{Program, ProgramSerdeError, SPEC_VERSION};
use gpt_rs::ptir_program;

fn sample_program() -> Program {
    ptir_program!(
        r#"
func @single_result(%x: tensor<f32, 1x1>) -> tensor<f32, 1x1> {
  %sg = stop_gradient %x -> tensor<f32, 1x1>
  return %sg
}
"#
    )
}

fn unique_path(ext: &str) -> PathBuf {
    let mut path = env::temp_dir();
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time before unix epoch")
        .as_nanos();
    path.push(format!("gpt_rs_ptir_{timestamp}.{ext}"));
    path
}

#[test]
fn program_builder_emits_single_result_instructions() {
    let program = sample_program();
    let function = program
        .functions
        .first()
        .expect("sample program should contain a function");
    assert_eq!(function.body.len(), 1);
    let instruction = &function.body[0];
    assert_eq!(instruction.id, function.result_ids[0]);
    assert_eq!(instruction.output, function.results[0]);
}

#[test]
fn program_display_renders_ir() {
    let program = sample_program();
    let rendered = format!("{program}");
    assert!(
        rendered.contains("program @single_result"),
        "rendered IR missing program header:\n{rendered}"
    );
    assert!(
        rendered.contains("%1 = StopGradient(%0)"),
        "rendered IR missing instruction:\n{rendered}"
    );
    assert!(
        rendered.contains("tensor<F32 x 1x1>"),
        "rendered IR missing value type:\n{rendered}"
    );
}

#[test]
fn program_json_roundtrip_preserves_structure() {
    let program = sample_program();
    let json = program.to_json_string().expect("json serialization");
    let parsed = Program::from_json_str(&json).expect("json deserialization");
    assert_eq!(parsed, program);
}

#[test]
fn program_bincode_roundtrip_preserves_structure() {
    let program = sample_program();
    let bytes = program.to_bincode_bytes().expect("bincode serialization");
    let parsed = Program::from_bincode_slice(&bytes).expect("bincode deserialization");
    assert_eq!(parsed, program);
}

#[test]
fn program_json_missing_spec_version_defaults() {
    let program = sample_program();
    let mut value = serde_json::to_value(&program).expect("serialize to json value");
    value
        .as_object_mut()
        .expect("json object")
        .remove("spec_version");
    let json = serde_json::to_string_pretty(&value).expect("encode json");
    let parsed = Program::from_json_str(&json).expect("parsed without spec version");
    assert_eq!(parsed.spec_version, SPEC_VERSION);
}

#[test]
fn program_json_spec_version_mismatch_errors() {
    let program = sample_program();
    let mut value = serde_json::to_value(&program).expect("serialize to json value");
    value["spec_version"] = serde_json::Value::String("ptir.v999".to_string());
    let json = serde_json::to_string_pretty(&value).expect("encode json");
    let err = Program::from_json_str(&json).expect_err("expected spec version mismatch");
    match err {
        ProgramSerdeError::SpecVersionMismatch { found, expected } => {
            assert_eq!(found, "ptir.v999");
            assert_eq!(expected, SPEC_VERSION);
        }
        other => panic!("unexpected error: {other:?}"),
    }
}

#[test]
fn program_file_roundtrip_json_and_bincode() {
    let program = sample_program();
    let json_path = unique_path("json");
    let bin_path = unique_path("bin");

    program
        .save_json(&json_path)
        .expect("save json to disk succeeds");
    program
        .save_bincode(&bin_path)
        .expect("save bincode to disk succeeds");

    let from_json = Program::load_json(&json_path).expect("load json program");
    let from_bincode = Program::load_bincode(&bin_path).expect("load bincode program");

    assert_eq!(from_json, program);
    assert_eq!(from_bincode, program);

    let _ = fs::remove_file(json_path);
    let _ = fs::remove_file(bin_path);
}
