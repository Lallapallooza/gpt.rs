use std::sync::Arc;

use gpt_rs::backend::conversion::{ConversionOptions, ConversionTarget};
use gpt_rs::backend::pattern::OperationView;
use gpt_rs::backend::rewriter::ProgramRewriter;
use gpt_rs::backend::spec::{Function, Operation, Program};
use gpt_rs::module::Layer;
use gpt_rs::ops::functional;
use gpt_rs::ops::functional::activation::SoftmaxLastDimPattern;
use gpt_rs::tensor::{DeviceTensor, Shape, Tensor};
use gpt_rs_backend_tests::load_layer;
use gpt_rs_backend_tests::recording_backend::RecordingBackend;

use gpt_rs_backend_triton::TritonConversionTarget;

const DOT_BIAS_TARGET: &str = "gpt_rs.triton.fused_dot_bias.f32.v1";
const SOFTMAX_TARGET: &str = "gpt_rs.triton.fused_softmax_last_axis.f32.v1";

#[test]
fn linear_forward_lowers_to_dot_bias_custom_call() {
    let backend = Arc::new(RecordingBackend::default());
    let tensors = [
        ("fc.weight".to_string(), Tensor::ones(Shape::new([4, 3]))),
        ("fc.bias".to_string(), Tensor::zeros(Shape::new([4]))),
    ];
    let linear =
        load_layer(&backend, tensors, |p| p.linear("fc", 3, 4, true)).expect("linear init");
    let input = DeviceTensor::from_host(
        Arc::clone(&backend),
        Tensor::from_vec(Shape::new([2, 3]), vec![1.0; 6]).expect("input tensor"),
    )
    .expect("input upload");

    let out = linear.call(&input).expect("linear forward");
    out.materialize().expect("materialize linear output");

    let lowered = lower_recorded_program(&backend.recorded_program_or_panic());
    let entry = entry_function(&lowered);
    assert!(
        has_custom_call_target(entry, DOT_BIAS_TARGET),
        "expected lowered program to include {DOT_BIAS_TARGET}"
    );
}

#[test]
fn softmax_lowers_to_softmax_custom_call() {
    let backend = Arc::new(RecordingBackend::default());
    let input = DeviceTensor::from_host(
        Arc::clone(&backend),
        Tensor::from_vec(Shape::new([2, 4]), vec![1.0; 8]).expect("input tensor"),
    )
    .expect("input upload");

    let out = functional::softmax_last_dim(&input).expect("softmax");
    out.materialize().expect("materialize softmax output");

    let lowered = lower_recorded_program(&backend.recorded_program_or_panic());
    let entry = entry_function(&lowered);
    assert!(
        has_custom_call_target(entry, SOFTMAX_TARGET),
        "expected lowered program to include {SOFTMAX_TARGET}"
    );
}

#[test]
fn softmax_pattern_extracts_on_captured_graph() {
    let backend = Arc::new(RecordingBackend::default());
    let input = DeviceTensor::from_host(
        Arc::clone(&backend),
        Tensor::from_vec(Shape::new([2, 4]), vec![1.0; 8]).expect("input tensor"),
    )
    .expect("input upload");

    let out = functional::softmax_last_dim(&input).expect("softmax");
    out.materialize().expect("materialize softmax output");

    let recorded = backend.recorded_program_or_panic();
    let mut entry = entry_function(&recorded).clone();
    let rewriter = ProgramRewriter::new(&mut entry).expect("build rewriter");
    let matched = rewriter
        .insts_in_order()
        .into_iter()
        .any(|inst| SoftmaxLastDimPattern::extract(inst, &rewriter).is_some());
    assert!(
        matched,
        "expected SoftmaxLastDimPattern extractor to match captured softmax graph"
    );
}

#[test]
fn layer_norm_lowers_to_layer_norm_custom_call() {
    let backend = Arc::new(RecordingBackend::default());
    let tensors = [
        ("norm.weight".to_string(), Tensor::ones(Shape::new([4]))),
        ("norm.bias".to_string(), Tensor::zeros(Shape::new([4]))),
    ];
    let layer_norm =
        load_layer(&backend, tensors, |p| p.layer_norm("norm", 4, 1e-5)).expect("layer norm init");
    let input = DeviceTensor::from_host(
        Arc::clone(&backend),
        Tensor::from_vec(Shape::new([2, 4]), vec![0.25; 8]).expect("input tensor"),
    )
    .expect("input upload");

    let out = layer_norm.call(&input).expect("layer norm forward");
    out.materialize().expect("materialize layer norm output");

    let recorded = backend.recorded_program_or_panic();
    let lowered = lower_recorded_program(&recorded);
    let entry = entry_function(&lowered);
    let targets = custom_call_targets(entry);
    assert!(
        has_custom_call_target(entry, "gpt_rs.triton.fused_layer_norm.f32.v1"),
        "expected lowered program to include gpt_rs.triton.fused_layer_norm.f32.v1; custom calls present: {targets:?}",
    );
}

#[test]
fn attention_kv_cache_lowers_softmax_custom_call() {
    let backend = Arc::new(RecordingBackend::default());
    let (query_heads, kv_heads, head_dim, capacity) = (2usize, 2usize, 4usize, 4usize);
    let upload = |dims: &[usize], value: f32| {
        let len = dims.iter().product();
        DeviceTensor::from_host(
            Arc::clone(&backend),
            Tensor::from_vec(Shape::new(dims.to_vec()), vec![value; len]).expect("host tensor"),
        )
        .expect("tensor upload")
    };
    let q = upload(&[query_heads, 1, head_dim], 0.1);
    let k = upload(&[kv_heads, 1, head_dim], 0.1);
    let v = upload(&[kv_heads, 1, head_dim], 0.1);
    let cache = functional::DecodeKvCache::new(
        upload(&[kv_heads, capacity, head_dim], 0.0),
        upload(&[kv_heads, capacity, head_dim], 0.0),
        0,
    )
    .expect("kv cache");
    let update_starts = DeviceTensor::from_host(
        Arc::clone(&backend),
        Tensor::from_i32(Shape::new([3]), vec![0, 0, 0]).expect("update starts"),
    )
    .expect("update starts upload");

    let out = functional::attention_kv_cache(&q, &k, &v, &cache, &update_starts)
        .expect("kv-cache attention");
    out.output
        .materialize()
        .expect("materialize attention output");

    let lowered = lower_recorded_program(&backend.recorded_program_or_panic());
    let entry = entry_function(&lowered);
    let targets = custom_call_targets(entry);
    assert!(
        has_custom_call_target(entry, SOFTMAX_TARGET),
        "expected lowered kv-cache attention graph to include {SOFTMAX_TARGET}; custom calls present: {targets:?}",
    );
}

fn lower_recorded_program(program: &Program) -> Program {
    let target = TritonConversionTarget::new();
    let converted = target
        .convert(program, &ConversionOptions::default())
        .expect("triton conversion should succeed");
    let artifact_json: serde_json::Value =
        serde_json::from_str(&converted.module).expect("artifact json");
    serde_json::from_value(artifact_json["program"].clone()).expect("artifact program")
}

fn entry_function(program: &Program) -> &Function {
    program
        .functions
        .iter()
        .find(|function| function.name == program.entry)
        .expect("entry function must exist")
}

fn has_custom_call_target(function: &Function, target: &str) -> bool {
    function
        .body
        .iter()
        .any(|instruction| match &instruction.op {
            Operation::CustomCall(spec) => spec.target == target,
            _ => false,
        })
}

fn custom_call_targets(function: &Function) -> Vec<String> {
    function
        .body
        .iter()
        .filter_map(|instruction| match &instruction.op {
            Operation::CustomCall(spec) => Some(spec.target.clone()),
            _ => None,
        })
        .collect::<Vec<_>>()
}
