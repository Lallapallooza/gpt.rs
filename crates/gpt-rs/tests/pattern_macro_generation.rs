use std::sync::Arc;

use anyhow::Result;
use gpt_rs::backend::pattern::{all_pattern_defs, OperationView};
use gpt_rs::backend::rewriter::ProgramRewriter;
use gpt_rs::backend::spec::{ElementwiseBinaryOp, Function, Operation, PortableBackend, Program};
use gpt_rs::ops::functional::CaptureIntoDeviceTensor;
use gpt_rs::ops::ptir;
use gpt_rs::tensor::{DeviceTensor, Shape, Tensor};
use gpt_rs_backend_tests::recording_backend::RecordingBackend;
use gpt_rs_macros::{capture_ptir, ptir_pattern};

fn softmax_last_axis_helper<'ctx, 'gb, B: PortableBackend + 'static>(
    input: &ptir::Tensor<'ctx, 'gb, B>,
    axis: usize,
) -> ptir::Tensor<'ctx, 'gb, B> {
    let max = input.reduce_max([axis], true);
    let shifted = *input - max.broadcast_like(input);
    let exp_values = shifted.exp();
    let sum = exp_values.reduce_sum([axis], true);
    exp_values / sum.broadcast_like(input)
}

#[ptir_pattern(target = "gpt_rs.test.helper_softmax", anchor = output)]
fn helper_softmax<B: PortableBackend + 'static>(
    _backend: &B,
    x: &DeviceTensor<B>,
) -> Result<DeviceTensor<B>> {
    let axis = x.shape().rank() - 1;
    capture_ptir!({ input = x }, |_session| {
        let output = softmax_last_axis_helper(&input, axis);
        Ok(output.id())
    })?
    .into_device_tensor()
}

#[test]
fn helper_softmax_pattern_extracts_on_captured_graph() {
    let backend = Arc::new(RecordingBackend::default());
    let input = DeviceTensor::from_host(
        Arc::clone(&backend),
        Tensor::from_vec(Shape::new([2, 4]), vec![1.0; 8]).expect("input tensor"),
    )
    .expect("input upload");

    let out = helper_softmax(backend.as_ref(), &input).expect("helper softmax");
    out.materialize().expect("materialize output");

    let recorded = backend.recorded_program_or_panic();
    let mut entry = entry_function(&recorded).clone();
    let rewriter = ProgramRewriter::new(&mut entry).expect("build rewriter");
    let view = rewriter
        .insts_in_order()
        .into_iter()
        .find_map(|inst| HelperSoftmaxPattern::extract(inst, &rewriter));
    assert!(
        view.is_some(),
        "expected HelperSoftmaxPattern extractor to match helper-based softmax graph"
    );
    let view = view.expect("expected softmax helper pattern view");
    assert!(
        matches!(
            view.output.op,
            Operation::ElementwiseBinary(ElementwiseBinaryOp::Div)
        ),
        "expected helper-call bound output to resolve as an AnyOpView wrapping div op"
    );
}

#[test]
fn helper_softmax_pattern_registers_any_view_field() {
    let defs = all_pattern_defs();
    let Some(def) = defs
        .iter()
        .find(|def| def.target == "gpt_rs.test.helper_softmax")
    else {
        panic!("expected gpt_rs.test.helper_softmax pattern definition to be registered");
    };
    let fields = def
        .fields
        .iter()
        .map(|field| (field.name, field.view))
        .collect::<Vec<_>>();
    assert!(
        fields.contains(&("output", "AnyOpView")),
        "expected helper softmax pattern to register output as AnyOpView; fields: {:?}",
        fields,
    );
    assert!(
        fields.len() == 1,
        "expected helper softmax pattern to bind helper-call output only; fields: {:?}",
        fields,
    );
}

#[test]
fn builtin_softmax_pattern_registers_helper_call_output_field() {
    let defs = all_pattern_defs();
    let Some(def) = defs
        .iter()
        .find(|def| def.target == "gpt_rs.softmax_last_dim_f32")
    else {
        panic!("expected gpt_rs.softmax_last_dim_f32 pattern definition to be registered");
    };
    let fields = def
        .fields
        .iter()
        .map(|field| (field.name, field.view))
        .collect::<Vec<_>>();
    assert!(
        fields.contains(&("output", "AnyOpView")),
        "expected builtin softmax helper-call output AnyOpView field; fields: {:?}",
        fields,
    );
}

fn entry_function(program: &Program) -> &Function {
    program
        .functions
        .iter()
        .find(|function| function.name == program.entry)
        .expect("entry function must exist")
}
