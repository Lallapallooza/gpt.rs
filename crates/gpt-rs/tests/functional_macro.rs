//! The pattern views `#[functional]` generates from the values its `capture!` bodies bind.

use std::sync::Arc;

use anyhow::Result;
use gpt_rs::backend::index::InstId;
use gpt_rs::backend::pattern::{all_pattern_defs, OperationView, PatternDef};
use gpt_rs::backend::rewriter::ProgramRewriter;
use gpt_rs::backend::spec::{ElementwiseBinaryOp, Function, Operation, PortableBackend, Program};
use gpt_rs::ops::ptir;
use gpt_rs::tensor::{DeviceTensor, Shape, Tensor as HostTensor};
use gpt_rs::{capture, functional};
use gpt_rs_backend_ref_cpu::CpuPortableBackend;
use gpt_rs_backend_tests::recording_backend::RecordingBackend;

#[functional]
fn mul_mul_add(t1: &Tensor, t2: &Tensor, t3: &Tensor) -> Result<Tensor> {
    capture!(|t1, t2, t3| {
        let mul1 = t1 * t2;
        let mul2 = mul1 * t3;
        mul2 + mul1
    })
}

fn softmax_helper<'ctx, 'gb, B: PortableBackend + 'static>(
    input: &ptir::Tensor<'ctx, 'gb, B>,
    axis: usize,
) -> ptir::Tensor<'ctx, 'gb, B> {
    let max = input.reduce_max([axis], true);
    let shifted = *input - max.broadcast_like(input);
    let exp_values = shifted.exp();
    let sum = exp_values.reduce_sum([axis], true);
    exp_values / sum.broadcast_like(input)
}

#[functional]
fn helper_softmax(x: &Tensor) -> Result<Tensor> {
    let axis = x.shape().rank() - 1;
    capture!(|x| softmax_helper(&x, axis))
}

fn device<B: PortableBackend + 'static>(backend: &Arc<B>, dims: &[usize]) -> DeviceTensor<B> {
    let len = dims.iter().product();
    let host = HostTensor::from_vec(Shape::new(dims.to_vec()), vec![1.0; len]).expect("host");
    DeviceTensor::from_host(Arc::clone(backend), host).expect("upload")
}

fn pattern_def(target: &str) -> &'static PatternDef {
    all_pattern_defs()
        .iter()
        .find(|def| def.target == target)
        .unwrap_or_else(|| panic!("no pattern definition for {target}"))
}

/// `(name, view, optional)` of each field of a pattern definition.
fn fields(def: &PatternDef) -> Vec<(&str, &str, bool)> {
    def.fields
        .iter()
        .map(|field| (field.name, field.view, field.optional))
        .collect()
}

fn entry_function(program: &Program) -> &Function {
    program
        .functions
        .iter()
        .find(|function| function.name == program.entry)
        .expect("entry function must exist")
}

#[test]
fn pattern_view_binds_lets_and_the_result() -> Result<()> {
    let backend = Arc::new(CpuPortableBackend::new());
    let [t1, t2, t3] = [0; 3].map(|_| device(&backend, &[2, 2]));
    mul_mul_add(&t1, &t2, &t3)?;

    let program = gpt_rs::ptir_program!(
        r#"
func @mul_mul_add(%a: tensor<f32, 2>, %b: tensor<f32, 2>, %c: tensor<f32, 2>) -> tensor<f32, 2> {
  %mul1 = mul %a, %b -> tensor<f32, 2>
  %mul2 = mul %mul1, %c -> tensor<f32, 2>
  %add = add %mul2, %mul1 -> tensor<f32, 2>
  return %add
}
"#
    );
    let mut function = program.functions[0].clone();
    let rewriter = ProgramRewriter::new(&mut function)?;
    // Among the equally specific binds, the anchor is the first by name: `mul1`.
    let view = MulMulAddPattern::extract(InstId(0), &rewriter).expect("pattern match");
    assert_eq!(view.mul1.root, InstId(0));
    assert_eq!(view.mul2.root, InstId(1));
    assert_eq!(view.output.root, InstId(2));
    assert_eq!(view.output(), rewriter.value_of(InstId(2)));
    assert_eq!(MulMulAddPattern::TARGET, "gpt_rs.mul_mul_add");
    Ok(())
}

#[test]
fn helper_call_result_is_bound_as_any_op() {
    let backend = Arc::new(RecordingBackend::default());
    let out = helper_softmax(&device(&backend, &[2, 4])).expect("helper softmax");
    out.materialize().expect("materialize output");

    let recorded = backend.recorded_program_or_panic();
    let mut entry = entry_function(&recorded).clone();
    let rewriter = ProgramRewriter::new(&mut entry).expect("build rewriter");
    let view = rewriter
        .insts_in_order()
        .into_iter()
        .find_map(|inst| HelperSoftmaxPattern::extract(inst, &rewriter))
        .expect("HelperSoftmaxPattern matches the captured graph");
    assert!(
        matches!(
            view.output.op,
            Operation::ElementwiseBinary(ElementwiseBinaryOp::Div)
        ),
        "the helper's result is its final div"
    );
    assert_eq!(
        fields(pattern_def("gpt_rs.helper_softmax")),
        [("output", "AnyOpView", false)]
    );
}

#[test]
fn functionals_register_their_pattern_views() {
    assert!(fields(pattern_def("gpt_rs.softmax_last_dim")).contains(&(
        "output",
        "AnyOpView",
        false
    )));
    let conv = fields(pattern_def("gpt_rs.conv2d"));
    for field in [
        ("patches", "ExtractPatchesOpView", false),
        ("patches_reshape", "ReshapeOpView", false),
        ("out", "DotGeneralOpView", false),
        ("out_add", "AddOpView", true),
        ("weight", "ReshapeOpView", true),
    ] {
        assert!(conv.contains(&field), "{field:?} missing from {conv:?}");
    }
}

#[test]
fn a_bind_whose_op_differs_between_capture_sites_is_any_op() {
    // `clamp` captures `max(x, min)`, `min(x, max)` or both, depending on the bounds given.
    assert_eq!(
        fields(pattern_def("gpt_rs.clamp")),
        [
            ("lower", "MaximumOpView", true),
            ("output", "AnyOpView", false)
        ]
    );
}
