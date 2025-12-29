use std::sync::Arc;

use anyhow::Result;
use gpt_rs::backend::{index::InstId, pattern::OperationView, rewriter::ProgramRewriter};
use gpt_rs::ops::functional::CaptureIntoDeviceTensor;
use gpt_rs::tensor::{DeviceTensor, Shape, Tensor};
use gpt_rs_backend_ref_cpu::CpuPortableBackend;
use gpt_rs_macros::{capture_ptir, ptir_pattern};

#[ptir_pattern(target = "gpt_rs.mul_mul_add_f32", anchor = mul1)]
fn mul_mul_add_pattern<B: gpt_rs::backend::spec::PortableBackend + 'static>(
    _backend: &B,
    t1: &DeviceTensor<B>,
    t2: &DeviceTensor<B>,
    t3: &DeviceTensor<B>,
) -> Result<DeviceTensor<B>> {
    capture_ptir!({ t1, t2, t3 }, |_session| {
        let mul1 = t1 * t2;
        let mul2 = mul1 * t3;
        let add = mul2 + mul1;
        Ok(add.id())
    })?
    .into_device_tensor()
}

fn rewriter_from_program(src: &str) -> ProgramRewriter<'static> {
    let program = gpt_rs::ptir_program!(src);
    let function = program
        .functions
        .into_iter()
        .next()
        .expect("program must define a function");
    let function_ref: &'static mut _ = Box::leak(Box::new(function));
    ProgramRewriter::new(function_ref).expect("build rewriter")
}

#[test]
fn ptir_pattern_macro_generates_working_operation_view() -> Result<()> {
    let backend = Arc::new(CpuPortableBackend::new());

    let t1 = DeviceTensor::from_host(
        Arc::clone(&backend),
        Tensor::from_vec(Shape::new([2, 2]), vec![1.0, 2.0, 3.0, 4.0])?,
    )?;
    let t2 = DeviceTensor::from_host(
        Arc::clone(&backend),
        Tensor::from_vec(Shape::new([2, 2]), vec![5.0, 6.0, 7.0, 8.0])?,
    )?;
    let t3 = DeviceTensor::from_host(
        Arc::clone(&backend),
        Tensor::from_vec(Shape::new([2, 2]), vec![1.0, 1.0, 1.0, 1.0])?,
    )?;

    let _ = mul_mul_add_pattern(backend.as_ref(), &t1, &t2, &t3)?;

    const PROGRAM: &str = r#"
func @mul_mul_add(%a: tensor<f32, 2>, %b: tensor<f32, 2>, %c: tensor<f32, 2>) -> tensor<f32, 2> {
  %mul1 = mul %a, %b -> tensor<f32, 2>
  %mul2 = mul %mul1, %c -> tensor<f32, 2>
  %add = add %mul2, %mul1 -> tensor<f32, 2>
  return %add
}
"#;

    let rewriter = rewriter_from_program(PROGRAM);
    let view = MulMulAddPattern::extract(InstId(0), &rewriter).expect("expected pattern match");

    assert_eq!(view.mul1.root, InstId(0));
    assert_eq!(view.mul2.root, InstId(1));
    assert_eq!(view.add.root, InstId(2));
    assert_eq!(view.output(), rewriter.value_of(InstId(2)));

    Ok(())
}
