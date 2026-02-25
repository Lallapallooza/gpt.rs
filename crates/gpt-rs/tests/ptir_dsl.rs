use std::{
    f32::consts::SQRT_2,
    fs,
    path::{Path, PathBuf},
    sync::Arc,
};

use gpt_rs::backend::spec::{
    DType as BackendDType, Dimension, ElementwiseBinaryOp, ElementwiseUnaryOp, Operation,
    ReduceKind,
};
use gpt_rs::ops::graph::GraphArena;
use gpt_rs::ops::ptir::{tensor, DotAttrs, DotDims, PtirGraph, PtirSession};
use gpt_rs::tensor::{DeviceTensor, Shape as DeviceShape};
use gpt_rs::{axes, DType as FrontendDType};
use gpt_rs_backend_tests::recording_backend::RecordingBackend;

#[test]
fn dsl_elementwise_chain_emits_expected_program() {
    let backend = Arc::new(RecordingBackend::default());
    let arena = GraphArena::new(Arc::clone(&backend));

    let result = arena
        .capture(|ctx| {
            let device_shape = DeviceShape::new(vec![2, 4]);
            let placeholder = tensor::<f32>(device_shape.clone());

            let lhs_tensor = DeviceTensor::from_handle(
                Arc::clone(&backend),
                device_shape.clone(),
                FrontendDType::F32,
                (),
            );
            let rhs_tensor = DeviceTensor::from_handle(
                Arc::clone(&backend),
                device_shape.clone(),
                FrontendDType::F32,
                (),
            );

            let lhs_id = ctx.import(&lhs_tensor)?;
            let rhs_id = ctx.import(&rhs_tensor)?;

            let mut graph = PtirGraph::new(ctx);
            let lhs = graph.import("lhs", lhs_id, &placeholder);
            let rhs = graph.import("rhs", rhs_id, &placeholder);

            let sum = lhs.add(&mut graph, &rhs)?;
            let exp = sum.exp(&mut graph)?;
            let reduced = exp.reduce_sum(&mut graph, [1usize], true)?;
            let broadcast = reduced.broadcast_like(&mut graph, &exp)?;
            let normalized = exp.div(&mut graph, &broadcast)?;
            Ok(normalized.id())
        })
        .expect("graph capture succeeds");

    arena
        .flush_until(result)
        .expect("graph flush should succeed");

    let recorded = backend
        .recorded_program()
        .expect("backend should record a program");

    let function = recorded
        .functions
        .iter()
        .find(|func| func.name == recorded.entry)
        .expect("captured function present");

    assert_eq!(function.parameters.len(), 2, "expected two parameters");
    assert_eq!(function.body.len(), 5, "expected five operations in chain");

    match &function.body[0].op {
        Operation::ElementwiseBinary(ElementwiseBinaryOp::Add) => {}
        other => panic!("expected add as first op, got {other:?}"),
    }

    match &function.body[1].op {
        Operation::ElementwiseUnary(ElementwiseUnaryOp::Exp) => {}
        other => panic!("expected exp as second op, got {other:?}"),
    }

    match &function.body[2].op {
        Operation::Reduce(spec) => {
            assert_eq!(spec.kind, ReduceKind::Sum);
            assert_eq!(spec.axes, vec![1]);
            assert!(spec.keepdims);
        }
        other => panic!("expected reduce_sum as third op, got {other:?}"),
    }

    match &function.body[3].op {
        Operation::BroadcastTo(spec) => {
            let dims = spec.result_shape.dims().to_vec();
            assert_eq!(
                dims,
                vec![Dimension::Static(2), Dimension::Static(4)],
                "broadcast result shape should match input"
            );
        }
        other => panic!("expected broadcast_to, got {other:?}"),
    }

    match &function.body[4].op {
        Operation::ElementwiseBinary(ElementwiseBinaryOp::Div) => {}
        other => panic!("expected div as final op, got {other:?}"),
    }

    assert_eq!(
        function.result_ids.last().copied(),
        Some(function.body.last().expect("body non-empty").id),
        "final result should correspond to last instruction",
    );
}

#[test]
fn dsl_dot_general_emits_expected_program() {
    let backend = Arc::new(RecordingBackend::default());
    let arena = GraphArena::new(Arc::clone(&backend));

    let result = arena
        .capture(|ctx| {
            let lhs_shape = DeviceShape::new(vec![2, 3]);
            let rhs_shape = DeviceShape::new(vec![3, 4]);

            let lhs_placeholder = tensor::<f32>(lhs_shape.clone());
            let rhs_placeholder = tensor::<f32>(rhs_shape.clone());

            let lhs_tensor = DeviceTensor::from_handle(
                Arc::clone(&backend),
                lhs_shape.clone(),
                FrontendDType::F32,
                (),
            );
            let rhs_tensor = DeviceTensor::from_handle(
                Arc::clone(&backend),
                rhs_shape.clone(),
                FrontendDType::F32,
                (),
            );

            let lhs_id = ctx.import(&lhs_tensor)?;
            let rhs_id = ctx.import(&rhs_tensor)?;

            let mut graph = PtirGraph::new(ctx);
            let lhs = graph.import("lhs", lhs_id, &lhs_placeholder);
            let rhs = graph.import("rhs", rhs_id, &rhs_placeholder);

            let dims = DotDims::new(axes![], axes![1usize], axes![0usize]);
            let attrs = DotAttrs::default();
            let dot = lhs.dot_general(&mut graph, &rhs, &dims, &attrs)?;
            Ok(dot.id())
        })
        .expect("graph capture succeeds");

    arena
        .flush_until(result)
        .expect("graph flush should succeed");

    let recorded = backend
        .recorded_program()
        .expect("backend should record a program");

    let function = recorded
        .functions
        .iter()
        .find(|func| func.name == recorded.entry)
        .expect("captured function present");

    assert_eq!(function.body.len(), 1, "expected single dot_general op");
    match &function.body[0].op {
        Operation::DotGeneral(spec) => {
            assert!(spec.batch_lhs.is_empty());
            assert!(spec.batch_rhs.is_empty());
            assert_eq!(spec.contract_lhs, vec![1]);
            assert_eq!(spec.contract_rhs, vec![0]);
            assert_eq!(spec.accum_dtype, None);
        }
        other => panic!("expected dot_general, got {other:?}"),
    }

    assert_eq!(
        function.result_ids,
        vec![function.body[0].id],
        "dot_general should return its sole instruction",
    );
}

#[test]
fn dsl_gelu_sequence_uses_capture_helpers() {
    let backend = Arc::new(RecordingBackend::default());
    let arena = GraphArena::new(Arc::clone(&backend));

    let result = arena
        .capture(|ctx| {
            let device_shape = DeviceShape::new(vec![2, 4]);
            let placeholder = tensor::<f32>(device_shape.clone());

            let input = DeviceTensor::from_handle(
                Arc::clone(&backend),
                device_shape.clone(),
                FrontendDType::F32,
                (),
            );

            let input_id = ctx.import(&input)?;
            let session = PtirSession::new(ctx);
            let x = session.import("x", input_id, &placeholder);

            let normalized = x.div_scalar(SQRT_2);
            let erf = normalized.erf();
            let shifted = erf.add_scalar(1.0);
            let product = x.mul(&shifted);
            let gelu = product.mul_scalar(0.5);

            Ok(gelu.id())
        })
        .expect("graph capture succeeds");

    arena
        .flush_until(result)
        .expect("graph flush should succeed");

    let recorded = backend
        .recorded_program()
        .expect("backend should record a program");
    let function = recorded
        .functions
        .iter()
        .find(|func| func.name == recorded.entry)
        .expect("captured function present");

    assert!(
        function.body.iter().any(|inst| matches!(
            inst.op,
            Operation::ElementwiseUnary(ElementwiseUnaryOp::Erf)
        )),
        "erf op missing from gelu PTIR chain"
    );

    let mul_ops = function
        .body
        .iter()
        .filter(|inst| {
            matches!(
                inst.op,
                Operation::ElementwiseBinary(ElementwiseBinaryOp::Mul)
            )
        })
        .count();
    assert!(
        mul_ops >= 2,
        "gelu graph should emit at least two mul operations"
    );

    let last_op = function.body.last().expect("non-empty PTIR body");
    assert!(matches!(
        last_op.op,
        Operation::ElementwiseBinary(ElementwiseBinaryOp::Mul)
    ));
}

#[test]
fn dsl_value_metadata_tracks_shape_and_dtype() {
    let backend = Arc::new(RecordingBackend::default());
    let arena = GraphArena::new(Arc::clone(&backend));

    let result = arena
        .capture(|ctx| {
            let shape = DeviceShape::new(vec![2, 3, 4]);
            let placeholder = tensor::<f32>(shape.clone());

            let lhs_tensor = DeviceTensor::from_handle(
                Arc::clone(&backend),
                shape.clone(),
                FrontendDType::F32,
                (),
            );
            let rhs_tensor = DeviceTensor::from_handle(
                Arc::clone(&backend),
                shape.clone(),
                FrontendDType::F32,
                (),
            );

            let lhs_id = ctx.import(&lhs_tensor)?;
            let rhs_id = ctx.import(&rhs_tensor)?;

            let mut graph = PtirGraph::new(ctx);
            let lhs = graph.import("lhs", lhs_id, &placeholder);
            let rhs = graph.import("rhs", rhs_id, &placeholder);

            assert_eq!(lhs.rank(), 3);
            assert_eq!(lhs.dtype(), BackendDType::F32);
            assert_eq!(lhs.dims().expect("static dims"), &[2, 3, 4]);

            let sum = lhs.add(&mut graph, &rhs)?;
            assert_eq!(sum.rank(), 3);
            assert_eq!(sum.dtype(), BackendDType::F32);
            assert_eq!(sum.dims().expect("sum dims"), &[2, 3, 4]);

            let reduced = sum.reduce_sum(&mut graph, [1usize], false)?;
            assert_eq!(reduced.rank(), 2);
            assert_eq!(reduced.dtype(), BackendDType::F32);
            assert_eq!(reduced.dims().expect("reduced dims"), &[2, 4]);

            Ok(reduced.id())
        })
        .expect("graph capture succeeds");

    arena
        .flush_until(result)
        .expect("graph flush should succeed");
}

#[test]
fn functional_ops_should_not_invoke_ptir_session_new() {
    let offenders = scan_functional_sources(&["PtirSession"]);
    assert!(
        offenders.is_empty(),
        "functional ops still reference PtirSession directly: \n{}",
        offenders.join("\n")
    );
}

fn scan_functional_sources(terms: &[&str]) -> Vec<String> {
    let mut matches = Vec::new();
    for file in functional_source_files() {
        let contents = fs::read_to_string(&file).unwrap_or_else(|err| {
            panic!("failed to read {}: {err}", file.display());
        });
        for (line_idx, line) in contents.lines().enumerate() {
            for term in terms {
                if line.contains(term) {
                    matches.push(format!("{}:{}:{term}", file.display(), line_idx + 1));
                }
            }
        }
    }
    matches
}

fn functional_source_files() -> Vec<PathBuf> {
    let mut files = Vec::new();
    collect_rust_sources(&functional_source_dir(), &mut files);
    files
}

fn functional_source_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("src/ops/functional")
}

fn collect_rust_sources(dir: &Path, files: &mut Vec<PathBuf>) {
    let entries = fs::read_dir(dir).unwrap_or_else(|err| {
        panic!("failed to read directory {}: {err}", dir.display());
    });
    for entry_result in entries {
        let entry = entry_result.unwrap_or_else(|err| {
            panic!("failed to access entry under {}: {err}", dir.display());
        });
        let path = entry.path();
        if path.is_dir() {
            collect_rust_sources(&path, files);
        } else if path
            .extension()
            .and_then(|ext| ext.to_str())
            .map(|ext| ext.eq_ignore_ascii_case("rs"))
            .unwrap_or(false)
        {
            files.push(path);
        }
    }
}
