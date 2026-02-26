use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use gpt_rs::backend::{
    driver::{apply_patterns_and_fold_greedily, GreedyConfig},
    fusion::{FusionCandidate, HintCostModel, HintLegalizer},
    optimizer::{EntryParam, EntrySignature, OptimizeConfig, OptimizeContext, OptimizeServices},
    param_resolver::InMemoryParamResolver,
    passes::{
        BroadcastCanonicalizationPass, CastCanonicalizationPass, EliminateIdentityBroadcast,
        EliminateRedundantCast, FunctionPass, FusionHintPass,
    },
    pattern::{BroadcastOpView, CastOpView, PatternSet},
    spec::{
        DType, Dimension, DotGeneralSpec, Function, HintKind, HintPolicy, Operand, Operation,
        PortableBackend, Program, ProgramBuilder, Shape, TensorSpec, ValueType,
    },
};
use gpt_rs::ptir_program;
use gpt_rs::tensor::InputRole;
use gpt_rs_backend_ref_cpu::CpuPortableBackend;

const REDUNDANT_CAST_INPUT: &str = r#"
func @redundant_cast(%x: tensor<f32, 2x2>) -> tensor<f32, 2x2> {
  %cast = cast %x -> tensor<f32, 2x2>
  return %cast
}
"#;

const REDUNDANT_CAST_EXPECTED: &str = r#"
func @redundant_cast(%x: tensor<f32, 2x2>) -> tensor<f32, 2x2> {
  return %x
}
"#;

const REAL_CAST_INPUT: &str = r#"
func @real_cast(%x: tensor<f32, 8>) -> tensor<f16, 8> {
  %cast = cast %x -> tensor<f16, 8>
  return %cast
}
"#;

const IDENTITY_BROADCAST_INPUT: &str = r#"
func @id(%x: tensor<f32, 2x3>) -> tensor<f32, 2x3> {
  %b = broadcast_to(%x) shape[2, 3] -> tensor<f32, 2x3>
  return %b
}
"#;

const IDENTITY_BROADCAST_EXPECTED: &str = r#"
func @id(%x: tensor<f32, 2x3>) -> tensor<f32, 2x3> {
  return %x
}
"#;

const NESTED_BROADCAST_INPUT: &str = r#"
func @chain(%x: tensor<f32, 3>) -> tensor<f32, 2x3> {
  %b0 = broadcast_to(%x) shape[1, 3] -> tensor<f32, 1x3>
  %b1 = broadcast_to(%b0) shape[2, 3] -> tensor<f32, 2x3>
  return %b1
}
"#;

#[test]
fn redundant_cast_pattern_eliminates_self_cast() {
    let mut program = ptir_program!(REDUNDANT_CAST_INPUT);
    let expected = ptir_program!(REDUNDANT_CAST_EXPECTED);
    {
        let function = entry_function_mut(&mut program);
        let mut patterns = PatternSet::new();
        patterns.insert_view::<CastOpView, _>(EliminateRedundantCast);
        let frozen = patterns.freeze();
        let stats = apply_patterns_and_fold_greedily(function, &frozen, &GreedyConfig::default());
        assert_eq!(stats.applied, 1);
        assert_eq!(function.body.len(), 0, "cast should be erased");
    }
    assert_eq!(program, expected);
}

#[test]
fn cast_canonicalization_pass_removes_redundant_casts() {
    let mut program = ptir_program!(REDUNDANT_CAST_INPUT);
    let expected = ptir_program!(REDUNDANT_CAST_EXPECTED);
    let result = {
        let function = entry_function_mut(&mut program);
        run_pass(
            &CpuPortableBackend::new(),
            &CastCanonicalizationPass::default(),
            function,
        )
    };
    assert!(result.changed);
    assert_eq!(result.rewrites_applied, 1);
    assert_eq!(program, expected);
}

#[test]
fn cast_canonicalization_pass_preserves_real_casts() {
    let mut program = ptir_program!(REAL_CAST_INPUT);
    let expected = program.clone();
    let result = {
        let function = entry_function_mut(&mut program);
        run_pass(
            &CpuPortableBackend::new(),
            &CastCanonicalizationPass::default(),
            function,
        )
    };
    assert!(!result.changed);
    assert_eq!(result.rewrites_applied, 0);
    assert_eq!(program, expected);
}

#[test]
fn identity_broadcast_is_removed() {
    let mut program = ptir_program!(IDENTITY_BROADCAST_INPUT);
    let expected = ptir_program!(IDENTITY_BROADCAST_EXPECTED);
    {
        let function = entry_function_mut(&mut program);
        let mut patterns = PatternSet::new();
        patterns.insert_view::<BroadcastOpView, _>(EliminateIdentityBroadcast);
        let frozen = patterns.freeze();
        let stats = apply_patterns_and_fold_greedily(function, &frozen, &GreedyConfig::default());
        assert_eq!(stats.applied, 1);
        assert_eq!(
            function.body.len(),
            0,
            "identity broadcast should be erased"
        );
    }
    assert_eq!(program, expected);
}

#[test]
fn broadcast_canonicalization_folds_nested_chain() {
    let mut program = ptir_program!(NESTED_BROADCAST_INPUT);
    let result = {
        let function = entry_function_mut(&mut program);
        run_pass(
            &CpuPortableBackend::new(),
            &BroadcastCanonicalizationPass::default(),
            function,
        )
    };
    assert!(result.changed);

    let function = entry_function_mut(&mut program);
    assert_eq!(
        function.body.len(),
        1,
        "broadcast chain should be fused to one op"
    );
    let inst = &function.body[0];
    match &inst.op {
        Operation::BroadcastTo(spec) => {
            assert_eq!(
                spec.result_shape.dims(),
                &[Dimension::from_usize(2), Dimension::from_usize(3)]
            );
        }
        other => panic!("expected broadcast op, got {other:?}"),
    }
    assert_eq!(inst.operands.len(), 1);
}

fn entry_function_mut(program: &mut Program) -> &mut Function {
    let entry = program.entry.clone();
    program
        .functions
        .iter_mut()
        .find(|func| func.name == entry)
        .expect("entry function must exist")
}

fn run_pass<B: PortableBackend + 'static, P: FunctionPass<B>>(
    backend: &B,
    pass: &P,
    function: &mut Function,
) -> gpt_rs::backend::optimizer::PassResult {
    let resolver = InMemoryParamResolver::<B::TensorHandle>::new();
    let entry_params = function
        .parameter_ids
        .iter()
        .copied()
        .zip(function.parameters.iter().cloned())
        .map(|(id, ty)| EntryParam {
            id,
            ty,
            role: InputRole::Arg,
            stable_id: Some(u128::from(id.0)),
        })
        .collect::<Vec<_>>();
    let entry = EntrySignature::new(entry_params);
    let services = OptimizeServices {
        params: Some(&resolver),
    };
    let cfg = OptimizeConfig::default();
    let mut cx = OptimizeContext::new(backend, services, entry, cfg);
    pass.run(function, &mut cx)
}

struct AlwaysLegal;

impl HintLegalizer<CpuPortableBackend> for AlwaysLegal {
    fn can_fuse(
        &self,
        _function: &Function,
        _candidate: &FusionCandidate,
        _cx: &OptimizeContext<CpuPortableBackend>,
    ) -> Result<HintPolicy, gpt_rs::backend::fusion::FusionRejectReason> {
        Ok(HintPolicy::Preferred)
    }
}

struct PreferDotCost;

impl HintCostModel<CpuPortableBackend> for PreferDotCost {
    fn score(
        &self,
        _function: &Function,
        candidate: &FusionCandidate,
        _cx: &OptimizeContext<CpuPortableBackend>,
    ) -> i64 {
        match candidate.kind {
            HintKind::DotEpilogue => 200,
            HintKind::ElementwiseDag => 50,
            HintKind::ReductionChain => 0,
        }
    }
}

struct CountingLegalizer {
    calls: Arc<AtomicUsize>,
}

impl HintLegalizer<CpuPortableBackend> for CountingLegalizer {
    fn can_fuse(
        &self,
        _function: &Function,
        _candidate: &FusionCandidate,
        _cx: &OptimizeContext<CpuPortableBackend>,
    ) -> Result<HintPolicy, gpt_rs::backend::fusion::FusionRejectReason> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Ok(HintPolicy::Preferred)
    }
}

struct CountingCost {
    calls: Arc<AtomicUsize>,
}

impl HintCostModel<CpuPortableBackend> for CountingCost {
    fn score(
        &self,
        _function: &Function,
        candidate: &FusionCandidate,
        _cx: &OptimizeContext<CpuPortableBackend>,
    ) -> i64 {
        self.calls.fetch_add(1, Ordering::SeqCst);
        match candidate.kind {
            HintKind::DotEpilogue => 200,
            HintKind::ElementwiseDag => 50,
            HintKind::ReductionChain => 0,
        }
    }
}

#[test]
fn fusion_hint_pass_materializes_dot_epilogue_hint_region() {
    let spec_a = TensorSpec::new(
        DType::F32,
        Shape::new(vec![Dimension::from_usize(2), Dimension::from_usize(3)]),
    );
    let spec_b = TensorSpec::new(
        DType::F32,
        Shape::new(vec![Dimension::from_usize(3), Dimension::from_usize(4)]),
    );
    let spec_out = TensorSpec::new(
        DType::F32,
        Shape::new(vec![Dimension::from_usize(2), Dimension::from_usize(4)]),
    );
    let mut builder = ProgramBuilder::new();
    let lhs = builder.add_parameter(ValueType::Tensor(spec_a.clone()));
    let rhs = builder.add_parameter(ValueType::Tensor(spec_b.clone()));
    let bias = builder.add_parameter(ValueType::Tensor(spec_out.clone()));
    let dot = builder.emit_single(
        Operation::DotGeneral(DotGeneralSpec {
            batch_lhs: vec![],
            batch_rhs: vec![],
            contract_lhs: vec![1],
            contract_rhs: vec![0],
            accum_dtype: None,
            out_dtype: None,
        }),
        vec![Operand::Value(lhs), Operand::Value(rhs)],
        ValueType::Tensor(spec_out.clone()),
    );
    let biased = builder.emit_single(
        Operation::ElementwiseBinary(gpt_rs::backend::spec::ElementwiseBinaryOp::Add),
        vec![Operand::Value(dot), Operand::Value(bias)],
        ValueType::Tensor(spec_out.clone()),
    );
    let out = builder.emit_single(
        Operation::ElementwiseUnary(gpt_rs::backend::spec::ElementwiseUnaryOp::Tanh),
        vec![Operand::Value(biased)],
        ValueType::Tensor(spec_out),
    );
    let mut function = builder.finish("main", vec![out]);

    let pass = FusionHintPass::new(Arc::new(AlwaysLegal), Arc::new(PreferDotCost));
    let backend = CpuPortableBackend::new();
    let result = run_pass(&backend, &pass, &mut function);
    assert!(result.changed);
    assert_eq!(result.rewrites_applied, 1);
    assert_eq!(function.hints.len(), 1);
    let hint = &function.hints[0];
    assert_eq!(hint.kind, HintKind::DotEpilogue);
    assert_eq!(hint.inputs.len(), 3);
    assert_eq!(hint.exports, vec![biased]);
    assert_eq!(hint.body.len(), 2);
    assert_eq!(hint.body[0].id, dot);
    assert_eq!(hint.body[1].id, biased);
}

fn build_dot_epilogue_function() -> Function {
    let spec_a = TensorSpec::new(
        DType::F32,
        Shape::new(vec![Dimension::from_usize(2), Dimension::from_usize(3)]),
    );
    let spec_b = TensorSpec::new(
        DType::F32,
        Shape::new(vec![Dimension::from_usize(3), Dimension::from_usize(4)]),
    );
    let spec_out = TensorSpec::new(
        DType::F32,
        Shape::new(vec![Dimension::from_usize(2), Dimension::from_usize(4)]),
    );
    let mut builder = ProgramBuilder::new();
    let lhs = builder.add_parameter(ValueType::Tensor(spec_a.clone()));
    let rhs = builder.add_parameter(ValueType::Tensor(spec_b.clone()));
    let bias = builder.add_parameter(ValueType::Tensor(spec_out.clone()));
    let dot = builder.emit_single(
        Operation::DotGeneral(DotGeneralSpec {
            batch_lhs: vec![],
            batch_rhs: vec![],
            contract_lhs: vec![1],
            contract_rhs: vec![0],
            accum_dtype: None,
            out_dtype: None,
        }),
        vec![Operand::Value(lhs), Operand::Value(rhs)],
        ValueType::Tensor(spec_out.clone()),
    );
    let biased = builder.emit_single(
        Operation::ElementwiseBinary(gpt_rs::backend::spec::ElementwiseBinaryOp::Add),
        vec![Operand::Value(dot), Operand::Value(bias)],
        ValueType::Tensor(spec_out.clone()),
    );
    let out = builder.emit_single(
        Operation::ElementwiseUnary(gpt_rs::backend::spec::ElementwiseUnaryOp::Tanh),
        vec![Operand::Value(biased)],
        ValueType::Tensor(spec_out),
    );
    builder.finish("main", vec![out])
}

#[test]
fn fusion_hint_pass_selected_memo_skips_legalize_and_cost_on_repeat_graph() {
    let legalizer_calls = Arc::new(AtomicUsize::new(0));
    let cost_calls = Arc::new(AtomicUsize::new(0));
    let pass = FusionHintPass::new(
        Arc::new(CountingLegalizer {
            calls: Arc::clone(&legalizer_calls),
        }),
        Arc::new(CountingCost {
            calls: Arc::clone(&cost_calls),
        }),
    )
    .with_min_score(97);
    let backend = CpuPortableBackend::new();

    let mut function_first = build_dot_epilogue_function();
    let first = run_pass(&backend, &pass, &mut function_first);
    assert!(first.changed);
    let legalizer_after_first = legalizer_calls.load(Ordering::SeqCst);
    let cost_after_first = cost_calls.load(Ordering::SeqCst);
    assert!(legalizer_after_first > 0);
    assert!(cost_after_first > 0);

    let mut function_second = build_dot_epilogue_function();
    let second = run_pass(&backend, &pass, &mut function_second);
    assert!(second.changed);

    let legalizer_after_second = legalizer_calls.load(Ordering::SeqCst);
    let cost_after_second = cost_calls.load(Ordering::SeqCst);
    assert_eq!(
        legalizer_after_second, legalizer_after_first,
        "selected memoized hints should skip legalizer on repeated graph"
    );
    assert_eq!(
        cost_after_second, cost_after_first,
        "selected memoized hints should skip cost model on repeated graph"
    );
    assert_eq!(function_first.hints, function_second.hints);
}
