use gpt_rs::backend::{
    driver::{apply_patterns_and_fold_greedily, GreedyConfig},
    optimizer::{EntryParam, EntrySignature, OptimizeConfig, OptimizeContext, OptimizeServices},
    param_resolver::InMemoryParamResolver,
    passes::{
        BroadcastCanonicalizationPass, CastCanonicalizationPass, EliminateIdentityBroadcast,
        EliminateRedundantCast, FunctionPass,
    },
    pattern::{BroadcastOpView, CastOpView, PatternSet},
    spec::{Dimension, Function, Operation, PortableBackend, Program},
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
