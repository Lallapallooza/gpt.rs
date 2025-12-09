use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};

use gpt_rs::register_patterns_for_view;

use gpt_rs::backend::{
    index::InstId,
    pattern::{filters, CastOpView, OpRewritePattern, Pattern, PatternSet},
    ptir_utils::{tensor_literal_zeros, tensor_spec_static},
    rewriter::ProgramRewriter,
    spec::{CastSpec, DType, Operation},
};
use gpt_rs::ptir_program;

fn sample_function() -> (ProgramRewriter<'static>, Arc<AtomicUsize>) {
    const PROGRAM: &str = r#"
func @pattern_test(%x: tensor<f32, 2>) -> tensor<f32, 2> {
  %sg = stop_gradient %x -> tensor<f32, 2>
  %cast = cast %sg -> tensor<f32, 2>
  return %cast
}
"#;
    let function = ptir_program!(PROGRAM)
        .functions
        .into_iter()
        .next()
        .expect("program must define function");
    let function_ref: &'static mut _ = Box::leak(Box::new(function));
    let rewriter = ProgramRewriter::new(function_ref).expect("build rewriter");
    (rewriter, Arc::new(AtomicUsize::new(0)))
}

struct StopGradientSpy {
    calls: Arc<AtomicUsize>,
}

impl Pattern for StopGradientSpy {
    fn matches_operation(&self, op: &Operation) -> bool {
        filters::stop_gradient(op)
    }

    fn benefit(&self) -> u16 {
        5
    }

    fn match_and_rewrite(&self, _root: InstId, _rewriter: &mut ProgramRewriter) -> bool {
        self.calls.fetch_add(1, Ordering::Relaxed);
        false
    }
}

struct CastNoOp;

impl OpRewritePattern<CastOpView> for CastNoOp {
    fn benefit(&self) -> u16 {
        3
    }

    fn may_match(&self, op: &CastOpView, _rewriter: &ProgramRewriter) -> bool {
        op.result.0 == 1
    }

    fn match_and_rewrite(&self, _op: CastOpView, _rewriter: &mut ProgramRewriter) -> bool {
        false
    }
}

#[test]
fn pattern_set_orders_by_benefit_and_filters_roots() {
    let (mut rewriter, calls) = sample_function();

    let mut set = PatternSet::new();
    set.add(StopGradientSpy {
        calls: Arc::clone(&calls),
    });
    set.insert_view::<CastOpView, _>(CastNoOp);

    let frozen = set.freeze();

    let op0 = rewriter.op(InstId(0)).clone();
    let stop_patterns: Vec<(usize, &dyn Pattern)> = frozen.matching(&op0).collect();
    assert_eq!(stop_patterns.len(), 1);
    assert_eq!(stop_patterns[0].1.benefit(), 5);

    let op1 = rewriter.op(InstId(1)).clone();
    let cast_patterns: Vec<(usize, &dyn Pattern)> = frozen.matching(&op1).collect();
    assert_eq!(cast_patterns.len(), 1);
    assert_eq!(cast_patterns[0].1.benefit(), 3);

    let constant_count = frozen
        .matching(&Operation::Constant(tensor_literal_zeros(
            tensor_spec_static(DType::F32, &[1]),
        )))
        .count();
    assert_eq!(constant_count, 0);

    stop_patterns[0]
        .1
        .match_and_rewrite(InstId(0), &mut rewriter);
    assert_eq!(calls.load(Ordering::Relaxed), 1);
}

#[test]
fn typed_adapter_calls_may_match() {
    let (mut rewriter, _) = sample_function();

    struct RejectingPattern {
        called: Arc<AtomicUsize>,
    }

    impl OpRewritePattern<CastOpView> for RejectingPattern {
        fn may_match(&self, _op: &CastOpView, _rewriter: &ProgramRewriter) -> bool {
            self.called.fetch_add(1, Ordering::Relaxed);
            false
        }

        fn match_and_rewrite(&self, _op: CastOpView, _rewriter: &mut ProgramRewriter) -> bool {
            panic!("should not be invoked when may_match returns false");
        }
    }

    let called = Arc::new(AtomicUsize::new(0));
    let mut set = PatternSet::new();
    set.insert_view::<CastOpView, _>(RejectingPattern {
        called: Arc::clone(&called),
    });
    let frozen = set.freeze();

    let op1 = rewriter.op(InstId(1)).clone();
    let mut patterns: Vec<(usize, &dyn Pattern)> = frozen.matching(&op1).collect();
    let (_, pattern) = patterns.remove(0);
    pattern.match_and_rewrite(InstId(1), &mut rewriter);
    assert_eq!(called.load(Ordering::Relaxed), 1);
}

#[test]
fn register_patterns_macro_registers_multiple_variants() {
    let mut set = PatternSet::new();
    register_patterns_for_view!(set, CastOpView, CastNoOp, AlternateCastPattern);
    let frozen = set.freeze();
    let matches: Vec<_> = frozen
        .matching(&Operation::Cast(CastSpec { dtype: DType::F32 }))
        .collect();
    assert_eq!(matches.len(), 2);
}

#[test]
fn match_any_patterns_are_supported() {
    let hits = Arc::new(AtomicUsize::new(0));
    let mut set = PatternSet::new();
    set.insert_match_any(
        MatchAnyPattern {
            hits: Arc::clone(&hits),
        },
        |_root, _| Some(()),
    );
    let frozen = set.freeze();
    assert_eq!(
        frozen
            .matching(&Operation::Cast(CastSpec { dtype: DType::F32 }))
            .count(),
        1
    );
    assert_eq!(hits.load(Ordering::Relaxed), 0);
}

struct AlternateCastPattern;

impl OpRewritePattern<CastOpView> for AlternateCastPattern {
    fn benefit(&self) -> u16 {
        2
    }

    fn may_match(&self, op: &CastOpView, _rewriter: &ProgramRewriter) -> bool {
        !op.operands.is_empty()
    }

    fn match_and_rewrite(&self, _op: CastOpView, _rewriter: &mut ProgramRewriter) -> bool {
        false
    }
}

struct MatchAnyPattern {
    hits: Arc<AtomicUsize>,
}

impl OpRewritePattern<()> for MatchAnyPattern {
    fn match_and_rewrite(&self, _op: (), _rewriter: &mut ProgramRewriter) -> bool {
        self.hits.fetch_add(1, Ordering::Relaxed);
        false
    }
}
