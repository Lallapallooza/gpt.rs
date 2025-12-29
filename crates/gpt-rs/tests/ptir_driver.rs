use std::{
    fs,
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
};

use gpt_rs::{
    backend::{
        driver::{apply_patterns_and_fold_greedily, GreedyConfig},
        index::InstId,
        pattern::{filters, Pattern, PatternSet},
        rewriter::ProgramRewriter,
        spec::{ElementwiseBinaryOp, ElementwiseUnaryOp, Function, Operand, Operation},
    },
    ptir_program,
};

const SG_PROGRAM: &str = r#"
func @sg(%x: tensor<f32, 2>) -> tensor<f32, 2> {
  %sg = stop_gradient %x -> tensor<f32, 2>
  %cast = cast %sg -> tensor<f32, 2>
  return %cast
}
"#;

fn make_stop_gradient_function() -> Function {
    ptir_program!(SG_PROGRAM)
        .functions
        .into_iter()
        .next()
        .expect("program must define a function")
}

struct FoldStopGradient;

impl Pattern for FoldStopGradient {
    fn matches_operation(&self, op: &Operation) -> bool {
        filters::stop_gradient(op)
    }

    fn match_and_rewrite(&self, root: InstId, rewriter: &mut ProgramRewriter) -> bool {
        let operands = rewriter.operands(root);
        let [Operand::Value(src)] = operands else {
            return false;
        };
        let produced = rewriter.value_of(root);
        rewriter.replace_all_uses(produced, *src);
        rewriter
            .erase_inst(root)
            .expect("stop_gradient erase should succeed");
        true
    }
}

#[test]
fn greedy_driver_applies_pattern_and_dce() {
    let mut function = make_stop_gradient_function();
    let mut set = PatternSet::new();
    set.add(FoldStopGradient);
    let frozen = set.freeze();

    let stats = apply_patterns_and_fold_greedily(
        &mut function,
        &frozen,
        &GreedyConfig {
            max_iterations: 10,
            enable_dce: true,
        },
    );

    assert_eq!(stats.applied, 1);
    assert!(function
        .body
        .iter()
        .all(|inst| !matches!(inst.op, Operation::StopGradient)));
    assert_eq!(
        function.body.len(),
        1,
        "DCE should remove the erased producer"
    );
}

struct FailingPattern {
    attempts: Arc<AtomicUsize>,
}

impl Pattern for FailingPattern {
    fn matches_operation(&self, op: &Operation) -> bool {
        filters::cast(op)
    }

    fn match_and_rewrite(&self, _root: InstId, _rewriter: &mut ProgramRewriter) -> bool {
        self.attempts.fetch_add(1, Ordering::Relaxed);
        false
    }
}

#[test]
fn failure_cache_prevents_repeated_attempts() {
    let mut function = make_stop_gradient_function();
    let attempts = Arc::new(AtomicUsize::new(0));
    let mut set = PatternSet::new();
    set.add(FailingPattern {
        attempts: Arc::clone(&attempts),
    });
    let frozen = set.freeze();

    let stats = apply_patterns_and_fold_greedily(
        &mut function,
        &frozen,
        &GreedyConfig {
            max_iterations: 5,
            enable_dce: false,
        },
    );

    assert_eq!(stats.applied, 0);
    assert_eq!(attempts.load(Ordering::Relaxed), 1);
}

const CAPTURE_STYLE_PROGRAM: &str = r#"
func @capture_style(%x: tensor<f32, 4x4>) -> tensor<f32, 4x4> {
  %square = mul %x, %x -> tensor<f32, 4x4>
  %exp = exp %square -> tensor<f32, 4x4>
  %result = add %exp, %x -> tensor<f32, 4x4>
  return %result
}
"#;

#[test]
fn greedy_driver_handles_capture_style_programs() {
    let mut function = ptir_program!(CAPTURE_STYLE_PROGRAM)
        .functions
        .into_iter()
        .next()
        .expect("program must define a function");

    let set = PatternSet::new();
    let frozen = set.freeze();

    let stats = apply_patterns_and_fold_greedily(
        &mut function,
        &frozen,
        &GreedyConfig {
            max_iterations: 3,
            enable_dce: true,
        },
    );

    assert_eq!(
        stats.applied, 0,
        "capture style program should remain unchanged"
    );
    assert_eq!(function.body.len(), 3, "expected erf, mul, add chain");

    let mut saw_exp = false;
    let mut mul_count = 0;
    for inst in &function.body {
        match inst.op {
            Operation::ElementwiseUnary(ElementwiseUnaryOp::Exp) => saw_exp = true,
            Operation::ElementwiseBinary(ElementwiseBinaryOp::Mul) => mul_count += 1,
            Operation::ElementwiseBinary(ElementwiseBinaryOp::Add) => {}
            ref other => panic!("unexpected op {other:?} in capture-style test"),
        }
    }

    assert!(
        saw_exp,
        "PTIR program should include exp for capture-style coverage"
    );
    assert_eq!(mul_count, 1, "exactly one mul is emitted before the add");
}

#[test]
fn functional_ops_should_not_reference_ptir_sessions_directly() {
    let offenders = scan_functional_sources(&["PtirSession"]);
    assert!(
        offenders.is_empty(),
        "functional ops still reference PtirSession: \n{}",
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
