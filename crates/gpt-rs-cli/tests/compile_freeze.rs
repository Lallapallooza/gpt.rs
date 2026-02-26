#[path = "../src/benchmark/compile_freeze.rs"]
mod compile_freeze;

use compile_freeze::{compile_ms_from_report_json, evaluate_compile_freeze, CompileFreezeGate};

#[test]
fn compile_ms_from_report_json_sums_compilation_tables() {
    let report = r#"
{
  "sections": [
    {
      "section": "run",
      "tables": {
        "compilation": [{"excl_ms": 1.25}],
        "compile_passes": [{"excl_ms": 0.75}]
      }
    },
    {
      "section": "other",
      "tables": {
        "compilation": [{"excl_ms": 2.00}],
        "compile_passes": []
      }
    }
  ]
}
"#;

    let total = compile_ms_from_report_json(report).expect("compile ms total");
    assert!((total - 4.0).abs() < 1e-6);
}

#[test]
fn evaluate_compile_freeze_reports_positive_delta() {
    let report = r#"
{
  "sections": [
    {
      "section": "run",
      "tables": {
        "compilation": [{"excl_ms": 3.50}],
        "compile_passes": [{"excl_ms": 1.00}]
      }
    }
  ]
}
"#;

    let gate = CompileFreezeGate {
        measured_token_threshold: 16,
        strict: true,
        baseline_compile_ms: Some(2.0),
    };
    assert_eq!(gate.measured_token_threshold, 16);
    assert!(gate.strict);
    let observation = evaluate_compile_freeze(report, gate).expect("freeze observation");
    assert!((observation.compile_ms_after_threshold - 2.5).abs() < 1e-6);
    assert!(observation.has_anomaly());
}

#[test]
fn evaluate_compile_freeze_clamps_negative_delta_to_zero() {
    let report = r#"
{
  "sections": [
    {
      "section": "run",
      "tables": {
        "compilation": [{"excl_ms": 1.00}],
        "compile_passes": [{"excl_ms": 0.25}]
      }
    }
  ]
}
"#;

    let gate = CompileFreezeGate {
        measured_token_threshold: 8,
        strict: false,
        baseline_compile_ms: Some(5.0),
    };
    assert_eq!(gate.measured_token_threshold, 8);
    assert!(!gate.strict);
    let observation = evaluate_compile_freeze(report, gate).expect("freeze observation");
    assert_eq!(observation.compile_ms_after_threshold, 0.0);
    assert!(!observation.has_anomaly());
}
