use serde_json::Value;

#[derive(Clone, Copy, Debug)]
pub struct CompileFreezeGate {
    pub measured_token_threshold: usize,
    pub strict: bool,
    pub baseline_compile_ms: Option<f64>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CompileFreezeObservation {
    pub compile_ms_after_threshold: f64,
}

impl CompileFreezeObservation {
    pub fn has_anomaly(&self) -> bool {
        self.compile_ms_after_threshold > 0.0
    }
}

pub fn compile_ms_from_report_json(report_json: &str) -> Option<f64> {
    let parsed: Value = serde_json::from_str(report_json).ok()?;
    let sections = parsed.get("sections")?.as_array()?;
    let mut total = 0.0f64;

    for section in sections {
        let tables = section.get("tables")?;
        for key in ["compilation", "compile_passes"] {
            if let Some(rows) = tables.get(key).and_then(Value::as_array) {
                for row in rows {
                    total += row.get("excl_ms").and_then(Value::as_f64).unwrap_or(0.0);
                }
            }
        }
    }

    Some(total)
}

pub fn evaluate_compile_freeze(
    report_json: &str,
    gate: CompileFreezeGate,
) -> Option<CompileFreezeObservation> {
    let baseline_compile_ms = gate.baseline_compile_ms?;
    let total_compile_ms = compile_ms_from_report_json(report_json)?;
    Some(CompileFreezeObservation {
        compile_ms_after_threshold: (total_compile_ms - baseline_compile_ms).max(0.0),
    })
}
