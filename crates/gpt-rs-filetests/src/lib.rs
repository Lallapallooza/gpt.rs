//! The filetest runner. It runs the `.ptir` files under `filetests/` at the repository root.
//!
//! Each crate whose passes have filetests runs its own directory from its `tests/filetests.rs`:
//! `gpt-rs-ir` runs `filetests/tensor/`, and a backend crate runs `filetests/backend/<backend>/`
//! through `backend::run` (feature `backend`). A pass case names the passes that `#[pass]`
//! registers in the linked crates, and in a backend's filetests `default` names the backend's
//! pipeline.
//!
//! A file starts with `// RUN: <mode>` and holds cases separated by `// -----`. A case starts at
//! its first non-blank line, and the line numbers of parse errors count from there. Modes:
//!
//! - `roundtrip`: the case is canonical text. It must parse, verify, and print back unchanged. A
//!   case that starts with the spec version prints with it, as a stored program does.
//! - `parse`: the case fails to parse with exactly the message of its `// ERROR: <message>` line.
//! - `verify`: the case parses and fails verification with exactly the message of its
//!   `// ERROR: <message>` line.
//! - `pass <passes>`: runs a pass list such as `canonicalize {fuel = 10}, cse`, or `default`,
//!   verifying after each pass. The output must parse, print
//!   back unchanged and satisfy the case's CHECK lines, of which it needs at least one. The passes
//!   must report a change exactly when the text changes. Every run-time function must compute the
//!   same results before and after on random inputs, as [`bits_eq`] compares them, unless the case
//!   has a `// NO-INTERP: <reason>` line.
//!
//!   The inputs start with the special values of [`Specials::All`]. When the function before the
//!   passes cannot run on them, as when they are out-of-range indices, plain random values replace
//!   them. Every parameter that a function reads gets one value before and after, and the
//!   load-time functions run first: their results bind the parameters they compute. A case with a
//!   `// ROUNDING: <reason>` line is for a pass whose rewrite holds in real arithmetic but rounds
//!   differently. Its inputs start with [`Specials::Finite`], and f32 results may differ by up to
//!   1e-5 relative plus 1e-6 absolute error.
//!
//! CHECK lines: `CHECK:` finds the pattern after the previous match, `CHECK-NEXT:` on the next
//! line, `CHECK-NOT:` nowhere between the previous match and the next one. Any other comment line
//! that names CHECK is an error. In a pattern, `[[NAME:regex]]` captures, `[[NAME]]` reuses a
//! capture and `{{regex}}` matches a regex. Everything else is literal.

#[cfg(feature = "backend")]
pub mod backend;
mod check;
mod differential;

use std::fs;
use std::path::{Path, PathBuf};

use gpt_rs_ir::module::Module;
use gpt_rs_ir::passes::{self, Pipeline};
use gpt_rs_ir::verify::verify;

pub use differential::{bits_eq, random_value, rotation, run_dims, Specials, ROUNDS};

/// The `filetests/` directory at the repository root.
pub fn filetests_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../filetests")
}

/// Every `.ptir` file under `dir`, in sorted order.
pub fn collect(dir: &Path, out: &mut Vec<PathBuf>) {
    let mut entries: Vec<PathBuf> = fs::read_dir(dir)
        .unwrap_or_else(|err| panic!("cannot read {}: {err}", dir.display()))
        .map(|e| e.expect("dir entry").path())
        .collect();
    entries.sort();
    for path in entries {
        if path.is_dir() {
            collect(&path, out);
        } else if path.extension().is_some_and(|e| e == "ptir") {
            out.push(path);
        }
    }
}

/// A case of a filetest.
pub struct Case {
    /// 1-based line of the case's first non-blank line in the file.
    pub line: usize,
    /// The case from its first non-blank line on.
    pub text: String,
}

/// The mode of a filetest's `// RUN: <mode>` line, and its cases, which `// -----` lines separate.
pub fn split_cases(text: &str) -> (String, Vec<Case>) {
    let mut lines = text.lines().enumerate();
    let run = loop {
        match lines.next() {
            Some((_, line)) if line.trim().is_empty() => continue,
            Some((_, line)) => break line.trim().strip_prefix("// RUN:").map(str::trim),
            None => break None,
        }
    }
    .unwrap_or_else(|| panic!("a filetest starts with `// RUN: <mode>`"))
    .to_string();
    let mut cases = Vec::new();
    let mut current = String::new();
    let mut start = 0usize;
    for (i, line) in lines {
        if line.trim() == "// -----" {
            cases.push(Case {
                line: start + 1,
                text: std::mem::take(&mut current),
            });
            continue;
        }
        if current.trim().is_empty() {
            start = i;
            current.clear();
        }
        current.push_str(line);
        current.push('\n');
    }
    if !current.trim().is_empty() {
        cases.push(Case {
            line: start + 1,
            text: current,
        });
    }
    (run, cases)
}

/// The pipeline that `default` names in a pass case.
pub type DefaultPipeline<'a> = Option<&'a dyn Fn() -> Pipeline>;

/// Runs every filetest under `filetests/<dir>`, with `default` as the pipeline that `default`
/// names, if any, and fails with the list of failed cases.
pub fn run(dir: &str, default: DefaultPipeline<'_>) {
    let root = filetests_root();
    let mut files = Vec::new();
    collect(&root.join(dir), &mut files);
    assert!(!files.is_empty(), "no filetests under filetests/{dir}");
    let mut failures = Vec::new();
    let mut cases = 0;
    for path in &files {
        let text = fs::read_to_string(path).expect("read filetest");
        let (mode, file_cases) = split_cases(&text);
        for case in &file_cases {
            cases += 1;
            if let Err(err) = run_case(&mode, case, default) {
                let rel = path.strip_prefix(&root).unwrap_or(path);
                failures.push(format!("filetests/{}:{}: {err}", rel.display(), case.line));
            }
        }
    }
    assert!(
        failures.is_empty(),
        "{} of {cases} filetest cases failed:\n\n{}",
        failures.len(),
        failures.join("\n\n")
    );
}

fn directive<'a>(case: &'a str, name: &str) -> Vec<&'a str> {
    let prefix = format!("// {name}:");
    case.lines()
        .filter_map(|l| l.trim().strip_prefix(prefix.as_str()).map(str::trim))
        .collect()
}

/// The case without comment lines and blank lines: what a canonical printer emits.
fn code_lines(case: &str) -> String {
    let mut out = String::new();
    for line in case.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with("//") {
            continue;
        }
        out.push_str(line.trim_end());
        out.push('\n');
    }
    out
}

/// Checks that `result` is an error with exactly the message of the case's `// ERROR:` line.
fn expect_error<T>(case: &str, result: anyhow::Result<T>) -> Result<(), String> {
    let expected = directive(case, "ERROR");
    let [expected] = expected.as_slice() else {
        return Err("the case needs exactly one `// ERROR:` line".into());
    };
    match result {
        Ok(_) => Err(format!("no error, expected `{expected}`")),
        Err(err) if format!("{err:#}") == *expected => Ok(()),
        Err(err) => Err(format!("error is `{err:#}`, expected `{expected}`")),
    }
}

fn run_case(mode: &str, case: &Case, default: DefaultPipeline<'_>) -> Result<(), String> {
    let (mode, args) = mode
        .split_once(' ')
        .map_or((mode, ""), |(m, rest)| (m, rest.trim()));
    let parsed = Module::from_text(&case.text);
    if mode == "parse" {
        return expect_error(&case.text, parsed);
    }
    let module = parsed.map_err(|err| format!("parse error: {err:#}"))?;
    match mode {
        "roundtrip" => {
            verify(&module).map_err(|err| format!("verify error: {err}"))?;
            let expected = code_lines(&case.text);
            let printed = if expected.starts_with("spec_version") {
                module.to_versioned_text()
            } else {
                module.to_text()
            };
            if printed != expected {
                return Err(format!(
                    "printed text differs\n--- expected\n{expected}--- printed\n{printed}"
                ));
            }
            Ok(())
        }
        "verify" => expect_error(&case.text, verify(&module)),
        "pass" => pass_case(module, args, case, default),
        other => Err(format!("unknown RUN mode `{other}`")),
    }
}

fn pass_case(
    before: Module,
    list: &str,
    case: &Case,
    default: DefaultPipeline<'_>,
) -> Result<(), String> {
    verify(&before).map_err(|err| format!("the input does not verify: {err}"))?;
    let checks = check::check_lines(&case.text)?;
    if checks.is_empty() {
        return Err("a pass case needs a CHECK line".into());
    }
    let pipeline = match (list, default) {
        ("default", Some(default)) => default(),
        ("default", None) => {
            return Err("`default` names a backend's pipeline, and there is none here".into())
        }
        (list, _) => passes::parse(list).map_err(|err| format!("pass list: {err:#}"))?,
    };
    let mut after = before.clone();
    let changed = pipeline.run(&mut after).map_err(|err| format!("{err:#}"))?;
    let output = after.to_text();
    let text_changed = output != before.to_text();
    if changed != text_changed {
        return Err(format!(
            "the passes report changed = {changed}, but the text {}",
            if text_changed {
                "changed"
            } else {
                "did not change"
            }
        ));
    }
    let reprinted = Module::from_text(&output).map(|m| m.to_text());
    if reprinted.as_ref().ok() != Some(&output) {
        return Err(format!(
            "the output does not print back unchanged: {reprinted:?}\n--- output\n{output}"
        ));
    }
    check::file_check(&checks, &output).map_err(|err| format!("{err}\n--- output\n{output}"))?;
    if directive(&case.text, "NO-INTERP").is_empty() {
        let rounding = !directive(&case.text, "ROUNDING").is_empty();
        differential::interp_differential(&before, &after, case.line as u64, rounding)?;
    }
    Ok(())
}
