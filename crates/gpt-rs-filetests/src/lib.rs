//! The filetest runner. It runs the `.ptir` files under `filetests/` at the repository root.
//!
//! Each crate whose passes have filetests runs its own directory from its `tests/filetests.rs`:
//! `gpt-rs-ir` runs the directory of each dialect, `filetests/<dialect>/`, and a backend crate
//! runs `filetests/backend/<backend>/` through `backend::run` (feature `backend`). A pass case
//! names the passes that `#[pass]` registers in the linked crates, and in a backend's filetests
//! `default` names the backend's pipeline.
//!
//! A file starts with `// RUN: <mode>` and holds cases separated by `// -----`. A case starts at
//! its first non-blank line, and the line numbers of parse errors count from there. Modes:
//!
//! - `roundtrip`: the case must parse and verify. Its printed text, which holds the locations of
//!   its ops, must parse, print back unchanged and satisfy the case's CHECK lines. The case needs
//!   at least one CHECK line. A case that starts with the spec version prints with it, as a stored
//!   program does.
//! - `parse`: the case fails to parse with an error that its `// ERROR: <message>` line matches
//!   (see the ERROR rules below).
//! - `verify`: the case parses and fails verification with an error that its
//!   `// ERROR: <message>` line matches (see the ERROR rules below).
//! - `pass <passes>`: runs a pass list such as `canonicalize {fuel = 10}, cse`, or `default`,
//!   verifying after each pass. The output must parse, print
//!   back unchanged and satisfy the case's CHECK lines, of which it needs at least one. The passes
//!   must report a change exactly when the text changes. Every run-time function must compute the
//!   same results before and after on random inputs, as [`bits_eq`] compares them, unless the case
//!   has a `// NO-INTERP: <reason>` line.
//!
//!   A pass case with an `// ERROR: <message>` line instead fails in its passes with an error
//!   that this line matches (see the ERROR rules below). One with a `// LOCATIONS: <reason>` line
//!   prints its output with the location of each op, trails included, for its CHECK lines.
//!
//!   The inputs start with the special values of [`Specials::All`]. When the function before the
//!   passes cannot run on them, as when they are out-of-range indices, plain random values replace
//!   them. Every parameter that a function reads gets one value before and after, and the
//!   load-time functions run first: their results bind the parameters they compute. A case with a
//!   `// ROUNDING: <classes>: <reason>` line is for a pass whose rewrite holds in real arithmetic
//!   but rounds differently, in the ways its classes name ([`Rounding`]). Its inputs start with
//!   [`Specials::Finite`], and each result may differ from the reference as far as its loosest
//!   class allows.
//!
//! An `// ERROR:` message must match the whole error. As in CHECK lines, a `{{regex}}` in it
//! matches a regex. The regex stands for text that the case cannot know in advance, such as the
//! number of an op that an earlier pass made.
//!
//! In a filetest, `%S` stands for the directory of the file, so a pass option can name a file
//! beside it, `pass schedule {target = "%S/one-core.toml"}`, and a message or a CHECK line can
//! name the same file.
//!
//! `GPTRS_FILETEST=<text>` runs only the files whose path contains `<text>`.
//!
//! CHECK lines: `CHECK:` finds the pattern after the previous match, `CHECK-NEXT:` on the next
//! line, `CHECK-NOT:` nowhere between the previous match and the next one. Any other comment line
//! that names CHECK is an error. In a pattern, `[[NAME:regex]]` captures, `[[NAME]]` reuses a
//! capture and `{{regex}}` matches a regex. Everything else is literal.
//!
//! Values have no names in the IR: the parser reads any name, and the printer numbers the values.
//! A case names each value for what it holds (`%weight`, `%scores`). Its CHECK lines capture a
//! printed value under such a name and reuse it. They never match a printed number:
//! `[[SUM:%[0-9]+]] = tensor.add([[LHS]], [[RHS]])`.

#[cfg(feature = "backend")]
pub mod backend;
mod check;
mod differential;

use std::fs;
use std::path::{Path, PathBuf};

use gpt_rs_ir::ir::{verify, Module};
use gpt_rs_ir::passes::{self, Pipeline};
use gpt_rs_ir::text::TextOptions;

pub use self::differential::{
    bits_eq, interp_differential, random_value, rotation, run_dims, Rounding, Specials, ROUNDS,
};

/// The `filetests/` directory at the repository root.
pub fn filetests_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../filetests")
}

/// Every `.ptir` file under `dir`, in sorted order.
pub fn collect(dir: &Path, out: &mut Vec<PathBuf>) {
    let mut entries: Vec<PathBuf> = fs::read_dir(dir)
        .unwrap_or_else(|err| panic!("cannot read {}: {err}", dir.display()))
        .map(|entry| entry.expect("dir entry").path())
        .collect();
    entries.sort();
    for path in entries {
        if path.is_dir() {
            collect(&path, out);
        } else if path
            .extension()
            .is_some_and(|extension| extension == "ptir")
        {
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

/// `text`, of the filetest at `path`, as it runs: `%S` replaced by the file's directory.
pub fn substitute(text: &str, path: &Path) -> String {
    let directory = path.parent().expect("a filetest is in a directory");
    text.replace("%S", &directory.display().to_string())
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
    for (line_index, line) in lines {
        if line.trim() == "// -----" {
            cases.push(Case {
                line: start + 1,
                text: std::mem::take(&mut current),
            });
            continue;
        }
        if current.trim().is_empty() {
            start = line_index;
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
pub fn run(dir: &str, default: DefaultPipeline) {
    let root = filetests_root();
    let mut files = Vec::new();
    collect(&root.join(dir), &mut files);
    assert!(!files.is_empty(), "no filetests under filetests/{dir}");
    if let Ok(filter) = std::env::var("GPTRS_FILETEST") {
        files.retain(|path| path.to_string_lossy().contains(&filter));
    }
    let mut failures = Vec::new();
    let mut cases = 0;
    for path in &files {
        let text = substitute(&fs::read_to_string(path).expect("read filetest"), path);
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

/// The text after each `// <name>:` line of a case, such as the message of its `// ERROR:` line.
pub fn directive<'a>(case: &'a str, name: &str) -> Vec<&'a str> {
    let prefix = format!("// {name}:");
    case.lines()
        .filter_map(|line| line.trim().strip_prefix(prefix.as_str()).map(str::trim))
        .collect()
}

/// The case without comment lines and blank lines: what a canonical printer emits.
pub fn code_lines(case: &str) -> String {
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

/// Checks that `result` is an error that the case's `// ERROR:` line matches, by the ERROR rules
/// of the crate doc.
fn expect_error<T>(case: &str, result: anyhow::Result<T>) -> Result<(), String> {
    let messages = directive(case, "ERROR");
    let [expected] = messages.as_slice() else {
        return Err("the case needs exactly one `// ERROR:` line".into());
    };
    let err = match result {
        Ok(_) => return Err(format!("no error, expected `{expected}`")),
        Err(err) => format!("{err:#}"),
    };
    if check::message_matches(expected, &err)? {
        Ok(())
    } else {
        Err(format!("error is `{err}`, expected `{expected}`"))
    }
}

fn run_case(run: &str, case: &Case, default: DefaultPipeline) -> Result<(), String> {
    let (mode, args) = run
        .split_once(' ')
        .map_or((run, ""), |(name, rest)| (name, rest.trim()));
    let parsed = Module::from_text(&case.text);
    if mode == "parse" {
        return expect_error(&case.text, parsed);
    }
    let module = parsed.map_err(|err| format!("parse error: {err:#}"))?;
    match mode {
        "roundtrip" => {
            verify(&module).map_err(|err| format!("verify error: {err}"))?;
            let checks = check::check_lines(&case.text)?;
            if checks.is_empty() {
                return Err("a roundtrip case needs a CHECK line".into());
            }
            let options = TextOptions {
                versioned: gpt_rs_ir::text::is_versioned(&case.text),
                locations: true,
                numbers: false,
            };
            let printed = module.print(options);
            let reprinted = Module::from_text(&printed).map(|reparsed| reparsed.print(options));
            if reprinted.as_ref().ok() != Some(&printed) {
                return Err(format!(
                    "the printed text does not print back unchanged: {reprinted:?}\n--- printed\n\
                     {printed}"
                ));
            }
            check::file_check(&checks, &printed)
                .map_err(|err| format!("{err}\n--- printed\n{printed}"))
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
    default: DefaultPipeline,
) -> Result<(), String> {
    verify(&before).map_err(|err| format!("the input does not verify: {err}"))?;
    let pipeline = match (list, default) {
        ("default", Some(build_default)) => build_default(),
        ("default", None) => {
            return Err("`default` names a backend's pipeline, and there is none here".into())
        }
        _ => passes::parse(list).map_err(|err| format!("pass list: {err:#}"))?,
    };
    if !directive(&case.text, "ERROR").is_empty() {
        let mut after = before.clone();
        return expect_error(&case.text, pipeline.run(&mut after));
    }
    let checks = check::check_lines(&case.text)?;
    if checks.is_empty() {
        return Err("a pass case needs a CHECK line".into());
    }
    let mut after = before.clone();
    let changed = pipeline.run(&mut after).map_err(|err| format!("{err:#}"))?;
    let text_changed = after.to_text() != before.to_text();
    let options = TextOptions {
        locations: !directive(&case.text, "LOCATIONS").is_empty(),
        ..TextOptions::default()
    };
    let output = after.print(options);
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
    let reprinted = Module::from_text(&output).map(|reparsed| reparsed.print(options));
    if reprinted.as_ref().ok() != Some(&output) {
        return Err(format!(
            "the output does not print back unchanged: {reprinted:?}\n--- output\n{output}"
        ));
    }
    check::file_check(&checks, &output).map_err(|err| format!("{err}\n--- output\n{output}"))?;
    if directive(&case.text, "NO-INTERP").is_empty() {
        let rounding = match directive(&case.text, "ROUNDING").as_slice() {
            [] => None,
            [line] => Some(Rounding::of_directive(line)?),
            _ => return Err("a case has at most one `// ROUNDING:` line".into()),
        };
        differential::interp_differential(&before, &after, case.line as u64, rounding)?;
    }
    Ok(())
}
