//! The filetests the pages draw on: their cases, and the rewrites that fire in them.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;

use anyhow::{Context, Result};
use gpt_rs_ir::doc::{record_fired, Fired};
use gpt_rs_ir::module::Module;
use gpt_rs_ir::op::OpDyn;
use gpt_rs_ir::passes::{self, Pipeline};

/// One filetest case: its file, relative to the repository, its mode, its text, and what the
/// module it holds names, when it parses.
pub struct Case {
    pub file: String,
    pub mode: String,
    pub text: String,
    /// The names of the ops it holds, at any depth.
    pub ops: BTreeSet<String>,
    /// The symbols it defines: the names of its functions and dispatches.
    pub symbols: BTreeSet<String>,
    /// The messages of its `// ERROR:` lines.
    pub errors: Vec<String>,
}

/// The names of the ops and the symbols of the module that `text` holds, or none when it does not
/// parse.
fn names(text: &str) -> (BTreeSet<String>, BTreeSet<String>) {
    let mut ops = BTreeSet::new();
    let mut symbols = BTreeSet::new();
    let Ok(module) = Module::from_text(text) else {
        return (ops, symbols);
    };
    for inst in module.walk(module.body()) {
        let op = module.op(inst);
        ops.insert(op.name().to_string());
        if let Some(symbol) = op.dyn_symbol() {
            symbols.insert(symbol.symbol().to_string());
        }
    }
    (ops, symbols)
}

/// The cases of every filetest, in file order, as the filetest runner splits them.
pub fn read() -> Result<Vec<Case>> {
    let root = gpt_rs_filetests::filetests_root();
    let mut files = Vec::new();
    gpt_rs_filetests::collect(&root, &mut files);
    let mut cases = Vec::new();
    for path in files {
        let text = fs::read_to_string(&path)?;
        let relative = path.strip_prefix(&root).unwrap_or(&path);
        let file = Path::new("filetests").join(relative).display().to_string();
        let (mode, file_cases) = gpt_rs_filetests::split_cases(&text);
        for case in file_cases {
            let (ops, symbols) = names(&case.text);
            let mut errors = Vec::new();
            for message in gpt_rs_filetests::directive(&case.text, "ERROR") {
                errors.push(message.to_string());
            }
            cases.push(Case {
                file: file.clone(),
                mode: mode.clone(),
                text: case.text,
                ops,
                symbols,
                errors,
            });
        }
    }
    Ok(cases)
}

/// Where each rewrite fired: by pass and rewrite, the files whose cases it changed an op in.
pub struct Fires(BTreeMap<(&'static str, Fired), BTreeSet<String>>);

impl Fires {
    /// Runs every pass case of `filetests` and records which rewrites fire under which pass. A
    /// case that runs `default` runs the pipeline of the backend whose filetests hold it.
    pub fn record(filetests: &[Case], backends: &[(&str, Pipeline)]) -> Result<Self> {
        let mut fires: BTreeMap<_, BTreeSet<String>> = BTreeMap::new();
        for case in filetests {
            let Some(list) = case.mode.strip_prefix("pass ") else {
                continue;
            };
            // A case with an `ERROR:` line fails in its passes, so nothing fires to the end.
            if !case.errors.is_empty() {
                continue;
            }
            let parsed;
            let pipeline = if list == "default" {
                let (_, pipeline) = backends
                    .iter()
                    .find(|(name, _)| case.file.contains(&format!("/backend/{name}/")))
                    .with_context(|| format!("{}: no backend runs `default`", case.file))?;
                pipeline
            } else {
                parsed = passes::parse(list).with_context(|| case.file.clone())?;
                &parsed
            };
            let mut module = Module::from_text(&case.text).with_context(|| case.file.clone())?;
            for pass in pipeline.passes() {
                let (result, fired) = record_fired(|| passes::run_pass(pass.as_ref(), &mut module));
                result.with_context(|| format!("{}: {}", case.file, pass.label()))?;
                for rewrite in fired {
                    fires
                        .entry((pass.name(), rewrite))
                        .or_default()
                        .insert(case.file.clone());
                }
            }
        }
        Ok(Self(fires))
    }

    /// The files in which `rewrite` fired under `pass`, or under any pass, once each.
    pub fn files(&self, pass: Option<&str>, rewrite: Fired) -> Vec<&str> {
        let files: BTreeSet<&str> = self
            .0
            .iter()
            .filter(|((p, r), _)| pass.is_none_or(|pass| *p == pass) && *r == rewrite)
            .flat_map(|(_, files)| files.iter().map(String::as_str))
            .collect();
        files.into_iter().collect()
    }

    /// The traits whose folds fired, by name, once each.
    pub fn trait_folds(&self) -> Vec<&'static str> {
        let names: BTreeSet<&'static str> = self
            .0
            .keys()
            .filter_map(|(_, rewrite)| match rewrite {
                Fired::TraitFold(name) => Some(*name),
                _ => None,
            })
            .collect();
        names.into_iter().collect()
    }
}
