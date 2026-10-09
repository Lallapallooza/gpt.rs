//! The filetests the pages draw on: their cases, and the rewrites that fire in them.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;

use anyhow::{Context, Result};
use gpt_rs_ir::doc::{record_fired, Fired};
use gpt_rs_ir::module::Module;
use gpt_rs_ir::passes::{self, Pipeline};

/// One filetest case: its file, relative to the repository, its mode and the text of the case.
pub struct Case {
    pub file: String,
    pub mode: String,
    pub text: String,
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
        cases.extend(file_cases.into_iter().map(|case| Case {
            file: file.clone(),
            mode: mode.clone(),
            text: case.text,
        }));
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
