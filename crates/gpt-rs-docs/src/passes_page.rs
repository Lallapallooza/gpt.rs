//! `docs/ptir/passes.md`: how passes are written, the pipeline of each backend, then each pass
//! with its options, statistics and the folds and patterns it applies.

use std::fmt::Write as _;
use std::fs;
use std::path::Path;

use anyhow::{bail, Result};
use gpt_rs_ir::ir::FOLDS;
use gpt_rs_ir::passes::{self, PassEntry, Pipeline};
use gpt_rs_ir::rewrite::{Fired, PatternSet, Rewrites};

use crate::filetests::{Case, Fires};
use crate::{cell, defined_by, file_links, readme, Catalog, HEADER};

pub fn write(
    root: &Path,
    filetests: &[Case],
    fires: &Fires,
    catalog: &Catalog,
    backends: &[(&str, Pipeline)],
) -> Result<String> {
    let intro = fs::read_to_string(root.join("crates/gpt-rs-ir/src/passes/README.md"))?;
    let mut out = String::from(HEADER);
    out.push_str(readme(&intro).trim_end());

    out.push_str("\n\n## The pipelines of the backends\n");
    // Each distinct pipeline once, after the backends that run it.
    let mut groups: Vec<(Vec<&str>, Vec<String>)> = Vec::new();
    for (name, pipeline) in backends {
        let labels = labels(pipeline);
        match groups.iter_mut().find(|(_, other)| *other == labels) {
            Some((names, _)) => names.push(name),
            None => groups.push((vec![name], labels)),
        }
    }
    for (names, labels) in &groups {
        let quoted: Vec<String> = names.iter().map(|name| format!("`{name}`")).collect();
        writeln!(out, "\n{}:\n", quoted.join(" and "))?;
        for (index, label) in labels.iter().enumerate() {
            writeln!(out, "{}. `{label}`", index + 1)?;
        }
    }

    let mut unfired = Vec::new();
    folds_section(&mut out, fires, catalog, &mut unfired)?;
    out.push_str("\n## Passes\n");
    for (entry, rewrites) in &catalog.passes {
        out.push('\n');
        pass_section(&mut out, entry, rewrites, filetests, fires, catalog)?;
        for set in [&rewrites.patterns, &rewrites.once] {
            for pattern in set.iter() {
                let fired = !fires
                    .files(Some(entry.name), Fired::Pattern(pattern.name()))
                    .is_empty();
                let missing = format!("`{}` under `{}`", pattern.name(), entry.name);
                if !fired && !unfired.contains(&missing) {
                    unfired.push(missing);
                }
            }
        }
    }
    if !unfired.is_empty() {
        bail!(
            "these rewrites fire in no filetest; add a case that exercises each:\n  {}",
            unfired.join("\n  ")
        );
    }
    Ok(out)
}

/// The table of the patterns of `set` under `title`, one row per pattern with its roots, when it
/// has any.
fn patterns_table(
    out: &mut String,
    title: &str,
    set: &PatternSet,
    catalog: &Catalog,
    fired_in: &dyn Fn(Fired) -> String,
) -> Result<()> {
    let mut patterns: Vec<(&str, Vec<String>, &str)> = Vec::new();
    for pattern in set.iter() {
        let root = catalog.root_label(pattern.root());
        match patterns
            .iter_mut()
            .find(|(name, ..)| *name == pattern.name())
        {
            Some((_, roots, _)) => roots.push(root),
            None => patterns.push((pattern.name(), vec![root], pattern.doc())),
        }
    }
    if patterns.is_empty() {
        return Ok(());
    }
    writeln!(
        out,
        "\n{title}:\n\n| Pattern | Root | Rewrite | Fires in |\n|---|---|---|---|"
    )?;
    for (name, roots, doc) in patterns {
        let fired = fired_in(Fired::Pattern(name));
        let root_list = roots.join(", ");
        writeln!(out, "| `{name}` | {root_list} | {} | {fired} |", cell(doc))?;
    }
    Ok(())
}

/// The folds that `canonicalize` and `fold` apply: each op's own folder, then the folds of its
/// traits. A folder that fires in no filetest goes to `unfired`.
fn folds_section(
    out: &mut String,
    fires: &Fires,
    catalog: &Catalog,
    unfired: &mut Vec<String>,
) -> Result<()> {
    out.push_str(
        "\n## Folds\n\nThe passes that fold an op try its own `#[fold]` folder, then the folds \
         of its traits. A fold replaces the op by an existing value or by a constant that repeats \
         or moves constant bytes; it never computes on elements.\n",
    );
    let fired_in = |rewrite: Fired| match fires.files(None, rewrite) {
        files if files.is_empty() => "none".to_string(),
        files => file_links(&files),
    };
    writeln!(out, "\n| Op | Folds | Fires in |\n|---|---|---|")?;
    let mut folds: Vec<_> = FOLDS
        .iter()
        .map(|fold| (format!("`{}`", fold.name), fold))
        .collect();
    folds.sort_by(|(left, _), (right, _)| left.cmp(right));
    for (op, fold) in folds {
        let fired = fired_in(Fired::Fold(fold.name));
        if fired == "none" {
            unfired.push(format!("the fold of {op}"));
        }
        writeln!(out, "| {op} | {} | {fired} |", cell(fold.doc))?;
    }
    writeln!(
        out,
        "\nThe traits whose folds fire in the filetests; [tensor.md](tensor.md#traits) describes \
         every trait.\n\n| Trait | | Fires in |\n|---|---|---|"
    )?;
    for name in fires.trait_folds() {
        let doc = catalog
            .traits
            .get(name)
            .map_or("", |op_trait| op_trait.doc());
        let fired = fired_in(Fired::TraitFold(name));
        writeln!(out, "| `{name}` | {} | {fired} |", cell(doc))?;
    }
    Ok(())
}

fn labels(pipeline: &Pipeline) -> Vec<String> {
    pipeline.passes().iter().map(|pass| pass.label()).collect()
}

fn pass_section(
    out: &mut String,
    entry: &PassEntry,
    rewrites: &Rewrites,
    filetests: &[Case],
    fires: &Fires,
    catalog: &Catalog,
) -> Result<()> {
    writeln!(out, "### `{}`\n", entry.name)?;
    writeln!(out, "{}\n", entry.doc.trim())?;
    writeln!(out, "{}", defined_by(entry.rust, entry.file))?;
    if !entry.options.is_empty() {
        writeln!(out, "\n| Option | Type | Default | |\n|---|---|---|---|")?;
        for option in entry.options {
            writeln!(
                out,
                "| `{}` | `{}` | `{}` | {} |",
                option.name,
                option.ty,
                (option.default)(),
                cell(option.doc)
            )?;
        }
    }
    if !entry.stats.is_empty() {
        writeln!(out, "\n| Statistic | |\n|---|---|")?;
        for stat in entry.stats {
            writeln!(out, "| `{}` | {} |", stat.name, stat.doc)?;
        }
    }
    let fired_in = |rewrite: Fired| match fires.files(Some(entry.name), rewrite) {
        files if files.is_empty() => "none".to_string(),
        files => file_links(&files),
    };
    if rewrites.folds {
        writeln!(out, "\nIt applies every [fold](#folds).")?;
    }
    patterns_table(out, "Patterns", &rewrites.patterns, catalog, &fired_in)?;
    patterns_table(
        out,
        "Once-rewrites, which rewrite each root once, after the patterns",
        &rewrites.once,
        catalog,
        &fired_in,
    )?;
    let mut files: Vec<&str> = filetests
        .iter()
        .filter(|case| runs(&case.mode, entry.name))
        .map(|case| case.file.as_str())
        .collect();
    files.dedup();
    if files.is_empty() {
        bail!("no filetest runs the pass `{}`", entry.name);
    }
    writeln!(out, "\nFiletests: {}.", file_links(&files))?;
    Ok(())
}

/// Whether the pass list of a `pass` mode names `pass`.
fn runs(mode: &str, pass: &str) -> bool {
    let Some(list) = mode.strip_prefix("pass ") else {
        return false;
    };
    passes::parse(list)
        .map(|pipeline| pipeline.passes().iter().any(|listed| listed.name() == pass))
        .unwrap_or(false)
}
