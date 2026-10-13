//! `ir-stats` and `ir-diff`: what the functions of a PTIR program cost, op by op, and how two
//! versions of a program differ, such as a dumped `program.ptir` before and after a change.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{anyhow, Context, Result};
use clap::{Args as ClapArgs, ValueEnum};
use gpt_rs_ir::analysis::{Cost, CostSummary};
use gpt_rs_ir::dialects::func::Func;
use gpt_rs_ir::ir::{Inst, Module};

/// How the tables group the ops.
#[derive(Clone, Copy, ValueEnum)]
pub enum Grouping {
    /// By op name.
    Op,
    /// By the module path of the layer that recorded them, the layer indices merged, such as
    /// `model.layers.*.linear_attn`.
    Scope,
    /// By layer: the module path up to the first index, such as `model.layers.3`.
    Layer,
}

#[derive(ClapArgs)]
pub struct IrStatsArgs {
    /// PTIR text module, such as a `program.ptir` that `--dump-dir` writes.
    #[arg(long)]
    ptir: PathBuf,

    /// Rows of the op table per function, the costliest first by bytes moved.
    #[arg(long, default_value_t = 20)]
    top: usize,

    /// Also show the load-time functions, which compute from parameters only.
    #[arg(long)]
    load_time: bool,

    /// How the table groups the ops. `scope` and `layer` read the locations that `--dump-dir`
    /// dumps keep.
    #[arg(long, value_enum, default_value = "op")]
    by: Grouping,
}

#[derive(ClapArgs)]
pub struct IrDiffArgs {
    /// The PTIR text module before the change.
    #[arg(long)]
    before: PathBuf,

    /// The PTIR text module after the change.
    #[arg(long)]
    after: PathBuf,

    /// Rows of the op table per function, the largest change first by bytes moved.
    #[arg(long, default_value_t = 20)]
    top: usize,

    /// Also compare the load-time functions, which compute from parameters only.
    #[arg(long)]
    load_time: bool,

    /// How the table groups the ops.
    #[arg(long, value_enum, default_value = "op")]
    by: Grouping,
}

/// Prints the cost of each function of the module, in all and for its costliest ops.
pub fn run_stats(args: IrStatsArgs) -> Result<()> {
    let functions = summaries(&args.ptir, args.load_time, args.by)?;
    for (name, summary) in &functions {
        println!("{}", headline(name, summary));
        let mut rows: Vec<(&String, &(usize, Cost))> = summary.groups.iter().collect();
        rows.sort_by_key(|(_, (_, cost))| std::cmp::Reverse(cost.bytes()));
        println!(
            "  {:<40} {:>7} {:>11} {:>11} {:>11}",
            "group", "ops", "GFLOP", "read MB", "written MB"
        );
        for (group, (count, cost)) in rows.into_iter().take(args.top) {
            println!(
                "  {group:<40} {count:>7} {:>11.3} {:>11.1} {:>11.1}",
                gflop(cost.flops),
                megabytes(cost.bytes_read),
                megabytes(cost.bytes_written)
            );
        }
        println!();
    }
    Ok(())
}

/// Prints, for each function of either module, how its ops and their cost changed.
pub fn run_diff(args: IrDiffArgs) -> Result<()> {
    let before = summaries(&args.before, args.load_time, args.by)?;
    let after = summaries(&args.after, args.load_time, args.by)?;
    let mut names: Vec<&String> = before.keys().collect();
    for name in after.keys() {
        if !before.contains_key(name) {
            names.push(name);
        }
    }
    let empty = CostSummary::default();
    for name in names {
        let old = before.get(name).unwrap_or(&empty);
        let new = after.get(name).unwrap_or(&empty);
        println!("before: {}", headline(name, old));
        println!("after:  {}", headline(name, new));
        let mut groups: Vec<&String> = old.groups.keys().collect();
        for group in new.groups.keys() {
            if !old.groups.contains_key(group) {
                groups.push(group);
            }
        }
        let mut rows = Vec::new();
        for group in groups {
            let (old_count, old_cost) = old.groups.get(group).copied().unwrap_or_default();
            let (new_count, new_cost) = new.groups.get(group).copied().unwrap_or_default();
            let moved = new_cost.bytes() as i128 - old_cost.bytes() as i128;
            if old_count != new_count || moved != 0 || old_cost.flops != new_cost.flops {
                rows.push((group, old_count, new_count, old_cost, new_cost, moved));
            }
        }
        rows.sort_by_key(|row| std::cmp::Reverse(row.5.abs()));
        println!(
            "  {:<40} {:>15} {:>12} {:>12}",
            "group", "ops", "GFLOP change", "MB change"
        );
        for (group, old_count, new_count, old_cost, new_cost, moved) in
            rows.into_iter().take(args.top)
        {
            let flops = new_cost.flops as f64 - old_cost.flops as f64;
            println!(
                "  {group:<40} {:>15} {:>+12.3} {:>+12.1}",
                format!("{old_count} -> {new_count}"),
                flops / 1e9,
                moved as f64 / 1e6
            );
        }
        println!();
    }
    Ok(())
}

/// The cost summary of each run-time function of the module at `path`, and of each load-time one
/// with `load_time`, by name, its ops grouped `by`.
fn summaries(path: &Path, load_time: bool, by: Grouping) -> Result<BTreeMap<String, CostSummary>> {
    let src =
        fs::read_to_string(path).with_context(|| format!("failed to read {}", path.display()))?;
    let module = Module::from_text(&src).with_context(|| format!("{}", path.display()))?;
    let mut functions = BTreeMap::new();
    for func in module.funcs() {
        let op = module
            .op(func)
            .get::<Func>()
            .ok_or_else(|| anyhow!("{}: a function is not a func.func", path.display()))?;
        if op.is_load_time() && !load_time {
            continue;
        }
        let body = module.regions(func)[0];
        let summary = module.cost_summary(body, |inst| group(&module, inst, by));
        functions.insert(op.name.clone(), summary);
    }
    Ok(functions)
}

/// The group of the op of `inst`.
fn group(module: &Module, inst: Inst, by: Grouping) -> String {
    let scope = || {
        let location = module.location(inst)?;
        location.scope.as_deref().map(str::to_string)
    };
    match by {
        Grouping::Op => module.op(inst).name().into_owned(),
        Grouping::Scope => match scope() {
            Some(scope) => merged_indices(&scope),
            None => "(no location)".to_string(),
        },
        Grouping::Layer => match scope() {
            Some(scope) => up_to_index(&scope),
            None => "(no location)".to_string(),
        },
    }
}

/// Whether a segment of a module path is an index into a list of layers, such as the `3` of
/// `model.layers.3.mlp`.
fn is_index(segment: &str) -> bool {
    !segment.is_empty() && segment.chars().all(|c| c.is_ascii_digit())
}

/// `scope` with each index segment as `*`.
fn merged_indices(scope: &str) -> String {
    let mut segments = Vec::new();
    for segment in scope.split('.') {
        segments.push(if is_index(segment) { "*" } else { segment });
    }
    segments.join(".")
}

/// `scope` up to its first index segment, such as `model.layers.3` of `model.layers.3.mlp`, or
/// all of it when it has none.
fn up_to_index(scope: &str) -> String {
    let mut segments = Vec::new();
    for segment in scope.split('.') {
        segments.push(segment);
        if is_index(segment) {
            break;
        }
    }
    segments.join(".")
}

/// One line with the function's name and what its ops cost in all.
fn headline(name: &str, summary: &CostSummary) -> String {
    let count: usize = summary.groups.values().map(|(count, _)| count).sum();
    let mut line = format!(
        "@{name}: {count} ops, {:.3} GFLOP, {:.1} MB read, {:.1} MB written",
        gflop(summary.total.flops),
        megabytes(summary.total.bytes_read),
        megabytes(summary.total.bytes_written)
    );
    if summary.unknown > 0 {
        line.push_str(&format!(
            ", {} ops of unknown cost left out",
            summary.unknown
        ));
    }
    line
}

fn gflop(flops: u64) -> f64 {
    flops as f64 / 1e9
}

fn megabytes(bytes: u64) -> f64 {
    bytes as f64 / 1e6
}
