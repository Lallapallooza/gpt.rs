//! Writes the reference pages of the IR from the code: `docs/ptir/<dialect>.md` for each dialect
//! with a specification (`tensor`, `arith`, `tile`, `host`), and `docs/ptir/passes.md`, the
//! passes, what they rewrite and the pipelines of the backends. With `--check`, it writes nothing
//! and fails when the pages in the repository differ from what the code gives.
//!
//! An op's entry comes from its struct: the doc comment, and what `#[op]` records of its groups,
//! attributes, traits and effects. Its example is a roundtrip filetest that uses it, and
//! its rejections are the messages its verify filetests pin. A pass's entry comes from what
//! `#[pass]` records, and lists the folds and patterns it applies with their doc comments, which
//! `#[fold]`, `#[pattern]` and `#[derive(Doc)]` record. The generator runs the pass filetests to
//! learn where each rewrite fires, and fails on a folder or pattern that fires in none. It also
//! fails on a pass that is not alone in a file named after it.

mod dialect_page;
mod filetests;
mod passes_page;

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use anyhow::{bail, ensure, Context, Result};
use gpt_rs::backend::PortableBackend;
use gpt_rs_ir::dialects::arith::ArithOp;
use gpt_rs_ir::dialects::func::FuncOp;
use gpt_rs_ir::dialects::host::HostOp;
use gpt_rs_ir::dialects::tensor::TensorOp;
use gpt_rs_ir::dialects::tile::TileOp;
use gpt_rs_ir::op::OpInfo;
use gpt_rs_ir::passes::{self, PassEntry, Pipeline, Rewrites};
use gpt_rs_ir::rewrite::RootInfo;
use gpt_rs_ir::traits::OpTrait;

const HEADER: &str = "<!-- Generated from the code by `cargo run -p gpt-rs-docs`. Do not edit: \
                      change the code and run it again. -->\n\n";

fn main() -> Result<()> {
    let check = std::env::args().skip(1).any(|arg| arg == "--check");
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let filetests = filetests::read()?;
    let backends = backends();
    let fires = filetests::Fires::record(&filetests, &backends)?;
    let catalog = Catalog::new()?;
    let dialects = [
        ("tensor", TensorOp::infos()),
        ("arith", ArithOp::infos()),
        ("tile", TileOp::infos()),
        ("host", HostOp::infos()),
    ];
    let mut pages = Vec::new();
    for (name, infos) in &dialects {
        let text = dialect_page::write(&root, name, infos, &filetests, &catalog)?;
        pages.push((format!("docs/ptir/{name}.md"), text));
    }
    pages.push((
        "docs/ptir/passes.md".to_string(),
        passes_page::write(&root, &filetests, &fires, &catalog, &backends)?,
    ));
    let mut stale = Vec::new();
    for (path, text) in pages {
        let file = root.join(&path);
        if check {
            if fs::read_to_string(&file).ok().as_deref() != Some(text.as_str()) {
                stale.push(path);
            }
            continue;
        }
        fs::create_dir_all(file.parent().expect("a docs directory"))?;
        fs::write(&file, text).with_context(|| format!("writing {path}"))?;
    }
    if !stale.is_empty() {
        bail!(
            "{} out of date with the code; run `cargo run -p gpt-rs-docs`",
            stale.join(" and ")
        );
    }
    Ok(())
}

/// Fails unless the pass is alone in a file named after it, as a struct named after it: pass
/// `merge-contracted-axes` is `MergeContractedAxes` in `merge_contracted_axes.rs`, or in
/// `merge_contracted_axes/mod.rs`. The name of a backend's pass starts with the backend,
/// `c-pack-weights`, which its file and struct leave out.
fn check_pass_layout(entry: &PassEntry) -> Result<()> {
    let path = Path::new(entry.file);
    let stem = match path.file_stem().and_then(|s| s.to_str()) {
        Some("mod") => path
            .parent()
            .and_then(Path::file_name)
            .and_then(|s| s.to_str()),
        stem => stem,
    };
    let backend = entry
        .file
        .strip_prefix("crates/gpt-rs-backend-")
        .and_then(|rest| rest.split('/').next());
    let name = backend
        .and_then(|backend| entry.name.strip_prefix(&format!("{backend}-")))
        .unwrap_or(entry.name);
    let file = name.replace('-', "_");
    ensure!(
        stem == Some(file.as_str()),
        "pass `{}` is in {}; each pass is alone in a file named after it, `{file}.rs`",
        entry.name,
        entry.file
    );
    let rust: String = name
        .split('-')
        .map(|word| {
            let mut chars = word.chars();
            chars.next().map_or_else(String::new, |first| {
                first.to_ascii_uppercase().to_string() + chars.as_str()
            })
        })
        .collect();
    ensure!(
        entry.rust == rust,
        "pass `{}` is the struct `{}`; a pass's struct is named after it, `{rust}`",
        entry.name,
        entry.rust
    );
    Ok(())
}

/// The backends of the repository, by name, with their pipelines.
fn backends() -> Vec<(&'static str, Pipeline)> {
    vec![
        (
            "ref-cpu",
            gpt_rs_backend_ref_cpu::CpuPortableBackend::new().pipeline(),
        ),
        (
            "faer",
            gpt_rs_backend_faer::FaerPortableBackend::new().pipeline(),
        ),
        ("c", gpt_rs_backend_c::CBackend::new().pipeline()),
    ]
}

/// What the pages know of the IR: the ops, every registered pass with the rewrites it applies,
/// and the op traits by name.
pub struct Catalog {
    pub ops: Vec<OpInfo>,
    pub passes: Vec<(&'static PassEntry, Rewrites)>,
    pub traits: BTreeMap<&'static str, &'static dyn OpTrait>,
}

impl Catalog {
    fn new() -> Result<Self> {
        let mut ops = TensorOp::infos();
        ops.extend(FuncOp::infos());
        ops.extend(ArithOp::infos());
        ops.extend(TileOp::infos());
        ops.extend(HostOp::infos());
        let mut traits = BTreeMap::new();
        for info in &ops {
            for &t in info.traits {
                traits.entry(t.name()).or_insert(t);
            }
        }
        let mut entries = Vec::new();
        for entry in passes::registered() {
            check_pass_layout(entry)?;
            let pipeline = passes::parse(entry.name)?;
            let [pass] = pipeline.passes() else {
                bail!("`{}` is not one pass", entry.name);
            };
            entries.push((entry, pass.rewrites()));
        }
        Ok(Self {
            ops,
            passes: entries,
            traits,
        })
    }

    /// How a page names the op struct `rust`: `tensor.transpose`.
    pub fn op_label(&self, rust: &str) -> String {
        match self.ops.iter().find(|info| info.rust == rust) {
            Some(info) => format!("`{}`", info.name),
            None => format!("`{rust}`"),
        }
    }

    pub fn root_label(&self, root: RootInfo) -> String {
        match root {
            RootInfo::Op(rust) => self.op_label(rust),
            RootInfo::Trait(name) => format!("every op with `{name}`"),
        }
    }

    /// The patterns whose root `at` takes, each with the pass that runs it and its root, once
    /// each.
    pub fn patterns_at(
        &self,
        at: impl Fn(RootInfo) -> bool,
    ) -> Vec<(&'static str, &'static str, RootInfo)> {
        let mut found = Vec::new();
        for (entry, rewrites) in &self.passes {
            for pattern in rewrites.patterns.iter() {
                let item = (pattern.name(), entry.name, pattern.root());
                if at(item.2) && !found.contains(&item) {
                    found.push(item);
                }
            }
        }
        found
    }
}

/// A README of the crate as a page shows it: its Rust blocks as rustdoc renders them, without the
/// lines that start with `# ` and with `rust` for their `rust,no_run` or `rust,ignore`.
fn readme(text: &str) -> String {
    let mut out = String::new();
    let mut in_rust = false;
    for line in text.trim_end().lines() {
        let fence = line.trim_start().strip_prefix("```");
        match fence {
            Some(info) if !in_rust && info.starts_with("rust") => {
                in_rust = true;
                out.push_str("```rust\n");
                continue;
            }
            Some("") if in_rust => in_rust = false,
            _ if in_rust && (line.trim() == "#" || line.trim_start().starts_with("# ")) => {
                continue;
            }
            _ => {}
        }
        out.push_str(line);
        out.push('\n');
    }
    out
}

/// `text` in one markdown table cell: paragraphs apart, lines joined, pipes escaped.
fn cell(text: &str) -> String {
    text.trim()
        .split("\n\n")
        .map(|paragraph| paragraph.split('\n').collect::<Vec<_>>().join(" "))
        .collect::<Vec<_>>()
        .join("<br><br>")
        .replace('|', "\\|")
}

fn inline_list(names: &[String]) -> String {
    names
        .iter()
        .map(|name| format!("`{name}`"))
        .collect::<Vec<_>>()
        .join(", ")
}

/// Where the struct `rust` is defined, as a link from `docs/ptir/` to its file.
fn defined_by(rust: &str, file: &str) -> String {
    format!("Defined by `{rust}` in [{file}](../../{file}).")
}

/// Links from `docs/ptir/` to filetest files, each shown without the `filetests/` prefix.
fn file_links(files: &[&str]) -> String {
    files
        .iter()
        .map(|file| {
            let shown = file.strip_prefix("filetests/").unwrap_or(file);
            format!("[{shown}](../../{file})")
        })
        .collect::<Vec<_>>()
        .join(", ")
}
