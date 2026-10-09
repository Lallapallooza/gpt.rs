//! `docs/ptir/tensor.md`: the general rules of the `tensor` dialect, then each op, then each
//! trait.

use std::fmt::Write as _;
use std::fs;
use std::path::Path;

use anyhow::{bail, Result};
use gpt_rs_ir::attr::ATTR_ENUMS;
use gpt_rs_ir::dialects::tensor::TensorOp;
use gpt_rs_ir::op::{Arity, Effects, Group, Kind, OpInfo, FOLDS};
use gpt_rs_ir::rewrite::RootInfo;
use gpt_rs_ir::traits::OpTrait;

use crate::filetests::Case;
use crate::{cell, defined_by, inline_list, readme, Catalog, HEADER};

pub fn write(root: &Path, filetests: &[Case], catalog: &Catalog) -> Result<String> {
    let spec = fs::read_to_string(root.join("crates/gpt-rs-ir/src/dialects/tensor/README.md"))?;
    let mut out = String::from(HEADER);
    out.push_str(readme(&spec).trim_end());
    out.push_str("\n\n## Ops\n\n");
    let infos = TensorOp::infos();
    for info in &infos {
        writeln!(out, "- {}", inline_list(&op_names(info)))?;
    }
    for info in &infos {
        out.push('\n');
        op_section(&mut out, info, filetests, catalog)?;
    }
    traits_section(&mut out, &infos, catalog)?;
    Ok(out)
}

/// The names of the op, or of every op of the family.
fn op_names(info: &OpInfo) -> Vec<String> {
    let kinds = (info.kinds)();
    if kinds.is_empty() {
        return vec![info.name.to_string()];
    }
    // The placeholder names the family's field: `{kind}`, `{distribution}`.
    let (prefix, rest) = info
        .name
        .split_once('{')
        .expect("a family name has a placeholder");
    let suffix = rest.split_once('}').map_or("", |(_, suffix)| suffix);
    kinds
        .iter()
        .map(|kind| format!("{prefix}{}{suffix}", kind.mnemonic))
        .collect()
}

fn op_section(
    out: &mut String,
    info: &OpInfo,
    filetests: &[Case],
    catalog: &Catalog,
) -> Result<()> {
    let names = op_names(info);
    writeln!(out, "### {}\n", catalog.op_label(info.rust))?;
    writeln!(out, "{}\n", body(info.doc))?;
    writeln!(out, "{}\n", defined_by(info.rust, info.file))?;
    writeln!(out, "| | |\n|---|---|")?;
    writeln!(out, "| Operands | {} |", groups(info.operands))?;
    writeln!(out, "| Results | {} |", groups(info.results))?;
    if !info.regions.is_empty() {
        let regions: Vec<String> = info.regions.iter().map(|r| r.to_string()).collect();
        writeln!(out, "| Regions | {} |", inline_list(&regions))?;
    }
    if !info.attrs.is_empty() {
        let attrs: Vec<String> = info
            .attrs
            .iter()
            .map(|attr| {
                let optional = if attr.default { ", optional" } else { "" };
                format!("`{}`: `{}`{optional}", attr.name, pretty_type(attr.ty))
            })
            .collect();
        writeln!(out, "| Attributes | {} |", attrs.join("; "))?;
    }
    if !info.traits.is_empty() {
        writeln!(out, "| Traits | {} |", traits(info.traits))?;
    }
    writeln!(
        out,
        "| Effects | {} |",
        effects(&info.effects, info.operands)
    )?;
    if let Some(fold) = FOLDS.iter().find(|entry| entry.rust == info.rust) {
        writeln!(out, "| Folds | {} |", cell(fold.doc))?;
    }
    let patterns = catalog.patterns_at(RootInfo::Op(info.rust));
    if !patterns.is_empty() {
        writeln!(out, "| Patterns | {} |", pattern_list(&patterns))?;
    }
    for attr in info.attrs {
        let ty = attr
            .ty
            .strip_prefix("Option<")
            .and_then(|ty| ty.strip_suffix('>'))
            .unwrap_or(attr.ty);
        let Some(values) = ATTR_ENUMS.iter().find(|entry| entry.name == ty) else {
            continue;
        };
        writeln!(
            out,
            "
| `{}` | |
|---|---|",
            attr.name
        )?;
        for (value, doc) in (values.values)() {
            writeln!(out, "| `{value}` | {} |", cell(doc))?;
        }
    }
    let kinds = (info.kinds)();
    let with_traits = kinds.iter().any(|kind| !kind.traits.is_empty());
    if with_traits {
        writeln!(out, "\n| Kind | Traits | |\n|---|---|---|")?;
    } else if !kinds.is_empty() {
        writeln!(out, "\n| Kind | |\n|---|---|")?;
    }
    for kind in &kinds {
        let doc = cell(kind.doc);
        if with_traits {
            let traits = traits(kind.traits);
            writeln!(out, "| `{}` | {traits} | {doc} |", kind.mnemonic)?;
        } else {
            writeln!(out, "| `{}` | {doc} |", kind.mnemonic)?;
        }
    }
    let untested: Vec<&String> = names
        .iter()
        .filter(|name| {
            !filetests
                .iter()
                .any(|case| case.mode == "roundtrip" && case.text.contains(&format!("{name}(")))
        })
        .collect();
    if !untested.is_empty() {
        bail!("no roundtrip filetest uses {untested:?}");
    }
    if let Some(example) = example(info, &names, filetests) {
        writeln!(
            out,
            "\nExample, from `{}`:\n\n```text\n{}```",
            example.file,
            code(&example.text)
        )?;
    }
    let rejections = rejections(&names, filetests);
    if !rejections.is_empty() {
        writeln!(out, "\nRejected, as the verify filetests pin it:\n")?;
        for message in rejections {
            writeln!(out, "- `{message}`")?;
        }
    }
    Ok(())
}

/// Patterns with the pass that runs each: `` `CollapseTransposeChain` (`canonicalize`) ``.
fn pattern_list(patterns: &[(&str, &str)]) -> String {
    patterns
        .iter()
        .map(|(pattern, pass)| format!("`{pattern}` (`{pass}`)"))
        .collect::<Vec<_>>()
        .join("; ")
}

/// Every trait that an op of the dialect, or one of its kinds, has: its doc comment, the ops
/// that have it, and the patterns rooted at it.
fn traits_section(out: &mut String, infos: &[OpInfo], catalog: &Catalog) -> Result<()> {
    out.push_str(
        "\n## Traits\n\nA trait states a fact about the ops that have it. The folds and patterns \
         it implies apply to every one of them; [passes.md](passes.md) says which pass runs \
         each.\n",
    );
    for (name, t) in &catalog.traits {
        let ops: Vec<String> = infos
            .iter()
            .flat_map(|info| {
                let names = op_names(info);
                let kinds = (info.kinds)();
                let has = |traits: &[&dyn OpTrait]| traits.iter().any(|o| o.name() == *name);
                if has(info.traits) {
                    return names;
                }
                names
                    .into_iter()
                    .zip(kinds.iter())
                    .filter(|(_, kind)| has(kind.traits))
                    .map(|(name, _)| name)
                    .collect()
            })
            .collect();
        if ops.is_empty() {
            continue;
        }
        writeln!(out, "\n### `{name}`\n")?;
        writeln!(out, "{}\n", t.doc())?;
        writeln!(out, "{}\n", defined_by(name, t.file()))?;
        writeln!(out, "Ops: {}.", inline_list(&ops))?;
        let patterns = catalog.patterns_at(RootInfo::Trait(name));
        if !patterns.is_empty() {
            writeln!(out, "\nPatterns rooted at it: {}.", pattern_list(&patterns))?;
        }
    }
    Ok(())
}

/// The lines of a case that are not comments or blank: the program it holds.
fn code(case: &str) -> String {
    case.lines()
        .filter(|line| !line.trim().is_empty() && !line.trim_start().starts_with("//"))
        .map(|line| format!("{line}\n"))
        .collect()
}

/// The doc comment without the name it starts with (`` `tensor.transpose`: ``), which the
/// heading shows.
fn body(doc: &str) -> String {
    let doc = doc.trim();
    let rest = match doc.split_once(": ") {
        Some((name, rest)) if name.starts_with('`') && !name.contains('\n') => rest,
        _ => doc,
    };
    let mut chars = rest.chars();
    match chars.next() {
        Some(first) => first.to_uppercase().chain(chars).collect(),
        None => String::new(),
    }
}

fn groups(groups: &[Group]) -> String {
    if groups.is_empty() {
        return "none".into();
    }
    groups
        .iter()
        .map(|group| {
            let kind = match group.kind {
                Kind::Any => "value",
                Kind::Tensor => "tensor",
                Kind::Index => "index",
                Kind::Scalar => "scalar",
            };
            let arity = match group.arity {
                Arity::One => kind.to_string(),
                Arity::Optional => format!("optional {kind}"),
                Arity::Variadic => format!("{kind}s"),
            };
            format!("`{}`: {arity}", group.name)
        })
        .collect::<Vec<_>>()
        .join("; ")
}

/// A Rust attribute type in the terms of the text form: `SmallVec<[usize;4]>` is `[usize]`.
fn pretty_type(ty: &str) -> String {
    if let Some(inner) = ty.strip_prefix("Option<").and_then(|t| t.strip_suffix('>')) {
        return pretty_type(inner);
    }
    if let Some(inner) = ty
        .strip_prefix("SmallVec<[")
        .and_then(|t| t.strip_suffix("]>"))
    {
        let element = inner.rsplit_once(';').map_or(inner, |(element, _)| element);
        return format!("[{}]", pretty_type(element));
    }
    if let Some(inner) = ty.strip_prefix("Vec<").and_then(|t| t.strip_suffix('>')) {
        return format!("[{}]", pretty_type(inner));
    }
    match ty {
        "DType" => "dtype".into(),
        "Dim" => "dim".into(),
        "Dense" => "dense".into(),
        "F64" => "f64".into(),
        "TensorType" => "tensor type".into(),
        "(usize,usize)" => "(usize, usize)".into(),
        other => other.to_string(),
    }
}

fn traits(traits: &[&dyn OpTrait]) -> String {
    traits
        .iter()
        .map(|t| t.to_string())
        .collect::<Vec<_>>()
        .join("; ")
}

fn effects(effects: &Effects, operands: &[Group]) -> String {
    let mut out = Vec::new();
    if effects.opaque {
        out.push("opaque".to_string());
    }
    if effects.rng {
        out.push("draws random numbers".to_string());
    }
    if effects.alloc {
        out.push("allocates".to_string());
    }
    for (bits, verb) in [(effects.reads, "reads"), (effects.writes, "writes")] {
        for (index, group) in operands.iter().enumerate() {
            if bits & (1 << index) != 0 {
                out.push(format!("{verb} the memory of `{}`", group.name));
            }
        }
    }
    if out.is_empty() {
        "none".into()
    } else {
        out.join(", ")
    }
}

/// A roundtrip case that uses one of `names`, the ops of a family or the one op. It prefers a case
/// whose function is named after the op, then one in `roundtrip/<stem>.ptir`, where `<stem>`
/// names the op's file, then one that uses more of the names, then the shortest.
fn example<'a>(info: &OpInfo, names: &[String], filetests: &'a [Case]) -> Option<&'a Case> {
    let stem = Path::new(info.file).file_stem()?.to_str()?;
    let own_file = format!("roundtrip/{stem}.ptir");
    filetests
        .iter()
        .filter(|case| case.mode == "roundtrip")
        .filter(|case| {
            names
                .iter()
                .any(|name| case.text.contains(&format!("{name}(")))
        })
        .min_by_key(|case| {
            let named = names.iter().any(|name| {
                let short = name.rsplit('.').next().unwrap_or(name);
                case.text.contains(&format!("func.func @{short}"))
            });
            let used = names
                .iter()
                .filter(|name| case.text.contains(&format!("{name}(")))
                .count();
            (
                !named,
                !case.file.ends_with(&own_file),
                std::cmp::Reverse(used),
                code(&case.text).lines().count(),
            )
        })
}

/// The messages of the verify filetests that reject one of `names`.
fn rejections(names: &[String], filetests: &[Case]) -> Vec<String> {
    filetests
        .iter()
        .filter(|case| case.mode == "verify")
        .flat_map(|case| case.text.lines())
        .filter_map(|line| line.trim().strip_prefix("// ERROR:"))
        .map(str::trim)
        .filter(|message| {
            names
                .iter()
                .any(|name| message.starts_with(&format!("{name}: ")))
        })
        .map(str::to_string)
        .collect()
}
