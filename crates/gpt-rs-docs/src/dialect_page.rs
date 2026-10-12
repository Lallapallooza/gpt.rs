//! `docs/ptir/<dialect>.md`: the general rules of a dialect, from its `README.md`, then each op
//! or family of ops, then each trait.

use std::fmt::Write as _;
use std::fs;
use std::path::Path;

use anyhow::{bail, Result};
use gpt_rs_ir::attr::{AttrFormat, ATTR_ENUMS, ATTR_FORMATS};
use gpt_rs_ir::op::{Arity, Effects, FamilyInfo, Group, Kind, OpInfo, FOLDS};
use gpt_rs_ir::rewrite::RootInfo;
use gpt_rs_ir::traits::OpTrait;

use crate::filetests::Case;
use crate::{cell, defined_by, inline_list, readme, Catalog, HEADER};

/// The page of the dialect `name`, whose ops are `infos`.
pub fn write(
    root: &Path,
    name: &str,
    infos: &[OpInfo],
    filetests: &[Case],
    catalog: &Catalog,
) -> Result<String> {
    let spec =
        fs::read_to_string(root.join(format!("crates/gpt-rs-ir/src/dialects/{name}/README.md")))?;
    let mut out = String::from(HEADER);
    out.push_str(readme(&spec).trim_end());
    out.push_str("\n\n## Ops\n\n");
    let entries = entries(infos);
    for entry in &entries {
        let names: Vec<String> = entry.ops().iter().map(|op| op.name.to_string()).collect();
        writeln!(out, "- {}", inline_list(&names))?;
    }
    for entry in &entries {
        out.push('\n');
        match entry {
            Entry::Op(info) => op_section(&mut out, info, filetests, catalog)?,
            Entry::Family(family, ops) => {
                family_section(&mut out, family, ops, filetests, catalog)?
            }
        }
    }
    traits_section(&mut out, infos, catalog)?;
    Ok(out)
}

/// An entry of the page: an op of its own, or the ops of a family, which share one entry.
enum Entry<'a> {
    Op(&'a OpInfo),
    Family(FamilyInfo, Vec<&'a OpInfo>),
}

impl<'a> Entry<'a> {
    fn ops(&self) -> Vec<&'a OpInfo> {
        match self {
            Entry::Op(info) => vec![info],
            Entry::Family(_, ops) => ops.clone(),
        }
    }
}

/// The entries of the ops `infos`, the ops of a family together, in order.
fn entries(infos: &[OpInfo]) -> Vec<Entry<'_>> {
    let mut entries: Vec<Entry<'_>> = Vec::new();
    for info in infos {
        let Some(family) = info.family else {
            entries.push(Entry::Op(info));
            continue;
        };
        match entries.last_mut() {
            Some(Entry::Family(last, ops)) if last.name == family.name => ops.push(info),
            _ => entries.push(Entry::Family(family, vec![info])),
        }
    }
    entries
}

/// The rows of the table of an entry that the ops of `ops` share: groups, attributes, effects.
fn shared_rows(out: &mut String, info: &OpInfo) -> Result<()> {
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
                format!("`{}`: `{}`{optional}", attr.name, (attr.type_name)())
            })
            .collect();
        writeln!(out, "| Attributes | {} |", attrs.join("; "))?;
    }
    Ok(())
}

/// The values of each attribute of `info` whose items are of an attribute enum.
fn attr_enum_tables(out: &mut String, info: &OpInfo) -> Result<()> {
    for attr in info.attrs {
        let Some(values) = ATTR_ENUMS.iter().find(|entry| entry.name == attr.item) else {
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
    Ok(())
}

/// The text form of each attribute of `info` whose type declares one, and of the types its
/// format writes in turn.
fn attr_format_tables(out: &mut String, info: &OpInfo) -> Result<()> {
    for attr in info.attrs {
        let mut rows = Vec::new();
        let mut seen = Vec::new();
        format_rows(attr.item, &mut rows, &mut seen);
        if rows.is_empty() {
            continue;
        }
        writeln!(out, "\n| `{}` | Text form |\n|---|---|", attr.name)?;
        for (name, format) in rows {
            writeln!(out, "| `{name}` | {} |", inline_code(format))?;
        }
    }
    Ok(())
}

/// The types of values and their extents, whose text forms the dialect specs give.
const VALUE_TYPES: [&str; 4] = ["Type", "TensorType", "BufferType", "Dim"];

/// The formats of the type `ty`, of its variants and of the types they write, each once. A format
/// that only writes a field, `$0`, gives no row of its own, and neither do [`VALUE_TYPES`].
fn format_rows(ty: &str, rows: &mut Vec<(&'static str, &'static str)>, seen: &mut Vec<String>) {
    if seen.iter().any(|done| done == ty) || VALUE_TYPES.contains(&ty) {
        return;
    }
    seen.push(ty.to_string());
    let variant = format!("{ty}::");
    let mut entries: Vec<&AttrFormat> = ATTR_FORMATS
        .iter()
        .filter(|entry| entry.name == ty || entry.name.starts_with(&variant))
        .collect();
    entries.sort_by_key(|entry| entry.name);
    for entry in &entries {
        if !matches!(entry.format, "" | "$0") {
            rows.push((entry.name, entry.format));
        }
    }
    for entry in entries {
        for field in entry.fields {
            format_rows(field, rows, seen);
        }
    }
}

/// `text` as markdown inline code, delimited by more backticks than it holds in a row.
fn inline_code(text: &str) -> String {
    let mut longest = 0;
    let mut run = 0;
    for character in text.chars() {
        run = if character == '`' { run + 1 } else { 0 };
        longest = longest.max(run);
    }
    let fence = "`".repeat(longest + 1);
    format!("{fence} {text} {fence}")
}

/// The roundtrip example, the check that a roundtrip case uses each of `names`, and the
/// rejections of the verify cases of `names`.
fn cases(out: &mut String, names: &[String], filetests: &[Case]) -> Result<()> {
    let untested: Vec<&String> = names
        .iter()
        .filter(|name| {
            !filetests
                .iter()
                .any(|case| case.mode == "roundtrip" && case.ops.contains(*name))
        })
        .collect();
    if !untested.is_empty() {
        bail!("no roundtrip filetest uses {untested:?}");
    }
    if let Some(example) = example(names, filetests) {
        writeln!(
            out,
            "\nExample, from `{}`:\n\n```text\n{}```",
            example.file,
            gpt_rs_filetests::code_lines(&example.text)
        )?;
    }
    let rejections = rejections(names, filetests);
    if !rejections.is_empty() {
        writeln!(out, "\nRejected, as the verify filetests pin it:\n")?;
        for message in rejections {
            writeln!(out, "- `{message}`")?;
        }
    }
    Ok(())
}

/// The entry of a family: its rules, what its ops share, then one row per op with its own
/// traits and doc comment.
fn family_section(
    out: &mut String,
    family: &FamilyInfo,
    ops: &[&OpInfo],
    filetests: &[Case],
    catalog: &Catalog,
) -> Result<()> {
    let names: Vec<String> = ops.iter().map(|op| op.name.to_string()).collect();
    writeln!(out, "### `{}`: {}\n", family.name, inline_list(&names))?;
    writeln!(out, "{}\n", body(family.doc))?;
    writeln!(out, "{}\n", defined_by(family.name, ops[0].file))?;
    shared_rows(out, ops[0])?;
    // The traits every op of the family has, with the same parameters, and each op's own.
    let shared: Vec<&dyn OpTrait> = ops[0]
        .traits
        .iter()
        .copied()
        .filter(|t| ops.iter().all(|op| has_same_trait(op, *t)))
        .collect();
    if !shared.is_empty() {
        writeln!(out, "| Traits | {} |", traits(&shared))?;
    }
    writeln!(
        out,
        "| Effects | {} |",
        effects(&ops[0].effects, ops[0].operands)
    )?;
    let mut folds = Vec::new();
    for op in ops {
        if let Some(fold) = FOLDS.iter().find(|entry| entry.name == op.name) {
            folds.push(format!("`{}`: {}", op.name, cell(fold.doc)));
        }
    }
    if !folds.is_empty() {
        writeln!(out, "| Folds | {} |", folds.join("; "))?;
    }
    let patterns = family_patterns(catalog, ops);
    if !patterns.is_empty() {
        writeln!(out, "| Patterns | {} |", patterns.join("; "))?;
    }
    attr_enum_tables(out, ops[0])?;
    attr_format_tables(out, ops[0])?;
    let own = |op: &OpInfo| -> Vec<&dyn OpTrait> {
        op.traits
            .iter()
            .copied()
            .filter(|t| !shared.iter().any(|s| s.to_string() == t.to_string()))
            .collect()
    };
    let with_traits = ops.iter().any(|op| !own(op).is_empty());
    if with_traits {
        writeln!(out, "\n| Op | Traits | |\n|---|---|---|")?;
    } else {
        writeln!(out, "\n| Op | |\n|---|---|")?;
    }
    for op in ops {
        let doc = cell(&body(op.doc));
        if with_traits {
            writeln!(out, "| `{}` | {} | {doc} |", op.name, traits(&own(op)))?;
        } else {
            writeln!(out, "| `{}` | {doc} |", op.name)?;
        }
    }
    cases(out, &names, filetests)
}

/// Whether `op` has the trait named `name`.
fn has_trait(op: &OpInfo, name: &str) -> bool {
    op.traits.iter().any(|t| t.name() == name)
}

/// Whether `op` has `wanted`, with its parameters.
fn has_same_trait(op: &OpInfo, wanted: &dyn OpTrait) -> bool {
    op.traits
        .iter()
        .any(|t| t.to_string() == wanted.to_string())
}

/// The patterns rooted at the ops of a family, each with its pass, and with the ops it is rooted
/// at when they are not all of the family.
fn family_patterns(catalog: &Catalog, ops: &[&OpInfo]) -> Vec<String> {
    let mut found: Vec<(&str, &str, Vec<&str>)> = Vec::new();
    for op in ops {
        for (pattern, pass, _) in catalog.patterns_at(|root| root == RootInfo::Op(op.name)) {
            match found
                .iter_mut()
                .find(|(p, s, _)| (*p, *s) == (pattern, pass))
            {
                Some((_, _, roots)) => roots.push(op.name),
                None => found.push((pattern, pass, vec![op.name])),
            }
        }
    }
    found
        .into_iter()
        .map(|(pattern, pass, roots)| {
            if roots.len() == ops.len() {
                format!("`{pattern}` (`{pass}`)")
            } else {
                let at: Vec<String> = roots.iter().map(|root| format!("`{root}`")).collect();
                format!("`{pattern}` (`{pass}`, at {})", at.join(", "))
            }
        })
        .collect()
}

fn op_section(
    out: &mut String,
    info: &OpInfo,
    filetests: &[Case],
    catalog: &Catalog,
) -> Result<()> {
    let names = vec![info.name.to_string()];
    writeln!(out, "### `{}`\n", info.name)?;
    writeln!(out, "{}\n", body(info.doc))?;
    writeln!(out, "{}\n", defined_by(info.rust, info.file))?;
    shared_rows(out, info)?;
    if !info.traits.is_empty() {
        writeln!(out, "| Traits | {} |", traits(info.traits))?;
    }
    writeln!(
        out,
        "| Effects | {} |",
        effects(&info.effects, info.operands)
    )?;
    if let Some(fold) = FOLDS.iter().find(|entry| entry.name == info.name) {
        writeln!(out, "| Folds | {} |", cell(fold.doc))?;
    }
    let patterns = catalog.patterns_at(|root| root == RootInfo::Op(info.name));
    if !patterns.is_empty() {
        writeln!(out, "| Patterns | {} |", pattern_list(&patterns))?;
    }
    attr_enum_tables(out, info)?;
    attr_format_tables(out, info)?;
    cases(out, &names, filetests)
}

/// Patterns with the pass that runs each: `` `CollapseTransposeChain` (`canonicalize`) ``.
fn pattern_list(patterns: &[(&str, &str, RootInfo)]) -> String {
    patterns
        .iter()
        .map(|(pattern, pass, _)| format!("`{pattern}` (`{pass}`)"))
        .collect::<Vec<_>>()
        .join("; ")
}

/// Every trait that an op of the dialect has: its doc comment, the ops that have it, and the
/// patterns rooted at it.
fn traits_section(out: &mut String, infos: &[OpInfo], catalog: &Catalog) -> Result<()> {
    out.push_str(
        "\n## Traits\n\nA trait states a fact about the ops that have it. The folds and patterns \
         it implies apply to every one of them; [passes.md](passes.md) says which pass runs \
         each.\n",
    );
    for (name, t) in &catalog.traits {
        let ops: Vec<String> = infos
            .iter()
            .filter(|info| has_trait(info, name))
            .map(|info| info.name.to_string())
            .collect();
        if ops.is_empty() {
            continue;
        }
        writeln!(out, "\n### `{name}`\n")?;
        writeln!(out, "{}\n", t.doc())?;
        writeln!(out, "{}\n", defined_by(name, t.file()))?;
        writeln!(out, "Ops: {}.", inline_list(&ops))?;
        let patterns = catalog.patterns_at(|root| root == RootInfo::Trait(name));
        if !patterns.is_empty() {
            writeln!(out, "\nPatterns rooted at it: {}.", pattern_list(&patterns))?;
        }
    }
    Ok(())
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
            let (kind, kinds) = match group.kind {
                Kind::Any => ("value", "values"),
                Kind::Tensor => ("tensor", "tensors"),
                Kind::Buffer => ("buffer", "buffers"),
                Kind::Index => ("index", "indices"),
                Kind::Scalar => ("scalar", "scalars"),
            };
            let arity = match group.arity {
                Arity::One => kind.to_string(),
                Arity::Optional => format!("optional {kind}"),
                Arity::Variadic => kinds.to_string(),
            };
            format!("`{}`: {arity}", group.name)
        })
        .collect::<Vec<_>>()
        .join("; ")
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
/// whose function is named after the op, then one that uses more of the names, then the shortest.
fn example<'a>(names: &[String], filetests: &'a [Case]) -> Option<&'a Case> {
    filetests
        .iter()
        .filter(|case| case.mode == "roundtrip")
        .filter(|case| names.iter().any(|name| case.ops.contains(name)))
        .min_by_key(|case| {
            let named = names.iter().any(|name| {
                let short = name.rsplit('.').next().unwrap_or(name);
                case.symbols.contains(short)
            });
            let used = names.iter().filter(|name| case.ops.contains(*name)).count();
            (
                !named,
                std::cmp::Reverse(used),
                gpt_rs_filetests::code_lines(&case.text).lines().count(),
            )
        })
}

/// The messages of the verify filetests that reject one of `names`.
fn rejections(names: &[String], filetests: &[Case]) -> Vec<String> {
    filetests
        .iter()
        .filter(|case| case.mode == "verify")
        .flat_map(|case| &case.errors)
        .filter(|message| {
            names
                .iter()
                .any(|name| message.starts_with(&format!("{name}: ")))
        })
        .cloned()
        .collect()
}
