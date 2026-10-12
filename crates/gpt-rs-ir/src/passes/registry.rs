//! The registry of the passes that `#[pass]` defines, and the parser of pass lists such as
//! `canonicalize {fuel = 10}, cse`.

use std::collections::HashSet;

use anyhow::Result;

use super::{Pass, Pipeline};
use crate::text::Cursor;

/// One pass in the registry: what `#[pass]` records about it.
pub struct PassEntry {
    pub name: &'static str,
    /// The pass's doc comment. Its first paragraph is the summary.
    pub doc: &'static str,
    /// The struct of the pass, and the file that defines it, relative to the workspace.
    pub rust: &'static str,
    pub file: &'static str,
    pub options: &'static [OptionInfo],
    pub stats: &'static [StatInfo],
    /// The pass with the options that follow its name in a pass list, `{name = value, ...}`.
    pub make: fn(&mut Cursor<'_>) -> Result<Box<dyn Pass>>,
}

/// An option of a pass: a field of its struct.
pub struct OptionInfo {
    pub name: &'static str,
    /// The field's Rust type.
    pub ty: &'static str,
    /// The default, in the text form of an attribute.
    pub default: fn() -> String,
    pub doc: &'static str,
}

/// A statistic that a pass counts as it runs.
pub struct StatInfo {
    pub name: &'static str,
    pub doc: &'static str,
}

/// The passes of every linked crate. `#[pass]` adds each one.
#[linkme::distributed_slice]
pub static PASSES: [PassEntry];

/// Every registered pass, by name. Panics when two passes share a name.
pub fn registered() -> Vec<&'static PassEntry> {
    let mut entries: Vec<&'static PassEntry> = PASSES.iter().collect();
    entries.sort_by_key(|entry| entry.name);
    let mut names = HashSet::new();
    for entry in &entries {
        assert!(
            names.insert(entry.name),
            "two passes are named `{}`",
            entry.name
        );
    }
    entries
}

/// The passes of a pass list such as `canonicalize {fuel = 10}, cse`, in order.
pub fn parse(text: &str) -> Result<Pipeline> {
    let entries = registered();
    let mut cursor = Cursor::new(text);
    let mut pipeline = Pipeline::new();
    loop {
        let start = cursor.position();
        let name = cursor.hyphenated()?;
        let Some(entry) = entries.iter().find(|entry| entry.name == name) else {
            return Err(cursor.error_at(start, format!("unknown pass `{name}`")));
        };
        pipeline.passes.push((entry.make)(&mut cursor)?);
        if cursor.at_end() {
            return Ok(pipeline);
        }
        cursor.expect(",")?;
    }
}
