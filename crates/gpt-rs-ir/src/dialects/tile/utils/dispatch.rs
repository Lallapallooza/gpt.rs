//! The text form of the signature of `tile.dispatch`.

use std::collections::BTreeMap;

use anyhow::Result;

use crate::attr::Attr;
use crate::dialects::tile::ir::*;
use crate::index::{IterList, SymFact};
use crate::op::{OpKind, ParsedSymbol, SymbolOp};
use crate::text::{named_arg, parse_result_types, results_text, Cursor};

/// `tile.dispatch @name(%a: T, ...) -> T {grid = [...], facts = {...}}`.
impl SymbolOp for Dispatch {
    fn symbol(&self) -> &str {
        &self.name
    }

    fn arg_suffix(&self, _index: usize) -> String {
        String::new()
    }

    fn results_text(&self) -> String {
        results_text(&self.results.iter().collect::<Vec<_>>())
    }

    /// The attributes that the signature does not write.
    fn attr_entries(&self) -> Vec<String> {
        let mut attrs = Vec::new();
        self.print_attrs(&mut attrs);
        let mut entries = Vec::new();
        for (key, text) in attrs {
            if !matches!(key, "name" | "args" | "results") {
                entries.push(format!("{key} = {text}"));
            }
        }
        entries
    }

    fn parse_signature(c: &mut Cursor<'_>) -> Result<ParsedSymbol<Self>> {
        let name = c.sigil_name('@')?.to_string();
        c.expect("(")?;
        let named = c.list(")", named_arg)?;
        c.expect("->")?;
        let results = parse_result_types(c)?;
        let mut grid = IterList::default();
        let mut facts = BTreeMap::new();
        if c.peek_attr_dict() {
            c.expect("{")?;
            c.list("}", |c| {
                let key = c.ident()?;
                c.expect("=")?;
                match key {
                    "grid" => grid = IterList::parse(c)?,
                    "facts" => facts = BTreeMap::<String, SymFact>::parse(c)?,
                    other => {
                        return Err(c.error(format!("tile.dispatch has no attribute `{other}`")))
                    }
                }
                Ok(())
            })?;
        }
        let mut args = Vec::with_capacity(named.len());
        for (_, ty) in &named {
            args.push(ty.clone());
        }
        let dispatch = Dispatch {
            name,
            args,
            results,
            grid,
            facts,
        };
        Ok((dispatch, named))
    }
}
