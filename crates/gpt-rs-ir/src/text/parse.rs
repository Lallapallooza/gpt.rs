use std::sync::Arc;

use anyhow::{bail, ensure, Result};

use super::Cursor;
use crate::attr::Attr;
use crate::location::{Location, SourcePosition};
use crate::module::{Module, NewInst, Region, Value};
use crate::scope::ScopedMap;
use crate::traits::IsolatedFromAbove;
use crate::types::Type;
use crate::Op;

struct Parser<'a> {
    c: Cursor<'a>,
    module: Module,
    names: ScopedMap<String, Value>,
}

/// A block argument or function argument: `%name: type`.
pub(crate) fn named_arg(c: &mut Cursor<'_>) -> Result<(String, Type)> {
    let name = c.sigil_name('%')?.to_string();
    c.expect(":")?;
    Ok((name, Type::parse(c)?))
}

/// `(T, ...)`.
fn types_list(c: &mut Cursor<'_>) -> Result<Vec<Type>> {
    c.expect("(")?;
    c.list(")", Type::parse)
}

/// The result types of an op: one type, or several in parentheses.
pub(crate) fn parse_result_types(c: &mut Cursor<'_>) -> Result<Vec<Type>> {
    if c.peek() == Some('(') {
        types_list(c)
    } else {
        Ok(vec![Type::parse(c)?])
    }
}

impl<'a> Parser<'a> {
    fn bind(&mut self, name: &str, value: Value) -> Result<()> {
        if self.names.insert(name.to_string(), value).is_some() {
            bail!(self.c.error(format!("%{name} is defined twice")));
        }
        Ok(())
    }

    /// Parses ops into `region` until `}` or the end of input.
    fn parse_ops(&mut self, region: Region) -> Result<()> {
        while !self.c.at_end() && self.c.peek() != Some('}') {
            self.parse_op(region)?;
        }
        Ok(())
    }

    fn parse_op(&mut self, region: Region) -> Result<()> {
        let mut result_names = Vec::new();
        if self.c.peek() == Some('%') {
            loop {
                result_names.push(self.c.sigil_name('%')?.to_string());
                if !self.c.eat(",") {
                    break;
                }
            }
            self.c.expect("=")?;
        }
        let name = self.c.ident()?;
        if let Some(signature) = Op::parse_symbol(name, &mut self.c) {
            ensure!(
                result_names.is_empty(),
                self.c.error(format!("{name} has no results"))
            );
            let (op, args) = signature?;
            self.c.expect("{")?;
            let body = self.block(&args, true)?;
            self.append_symbol(region, op, body);
            return Ok(());
        }

        self.c.expect("(")?;
        let names = &self.names;
        let operands = self.c.list(")", |c| {
            let name = c.sigil_name('%')?;
            names
                .get(name)
                .ok_or_else(|| c.error(format!("%{name} is not defined here")))
        })?;

        self.c.operand_segments = None;
        self.c.result_segments = None;
        let op = Op::parse(name, &mut self.c)
            .ok_or_else(|| self.c.error(format!("unknown op `{name}`")))??;
        let operand_segments = self.c.operand_segments.take();
        let result_segments = self.c.result_segments.take();

        let mut regions = Vec::new();
        if self.c.peek_pair('(', '{') {
            self.c.expect("(")?;
            let isolated = op.has_trait::<IsolatedFromAbove>();
            let labels = op.region_names();
            loop {
                let index = regions.len();
                regions.push(self.parse_region(isolated, labels.get(index).copied())?);
                if self.c.eat(")") {
                    break;
                }
                self.c.expect(",")?;
            }
        }

        self.c.expect(":")?;
        let operand_types = types_list(&mut self.c)?;
        self.c.expect("->")?;
        let result_types = parse_result_types(&mut self.c)?;
        let location = self.location()?;
        ensure!(
            operand_types.len() == operands.len(),
            self.c.error(format!(
                "{name} lists {} operand types for {} operands",
                operand_types.len(),
                operands.len()
            ))
        );
        for (i, (&value, ty)) in operands.iter().zip(&operand_types).enumerate() {
            let actual = self.module.ty(value);
            ensure!(
                actual == ty,
                self.c.error(format!(
                    "{name} operand {i} is {actual}, but the signature says {ty}"
                ))
            );
        }
        ensure!(
            result_types.len() == result_names.len(),
            self.c.error(format!(
                "{name} defines {} results but its signature has {} result types",
                result_names.len(),
                result_types.len()
            ))
        );

        let inst = self.module.create_inst(NewInst {
            op,
            operands,
            operand_segments,
            result_types,
            result_segments,
            regions,
            location,
        });
        self.module.append(region, inst);
        for (name, value) in result_names.iter().zip(self.module.results(inst).to_vec()) {
            self.bind(name, value)?;
        }
        Ok(())
    }

    fn parse_region(&mut self, isolated: bool, label: Option<&str>) -> Result<Region> {
        self.c.expect("{")?;
        let mut args = Vec::new();
        if self.c.peek() == Some('^') {
            let found = self.c.sigil_name('^')?;
            if let Some(label) = label {
                ensure!(
                    found == label,
                    self.c
                        .error(format!("expected region label ^{label}, got ^{found}"))
                );
            }
            self.c.expect("(")?;
            args = self.c.list(")", named_arg)?;
            self.c.expect(":")?;
        }
        self.block(&args, isolated)
    }

    /// A new region with the block arguments `args`, holding the ops up to its closing `}`. Unless
    /// `isolated`, its ops see the values of the enclosing regions.
    fn block(&mut self, args: &[(String, Type)], isolated: bool) -> Result<Region> {
        let region = self
            .module
            .new_region(args.iter().map(|(_, ty)| ty.clone()).collect());
        self.names.push(isolated);
        for ((name, _), value) in args.iter().zip(self.module.args(region).to_vec()) {
            self.bind(name, value)?;
        }
        self.parse_ops(region)?;
        self.names.pop();
        self.c.expect("}")?;
        Ok(region)
    }

    /// Appends to `parent` the symbol op `op` with its `body`.
    fn append_symbol(&mut self, parent: Region, op: Op, body: Region) {
        let inst = self.module.create_inst(NewInst {
            op,
            operands: Vec::new(),
            operand_segments: None,
            result_types: Vec::new(),
            result_segments: None,
            regions: vec![body],
            location: None,
        });
        self.module.append(parent, inst);
    }

    /// The location after an op's signature, if it has one: `loc("scope"("file":line:column))`,
    /// `loc("scope")` or `loc("file":line:column)`.
    fn location(&mut self) -> Result<Option<Location>> {
        if !self.c.eat("loc") {
            return Ok(None);
        }
        self.c.expect("(")?;
        let first = self.c.string()?;
        let location = if self.c.eat(":") {
            Location {
                scope: None,
                source: Some(self.source_position(first)?),
            }
        } else {
            let source = if self.c.eat("(") {
                let file = self.c.string()?;
                self.c.expect(":")?;
                let position = self.source_position(file)?;
                self.c.expect(")")?;
                Some(position)
            } else {
                None
            };
            Location {
                scope: Some(Arc::from(first)),
                source,
            }
        };
        self.c.expect(")")?;
        Ok(Some(location))
    }

    /// `line:column` after the `file` of a location and its colon.
    fn source_position(&mut self, file: String) -> Result<SourcePosition> {
        let line = self.c.int()?;
        self.c.expect(":")?;
        let column = self.c.int()?;
        let in_range =
            |n: i128| u32::try_from(n).map_err(|_| self.c.error("position out of range"));
        Ok(SourcePosition {
            file: Arc::from(file),
            line: in_range(line)?,
            column: in_range(column)?,
        })
    }
}

impl Module {
    /// Parses the text form. It does not verify the module: [`crate::verify::verify`] does.
    pub fn from_text(src: &str) -> Result<Module> {
        let mut parser = Parser {
            c: Cursor::new(src),
            module: Module::new(),
            names: ScopedMap::new(),
        };
        if parser.c.eat("spec_version") {
            parser.c.expect("=")?;
            let version = parser.c.string()?;
            ensure!(
                version == super::SPEC_VERSION,
                parser.c.error(format!(
                    "the program follows {version}, but this parser reads {}",
                    super::SPEC_VERSION
                ))
            );
        }
        let body = parser.module.body();
        parser.parse_ops(body)?;
        if !parser.c.at_end() {
            return Err(parser.c.error("unexpected `}`"));
        }
        Ok(parser.module)
    }
}
