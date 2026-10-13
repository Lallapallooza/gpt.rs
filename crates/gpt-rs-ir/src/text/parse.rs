use anyhow::{bail, ensure, Result};

use super::{Cursor, TokenKind};
use crate::ir::{
    parse_tuple, Attr, Location, Module, NamedArg, NewInst, Region, ScopedMap, Type, Value,
};
use crate::traits::IsolatedFromAbove;
use crate::Op;

struct Parser<'a> {
    c: Cursor<'a>,
    module: Module,
    names: ScopedMap<String, Value>,
}

/// `(T, ...)`, each type with its position.
fn types_list(c: &mut Cursor<'_>) -> Result<Vec<(usize, Type)>> {
    c.expect("(")?;
    c.list(")", |c| Ok((c.position(), Type::parse(c)?)))
}

/// The sizes of the operand and result groups of an op with more than one optional or variadic
/// group, which its attribute dictionary writes after its own attributes.
#[derive(Default)]
struct Segments {
    operands: Option<Vec<usize>>,
    results: Option<Vec<usize>>,
}

impl Segments {
    /// Reads the value of `key`, at `at` in the attribute dictionary of the op `op`, a key the op
    /// does not have, which must be `operand_segments` or `result_segments`.
    fn read(&mut self, op: &str, key: &str, c: &mut Cursor<'_>, at: usize) -> Result<()> {
        let slot = match key {
            "operand_segments" => &mut self.operands,
            "result_segments" => &mut self.results,
            _ => bail!(c.error_at(at, format!("{op} has no attribute `{key}`"))),
        };
        ensure!(
            slot.is_none(),
            c.error_at(at, format!("duplicate attribute `{key}`"))
        );
        *slot = Some(Vec::<usize>::parse(c)?);
        Ok(())
    }
}

impl<'a> Parser<'a> {
    /// Names `value` `name`, which the text writes at `at`.
    fn bind(&mut self, name: &str, at: usize, value: Value) -> Result<()> {
        if self.names.insert(name.to_string(), value).is_some() {
            bail!(self.c.error_at(at, format!("%{name} is defined twice")));
        }
        Ok(())
    }

    /// Parses ops into `region` until `}` or the end of input.
    fn parse_ops(&mut self, region: Region) -> Result<()> {
        while !self.c.at_end() && !self.c.peek_is("}") {
            self.parse_op(region)?;
        }
        Ok(())
    }

    /// A value's name, after its `%`.
    fn value_name(&mut self) -> Result<&'a str> {
        self.c.sigil_name(TokenKind::Value, "a value")
    }

    fn parse_op(&mut self, region: Region) -> Result<()> {
        let mut result_names = Vec::new();
        if self.c.peek_kind(TokenKind::Value) {
            loop {
                let at = self.c.position();
                result_names.push((self.value_name()?.to_string(), at));
                if !self.c.eat(",") {
                    break;
                }
            }
            self.c.expect("=")?;
        }
        let name_start = self.c.position();
        let name = self.c.ident()?;
        if let Some(signature) = Op::parse_symbol(name, &mut self.c) {
            ensure!(
                result_names.is_empty(),
                self.c
                    .error_at(name_start, format!("{name} has no results"))
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
            let start = c.position();
            let name = c.sigil_name(TokenKind::Value, "a value")?;
            names
                .get(name)
                .ok_or_else(|| c.error_at(start, format!("%{name} is not defined here")))
        })?;

        let mut segments = Segments::default();
        let op = Op::parse(name, &mut self.c, &mut |key, c, at| {
            segments.read(name, key, c, at)
        })
        .ok_or_else(|| self.c.error_at(name_start, format!("unknown op `{name}`")))??;

        let mut regions = Vec::new();
        if self.c.eat("(") {
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
        let operand_types_at = self.c.position();
        let operand_types = types_list(&mut self.c)?;
        self.c.expect("->")?;
        let result_types_at = self.c.position();
        let result_types = parse_tuple::<Vec<Type>>(&mut self.c)?;
        let location = if self.c.peek_is("loc") {
            Some(Location::parse(&mut self.c)?)
        } else {
            None
        };
        ensure!(
            operand_types.len() == operands.len(),
            self.c.error_at(
                operand_types_at,
                format!(
                    "{name} lists {} operand types for {} operands",
                    operand_types.len(),
                    operands.len()
                )
            )
        );
        for (index, (&value, (at, ty))) in operands.iter().zip(&operand_types).enumerate() {
            let actual = self.module.ty(value);
            ensure!(
                actual == ty,
                self.c.error_at(
                    *at,
                    format!("{name} operand {index} is {actual}, but the signature says {ty}")
                )
            );
        }
        ensure!(
            result_types.len() == result_names.len(),
            self.c.error_at(
                result_types_at,
                format!(
                    "{name} defines {} results but its signature has {} result types",
                    result_names.len(),
                    result_types.len()
                )
            )
        );

        let inst = self.module.create_inst(NewInst {
            op,
            operands,
            operand_segments: segments.operands,
            result_types,
            result_segments: segments.results,
            regions,
            location,
        });
        self.module.append(region, inst);
        for ((name, at), value) in result_names.iter().zip(self.module.results(inst).to_vec()) {
            self.bind(name, *at, value)?;
        }
        Ok(())
    }

    fn parse_region(&mut self, isolated: bool, label: Option<&str>) -> Result<Region> {
        self.c.expect("{")?;
        let mut args = Vec::new();
        if self.c.peek_kind(TokenKind::Label) {
            let start = self.c.position();
            let found = self.c.sigil_name(TokenKind::Label, "a region label")?;
            if let Some(label) = label {
                ensure!(
                    found == label,
                    self.c.error_at(
                        start,
                        format!("expected region label ^{label}, got ^{found}")
                    )
                );
            }
            self.c.expect("(")?;
            args = self.c.list(")", |c| {
                let at = c.position();
                let name = c.sigil_name(TokenKind::Value, "a value")?.to_string();
                c.expect(":")?;
                let ty = Type::parse(c)?;
                Ok(NamedArg { name, at, ty })
            })?;
            self.c.expect(":")?;
        }
        self.block(&args, isolated)
    }

    /// A new region with the block arguments `args`, holding the ops up to its closing `}`. Unless
    /// `isolated`, its ops see the values of the enclosing regions.
    fn block(&mut self, args: &[NamedArg], isolated: bool) -> Result<Region> {
        let mut types = Vec::with_capacity(args.len());
        for arg in args {
            types.push(arg.ty.clone());
        }
        let region = self.module.new_region(types);
        self.names.push(isolated);
        for (arg, value) in args.iter().zip(self.module.args(region).to_vec()) {
            self.bind(&arg.name, arg.at, value)?;
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
}

impl Module {
    /// Parses the text form. It does not verify the module: [`crate::ir::verify`] does.
    pub fn from_text(src: &str) -> Result<Module> {
        let mut parser = Parser {
            c: Cursor::new(src),
            module: Module::new(),
            names: ScopedMap::new(),
        };
        if parser.c.eat("spec_version") {
            parser.c.expect("=")?;
            let start = parser.c.position();
            let version = parser.c.string()?;
            ensure!(
                version == super::SPEC_VERSION,
                parser.c.error_at(
                    start,
                    format!(
                        "the program follows {version}, but this parser reads {}",
                        super::SPEC_VERSION
                    )
                )
            );
        }
        let body = parser.module.body();
        parser.parse_ops(body)?;
        if !parser.c.at_end() {
            return Err(parser.c.unexpected("an op"));
        }
        Ok(parser.module)
    }
}
