use anyhow::{bail, ensure, Result};

use super::{Cursor, TokenKind};
use crate::ir::{
    parse_tuple, Attr, Location, Module, NamedArg, NewInst, Region, ScopedMap, Type, Value,
};
use crate::rewrite::Builder;
use crate::traits::IsolatedFromAbove;
use crate::Op;

struct Parser<'a> {
    cursor: Cursor<'a>,
    module: Module,
    names: ScopedMap<String, Value>,
}

/// `(T, ...)`, each type with its position.
fn types_list(cursor: &mut Cursor) -> Result<Vec<(usize, Type)>> {
    cursor.expect("(")?;
    cursor.list(")", |list_cursor| {
        Ok((list_cursor.position(), Type::parse(list_cursor)?))
    })
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
    fn read(&mut self, op: &str, key: &str, cursor: &mut Cursor, at: usize) -> Result<()> {
        let slot = match key {
            "operand_segments" => &mut self.operands,
            "result_segments" => &mut self.results,
            _ => bail!(cursor.error_at(at, format!("{op} has no attribute `{key}`"))),
        };
        ensure!(
            slot.is_none(),
            cursor.error_at(at, format!("duplicate attribute `{key}`"))
        );
        *slot = Some(Vec::<usize>::parse(cursor)?);
        Ok(())
    }
}

impl<'a> Parser<'a> {
    /// Names `value` `name`, which the text writes at `at`.
    fn bind(&mut self, name: &str, at: usize, value: Value) -> Result<()> {
        if self.names.insert(name.to_string(), value).is_some() {
            bail!(self
                .cursor
                .error_at(at, format!("%{name} is defined twice")));
        }
        Ok(())
    }

    /// Parses ops into `region` until `}` or the end of input.
    fn parse_ops(&mut self, region: Region) -> Result<()> {
        while !self.cursor.at_end() && !self.cursor.peek_is("}") {
            self.parse_op(region)?;
        }
        Ok(())
    }

    /// A value's name, after its `%`.
    fn value_name(&mut self) -> Result<&'a str> {
        self.cursor.sigil_name(TokenKind::Value, "a value")
    }

    fn parse_op(&mut self, region: Region) -> Result<()> {
        let result_names = self.result_names()?;
        let name_start = self.cursor.position();
        let name = self.cursor.ident()?;
        if let Some(signature) = Op::parse_symbol(name, &mut self.cursor) {
            ensure!(
                result_names.is_empty(),
                self.cursor
                    .error_at(name_start, format!("{name} has no results"))
            );
            let (op, args) = signature?;
            self.cursor.expect("{")?;
            let body = self.block(&args, true)?;
            self.append_symbol(region, op, body);
            return Ok(());
        }

        let operands = self.operands()?;
        let mut segments = Segments::default();
        let op = Op::parse(name, &mut self.cursor, &mut |key, cursor, at| {
            segments.read(name, key, cursor, at)
        })
        .ok_or_else(|| {
            self.cursor
                .error_at(name_start, format!("unknown op `{name}`"))
        })??;
        let regions = self.op_regions(&op)?;

        self.cursor.expect(":")?;
        let operand_types_at = self.cursor.position();
        let operand_types = types_list(&mut self.cursor)?;
        self.cursor.expect("->")?;
        let result_types_at = self.cursor.position();
        let result_types = parse_tuple::<Vec<Type>>(&mut self.cursor)?;
        let location = if self.cursor.peek_is("loc") {
            Some(Location::parse(&mut self.cursor)?)
        } else {
            None
        };
        self.check_operand_types(name, &operands, operand_types_at, &operand_types)?;
        ensure!(
            result_types.len() == result_names.len(),
            self.cursor.error_at(
                result_types_at,
                format!(
                    "{name} defines {} results but its signature has {} result types",
                    result_names.len(),
                    result_types.len()
                )
            )
        );

        let inst = Builder::at_end(&mut self.module, region).create_typed(NewInst {
            op,
            operands,
            operand_segments: segments.operands,
            result_types,
            result_segments: segments.results,
            regions,
            location,
        });
        for ((result_name, at), value) in
            result_names.iter().zip(self.module.results(inst).to_vec())
        {
            self.bind(result_name, *at, value)?;
        }
        Ok(())
    }

    /// The names of the results an op defines, `%a, %b =`, each with its position, or none when
    /// no value name comes next.
    fn result_names(&mut self) -> Result<Vec<(String, usize)>> {
        let mut result_names = Vec::new();
        if !self.cursor.peek_kind(TokenKind::Value) {
            return Ok(result_names);
        }
        loop {
            let at = self.cursor.position();
            result_names.push((self.value_name()?.to_string(), at));
            if !self.cursor.eat(",") {
                break;
            }
        }
        self.cursor.expect("=")?;
        Ok(result_names)
    }

    /// The operands of an op, `(%a, ...)`, each a value defined where the op is.
    fn operands(&mut self) -> Result<Vec<Value>> {
        self.cursor.expect("(")?;
        let names = &self.names;
        self.cursor.list(")", |cursor| {
            let start = cursor.position();
            let value_name = cursor.sigil_name(TokenKind::Value, "a value")?;
            names
                .get(value_name)
                .ok_or_else(|| cursor.error_at(start, format!("%{value_name} is not defined here")))
        })
    }

    /// The regions of `op`, `({...}, ...)`, or none when no `(` comes next.
    fn op_regions(&mut self, op: &Op) -> Result<Vec<Region>> {
        let mut regions = Vec::new();
        if !self.cursor.eat("(") {
            return Ok(regions);
        }
        let isolated = op.has_trait::<IsolatedFromAbove>();
        let labels = op.region_names();
        loop {
            let index = regions.len();
            regions.push(self.parse_region(isolated, labels.get(index).copied())?);
            if self.cursor.eat(")") {
                break;
            }
            self.cursor.expect(",")?;
        }
        Ok(regions)
    }

    /// Checks that the signature of the op `name` lists, at `operand_types_at`, the types of
    /// `operands`.
    fn check_operand_types(
        &self,
        name: &str,
        operands: &[Value],
        operand_types_at: usize,
        operand_types: &[(usize, Type)],
    ) -> Result<()> {
        ensure!(
            operand_types.len() == operands.len(),
            self.cursor.error_at(
                operand_types_at,
                format!(
                    "{name} lists {} operand types for {} operands",
                    operand_types.len(),
                    operands.len()
                )
            )
        );
        for (index, (&value, (at, ty))) in operands.iter().zip(operand_types).enumerate() {
            let actual = self.module.ty(value);
            ensure!(
                actual == ty,
                self.cursor.error_at(
                    *at,
                    format!("{name} operand {index} is {actual}, but the signature says {ty}")
                )
            );
        }
        Ok(())
    }

    fn parse_region(&mut self, isolated: bool, label: Option<&str>) -> Result<Region> {
        self.cursor.expect("{")?;
        let mut args = Vec::new();
        if self.cursor.peek_kind(TokenKind::Label) {
            let start = self.cursor.position();
            let found = self.cursor.sigil_name(TokenKind::Label, "a region label")?;
            if let Some(expected) = label {
                ensure!(
                    found == expected,
                    self.cursor.error_at(
                        start,
                        format!("expected region label ^{expected}, got ^{found}")
                    )
                );
            }
            self.cursor.expect("(")?;
            args = self.cursor.list(")", |cursor| {
                let at = cursor.position();
                let name = cursor.sigil_name(TokenKind::Value, "a value")?.to_string();
                cursor.expect(":")?;
                let ty = Type::parse(cursor)?;
                Ok(NamedArg { name, at, ty })
            })?;
            self.cursor.expect(":")?;
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
        self.cursor.expect("}")?;
        Ok(region)
    }

    /// Appends to `parent` the symbol op `op` with its `body`.
    fn append_symbol(&mut self, parent: Region, op: Op, body: Region) {
        Builder::at_end(&mut self.module, parent).create_typed(NewInst {
            op,
            operands: Vec::new(),
            operand_segments: None,
            result_types: Vec::new(),
            result_segments: None,
            regions: vec![body],
            location: None,
        });
    }
}

impl Module {
    /// Parses the text form. It does not verify the module: [`crate::ir::verify`] does.
    pub fn from_text(src: &str) -> Result<Module> {
        let mut parser = Parser {
            cursor: Cursor::new(src),
            module: Module::new(),
            names: ScopedMap::new(),
        };
        if parser.cursor.eat("spec_version") {
            parser.cursor.expect("=")?;
            let start = parser.cursor.position();
            let version = parser.cursor.string()?;
            ensure!(
                version == super::SPEC_VERSION,
                parser.cursor.error_at(
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
        if !parser.cursor.at_end() {
            return Err(parser.cursor.unexpected("an op"));
        }
        Ok(parser.module)
    }
}
