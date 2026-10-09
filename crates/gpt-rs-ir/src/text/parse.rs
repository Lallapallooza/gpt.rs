use anyhow::{bail, ensure, Result};

use super::Cursor;
use crate::attr::Attr;
use crate::dialects::func::{Func, ParamId};
use crate::module::{Module, NewInst, Region, Value};
use crate::rewrite::bound_params;
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
fn arg(c: &mut Cursor<'_>) -> Result<(String, Type)> {
    let name = c.sigil_name('%')?.to_string();
    c.expect(":")?;
    Ok((name, Type::parse(c)?))
}

/// An optional ` {param = id}` after an argument or result type.
fn param_binding(c: &mut Cursor<'_>) -> Result<Option<ParamId>> {
    if !c.eat("{") {
        return Ok(None);
    }
    c.expect("param")?;
    c.expect("=")?;
    let param = ParamId::parse(c)?;
    c.expect("}")?;
    Ok(Some(param))
}

impl<'a> Parser<'a> {
    fn bind(&mut self, name: &str, value: Value) -> Result<()> {
        if self.names.insert(name.to_string(), value).is_some() {
            bail!(self.c.error(format!("%{name} is defined twice")));
        }
        Ok(())
    }

    fn types_list(&mut self) -> Result<Vec<Type>> {
        self.c.expect("(")?;
        self.c.list(")", Type::parse)
    }

    fn result_types(&mut self) -> Result<Vec<Type>> {
        if self.c.peek() == Some('(') {
            self.types_list()
        } else {
            Ok(vec![Type::parse(&mut self.c)?])
        }
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
        if name == "func.func" {
            ensure!(
                result_names.is_empty(),
                self.c.error("func.func has no results")
            );
            return self.parse_func(region);
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
        let operand_types = self.types_list()?;
        self.c.expect("->")?;
        let result_types = self.result_types()?;
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
            args = self.c.list(")", arg)?;
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

    /// A function's result types, and each result's parameter when it is a load-time function.
    /// A single run-time result may be written without parentheses.
    fn func_results(&mut self) -> Result<(Vec<Type>, Vec<ParamId>)> {
        if self.c.peek() != Some('(') {
            return Ok((vec![Type::parse(&mut self.c)?], Vec::new()));
        }
        self.c.expect("(")?;
        let results = self
            .c
            .list(")", |c| Ok((Type::parse(c)?, param_binding(c)?)))?;
        let (types, params) = results.into_iter().unzip();
        let params = bound_params(params).into_iter().collect::<Option<_>>();
        let unbound = "a load-time function binds every result to a parameter";
        Ok((types, params.ok_or_else(|| self.c.error(unbound))?))
    }

    fn parse_func(&mut self, parent: Region) -> Result<()> {
        let name = self.c.sigil_name('@')?.to_string();
        self.c.expect("(")?;
        let bound = self.c.list(")", |c| Ok((arg(c)?, param_binding(c)?)))?;
        let (named, params): (Vec<(String, Type)>, Vec<Option<ParamId>>) =
            bound.into_iter().unzip();
        self.c.expect("->")?;
        let (results, result_params) = self.func_results()?;
        self.c.expect("{")?;
        let body = self.block(&named, true)?;

        let func = Func {
            name,
            args: named.into_iter().map(|(_, ty)| ty).collect(),
            results,
            params: bound_params(params),
            result_params,
        };
        let inst = self.module.create_inst(NewInst {
            op: func.into(),
            operands: Vec::new(),
            operand_segments: None,
            result_types: Vec::new(),
            result_segments: None,
            regions: vec![body],
        });
        self.module.append(parent, inst);
        Ok(())
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
