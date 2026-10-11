//! The text form of parameter ids and of a function's signature, and reading a function.

use anyhow::Result;

use super::ir::*;
use crate::attr::Attr;
use crate::op::{ParsedSymbol, SymbolOp};
use crate::rewrite::bound_params;
use crate::text::{named_arg, results_text, Cursor};
use crate::types::Type;

impl Attr for ParamId {
    fn print(&self, out: &mut String) {
        out.push_str(&self.0.to_string());
    }
    fn parse(c: &mut Cursor<'_>) -> Result<Self> {
        c.uint128().map(ParamId)
    }
}

impl Func {
    /// Whether the runtime runs this function when it loads parameters, not per call.
    pub fn is_load_time(&self) -> bool {
        !self.result_params.is_empty()
    }

    /// The parameter that argument `index` binds.
    pub fn param(&self, index: usize) -> Option<ParamId> {
        self.params.get(index).copied().flatten()
    }
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

/// A function's result types, and each result's parameter when it is a load-time function. A
/// single run-time result may be written without parentheses.
fn func_results(c: &mut Cursor<'_>) -> Result<(Vec<Type>, Vec<ParamId>)> {
    if c.peek() != Some('(') {
        return Ok((vec![Type::parse(c)?], Vec::new()));
    }
    c.expect("(")?;
    let results = c.list(")", |c| Ok((Type::parse(c)?, param_binding(c)?)))?;
    let (types, params) = results.into_iter().unzip();
    let params = bound_params(params).into_iter().collect::<Option<_>>();
    let unbound = "a load-time function binds every result to a parameter";
    Ok((types, params.ok_or_else(|| c.error(unbound))?))
}

/// `func.func @name(%a: T {param = id}, ...) -> (T {param = id}, ...)`, the parameter bindings of
/// a load-time function written after their types.
impl SymbolOp for Func {
    fn symbol(&self) -> &str {
        &self.name
    }

    fn arg_suffix(&self, index: usize) -> String {
        match self.param(index) {
            Some(param) => format!(" {{param = {}}}", param.0),
            None => String::new(),
        }
    }

    fn results_text(&self) -> String {
        if !self.is_load_time() {
            return results_text(&self.results.iter().collect::<Vec<_>>());
        }
        let mut bound = Vec::with_capacity(self.results.len());
        for (index, ty) in self.results.iter().enumerate() {
            bound.push(match self.result_params.get(index) {
                Some(param) => format!("{ty} {{param = {}}}", param.0),
                None => ty.to_string(),
            });
        }
        format!("({})", bound.join(", "))
    }

    fn attr_entries(&self) -> Vec<String> {
        Vec::new()
    }

    fn parse_signature(c: &mut Cursor<'_>) -> Result<ParsedSymbol<Self>> {
        let name = c.sigil_name('@')?.to_string();
        c.expect("(")?;
        let bound = c.list(")", |c| Ok((named_arg(c)?, param_binding(c)?)))?;
        let (named, params): (Vec<(String, Type)>, Vec<Option<ParamId>>) =
            bound.into_iter().unzip();
        c.expect("->")?;
        let (results, result_params) = func_results(c)?;
        let mut args = Vec::with_capacity(named.len());
        for (_, ty) in &named {
            args.push(ty.clone());
        }
        let func = Func {
            name,
            args,
            results,
            params: bound_params(params),
            result_params,
        };
        Ok((func, named))
    }
}
