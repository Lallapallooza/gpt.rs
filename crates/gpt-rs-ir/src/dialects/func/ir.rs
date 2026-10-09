//! The `func` dialect's ops: their attributes, groups and type rules.

use anyhow::{ensure, Result};

use crate::attr::Attr;
use crate::module::Region;
use crate::op::{OpDef, OwnedTypes, RegionOpDef, RegionSig, TypeCx, Types};
use crate::text::Cursor;
use crate::traits::{IsolatedFromAbove, Terminator};
use crate::types::Type;
use crate::Op;

/// Stable identity of a model parameter: derived from its checkpoint name, or from the text of the
/// load-time function that computes it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ParamId(pub u128);

impl Attr for ParamId {
    fn print(&self, out: &mut String) {
        out.push_str(&self.0.to_string());
    }
    fn parse(c: &mut Cursor<'_>) -> Result<Self> {
        c.uint128().map(ParamId)
    }
}

/// `func.func`: a named function. Its body's block arguments are the function arguments, and its
/// `func.return` yields the results.
///
/// A function whose results are bound to parameters is a load-time function. It reads only
/// parameters, and the runtime runs it once when it loads them. The runtime binds each result to
/// its parameter wherever an argument names that parameter.
#[derive(Op, Debug, Clone, PartialEq, Eq, Hash)]
#[op("func.func", regions(body: Block<Return>), traits(IsolatedFromAbove))]
pub struct Func {
    pub name: String,
    pub args: Vec<Type>,
    pub results: Vec<Type>,
    /// For each argument, the parameter it binds, or `None` for a run-time input.
    #[attr(default)]
    pub params: Vec<Option<ParamId>>,
    /// For each result of a load-time function, the parameter it computes. Empty otherwise.
    #[attr(default)]
    pub result_params: Vec<ParamId>,
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

impl RegionOpDef for Func {
    fn region_signatures(&self, _ins: FuncOperands<Types<'_>>) -> Result<FuncRegions<RegionSig>> {
        Ok(FuncRegions {
            body: RegionSig::new(self.args.clone(), self.results.clone()),
        })
    }
}

impl OpDef for Func {
    fn result_types(
        &self,
        _ins: FuncOperands<Types<'_>>,
        _cx: &TypeCx<'_, FuncRegions<Region>>,
    ) -> Result<FuncResults<OwnedTypes>> {
        ensure!(
            self.params.is_empty() || self.params.len() == self.args.len(),
            "{} params for {} arguments",
            self.params.len(),
            self.args.len()
        );
        if self.is_load_time() {
            ensure!(
                self.result_params.len() == self.results.len(),
                "a load-time function binds every result to a parameter"
            );
            ensure!(
                (0..self.args.len()).all(|i| self.param(i).is_some()),
                "a load-time function reads only parameters"
            );
        }
        Ok(FuncResults::default())
    }
}

/// `func.return`: ends a function body. Its operands are the function results.
#[derive(Op, Debug, Clone, PartialEq, Eq, Hash)]
#[op("func.return", operands(values: Variadic<Any>), traits(Terminator))]
pub struct Return;

impl OpDef for Return {
    fn result_types(
        &self,
        _ins: ReturnOperands<Types<'_>>,
        _cx: &TypeCx<'_, crate::op::NoRegions>,
    ) -> Result<ReturnResults<OwnedTypes>> {
        Ok(ReturnResults::default())
    }
}

#[derive(crate::Dialect, Debug, Clone, PartialEq, Eq, Hash)]
#[dialect(name = "func", op = Func)]
pub enum FuncOp {
    Func(Func),
    Return(Return),
}

impl Op {
    /// The `func.func` attributes when this is a function.
    pub fn as_func(&self) -> Option<&Func> {
        <Func as crate::op::InOp>::cast(self)
    }
}
