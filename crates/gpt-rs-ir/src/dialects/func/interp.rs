//! Semantics of the func ops.

use anyhow::{bail, Result};

use super::ir::*;
use crate::module::Region;
use crate::op::{Eval, EvalCx, Host, OwnedHost};

impl Eval for Func {
    fn eval(
        &self,
        _ins: FuncOperands<Host<'_>>,
        _cx: &mut EvalCx<'_, FuncRegions<Region>>,
    ) -> Result<FuncResults<OwnedHost>> {
        bail!("func.func is entered through interp::run, not evaluated as an op")
    }
}

impl Eval for Return {
    fn eval(
        &self,
        _ins: ReturnOperands<Host<'_>>,
        _cx: &mut EvalCx<'_, crate::op::NoRegions>,
    ) -> Result<ReturnResults<OwnedHost>> {
        Ok(ReturnResults::default())
    }
}
