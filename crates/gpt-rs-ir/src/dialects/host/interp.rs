//! Semantics of `host.dispatch`: a call of its `tile.dispatch`.

use anyhow::Result;

use super::ir::*;
use crate::host::HostValue;
use crate::op::{Eval, EvalCx, Host, NoRegions, OwnedHost};

impl Eval for Dispatch {
    fn eval(
        &self,
        ins: DispatchOperands<Host<'_>>,
        cx: &mut EvalCx<'_, NoRegions>,
    ) -> Result<DispatchResults<OwnedHost>> {
        let args = HostValue::of_tensors(&ins.args);
        let outs = HostValue::into_tensors(cx.call(&self.callee.0, args)?)?;
        Ok(DispatchResults { outs })
    }
}
