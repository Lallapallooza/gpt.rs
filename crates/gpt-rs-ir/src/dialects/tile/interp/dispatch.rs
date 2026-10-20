//! Semantics of `tile.group` and `tile.dispatch`.

use anyhow::{bail, Result};

use crate::dialects::tile::ir::*;
use crate::interp::HostValue;
use crate::ir::{Eval, EvalCx, Host, OwnedHost, Region};

impl Eval for Group {
    fn eval(
        &self,
        _ins: GroupOperands<Host>,
        cx: &mut EvalCx<GroupRegions<Region>>,
    ) -> Result<GroupResults<OwnedHost>> {
        let outs = HostValue::into_tensors(cx.run(cx.regions.body, Vec::new())?)?;
        Ok(GroupResults { outs })
    }
}

impl Eval for Dispatch {
    fn eval(
        &self,
        _ins: DispatchOperands<Host>,
        _cx: &mut EvalCx<DispatchRegions<Region>>,
    ) -> Result<DispatchResults<OwnedHost>> {
        bail!("tile.dispatch is entered through host.dispatch, not evaluated as an op")
    }
}
