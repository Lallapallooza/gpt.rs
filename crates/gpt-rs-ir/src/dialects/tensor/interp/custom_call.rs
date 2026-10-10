//! Semantics of the ops in `ir/custom_call.rs`.

use anyhow::{bail, Result};

use crate::dialects::tensor::ir::{CustomCall, CustomCallOperands, CustomCallResults};
use crate::op::{Eval, EvalCx, Host, NoRegions, OwnedHost};

impl Eval for CustomCall {
    fn eval(
        &self,
        _ins: CustomCallOperands<Host<'_>>,
        _cx: &mut EvalCx<'_, NoRegions>,
    ) -> Result<CustomCallResults<OwnedHost>> {
        bail!(
            "`{}` is a backend kernel with no reference semantics",
            self.target
        )
    }
}
