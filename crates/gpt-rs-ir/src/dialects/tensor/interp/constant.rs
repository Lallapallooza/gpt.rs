//! Semantics of `tensor.constant`.

use std::sync::Arc;

use anyhow::Result;

use crate::dialects::tensor::ir::{Constant, ConstantOperands, ConstantResults};
use crate::host::{HostTensor, TensorData};
use crate::ir::{Eval, EvalCx, Host, NoRegions, OwnedHost};

impl Eval for Constant {
    fn eval(
        &self,
        _ins: ConstantOperands<Host<'_>>,
        _cx: &mut EvalCx<'_, NoRegions>,
    ) -> Result<ConstantResults<OwnedHost>> {
        // The result views the constant's bytes, which outlive it through the `Arc`.
        let value = &self.value;
        let count = value.dims.iter().product();
        let data = TensorData::read(value.dtype, count, Arc::new(value.bytes.clone()))?;
        Ok(ConstantResults {
            output: HostTensor::new(value.dims.clone(), data)?,
        })
    }
}
