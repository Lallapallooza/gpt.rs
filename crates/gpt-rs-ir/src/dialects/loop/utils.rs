//! The signature of `loop.func` as a call with tensors sees it.

use anyhow::{anyhow, Result};

use super::ir::*;
use crate::ir::TensorType;

impl Func {
    /// The tensor types of the arguments and of the results of a call with tensors: those of the
    /// `in` parameters and of the `out` parameters, each in order.
    pub fn call_types(&self) -> Result<(Vec<TensorType>, Vec<TensorType>)> {
        let mut args = Vec::new();
        let mut results = Vec::new();
        for (index, (ty, access)) in self.args.iter().zip(&self.access).enumerate() {
            let buffer = ty
                .as_buffer()
                .ok_or_else(|| anyhow!("parameter {index} of @{} is not a buffer", self.name))?;
            match access {
                Access::In => args.push(buffer.tensor_type()),
                Access::Out => results.push(buffer.tensor_type()),
            }
        }
        Ok((args, results))
    }
}
