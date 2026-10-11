//! The type rules of `tensor.custom_call`: the types each op gives, and what it rejects.

use anyhow::{ensure, Result};

use crate::dialects::tensor::ir::*;
use crate::op::{NoRegions, OpDef, OwnedTypes, TypeCx, Types};

impl OpDef for CustomCall {
    fn result_types(
        &self,
        _ins: CustomCallOperands<Types<'_>>,
        _cx: &TypeCx<'_, NoRegions>,
    ) -> Result<CustomCallResults<OwnedTypes>> {
        ensure!(!self.target.is_empty(), "the target is empty");
        Ok(CustomCallResults {
            outputs: self.results.clone(),
        })
    }
}
