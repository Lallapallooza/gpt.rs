//! The type rules of the `func` ops: the types each op gives, and what it rejects.

use anyhow::{ensure, Result};

use crate::dialects::func::ir::*;
use crate::module::Region;
use crate::op::{OpDef, OwnedTypes, RegionOpDef, RegionSig, TypeCx, Types};

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

impl OpDef for Return {
    fn result_types(
        &self,
        _ins: ReturnOperands<Types<'_>>,
        _cx: &TypeCx<'_, crate::op::NoRegions>,
    ) -> Result<ReturnResults<OwnedTypes>> {
        Ok(ReturnResults::default())
    }
}
