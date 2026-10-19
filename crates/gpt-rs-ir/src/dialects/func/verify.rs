//! The type rules of the `func` ops: the types each op gives, and what it rejects.

use anyhow::{ensure, Result};

use crate::dialects::func::ir::*;
use crate::ir::{OpDef, OwnedTypes, Region, RegionOpDef, RegionSig, TypeCx, Types};

impl RegionOpDef for Func {
    fn region_signatures(&self, _ins: FuncOperands<Types>) -> Result<FuncRegions<RegionSig>> {
        Ok(FuncRegions {
            body: RegionSig::new(self.args.clone(), self.results.clone()),
        })
    }
}

impl OpDef for Func {
    fn result_types(
        &self,
        _ins: FuncOperands<Types>,
        _cx: &TypeCx<FuncRegions<Region>>,
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
                (0..self.args.len()).all(|index| self.param(index).is_some()),
                "a load-time function reads only parameters"
            );
        }
        Ok(FuncResults::default())
    }
}

impl OpDef for Return {
    fn result_types(
        &self,
        _ins: ReturnOperands<Types>,
        _cx: &TypeCx<crate::ir::NoRegions>,
    ) -> Result<ReturnResults<OwnedTypes>> {
        Ok(ReturnResults::default())
    }
}
