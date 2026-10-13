//! The type rules of the control-flow ops: the types each op gives, and what it rejects.

use anyhow::{ensure, Result};

use crate::dialects::tensor::ir::*;
use crate::ir::{
    ranked, DType, Dim, NoRegions, OpDef, OwnedTypes, Region, RegionOpDef, RegionSig, TensorType,
    Type, TypeCx, Types,
};

impl OpDef for Yield {
    fn result_types(
        &self,
        _ins: YieldOperands<Types<'_>>,
        _cx: &TypeCx<'_, NoRegions>,
    ) -> Result<YieldResults<OwnedTypes>> {
        Ok(YieldResults::default())
    }
}

impl RegionOpDef for Cond {
    fn region_signatures(&self, ins: CondOperands<Types<'_>>) -> Result<CondRegions<RegionSig>> {
        let branch = || {
            RegionSig::new(
                tensors(&ins.args),
                self.results.iter().cloned().map(Type::Tensor),
            )
        };
        Ok(CondRegions {
            then_branch: branch(),
            else_branch: branch(),
        })
    }
}

impl OpDef for Cond {
    fn result_types(
        &self,
        ins: CondOperands<Types<'_>>,
        _cx: &TypeCx<'_, CondRegions<Region>>,
    ) -> Result<CondResults<OwnedTypes>> {
        ensure_predicate(ins.pred)?;
        Ok(CondResults {
            outputs: self.results.clone(),
        })
    }
}

impl RegionOpDef for While {
    fn region_signatures(&self, ins: WhileOperands<Types<'_>>) -> Result<WhileRegions<RegionSig>> {
        let carried = tensors(&ins.init);
        let predicate = Type::Tensor(TensorType::new(DType::I1, [] as [usize; 0]));
        let cond_yields = [vec![predicate], carried.clone()].concat();
        Ok(WhileRegions {
            cond: RegionSig::new(carried.clone(), cond_yields),
            body: RegionSig::new(carried.clone(), carried),
        })
    }
}

impl OpDef for While {
    fn result_types(
        &self,
        ins: WhileOperands<Types<'_>>,
        _cx: &TypeCx<'_, WhileRegions<Region>>,
    ) -> Result<WhileResults<OwnedTypes>> {
        Ok(WhileResults {
            outputs: ins.init.iter().map(|&ty| ty.clone()).collect(),
        })
    }
}

impl RegionOpDef for Scan {
    fn region_signatures(&self, ins: ScanOperands<Types<'_>>) -> Result<ScanRegions<RegionSig>> {
        steps(self, &ins.inputs)?;
        let (carried, scanned) = ins.inputs.split_at(self.carry_count);
        let mut args = tensors(carried);
        for input in scanned {
            let step_dims = input.dims[1..].iter().cloned().collect();
            args.push(Type::Tensor(input.with_dims(step_dims)));
        }
        let mut yields = tensors(carried);
        for output in &self.outputs {
            yields.push(Type::Tensor(output.clone()));
        }
        Ok(ScanRegions {
            body: RegionSig::new(args, yields),
        })
    }
}

impl OpDef for Scan {
    fn result_types(
        &self,
        ins: ScanOperands<Types<'_>>,
        _cx: &TypeCx<'_, ScanRegions<Region>>,
    ) -> Result<ScanResults<OwnedTypes>> {
        let steps = steps(self, &ins.inputs)?;
        let mut outputs: Vec<TensorType> = ins.inputs[..self.carry_count]
            .iter()
            .map(|&ty| ty.clone())
            .collect();
        for output in &self.outputs {
            let stacked_dims = [std::slice::from_ref(&steps), &output.dims[..]].concat();
            outputs.push(output.with_dims(ranked(stacked_dims)?));
        }
        Ok(ScanResults { outputs })
    }
}

fn tensors(types: &[&TensorType]) -> Vec<Type> {
    types.iter().map(|&ty| Type::Tensor(ty.clone())).collect()
}

/// Checks that `pred` is an `i1` scalar.
fn ensure_predicate(pred: &TensorType) -> Result<()> {
    ensure!(
        pred.dtype == DType::I1 && pred.rank() == 0,
        "the predicate must be an i1 scalar, got {pred}"
    );
    Ok(())
}

/// The steps: the leading extent that every scanned input shares.
fn steps(scan: &Scan, inputs: &[&TensorType]) -> Result<Dim> {
    ensure!(
        scan.carry_count <= inputs.len(),
        "carry_count {} exceeds the {} operands",
        scan.carry_count,
        inputs.len()
    );
    let scanned = &inputs[scan.carry_count..];
    ensure!(!scanned.is_empty(), "a scan needs a scanned input");
    let steps = scanned[0].dims.first().cloned();
    for input in scanned {
        ensure!(
            input.rank() >= 1 && input.dims.first().cloned() == steps,
            "the scanned inputs must share their leading extent, got {input}"
        );
    }
    Ok(steps.expect("a scanned input has rank 1 or more"))
}
