//! Control flow over whole tensor values: ops whose regions compute their results.

use anyhow::{ensure, Result};

use crate::module::Region;
use crate::op::{NoRegions, OpDef, OwnedTypes, RegionOpDef, RegionSig, TypeCx, Types};
use crate::traits::Terminator;
use crate::types::{DType, Dim, TensorType, Type};

/// `tensor.yield`: ends a region of a control-flow op. Its operands are the region's results.
///
/// - Each region of a control-flow op has its own arguments and ends with a `tensor.yield`. The
///   ops compute on whole tensor values, like the rest of the dialect.
#[derive(crate::Op, Debug, Clone, PartialEq, Eq, Hash)]
#[op("tensor.yield", operands(values: Variadic<Tensor>), traits(Terminator))]
pub struct Yield;

impl OpDef for Yield {
    fn result_types(
        &self,
        _ins: YieldOperands<Types<'_>>,
        _cx: &TypeCx<'_, NoRegions>,
    ) -> Result<YieldResults<OwnedTypes>> {
        Ok(YieldResults::default())
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

/// `tensor.cond`: runs `then_branch` on `args` when `pred` holds and `else_branch` otherwise,
/// and gives the results that the branch yields, of the types `results`.
///
/// - `pred` is an `i1` scalar, `i1[]`.
/// - Both branches take arguments of the types of `args` and MUST yield the types `results`.
#[derive(crate::Op, Debug, Clone, PartialEq, Eq, Hash)]
#[op(
    "tensor.cond",
    operands(pred: Tensor, args: Variadic<Tensor>),
    results(outputs: Variadic<Tensor>),
    regions(then_branch: Block<Yield>, else_branch: Block<Yield>)
)]
pub struct Cond {
    pub results: Vec<TensorType>,
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

/// `tensor.while`: a loop over the carried values `init`.
///
/// - `cond` takes the carried values and yields an `i1` scalar and new carried values:
///   `cond(carry...) -> (i1[], carry...)`.
/// - If the scalar is true, `body` runs on the carried values `cond` yielded, `body(carry...) ->
///   carry...`, and the loop repeats with the values `body` yields.
/// - If the scalar is false, the results are the carried values `cond` yielded.
#[derive(crate::Op, Debug, Clone, PartialEq, Eq, Hash)]
#[op(
    "tensor.while",
    operands(init: Variadic<Tensor>),
    results(outputs: Variadic<Tensor>),
    regions(cond: Block<Yield>, body: Block<Yield>)
)]
pub struct While;

impl RegionOpDef for While {
    fn region_signatures(&self, ins: WhileOperands<Types<'_>>) -> Result<WhileRegions<RegionSig>> {
        let carried = tensors(&ins.init);
        let predicate = Type::Tensor(TensorType::new(DType::I1, [] as [usize; 0]));
        Ok(WhileRegions {
            cond: RegionSig::new(
                carried.clone(),
                std::iter::once(predicate).chain(carried.clone()),
            ),
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

/// `tensor.scan`: a loop over the leading axis of the scanned inputs, like a functional for-loop.
///
/// - Its operands are `carry_count` carried values, then the scanned inputs, each `[T, ...]`:
///   at least one, all with the same leading extent `T`.
/// - Step `t` runs `body` on the carried values and slice `t` of each scanned input, and `body`
///   yields the next carried values, then one output of each type in `outputs`.
/// - The results are the final carried values, then each output stacked along a new leading axis
///   over the steps, `[T, ...]`.
#[derive(crate::Op, Debug, Clone, PartialEq, Eq, Hash)]
#[op(
    "tensor.scan",
    operands(inputs: Variadic<Tensor>),
    results(outputs: Variadic<Tensor>),
    regions(body: Block<Yield>)
)]
pub struct Scan {
    pub carry_count: usize,
    /// The type of each per-step output.
    pub outputs: Vec<TensorType>,
}

impl Scan {
    /// The steps: the leading extent that every scanned input shares.
    fn steps(&self, inputs: &[&TensorType]) -> Result<Dim> {
        ensure!(
            self.carry_count <= inputs.len(),
            "carry_count {} exceeds the {} operands",
            self.carry_count,
            inputs.len()
        );
        let scanned = &inputs[self.carry_count..];
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
}

impl RegionOpDef for Scan {
    fn region_signatures(&self, ins: ScanOperands<Types<'_>>) -> Result<ScanRegions<RegionSig>> {
        self.steps(&ins.inputs)?;
        let (carried, scanned) = ins.inputs.split_at(self.carry_count);
        let args = tensors(carried).into_iter().chain(
            scanned
                .iter()
                .map(|&ty| Type::Tensor(ty.with_dims(ty.dims[1..].iter().cloned().collect()))),
        );
        let yields = tensors(carried)
            .into_iter()
            .chain(self.outputs.iter().cloned().map(Type::Tensor));
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
        let steps = self.steps(&ins.inputs)?;
        let carried = ins.inputs[..self.carry_count].iter().map(|&ty| ty.clone());
        let stacked = self.outputs.iter().map(|ty| {
            ty.with_dims(
                std::iter::once(steps.clone())
                    .chain(ty.dims.iter().cloned())
                    .collect(),
            )
        });
        Ok(ScanResults {
            outputs: carried.chain(stacked).collect(),
        })
    }
}
