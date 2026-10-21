//! Semantics of the `loop` ops.

use anyhow::{anyhow, bail, ensure, Result};

use super::ir::*;
use crate::host::HostScalar;
use crate::interp::{HostBuffer, HostValue};
use crate::ir::{BufferType, Dim, Eval, EvalCx, Host, IndexEnv, NoRegions, OwnedHost, Region, Sym};

impl Eval for Func {
    fn eval(
        &self,
        _ins: FuncOperands<Host>,
        _cx: &mut EvalCx<FuncRegions<Region>>,
    ) -> Result<FuncResults<OwnedHost>> {
        bail!("loop.func runs when host.dispatch or host.launch calls it, not as an op")
    }
}

impl Eval for Return {
    fn eval(
        &self,
        _ins: ReturnOperands<Host>,
        _cx: &mut EvalCx<NoRegions>,
    ) -> Result<ReturnResults<OwnedHost>> {
        Ok(ReturnResults::default())
    }
}

impl Eval for For {
    fn eval(
        &self,
        ins: ForOperands<Host>,
        cx: &mut EvalCx<ForRegions<Region>>,
    ) -> Result<ForResults<OwnedHost>> {
        for index in ins.start..ins.end {
            cx.run(cx.regions.body, vec![HostValue::Index(index)])?;
        }
        Ok(ForResults::default())
    }
}

impl Eval for Forall {
    fn eval(
        &self,
        ins: ForallOperands<Host>,
        cx: &mut EvalCx<ForallRegions<Region>>,
    ) -> Result<ForallResults<OwnedHost>> {
        let units = (ins.start..ins.end)
            .map(|index| vec![HostValue::Index(index)])
            .collect();
        cx.run_units(cx.regions.body, units)?;
        Ok(ForallResults::default())
    }
}

impl Eval for If {
    fn eval(
        &self,
        ins: IfOperands<Host>,
        cx: &mut EvalCx<IfRegions<Region>>,
    ) -> Result<IfResults<OwnedHost>> {
        if ins.condition.as_int() != Some(0) {
            cx.run(cx.regions.then, Vec::new())?;
        }
        Ok(IfResults::default())
    }
}

impl Eval for Atom {
    fn eval(
        &self,
        ins: AtomOperands<Host>,
        cx: &mut EvalCx<AtomRegions<Region>>,
    ) -> Result<AtomResults<OwnedHost>> {
        let mut points: Vec<Vec<HostValue>> = vec![Vec::new()];
        for (&extent, &tile) in ins.extents.iter().zip(&self.tile) {
            ensure!(
                extent >= 1,
                "the extents {:?} run no point of the tile {:?}",
                ins.extents,
                self.tile
            );
            ensure!(
                extent <= tile as i64,
                "the extents {:?} exceed the tile {:?}",
                ins.extents,
                self.tile
            );
            let mut longer = Vec::with_capacity(points.len() * extent as usize);
            for point in &points {
                for coord in 0..extent {
                    longer.push([&point[..], &[HostValue::Index(coord)]].concat());
                }
            }
            points = longer;
        }
        cx.run_units(cx.regions.body, points)?;
        Ok(AtomResults::default())
    }
}

impl Eval for Yield {
    fn eval(
        &self,
        _ins: YieldOperands<Host>,
        _cx: &mut EvalCx<NoRegions>,
    ) -> Result<YieldResults<OwnedHost>> {
        Ok(YieldResults::default())
    }
}

/// What the expressions of `loop.index` and `loop.predicate` read: their operands, as the
/// iterators `di`, and the extents the run bound to the symbols they name.
struct Operands<'c, 'a, R> {
    values: Vec<i64>,
    cx: &'c EvalCx<'a, R>,
}

impl<R> IndexEnv for Operands<'_, '_, R> {
    fn iter(&self, position: usize) -> Option<i64> {
        self.values.get(position).copied()
    }

    fn sym(&self, sym: &Sym) -> Option<i64> {
        self.cx.sym(sym).map(|extent| extent as i64)
    }

    fn read(&self, operand: usize, _coords: &[i64]) -> Result<i64> {
        bail!("an index expression of the loop dialect reads no element of ${operand}")
    }
}

impl Eval for Index {
    fn eval(
        &self,
        ins: IndexOperands<Host>,
        cx: &mut EvalCx<NoRegions>,
    ) -> Result<IndexResults<OwnedHost>> {
        let operands = Operands {
            values: ins.ins,
            cx,
        };
        Ok(IndexResults {
            output: self.expr.eval(&operands)?,
        })
    }
}

impl Eval for Predicate {
    fn eval(
        &self,
        ins: PredicateOperands<Host>,
        cx: &mut EvalCx<NoRegions>,
    ) -> Result<PredicateResults<OwnedHost>> {
        let operands = Operands {
            values: ins.ins,
            cx,
        };
        let mut holds = true;
        for constraint in &self.constraints {
            holds &= constraint.holds(&operands)?;
        }
        Ok(PredicateResults {
            output: HostScalar::int(crate::ir::DType::I1, i128::from(holds))?,
        })
    }
}

impl Eval for Alloc {
    fn eval(
        &self,
        _ins: AllocOperands<Host>,
        _cx: &mut EvalCx<NoRegions>,
    ) -> Result<AllocResults<OwnedHost>> {
        let ty = BufferType {
            dtype: self.dtype,
            dims: self
                .shape
                .iter()
                .map(|&extent| Dim::Static(extent))
                .collect(),
            space: self.space,
        };
        Ok(AllocResults {
            output: HostBuffer::new(ty)?,
        })
    }
}

impl Eval for Pick {
    fn eval(
        &self,
        ins: PickOperands<Host>,
        _cx: &mut EvalCx<NoRegions>,
    ) -> Result<PickResults<OwnedHost>> {
        let position = ins.index;
        let picked = usize::try_from(position)
            .ok()
            .and_then(|index| ins.buffers.get(index))
            .ok_or_else(|| anyhow!("it picks buffer {position} of {}", ins.buffers.len()))?;
        Ok(PickResults {
            output: (*picked).clone(),
        })
    }
}

impl Eval for Load {
    fn eval(
        &self,
        ins: LoadOperands<Host>,
        cx: &mut EvalCx<NoRegions>,
    ) -> Result<LoadResults<OwnedHost>> {
        let output = match self.pad {
            Some(pad) if !ins.buffer.contains(&ins.indices) => {
                HostScalar::from_f64(ins.buffer.ty().dtype, pad.0)?
            }
            _ => ins.buffer.load(&ins.indices, cx.unit())?,
        };
        Ok(LoadResults { output })
    }
}

impl Eval for Store {
    fn eval(
        &self,
        ins: StoreOperands<Host>,
        cx: &mut EvalCx<NoRegions>,
    ) -> Result<StoreResults<OwnedHost>> {
        ins.buffer.store(&ins.indices, ins.value, cx.unit())?;
        Ok(StoreResults::default())
    }
}

impl Eval for Prefetch {
    fn eval(
        &self,
        _ins: PrefetchOperands<Host>,
        _cx: &mut EvalCx<NoRegions>,
    ) -> Result<PrefetchResults<OwnedHost>> {
        Ok(PrefetchResults::default())
    }
}
