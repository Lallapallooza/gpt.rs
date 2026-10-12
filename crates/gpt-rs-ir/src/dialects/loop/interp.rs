//! Semantics of the `loop` ops.

use std::collections::{BTreeSet, HashMap};

use anyhow::{anyhow, bail, Result};

use super::ir::*;
use crate::host::{HostBuffer, HostScalar, HostValue};
use crate::index::{Expr, IndexEnv, ReadKind};
use crate::module::Region;
use crate::op::{Eval, EvalCx, Host, NoRegions, OwnedHost};
use crate::types::{BufferType, Dim, Space, Sym};

impl Eval for Func {
    fn eval(
        &self,
        _ins: FuncOperands<Host<'_>>,
        _cx: &mut EvalCx<'_, FuncRegions<Region>>,
    ) -> Result<FuncResults<OwnedHost>> {
        bail!("loop.func runs when host.dispatch calls it, not as an op")
    }
}

impl Eval for Return {
    fn eval(
        &self,
        _ins: ReturnOperands<Host<'_>>,
        _cx: &mut EvalCx<'_, NoRegions>,
    ) -> Result<ReturnResults<OwnedHost>> {
        Ok(ReturnResults::default())
    }
}

impl Eval for For {
    fn eval(
        &self,
        ins: ForOperands<Host<'_>>,
        cx: &mut EvalCx<'_, ForRegions<Region>>,
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
        ins: ForallOperands<Host<'_>>,
        cx: &mut EvalCx<'_, ForallRegions<Region>>,
    ) -> Result<ForallResults<OwnedHost>> {
        cx.run_units(cx.regions.body, ins.start, ins.end)?;
        Ok(ForallResults::default())
    }
}

impl Eval for If {
    fn eval(
        &self,
        ins: IfOperands<Host<'_>>,
        cx: &mut EvalCx<'_, IfRegions<Region>>,
    ) -> Result<IfResults<OwnedHost>> {
        if ins.condition.as_int() != Some(0) {
            cx.run(cx.regions.then, Vec::new())?;
        }
        Ok(IfResults::default())
    }
}

impl Eval for Yield {
    fn eval(
        &self,
        _ins: YieldOperands<Host<'_>>,
        _cx: &mut EvalCx<'_, NoRegions>,
    ) -> Result<YieldResults<OwnedHost>> {
        Ok(YieldResults::default())
    }
}

/// What the expressions of `loop.index` and `loop.predicate` read: their operands, as the
/// iterators `di`, and the extents of the symbols they name.
struct Operands {
    values: Vec<i64>,
    syms: HashMap<Sym, i64>,
}

impl Operands {
    /// The operands `values`, and the extent the run bound to each symbol of `exprs`.
    fn new<'e, R: ReadKind, C>(
        values: Vec<i64>,
        exprs: impl IntoIterator<Item = &'e Expr<R>>,
        cx: &EvalCx<'_, C>,
    ) -> Result<Self> {
        let mut names = BTreeSet::new();
        for expr in exprs {
            expr.collect_syms(&mut names);
        }
        let mut syms = HashMap::new();
        for sym in names {
            let extent = cx
                .sym(&sym)
                .ok_or_else(|| anyhow!("the symbol {{{}}} is not bound", sym.as_str()))?;
            syms.insert(sym, extent as i64);
        }
        Ok(Self { values, syms })
    }
}

impl IndexEnv for Operands {
    fn iter(&self, position: usize) -> Option<i64> {
        self.values.get(position).copied()
    }

    fn sym(&self, sym: &Sym) -> Option<i64> {
        self.syms.get(sym).copied()
    }

    fn read(&self, operand: usize, _coords: &[i64]) -> Result<i64> {
        bail!("an index expression of the loop dialect reads no element of ${operand}")
    }
}

impl Eval for Index {
    fn eval(
        &self,
        ins: IndexOperands<Host<'_>>,
        cx: &mut EvalCx<'_, NoRegions>,
    ) -> Result<IndexResults<OwnedHost>> {
        let operands = Operands::new(ins.ins, [&self.expr], cx)?;
        Ok(IndexResults {
            output: self.expr.eval(&operands)?,
        })
    }
}

impl Eval for Predicate {
    fn eval(
        &self,
        ins: PredicateOperands<Host<'_>>,
        cx: &mut EvalCx<'_, NoRegions>,
    ) -> Result<PredicateResults<OwnedHost>> {
        let exprs = self
            .constraints
            .iter()
            .flat_map(|constraint| constraint.exprs());
        let operands = Operands::new(ins.ins, exprs, cx)?;
        let mut holds = true;
        for constraint in &self.constraints {
            holds &= constraint.holds(&operands)?;
        }
        Ok(PredicateResults {
            output: HostScalar::int(crate::types::DType::I1, i128::from(holds))?,
        })
    }
}

impl Eval for Alloc {
    fn eval(
        &self,
        _ins: AllocOperands<Host<'_>>,
        _cx: &mut EvalCx<'_, NoRegions>,
    ) -> Result<AllocResults<OwnedHost>> {
        let ty = BufferType {
            dtype: self.dtype,
            dims: self
                .shape
                .iter()
                .map(|&extent| Dim::Static(extent))
                .collect(),
            space: Space::Private,
        };
        Ok(AllocResults {
            output: HostBuffer::new(ty)?,
        })
    }
}

impl Eval for Load {
    fn eval(
        &self,
        ins: LoadOperands<Host<'_>>,
        cx: &mut EvalCx<'_, NoRegions>,
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
        ins: StoreOperands<Host<'_>>,
        cx: &mut EvalCx<'_, NoRegions>,
    ) -> Result<StoreResults<OwnedHost>> {
        ins.buffer.store(&ins.indices, ins.value, cx.unit())?;
        Ok(StoreResults::default())
    }
}
