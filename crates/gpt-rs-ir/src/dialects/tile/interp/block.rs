//! Semantics of `tile.block`, `tile.index` and `tile.yield`. A `tile.block` compiles its body
//! once into a scalar program, then runs it at every point of its domain in lexicographic order.

use std::collections::HashMap;

use anyhow::{anyhow, bail, ensure, Context, Result};

use super::combine::Acc;
use crate::dialects::tile::ir::*;
use crate::host::{HostScalar, HostTensor, HostValue};
use crate::index::{AccessExpr, IndexEnv, IterKind, Source, Write};
use crate::module::{Module, Region, Value};
use crate::op::{Eval, EvalCx, Host, NoRegions, OpDyn, OwnedHost, RegionRunner};
use crate::scalar::{self, Decoded};
use crate::traits::Terminator;
use crate::types::{DType, Dim, Sym, Type};
use crate::Op;

/// One op of a compiled block body, on the values of its slots.
struct Instr {
    op: Op,
    operands: Vec<usize>,
}

/// A block body as a flat program over value slots: the block arguments first, then one slot per
/// op result, and the slots its terminator yields.
struct Program {
    args: usize,
    instrs: Vec<Instr>,
    yields: Vec<usize>,
}

impl Program {
    fn compile(module: &Module, body: Region) -> Result<Program> {
        let mut slots: HashMap<Value, usize> = HashMap::new();
        for (index, &arg) in module.args(body).iter().enumerate() {
            slots.insert(arg, index);
        }
        let args = slots.len();
        let mut next = args;
        let mut instrs = Vec::new();
        for inst in module.insts(body) {
            let mut operands = Vec::new();
            for operand in module.operands(inst) {
                let slot = slots
                    .get(operand)
                    .copied()
                    .ok_or_else(|| anyhow!("a block body reads a value from outside"))?;
                operands.push(slot);
            }
            let op = module.op(inst);
            if op.has_trait::<Terminator>() {
                return Ok(Program {
                    args,
                    instrs,
                    yields: operands,
                });
            }
            for &result in module.results(inst) {
                slots.insert(result, next);
                next += 1;
            }
            instrs.push(Instr {
                op: op.clone(),
                operands,
            });
        }
        bail!("the block body ends without tile.yield")
    }

    /// Runs the program on `args` at the point of `runner` and returns what it yields.
    fn run(&self, args: &[HostScalar], runner: &mut BodyRunner<'_>) -> Result<Vec<HostScalar>> {
        let mut values: Vec<HostScalar> = Vec::with_capacity(self.args + self.instrs.len());
        values.extend_from_slice(args);
        for instr in &self.instrs {
            let mut ins = Vec::with_capacity(instr.operands.len());
            for &slot in &instr.operands {
                ins.push(HostValue::Scalar(values[slot]));
            }
            let ins: Vec<&HostValue> = ins.iter().collect();
            match instr.op.dyn_eval(&ins, None, &[], &[], runner)?.as_slice() {
                [HostValue::Scalar(result)] => values.push(*result),
                _ => bail!("{} gives one scalar", instr.op.name()),
            }
        }
        let mut yielded = Vec::with_capacity(self.yields.len());
        for &slot in &self.yields {
            yielded.push(values[slot]);
        }
        Ok(yielded)
    }
}

/// What the ops of a body run on: the point they run at, and no regions or calls.
struct BodyRunner<'a> {
    module: &'a Module,
    env: &'a dyn IndexEnv,
}

impl RegionRunner for BodyRunner<'_> {
    fn run_region(&mut self, _region: Region, _args: Vec<HostValue>) -> Result<Vec<HostValue>> {
        bail!("an op of a block body has no regions")
    }

    fn module(&self) -> &Module {
        self.module
    }

    fn sym(&self, _sym: &Sym) -> Option<usize> {
        None
    }

    fn call(&mut self, callee: &str, _args: Vec<HostValue>) -> Result<Vec<HostValue>> {
        bail!("an op of a block body calls nothing, not {callee}")
    }

    fn index_env(&self) -> Option<&dyn IndexEnv> {
        Some(self.env)
    }
}

/// The run-time view of a block at one point: iterator values, symbol extents and the integer
/// operands that index expressions read.
struct Env<'a> {
    point: &'a [i64],
    syms: &'a HashMap<Sym, i64>,
    operands: &'a [Decoded],
    dims: &'a [&'a [usize]],
}

/// The row-major position of `coords` in a tensor of shape `dims`, or `None` when a coordinate is
/// out of range.
fn position(coords: &[i64], dims: &[usize]) -> Option<usize> {
    if coords.len() != dims.len() {
        return None;
    }
    let mut at = 0usize;
    for (&coord, &extent) in coords.iter().zip(dims) {
        let index = usize::try_from(coord)
            .ok()
            .filter(|&index| index < extent)?;
        at = at * extent + index;
    }
    Some(at)
}

impl IndexEnv for Env<'_> {
    fn iter(&self, position: usize) -> Option<i64> {
        self.point.get(position).copied()
    }

    fn sym(&self, sym: &Sym) -> Option<i64> {
        self.syms.get(sym).copied()
    }

    fn read(&self, operand: usize, coords: &[i64]) -> Result<i64> {
        let dims = self.dims[operand];
        let at = position(coords, dims).ok_or_else(|| {
            anyhow!("index read ${operand} at {coords:?} is outside its shape {dims:?}")
        })?;
        self.operands[operand].get(at).as_index()
    }
}

/// What every point of a block shares, from which [`Ctx::at`] makes the view at one point.
struct Ctx<'a> {
    syms: &'a HashMap<Sym, i64>,
    operands: &'a [Decoded],
    dims: &'a [&'a [usize]],
}

impl Ctx<'_> {
    fn at<'b>(&'b self, point: &'b [i64]) -> Env<'b> {
        Env {
            point,
            syms: self.syms,
            operands: self.operands,
            dims: self.dims,
        }
    }
}

impl Env<'_> {
    fn eval_all(&self, exprs: &[AccessExpr]) -> Result<Vec<i64>> {
        exprs.iter().map(|expr| expr.eval(self)).collect()
    }
}

/// The extent of each symbol the block uses, in this run.
fn bind_syms(block: &Block, cx: &EvalCx<'_, BlockRegions<Region>>) -> Result<HashMap<Sym, i64>> {
    block
        .syms(cx.module(), cx.regions.body)
        .into_iter()
        .map(|sym| {
            let extent = cx
                .sym(&sym)
                .ok_or_else(|| anyhow!("the symbol {{{}}} is not bound", sym.as_str()))?;
            Ok((sym, extent as i64))
        })
        .collect()
}

/// The pad value of a read, as a scalar of the operand's dtype.
fn pad_scalar(pad: f64, dtype: DType) -> Result<HostScalar> {
    if dtype.is_float() {
        HostScalar::from_f64(dtype, pad)
    } else {
        HostScalar::int(dtype, pad as i128)
    }
}

/// The storage of one result while the block runs.
struct Out {
    dims: Vec<usize>,
    dtype: DType,
    elements: Vec<Option<HostScalar>>,
}

impl Out {
    fn set(&mut self, coords: &[i64], value: HostScalar, result: usize) -> Result<()> {
        let at = position(coords, &self.dims).ok_or_else(|| {
            anyhow!(
                "write {result} at {coords:?} is outside its result {:?}",
                self.dims
            )
        })?;
        self.elements[at] = Some(value);
        Ok(())
    }

    fn get(&self, coords: &[i64], result: usize) -> Result<Option<HostScalar>> {
        let at = position(coords, &self.dims).ok_or_else(|| {
            anyhow!(
                "write {result} at {coords:?} is outside its result {:?}",
                self.dims
            )
        })?;
        Ok(self.elements[at])
    }
}

/// A fold in progress: its accumulator, and the point that keyed it, at which its results are
/// written.
struct FoldGroup {
    acc: Acc,
    point: Vec<i64>,
}

/// One combine kind of a block: the yields and the results it takes.
struct Part {
    kind: CombineKind,
    yields: std::ops::Range<usize>,
    results: std::ops::Range<usize>,
    dtypes: Vec<DType>,
    /// The init operands, one per state component.
    init: Vec<usize>,
}

impl Eval for Block {
    fn eval(
        &self,
        ins: BlockOperands<Host<'_>>,
        cx: &mut EvalCx<'_, BlockRegions<Region>>,
    ) -> Result<BlockResults<OwnedHost>> {
        let module = cx.module();
        let program = Program::compile(module, cx.regions.body)?;
        let yield_types = Block::yield_types(module, cx.regions.body)?;
        let syms = bind_syms(self, cx)?;
        let extents = self
            .iters
            .iter()
            .map(|it| match &it.extent {
                Dim::Static(n) => Ok(*n as i64),
                Dim::Dynamic(sym) => Ok(syms[sym]),
            })
            .collect::<Result<Vec<i64>>>()?;
        let decoded: Vec<Decoded> = ins
            .ins
            .iter()
            .map(|tensor| Decoded::new(&tensor.data))
            .collect();
        let dims: Vec<&[usize]> = ins.ins.iter().map(|tensor| &tensor.dims[..]).collect();

        // The results, with `into` results starting as copies of their operands.
        let writes: Vec<&Write> = self.writes().collect();
        let result_types = cx.result_types().to_vec();
        let mut outs = Vec::new();
        for (result, (write, ty)) in writes.iter().zip(&result_types).enumerate() {
            let Type::Tensor(ty) = ty else {
                bail!("result {result} is not a tensor");
            };
            let dims = ty
                .static_dims()
                .ok_or_else(|| anyhow!("result {result} has unbound extents"))?
                .to_vec();
            let count = dims.iter().product();
            let elements = match write.into {
                Some(operand) => (0..count)
                    .map(|element| Some(decoded[operand].get(element)))
                    .collect(),
                None => vec![None; count],
            };
            outs.push(Out {
                dims,
                dtype: ty.dtype,
                elements,
            });
        }

        let mut parts = Vec::new();
        let (mut first_yield, mut first_result) = (0, 0);
        for fold in &self.folds {
            let kind = fold.combine.clone();
            let (yields, results) = (kind.rule().yields(), kind.rule().results());
            parts.push(Part {
                dtypes: yield_types[first_yield..first_yield + yields].to_vec(),
                yields: first_yield..first_yield + yields,
                results: first_result..first_result + results,
                init: fold.init.to_vec(),
                kind,
            });
            first_yield += yields;
            first_result += results;
        }
        let prefix = self
            .iters
            .iter()
            .position(|iter| iter.kind == IterKind::Prefix);

        // One accumulator map per folding part, keyed by the coordinates its key result writes.
        let mut groups: Vec<HashMap<Vec<i64>, FoldGroup>> =
            parts.iter().map(|_| HashMap::new()).collect();
        let ctx = Ctx {
            syms: &syms,
            operands: &decoded,
            dims: &dims,
        };
        // The key of a fold at a point: the coordinates of its key result, without the prefix
        // coordinate, which each prefix result names.
        let key_of = |part: &Part, env: &Env<'_>| -> Result<Vec<i64>> {
            let write = writes[part.results.start + (part.kind.rule().results() - 1)];
            let mut coords = Vec::with_capacity(write.coords.len());
            for coord in &write.coords {
                if prefix.is_some() && coord.as_iter() == prefix {
                    continue;
                }
                coords.push(coord.eval(env)?);
            }
            Ok(coords)
        };
        // The state of each component starts as the init operand at the element of the result
        // of the same position.
        let new_acc = |part: &Part, env: &Env<'_>| -> Result<Acc> {
            let mut init = Vec::with_capacity(part.init.len());
            for (component, &operand) in part.init.iter().enumerate() {
                let coords = env.eval_all(&writes[part.results.start + component].coords)?;
                let at = position(&coords, dims[operand])
                    .ok_or_else(|| anyhow!("init ${operand} at {coords:?} is out of range"))?;
                init.push(decoded[operand].get(at));
            }
            Acc::new(&part.kind, &part.dtypes, &init)
        };

        // Every element of a new folded result gets a fold, even one whose points are all masked
        // or whose reduction is empty: walk the parallel iterators once.
        let par: Vec<usize> = (0..self.iters.len())
            .filter(|&i| !self.iters[i].kind.is_reduced() && Some(i) != prefix)
            .collect();
        let folding: Vec<usize> = (0..parts.len())
            .filter(|&p| {
                parts[p].kind != CombineKind::None
                    && writes[parts[p].results.start].into.is_none()
                    && prefix.is_none()
            })
            .collect();
        if !folding.is_empty() {
            let mut point = vec![0i64; self.iters.len()];
            let mut done = par.iter().any(|&i| extents[i] == 0);
            while !done {
                let env = ctx.at(&point);
                for &p in &folding {
                    let key = key_of(&parts[p], &env)?;
                    if !groups[p].contains_key(&key) {
                        let acc = new_acc(&parts[p], &env)?;
                        groups[p].insert(
                            key,
                            FoldGroup {
                                acc,
                                point: point.clone(),
                            },
                        );
                    }
                }
                done = advance(&mut point, &extents, &par);
            }
        }

        // The points, in lexicographic order.
        let all: Vec<usize> = (0..self.iters.len()).collect();
        let mut point = vec![0i64; self.iters.len()];
        let mut done = extents.contains(&0);
        let mut args = Vec::with_capacity(self.reads.len());
        while !done {
            let env = ctx.at(&point);
            let mut kept = true;
            for constraint in &self.constraints {
                if !constraint.holds(&env)? {
                    kept = false;
                    break;
                }
            }
            let yields = if kept {
                args.clear();
                for (index, read) in self.reads.iter().enumerate() {
                    let coords = env.eval_all(&read.coords)?;
                    let operand = match &read.source {
                        Source::Operand(operand) => *operand,
                        Source::List { first, count, pick } => {
                            let picked = pick.eval(&env)?;
                            ensure!(
                                (0..*count as i64).contains(&picked),
                                "read {index} picks operand {picked} of {count}"
                            );
                            first + picked as usize
                        }
                    };
                    let value = match position(&coords, dims[operand]) {
                        Some(at) => decoded[operand].get(at),
                        None => match read.pad {
                            Some(pad) => pad_scalar(pad.0, decoded[operand].dtype())?,
                            None => bail!(
                                "read {index} at {coords:?} is outside operand {operand} of \
                                 shape {:?}",
                                dims[operand]
                            ),
                        },
                    };
                    args.push(value);
                }
                let mut runner = BodyRunner { module, env: &env };
                Some(
                    program
                        .run(&args, &mut runner)
                        .with_context(|| format!("at {point:?}"))?,
                )
            } else {
                None
            };
            for (part_index, part) in parts.iter().enumerate() {
                let first = part.results.start;
                if part.kind == CombineKind::None {
                    if let Some(yielded) = &yields {
                        let coords = env.eval_all(&writes[first].coords)?;
                        outs[first].set(&coords, yielded[part.yields.start], first)?;
                    }
                    continue;
                }
                if let Some(into) = writes[first].into {
                    // A fold into an operand starts from the element's value there.
                    let Some(yielded) = &yields else { continue };
                    let coords = env.eval_all(&writes[first].coords)?;
                    let current = outs[first]
                        .get(&coords, first)?
                        .ok_or_else(|| anyhow!("operand ${into} has no element at {coords:?}"))?;
                    let mut acc = Acc::new(&part.kind, &part.dtypes, &[current])?;
                    acc.push(&yielded[part.yields.clone()])?;
                    outs[first].set(&coords, acc.finish()?[0], first)?;
                    continue;
                }
                let key = key_of(part, &env)?;
                if !groups[part_index].contains_key(&key) {
                    let acc = new_acc(part, &env)?;
                    groups[part_index].insert(
                        key.clone(),
                        FoldGroup {
                            acc,
                            point: point.clone(),
                        },
                    );
                }
                let group = groups[part_index].get_mut(&key).expect("inserted above");
                if let Some(yielded) = &yields {
                    group.acc.push(&yielded[part.yields.clone()])?;
                }
                if prefix.is_some() {
                    write_results(&writes, part, &group.acc.finish()?, &env, &mut outs)?;
                }
            }
            done = advance(&mut point, &extents, &all);
        }

        // Finish the folds of the new results.
        if prefix.is_none() {
            for (part_index, part) in parts.iter().enumerate() {
                for group in groups[part_index].values() {
                    let env = ctx.at(&group.point);
                    write_results(&writes, part, &group.acc.finish()?, &env, &mut outs)?;
                }
            }
        }

        let tensors = outs
            .into_iter()
            .enumerate()
            .map(|(result, out)| {
                let mut elements = Vec::with_capacity(out.elements.len());
                for (element, value) in out.elements.into_iter().enumerate() {
                    elements.push(value.ok_or_else(|| {
                        anyhow!("no point writes element {element} of result {result}")
                    })?);
                }
                HostTensor::new(out.dims, scalar::encode(out.dtype, &elements)?)
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(BlockResults { outs: tensors })
    }
}

/// Writes the finished results of `part` at their coordinates at the point of `env`. A kind that
/// appends a dimension of extent `k`, `top_k` and `bottom_k`, gives `k` values per result, written
/// along it.
fn write_results(
    writes: &[&Write],
    part: &Part,
    finished: &[HostScalar],
    env: &Env<'_>,
    outs: &mut [Out],
) -> Result<()> {
    for (position, result) in part.results.clone().enumerate() {
        let base = env.eval_all(&writes[result].coords)?;
        let appended = part.kind.rule().extra_dims(position);
        let &[Dim::Static(count)] = &appended[..] else {
            outs[result].set(&base, finished[position], result)?;
            continue;
        };
        for rank in 0..count {
            let mut coords = base.clone();
            coords.push(rank as i64);
            outs[result].set(&coords, finished[position * count + rank], result)?;
        }
    }
    Ok(())
}

/// Steps `point` to the next point in lexicographic order over the iterators `which`, the last one
/// fastest, and returns true when it wraps around: every point has been visited.
fn advance(point: &mut [i64], extents: &[i64], which: &[usize]) -> bool {
    for &i in which.iter().rev() {
        point[i] += 1;
        if point[i] < extents[i] {
            return false;
        }
        point[i] = 0;
    }
    true
}

impl Eval for Index {
    fn eval(
        &self,
        _ins: IndexOperands<Host<'_>>,
        cx: &mut EvalCx<'_, NoRegions>,
    ) -> Result<IndexResults<OwnedHost>> {
        let env = cx
            .index_env()
            .ok_or_else(|| anyhow!("tile.index runs in the body of a tile.block, at its points"))?;
        let value = self.expr.eval(env)?;
        Ok(IndexResults {
            output: scalar::cast(HostScalar::int(DType::Si64, i128::from(value))?, self.dtype)?,
        })
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
