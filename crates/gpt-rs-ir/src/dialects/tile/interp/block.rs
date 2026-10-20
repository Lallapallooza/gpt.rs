//! Semantics of `tile.block`, `tile.index` and `tile.yield`. A `tile.block` compiles its body
//! once into a scalar program, then runs it at every point of its domain in lexicographic order.

use std::collections::HashMap;

use anyhow::{anyhow, bail, ensure, Context, Result};

use super::combine::Acc;
use crate::dialects::arith::math;
use crate::dialects::tile::ir::*;
use crate::dialects::tile::{IterKind, Source, Write};
use crate::host::{self, Decoded, HostScalar, HostTensor};
use crate::interp::HostValue;
use crate::ir::{
    AccessExpr, DType, Dim, Eval, EvalCx, Host, IndexEnv, Module, NoRegions, OpDyn, OperandRef,
    OwnedHost, Region, RegionRunner, Sym, Type, Value,
};
use crate::traits::Terminator;
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
    fn run(&self, args: &[HostScalar], runner: &mut BodyRunner) -> Result<Vec<HostScalar>> {
        let mut values: Vec<HostScalar> = Vec::with_capacity(self.args + self.instrs.len());
        values.extend_from_slice(args);
        for instr in &self.instrs {
            let mut operand_values = Vec::with_capacity(instr.operands.len());
            for &slot in &instr.operands {
                operand_values.push(HostValue::Scalar(values[slot]));
            }
            let ins: Vec<&HostValue> = operand_values.iter().collect();
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

/// What every point of a block shares, from which [`Ctx::at`] makes the view at one point: the
/// module, the symbol extents, the operands, the writes of the results and the position of the
/// prefix iterator.
struct Ctx<'a> {
    module: &'a Module,
    syms: &'a HashMap<Sym, i64>,
    operands: &'a [Decoded],
    dims: &'a [&'a [usize]],
    writes: &'a [&'a Write],
    prefix: Option<usize>,
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

    /// The key of a fold at the point of `env`: the coordinates of its key result, without the
    /// prefix coordinate, which each prefix result names.
    fn key_of(&self, part: &Part, env: &Env) -> Result<Vec<i64>> {
        let write = self.writes[part.results.start + (part.kind.rule().results() - 1)];
        let mut coords = Vec::with_capacity(write.coords.len());
        for coord in &write.coords {
            if self.prefix.is_some() && coord.as_iter() == self.prefix {
                continue;
            }
            coords.push(coord.eval(env)?);
        }
        Ok(coords)
    }

    /// A new accumulator of `part` at the point of `env`. The state of each component starts as
    /// the init operand at the element of the result of the same position.
    fn new_acc(&self, part: &Part, env: &Env) -> Result<Acc> {
        let mut init = Vec::with_capacity(part.init.len());
        for (component, &operand) in part.init.iter().enumerate() {
            let coords = env.eval_all(&self.writes[part.results.start + component].coords)?;
            let at = position(&coords, self.dims[operand])
                .ok_or_else(|| anyhow!("init ${operand} at {coords:?} is out of range"))?;
            init.push(self.operands[operand].get(at));
        }
        Acc::new(&part.kind, &part.dtypes, &init)
    }

    /// The fold of `part` that the point of `env` keys in `groups`, made there at that point
    /// when no point before keyed it.
    fn fold_group<'g>(
        &self,
        groups: &'g mut HashMap<Vec<i64>, FoldGroup>,
        part: &Part,
        env: &Env,
    ) -> Result<&'g mut FoldGroup> {
        let key = self.key_of(part, env)?;
        if !groups.contains_key(&key) {
            let acc = self.new_acc(part, env)?;
            groups.insert(
                key.clone(),
                FoldGroup {
                    acc,
                    point: env.point.to_vec(),
                },
            );
        }
        Ok(groups.get_mut(&key).expect("inserted above"))
    }

    /// The value that `read`, read `index` of the block, reads at the point of `env`: the
    /// element of its operand at its coordinates, or its pad outside the operand.
    fn read_value(&self, read: &Read, index: usize, env: &Env) -> Result<HostScalar> {
        let coords = env.eval_all(&read.coords)?;
        let operand = match &read.source {
            Source::Operand(operand) => *operand,
            Source::List { first, count, pick } => {
                let picked = pick.eval(env)?;
                ensure!(
                    (0..*count as i64).contains(&picked),
                    "read {index} picks operand {picked} of {count}"
                );
                first + picked as usize
            }
        };
        let value = match position(&coords, self.dims[operand]) {
            Some(at) => self.operands[operand].get(at),
            None => match read.pad {
                Some(pad) => HostScalar::from_f64(self.operands[operand].dtype(), pad.0)?,
                None => bail!(
                    "read {index} at {coords:?} is outside operand {operand} of shape {:?}",
                    self.dims[operand]
                ),
            },
        };
        Ok(value)
    }
}

impl Env<'_> {
    fn eval_all(&self, exprs: &[AccessExpr]) -> Result<Vec<i64>> {
        exprs.iter().map(|expr| expr.eval(self)).collect()
    }
}

/// The extent of each symbol the block uses, in this run.
fn bind_syms(block: &Block, cx: &EvalCx<BlockRegions<Region>>) -> Result<HashMap<Sym, i64>> {
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

/// The storage of one result while the block runs.
struct Out {
    dims: Vec<usize>,
    dtype: DType,
    elements: Vec<Option<HostScalar>>,
}

impl Out {
    /// The storage of result `result`, written by `write`, which starts as a copy of its `into`
    /// operand.
    fn new(write: &Write, result_type: &Type, result: usize, operands: &[Decoded]) -> Result<Out> {
        let Type::Tensor(tensor_type) = result_type else {
            bail!("result {result} is not a tensor");
        };
        let dims = tensor_type
            .static_dims()
            .ok_or_else(|| anyhow!("result {result} has unbound extents"))?
            .to_vec();
        let count = dims.iter().product();
        let elements = match write.into {
            Some(OperandRef(operand)) => (0..count)
                .map(|element| Some(operands[operand].get(element)))
                .collect(),
            None => vec![None; count],
        };
        Ok(Out {
            dims,
            dtype: tensor_type.dtype,
            elements,
        })
    }

    /// The tensor of result `result`, once every point that writes it has run.
    fn finish(self, result: usize) -> Result<HostTensor> {
        let mut elements = Vec::with_capacity(self.elements.len());
        for (element, value) in self.elements.into_iter().enumerate() {
            elements.push(
                value.ok_or_else(|| {
                    anyhow!("no point writes element {element} of result {result}")
                })?,
            );
        }
        HostTensor::new(self.dims, host::encode(self.dtype, &elements)?)
    }

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
        ins: BlockOperands<Host>,
        cx: &mut EvalCx<BlockRegions<Region>>,
    ) -> Result<BlockResults<OwnedHost>> {
        let module = cx.module();
        let program = Program::compile(module, cx.regions.body)?;
        let yield_types = Block::yield_types(module, cx.regions.body)?;
        let syms = bind_syms(self, cx)?;
        let extents = iter_extents(self, &syms);
        let decoded: Vec<Decoded> = ins
            .ins
            .iter()
            .map(|tensor| Decoded::new(&tensor.data))
            .collect();
        let dims: Vec<&[usize]> = ins.ins.iter().map(|tensor| &tensor.dims[..]).collect();

        // The results, with `into` results starting as copies of their operands.
        let writes: Vec<&Write> = self.writes().collect();
        let mut outs = Vec::new();
        for (result, (write, result_type)) in writes.iter().zip(cx.result_types()).enumerate() {
            outs.push(Out::new(write, result_type, result, &decoded)?);
        }

        let parts = fold_parts(self, &yield_types);
        let ctx = Ctx {
            module,
            syms: &syms,
            operands: &decoded,
            dims: &dims,
            writes: &writes,
            prefix: self
                .iters
                .iter()
                .position(|iter| iter.kind == IterKind::Prefix),
        };
        // One accumulator map per folding part, keyed by the coordinates its key result writes.
        let mut groups: Vec<HashMap<Vec<i64>, FoldGroup>> =
            parts.iter().map(|_| HashMap::new()).collect();
        seed_folds(self, &ctx, &parts, &extents, &mut groups)?;

        // The points, in lexicographic order.
        let all: Vec<usize> = (0..self.iters.len()).collect();
        let mut point = vec![0i64; self.iters.len()];
        let mut done = extents.contains(&0);
        let mut args = Vec::with_capacity(self.reads.len());
        while !done {
            let env = ctx.at(&point);
            let yields = yields_at(self, &program, &ctx, &env, &mut args)?;
            combine_at(
                &ctx,
                &parts,
                &mut groups,
                yields.as_deref(),
                &env,
                &mut outs,
            )?;
            done = advance(&mut point, &extents, &all);
        }

        // Finish the folds of the new results.
        if ctx.prefix.is_none() {
            for (part, part_groups) in parts.iter().zip(&groups) {
                for group in part_groups.values() {
                    let env = ctx.at(&group.point);
                    write_results(&writes, part, &group.acc.finish()?, &env, &mut outs)?;
                }
            }
        }

        let mut tensors = Vec::with_capacity(outs.len());
        for (result, out) in outs.into_iter().enumerate() {
            tensors.push(out.finish(result)?);
        }
        Ok(BlockResults { outs: tensors })
    }
}

/// The extent of each iterator of `block`, in this run.
fn iter_extents(block: &Block, syms: &HashMap<Sym, i64>) -> Vec<i64> {
    let mut extents = Vec::with_capacity(block.iters.len());
    for iter in block.iters.iter() {
        extents.push(match &iter.extent {
            Dim::Static(extent) => *extent as i64,
            Dim::Dynamic(sym) => syms[sym],
        });
    }
    extents
}

/// The combine kinds of `block`, in order, with the yields and the results each takes.
fn fold_parts(block: &Block, yield_types: &[DType]) -> Vec<Part> {
    let mut parts = Vec::new();
    for (fold, (yields, results)) in block.folds.iter().zip(block.fold_ranges()) {
        parts.push(Part {
            dtypes: yield_types[yields.clone()].to_vec(),
            yields,
            results,
            init: fold.init.iter().map(|operand| operand.0).collect(),
            kind: fold.combine.clone(),
        });
    }
    parts
}

/// Gives every element of a new folded result a fold in `groups`, even one whose points are all
/// masked or whose reduction is empty: walks the parallel iterators once.
fn seed_folds(
    block: &Block,
    ctx: &Ctx,
    parts: &[Part],
    extents: &[i64],
    groups: &mut [HashMap<Vec<i64>, FoldGroup>],
) -> Result<()> {
    let mut parallel = Vec::new();
    for (position, iter) in block.iters.iter().enumerate() {
        if !iter.kind.is_reduced() && Some(position) != ctx.prefix {
            parallel.push(position);
        }
    }
    let mut folding = Vec::new();
    for (part_index, part) in parts.iter().enumerate() {
        if part.kind != CombineKind::None
            && ctx.writes[part.results.start].into.is_none()
            && ctx.prefix.is_none()
        {
            folding.push(part_index);
        }
    }
    if folding.is_empty() {
        return Ok(());
    }
    let mut point = vec![0i64; block.iters.len()];
    let mut done = parallel.iter().any(|&position| extents[position] == 0);
    while !done {
        let env = ctx.at(&point);
        for &part_index in &folding {
            ctx.fold_group(&mut groups[part_index], &parts[part_index], &env)?;
        }
        done = advance(&mut point, extents, &parallel);
    }
    Ok(())
}

/// What the body of `block`, compiled to `program`, yields at the point of `env`, or `None` when
/// a constraint masks the point. `args` holds the values of the reads.
fn yields_at(
    block: &Block,
    program: &Program,
    ctx: &Ctx,
    env: &Env,
    args: &mut Vec<HostScalar>,
) -> Result<Option<Vec<HostScalar>>> {
    for constraint in &block.constraints {
        if !constraint.holds(env)? {
            return Ok(None);
        }
    }
    args.clear();
    for (index, read) in block.reads.iter().enumerate() {
        args.push(ctx.read_value(read, index, env)?);
    }
    let mut runner = BodyRunner {
        module: ctx.module,
        env,
    };
    let yielded = program
        .run(args, &mut runner)
        .with_context(|| format!("at {:?}", env.point))?;
    Ok(Some(yielded))
}

/// Combines what the body yields at the point of `env`, `None` at a masked point, into the
/// results of each part, or into its fold in `groups`.
fn combine_at(
    ctx: &Ctx,
    parts: &[Part],
    groups: &mut [HashMap<Vec<i64>, FoldGroup>],
    yields: Option<&[HostScalar]>,
    env: &Env,
    outs: &mut [Out],
) -> Result<()> {
    for (part, part_groups) in parts.iter().zip(groups.iter_mut()) {
        let first = part.results.start;
        if part.kind == CombineKind::None {
            if let Some(yielded) = yields {
                let coords = env.eval_all(&ctx.writes[first].coords)?;
                outs[first].set(&coords, yielded[part.yields.start], first)?;
            }
            continue;
        }
        if let Some(into) = ctx.writes[first].into {
            // A fold into an operand starts from the element's value there.
            let Some(yielded) = yields else { continue };
            let coords = env.eval_all(&ctx.writes[first].coords)?;
            let current = outs[first]
                .get(&coords, first)?
                .ok_or_else(|| anyhow!("operand ${into} has no element at {coords:?}"))?;
            let mut acc = Acc::new(&part.kind, &part.dtypes, &[current])?;
            acc.push(&yielded[part.yields.clone()])?;
            outs[first].set(&coords, acc.finish()?[0], first)?;
            continue;
        }
        let group = ctx.fold_group(part_groups, part, env)?;
        if let Some(yielded) = yields {
            group.acc.push(&yielded[part.yields.clone()])?;
        }
        if ctx.prefix.is_some() {
            write_results(ctx.writes, part, &group.acc.finish()?, env, outs)?;
        }
    }
    Ok(())
}

/// Writes the finished results of `part` at their coordinates at the point of `env`. A kind that
/// appends a dimension of extent `k`, `top_k` and `bottom_k`, gives `k` values per result, written
/// along it.
fn write_results(
    writes: &[&Write],
    part: &Part,
    finished: &[HostScalar],
    env: &Env,
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
    for &iter_index in which.iter().rev() {
        point[iter_index] += 1;
        if point[iter_index] < extents[iter_index] {
            return false;
        }
        point[iter_index] = 0;
    }
    true
}

impl Eval for Index {
    fn eval(
        &self,
        _ins: IndexOperands<Host>,
        cx: &mut EvalCx<NoRegions>,
    ) -> Result<IndexResults<OwnedHost>> {
        let env = cx
            .index_env()
            .ok_or_else(|| anyhow!("tile.index runs in the body of a tile.block, at its points"))?;
        Ok(IndexResults {
            output: math::from_index(self.expr.eval(env)?, self.dtype)?,
        })
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
