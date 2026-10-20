//! Reading and rewriting the attributes of `tile.block`.

use std::collections::BTreeSet;
use std::ops::Range;

use anyhow::{anyhow, bail, ensure, Result};
use smallvec::SmallVec;
use tinyvec::ArrayVec;

use crate::dialects::arith::{self, math};
use crate::dialects::func::Func;
use crate::dialects::tile::ir::*;
use crate::dialects::{isa, tensor};
use crate::host::HostScalar;
use crate::ir::{
    AccessExpr, DType, Dim, Inst, IterRef, Module, OperandRef, Region, Root, Sym, TensorType,
    Value, ValueRef,
};
use crate::rewrite::{self, RootInfo};
use crate::traits::Terminator;
use crate::Op;

/// Roots a pattern at each `tile.block` that is not scheduled ([`Block::scheduled`]) and heads
/// no run ([`heads_a_run`]): whose loops, clips, holdings and placement name none of its
/// iterators or modes by position, and whose iterators no placed block names. A rewrite that
/// renumbers them would leave those naming others. A pattern that renumbers them is rooted here,
/// not at `Block`.
pub struct UnscheduledBlock;

impl rewrite::Root for UnscheduledBlock {
    type OpRef<'m> = BlockOp<'m>;

    fn find(module: &Module, inst: Inst) -> Option<BlockOp<'_>> {
        let block = module.op_as::<Block>(inst)?;
        (!block.scheduled() && !heads_a_run(module, inst)).then_some(block)
    }

    fn info() -> RootInfo {
        <Block as rewrite::Root>::info()
    }
}

/// Whether a placed block reads a result of the block `inst`, which then heads a run: the placed
/// block's iterators are the first iterators of the run's first block, by position.
fn heads_a_run(module: &Module, inst: Inst) -> bool {
    for &result in module.results(inst) {
        for reader in module.uses(result) {
            let Some(block) = module.op(reader.inst).get::<Block>() else {
                continue;
            };
            if block.reads.iter().any(|read| read.at.is_some()) {
                return true;
            }
        }
    }
    false
}

impl Fold {
    /// Whether the fold is a `none` fold into new results: each point writes its element.
    pub fn maps_new(&self) -> bool {
        self.combine == CombineKind::None && self.writes[0].into.is_none()
    }
}

impl Block {
    /// The read `index` of this block, whose first result `read` reads, as a read of a block that
    /// takes this block's operands after its `offset` own. It reads at the point of this block that
    /// `read` reaches. There each iterator of this block is the coordinate of `read` at which the
    /// result's write takes it, or 0 where the write takes it at none. It keeps this read's pad.
    pub fn read_through(&self, index: usize, read: &Read, offset: usize) -> Read {
        let through = &self.reads[index];
        let mut point = vec![AccessExpr::constant(0); self.iters.len()];
        for (coord, expr) in self.folds[0].writes[0].coords.iter().zip(&read.coords) {
            if let Some(position) = coord.as_iter() {
                point[position] = expr.clone();
            }
        }
        let at_point = |position: usize| Some(point[position].clone());
        let moved = |expr: &AccessExpr| {
            expr.rename_operands(&|operand| operand + offset)
                .substitute(&at_point)
        };
        let mut coords = Coords::new();
        for coord in &through.coords {
            coords.push(moved(coord));
        }
        let source = match &through.source {
            Source::Operand(operand) => Source::Operand(operand + offset),
            Source::List { first, count, pick } => Source::List {
                first: first + offset,
                count: *count,
                pick: Box::new(moved(pick)),
            },
        };
        Read {
            source,
            coords,
            pad: through.pad,
            at: None,
            held: None,
        }
    }

    /// Whether the block's loops, clips or holdings name its iterators or modes by position, or a
    /// read places it in the loops of the first block of its run, whose iterators its own are by
    /// position: a rewrite that moves or removes them would leave those naming others.
    pub fn scheduled(&self) -> bool {
        !self.loops.is_empty()
            || !self.clip.is_empty()
            || self
                .reads
                .iter()
                .any(|read| read.held.is_some() || read.at.is_some())
    }

    /// The writes of the results, in result order.
    pub fn writes(&self) -> impl Iterator<Item = &Write> {
        self.folds.iter().flat_map(|fold| &fold.writes)
    }

    /// The writes of the results, in result order, to change.
    pub fn writes_mut(&mut self) -> impl Iterator<Item = &mut Write> {
        self.folds.iter_mut().flat_map(|fold| &mut fold.writes)
    }

    /// The number of results.
    pub fn result_count(&self) -> usize {
        self.writes().count()
    }

    /// The folds of the block that share one walk of its points, by their positions, in order:
    /// consecutive `none` folds into new results share one, which computes the body once at each
    /// point for all of them; every other fold walks the points alone, and computes the body
    /// again.
    pub fn fold_walks(&self) -> Vec<Range<usize>> {
        let mut walks: Vec<Range<usize>> = Vec::new();
        for (index, fold) in self.folds.iter().enumerate() {
            let joins = fold.maps_new()
                && walks
                    .last()
                    .is_some_and(|last| self.folds[last.start].maps_new());
            match (joins, walks.last_mut()) {
                (true, Some(last)) => last.end = index + 1,
                _ => walks.push(index..index + 1),
            }
        }
        walks
    }

    /// For each fold of the block, in order, the values it takes, by their positions among the
    /// values that the body yields, and the results it gives, by their positions among the
    /// block's results, as its combine kind says.
    pub fn fold_ranges(&self) -> Vec<(Range<usize>, Range<usize>)> {
        let mut ranges = Vec::with_capacity(self.folds.len());
        let (mut first_yield, mut first_result) = (0, 0);
        for fold in &self.folds {
            let rule = fold.combine.rule();
            let yields = first_yield..first_yield + rule.yields();
            let results = first_result..first_result + rule.results();
            first_yield = yields.end;
            first_result = results.end;
            ranges.push((yields, results));
        }
        ranges
    }

    /// The scalar types the body yields, from its terminator.
    pub fn yield_types(module: &Module, body: Region) -> Result<Vec<DType>> {
        let terminator = module
            .last_inst(body)
            .ok_or_else(|| anyhow!("the body is empty"))?;
        module
            .operand_types(terminator)
            .into_iter()
            .map(|ty| {
                ty.as_scalar()
                    .ok_or_else(|| anyhow!("the body yields scalars, got {ty}"))
            })
            .collect()
    }

    /// Every dimension symbol the block uses: in its extents, its expressions and its body's
    /// `tile.index` values.
    pub fn syms(&self, module: &Module, body: Region) -> BTreeSet<Sym> {
        let mut syms = BTreeSet::new();
        for iter in &self.iters {
            if let Dim::Dynamic(sym) = &iter.extent {
                syms.insert(sym.clone());
            }
        }
        for expr in self.exprs() {
            expr.collect_syms(&mut syms);
        }
        for inst in module.insts(body) {
            if let Some(index) = module.op(inst).get::<Index>() {
                index.expr.collect_syms(&mut syms);
            }
        }
        syms
    }

    /// Every index expression of the block: of its reads, its writes and its constraints.
    pub fn exprs(&self) -> Vec<&AccessExpr> {
        let mut exprs: Vec<&AccessExpr> = Vec::new();
        for read in &self.reads {
            exprs.extend(read.exprs());
        }
        for write in self.writes() {
            exprs.extend(&write.coords);
        }
        for constraint in &self.constraints {
            exprs.extend(constraint.exprs());
        }
        exprs
    }

    /// The extent of the iterator at `position`.
    pub fn extent(&self, position: usize) -> Option<Dim> {
        self.iters.get(position).map(|iter| iter.extent.clone())
    }

    /// The smallest and largest value of the iterator at `position`, when its extent is static.
    pub fn iter_range(&self, position: usize) -> Option<(i64, i64)> {
        match self.iters.get(position)?.extent {
            Dim::Static(extent) if extent > 0 => Some((0, extent as i64 - 1)),
            _ => None,
        }
    }

    /// The block with every index expression rewritten by `rewrite`: reads, writes and
    /// constraints.
    pub fn map_exprs(&self, rewrite: &dyn Fn(&AccessExpr) -> AccessExpr) -> Block {
        let mut block = self.clone();
        for (read, original) in block.reads.iter_mut().zip(&self.reads) {
            *read = original.map_exprs(rewrite);
        }
        for fold in &mut block.folds {
            for write in &mut fold.writes {
                for coord in &mut write.coords {
                    *coord = rewrite(coord);
                }
            }
        }
        for (constraint, original) in block.constraints.iter_mut().zip(&self.constraints) {
            *constraint = original.map_exprs(rewrite);
        }
        block
    }

    /// The block with operand `i` renamed to `rename(i)` everywhere it is named.
    pub fn rename_operands(&self, rename: &dyn Fn(usize) -> usize) -> Block {
        let mut block = self.map_exprs(&|expr| expr.rename_operands(rename));
        for read in &mut block.reads {
            read.source = read.source.rename_operands(rename);
        }
        for fold in &mut block.folds {
            for write in &mut fold.writes {
                write.into = write.into.map(|operand| OperandRef(rename(operand.0)));
            }
            for operand in &mut fold.init {
                operand.0 = rename(operand.0);
            }
        }
        block
    }

    /// The operands the block names: in its reads, index expressions, writes and init.
    pub fn named_operands(&self) -> BTreeSet<usize> {
        let mut named = self.read_operands();
        named.extend(
            self.writes()
                .filter_map(|write| write.into.map(|operand| operand.0)),
        );
        named
    }

    /// The operands whose elements the block reads: in its reads, its index expressions and the
    /// start states of its folds. An operand that a write only updates is not one of them.
    pub fn read_operands(&self) -> BTreeSet<usize> {
        let mut operands = BTreeSet::new();
        for expr in self.exprs() {
            expr.collect_operands(&mut operands);
        }
        for read in &self.reads {
            operands.extend(read.source.operands());
        }
        for fold in &self.folds {
            operands.extend(fold.init.iter().map(|operand| operand.0));
        }
        operands
    }

    /// What a point contributes to the block's one fold where read `read` falls outside its
    /// operand, when the body, given the read's pad, yields nothing the fold keeps, so that a loop
    /// may skip the point. None when the block folds into an operand or yields more than one
    /// value, or when the body yields anything else at the pad.
    pub fn pad_skip(&self, module: &Module, body: Region, read: usize) -> Option<PadSkip> {
        let pad = self.reads.get(read)?.pad?;
        let [fold] = self.folds.as_slice() else {
            return None;
        };
        if fold.writes.iter().any(|write| write.into.is_some()) {
            return None;
        }
        let &[value] = module.yielded(body) else {
            return None;
        };
        let arg = module.args(body)[read];
        let dtype = module.ty(value).as_scalar()?;
        if value == arg {
            let identity = match fold.combine {
                CombineKind::Sum => math::fold_init::<tensor::Add>(dtype),
                CombineKind::Max => math::fold_init::<tensor::Maximum>(dtype),
                CombineKind::Min => math::fold_init::<tensor::Minimum>(dtype),
                _ => return None,
            };
            let padded = HostScalar::from_f64(dtype, pad.0).ok()?;
            let same = identity.ok()?.to_data().ok()?.bytes() == padded.to_data().ok()?.bytes();
            return same.then_some(PadSkip::Exact);
        }
        if fold.combine != CombineKind::Sum || pad.0 != 0.0 {
            return None;
        }
        let product = module.value_ref(value).defining_op::<arith::Mul>()?;
        let takes_read = |factor: ValueRef<'_>| {
            factor.id() == arg
                || factor
                    .defining_op::<arith::Cast>()
                    .is_some_and(|cast| cast.input().id() == arg)
        };
        if !takes_read(product.lhs()) && !takes_read(product.rhs()) {
            return None;
        }
        match dtype.is_integer() {
            true => Some(PadSkip::Exact),
            false => Some(PadSkip::ZeroProducts),
        }
    }

    /// The reads of the block of `inst` that pad an operand where the pad adds nothing to its fold
    /// ([`Block::pad_skip`]): exactly, or as products of the pad 0 when `skip_zero_products`; each
    /// with the axes along which its coordinate can leave the operand.
    pub fn skipped_pads(
        &self,
        module: &Module,
        inst: Inst,
        skip_zero_products: bool,
    ) -> Vec<(usize, Vec<usize>)> {
        let body = module.regions(inst)[0];
        let operands = module.operands(inst);
        let mut skipped = Vec::new();
        for (index, read) in self.reads.iter().enumerate() {
            let Source::Operand(operand) = read.source else {
                continue;
            };
            let skips = match self.pad_skip(module, body, index) {
                Some(PadSkip::Exact) => true,
                Some(PadSkip::ZeroProducts) => skip_zero_products,
                None => false,
            };
            let ty = module.ty(operands[operand]);
            let Some(tensor) = ty.as_tensor().filter(|_| skips) else {
                continue;
            };
            let mut leaving = Vec::new();
            for (axis, coord) in read.coords.iter().enumerate() {
                let stays = tensor.dims[axis]
                    .as_static()
                    .is_some_and(|extent| self.stays_in(coord, extent));
                if !stays {
                    leaving.push(axis);
                }
            }
            skipped.push((index, leaving));
        }
        skipped
    }

    /// Whether `coord`, a coordinate of a read, stays inside an axis of `extent` elements at every
    /// point of the block.
    pub fn stays_in(&self, coord: &AccessExpr, extent: usize) -> bool {
        let range = |position: usize| {
            let points = self.iters.get(position)?.extent.as_static()? as i64;
            Some((0, points - 1))
        };
        coord
            .bounds(&range)
            .is_some_and(|(low, high)| low >= 0 && high < extent as i64)
    }

    /// The operand of the block of `after` that the one result of the block of `before` is.
    pub fn epilogue_input(module: &Module, before: Inst, after: Inst) -> Option<usize> {
        let &[result] = module.results(before) else {
            return None;
        };
        module
            .operands(after)
            .iter()
            .position(|&operand| operand == result)
    }

    /// Whether `next` can run on each element of the one result of this block as this block
    /// writes it, `next`'s operand `input` being that result: an epilogue. It runs on the elements
    /// of that result as [`Block::result_writes`] and [`Block::plain_reads_of`] say, writes each
    /// point where this block writes, has every parallel iterator of this block, and reads `input`
    /// only where this block writes it.
    pub fn takes_epilogue(&self, next: &Block, input: usize) -> bool {
        let (Ok((write, next_write)), Ok(reads)) =
            (self.result_writes(next), next.plain_reads_of(input))
        else {
            return false;
        };
        // Every parallel iterator of this block is one of `next`'s: each point of `next` is one
        // element.
        let parallel = self.iters.iter().filter(|iter| iter.kind == IterKind::Par);
        next_write.coords == write.coords
            && parallel.count() == next.iters.len()
            && reads.iter().all(|(_, read)| read.coords == write.coords)
    }

    /// Fails unless `next`, whose operand `input` is the one result of this block, can be placed
    /// after it, with the rule it breaks: run where this block finishes the elements of a piece of
    /// its run's loops that holds whole the iterators `whole` names, on each of its points whose
    /// reads take only those elements (see [`Placement`]).
    ///
    /// - `next` runs on the elements of that result as [`Block::result_writes`] and
    ///   [`Block::plain_reads_of`] say.
    /// - Each writes every point at an element of its own: each coordinate is one parallel
    ///   iterator or a constant, every parallel iterator in one.
    /// - Each read of `input` takes along each axis what this block writes there: the same
    ///   iterator; an iterator of this block that `next` lacks, only at a constant inside it and
    ///   when the piece holds it whole; or the same constant.
    pub fn placeable(
        &self,
        next: &Block,
        input: usize,
        whole: &dyn Fn(usize) -> bool,
    ) -> Result<()> {
        let (write, next_write) = self.result_writes(next)?;
        let reads = next.plain_reads_of(input)?;
        ensure!(
            self.writes_own_elements(write),
            "the block before it writes its points other than each at an element of its own"
        );
        ensure!(
            next.writes_own_elements(next_write),
            "it writes its points other than each at an element of its own"
        );
        let shared = next.iters.len();
        for (index, read) in reads {
            for (axis, (coord, written)) in read.coords.iter().zip(&write.coords).enumerate() {
                match (written.root(), coord.root()) {
                    (Root::Iter(position), _) if position < shared => ensure!(
                        coord == written,
                        "its read {index} takes axis {axis} of the result at {coord}, where the \
                         block before it writes {written}"
                    ),
                    (Root::Iter(position), Root::Const(value)) => {
                        let extent = self.iters[position].extent.as_static();
                        let inside =
                            extent.is_some_and(|points| (0..points as i64).contains(&value));
                        ensure!(
                            whole(position) && inside,
                            "its read {index} takes axis {axis} of the result at {value}, a \
                             constant along d{position}, which the piece where it is placed does \
                             not hold whole"
                        );
                    }
                    (Root::Const(written_value), Root::Const(value)) => ensure!(
                        written_value == value,
                        "its read {index} takes axis {axis} of the result at {value}, where the \
                         block before it writes {written_value}"
                    ),
                    _ => bail!(
                        "its read {index} takes axis {axis} of the result at {coord}, where the \
                         block before it writes {written}"
                    ),
                }
            }
        }
        Ok(())
    }

    /// The writes of the one results of this block and of `next`, when `next` can run on the
    /// elements of that result. Fails with the rule it breaks otherwise:
    ///
    /// - This block writes its one result as a new tensor, from no start state, one element per
    ///   point; `next` writes its one new result with no combine and has no constraints.
    /// - `next`'s iterators are the first of this block's, parallel in both, of the same extents.
    fn result_writes<'b>(&'b self, next: &'b Block) -> Result<(&'b Write, &'b Write)> {
        let write = self.result_write().ok_or_else(|| {
            anyhow!("the block before it gives other than one new result of one element per point")
        })?;
        let next_write = next.result_write().ok_or_else(|| {
            anyhow!("it gives other than one new result of one element per point")
        })?;
        let combine = &next.folds[0].combine;
        ensure!(
            *combine == CombineKind::None,
            "it folds its points with {}",
            combine.name()
        );
        ensure!(next.constraints.is_empty(), "it has constraints");
        ensure!(
            next.iters.len() <= self.iters.len(),
            "it has {} iterators, and the block before it {}",
            next.iters.len(),
            self.iters.len()
        );
        for (position, iter) in next.iters.iter().enumerate() {
            let own = &self.iters[position];
            ensure!(
                iter.kind == IterKind::Par
                    && own.kind == IterKind::Par
                    && iter.extent == own.extent,
                "its d{position} is not a parallel iterator of the same extent in both blocks"
            );
        }
        Ok((write, next_write))
    }

    /// The reads of operand `input`, the result of the block before this one, with their
    /// positions: reads of it alone, without a pad, at least one, and no index expression reads
    /// it. Fails with the rule it breaks otherwise.
    fn plain_reads_of(&self, input: usize) -> Result<Vec<(usize, &Read)>> {
        let mut reads = Vec::new();
        for (index, read) in self.reads.iter().enumerate() {
            if !read.source.operands().any(|named| named == input) {
                continue;
            }
            ensure!(
                read.source == Source::Operand(input) && read.pad.is_none(),
                "its read {index} takes the result of the block before it from a list or with a pad"
            );
            reads.push((index, read));
        }
        ensure!(
            !reads.is_empty(),
            "it does not read the result of the block before it"
        );
        let mut indexed = BTreeSet::new();
        for expr in self.exprs() {
            expr.collect_operands(&mut indexed);
        }
        ensure!(
            !indexed.contains(&input),
            "an index expression of it reads the result of the block before it"
        );
        Ok(reads)
    }

    /// Whether `next`, placed after this block, reads `input`, the one result of this block, only
    /// where this block writes at the same point: it runs at that point, on the element this
    /// block finishes there.
    pub fn read_at_point(&self, next: &Block, input: usize) -> bool {
        let Some(write) = self.result_write() else {
            return false;
        };
        next.reads
            .iter()
            .filter(|read| read.source.operands().any(|named| named == input))
            .all(|read| read.coords == write.coords)
    }

    /// The write of the one result of the block, when one fold writes it as a new tensor from no
    /// start state, one element per point.
    pub fn result_write(&self) -> Option<&Write> {
        let [fold] = self.folds.as_slice() else {
            return None;
        };
        let [write] = fold.writes.as_slice() else {
            return None;
        };
        let rule = fold.combine.rule();
        let one_element = rule.results() == 1 && rule.extra_dims(0).is_empty();
        (one_element && write.into.is_none() && fold.init.is_empty()).then_some(write)
    }

    /// Whether `write` puts each point of the block at an element of its own: each coordinate is
    /// one parallel iterator or a constant, and every parallel iterator is in one.
    fn writes_own_elements(&self, write: &Write) -> bool {
        let mut taken = BTreeSet::new();
        for coord in &write.coords {
            match coord.root() {
                Root::Const(_) => {}
                Root::Iter(position) if self.iters[position].kind == IterKind::Par => {
                    if !taken.insert(position) {
                        return false;
                    }
                }
                _ => return false,
            }
        }
        let parallel = self.iters.iter().filter(|iter| iter.kind == IterKind::Par);
        taken.len() == parallel.count()
    }

    /// The work of the vector code of the block of `inst` in a register tile whose lanes run along
    /// the iterator at `lane`: each `arith` op of its body as its lane-wise `isa` op, with the dtype
    /// of its lanes, and the dtypes that registers hold, of its reads, of those ops' results and of
    /// the block's results. An op computes once per vector, not in vectors, when it is a constant,
    /// an index that does not use the lanes, or computed from such values alone. Fails with an op
    /// that has no lane-wise form.
    pub fn vector_work(module: &Module, inst: Inst, lane: usize) -> Result<VectorWork> {
        let body = module.regions(inst)[0];
        let mut work = VectorWork::default();
        for &arg in module.args(body) {
            work.dtypes.extend(module.ty(arg).as_scalar());
        }
        let mut uniform = BTreeSet::new();
        for body_inst in module.insts(body) {
            let op = module.op(body_inst);
            if op.has_trait::<Terminator>() {
                continue;
            }
            let operands = module.operands(body_inst);
            let constant = op.get::<arith::Constant>().is_some();
            let other_iters = op
                .get::<Index>()
                .is_some_and(|index| !index.expr.uses_iter(lane));
            let from_uniform =
                !operands.is_empty() && operands.iter().all(|operand| uniform.contains(operand));
            if constant || other_iters || from_uniform {
                uniform.extend(module.results(body_inst).iter().copied());
                continue;
            }
            let lanewise =
                isa::lanewise(op).ok_or_else(|| anyhow!("{} has no lane-wise form", op.name()))?;
            // A select takes its mask first; its lanes are those of the values.
            let lanes_of = match op.get::<arith::Select>() {
                Some(_) => operands.get(1),
                None => operands.first(),
            };
            let dtype = lanes_of
                .and_then(|&value| module.ty(value).as_scalar())
                .ok_or_else(|| anyhow!("{} takes no scalar lanes", op.name()))?;
            work.ops.push((lanewise, dtype));
            for &result in module.results(body_inst) {
                work.dtypes.extend(module.ty(result).as_scalar());
            }
        }
        for &result in module.results(inst) {
            work.dtypes.extend(module.ty(result).dtype());
        }
        Ok(work)
    }

    /// The read of operand `operand`, or of the list that starts at it, when that is the only use
    /// of its operands in the block: one read takes them, and no index expression, write or start
    /// state names one.
    pub fn only_read_of(&self, operand: usize) -> Option<&Read> {
        let mut found: Option<&Read> = None;
        for read in &self.reads {
            if !read.source.operands().any(|named| named == operand) {
                continue;
            }
            if found.is_some() || read.source.first() != operand {
                return None;
            }
            found = Some(read);
        }
        let read = found?;
        let mut indexed = BTreeSet::new();
        for expr in self.exprs() {
            expr.collect_operands(&mut indexed);
        }
        for named in read.source.operands() {
            let others = self
                .reads
                .iter()
                .filter(|other| other.source.operands().contains(&named));
            let written = self
                .writes()
                .any(|write| write.into == Some(OperandRef(named)));
            let starts = self
                .folds
                .iter()
                .any(|fold| fold.init.contains(&OperandRef(named)));
            if others.count() > 1 || indexed.contains(&named) || written || starts {
                return None;
            }
        }
        Some(read)
    }

    /// The operands that a read takes from a run of operands: these keep their positions.
    pub fn listed_operands(&self) -> BTreeSet<usize> {
        let mut listed = BTreeSet::new();
        for read in &self.reads {
            if matches!(read.source, Source::List { .. }) {
                listed.extend(read.source.operands());
            }
        }
        listed
    }

    /// The block with operands `ins` compacted: operands that hold the same value merged into
    /// the first, and operands it does not name dropped. Operands of a run that a read picks from
    /// keep their places.
    pub fn compact(&self, ins: Vec<Value>) -> (Block, Vec<Value>) {
        let listed = self.listed_operands();
        let mut target = Vec::with_capacity(ins.len());
        for (index, value) in ins.iter().enumerate() {
            let first =
                (0..index).find(|&earlier| ins[earlier] == *value && !listed.contains(&earlier));
            target.push(match first {
                Some(earlier) if !listed.contains(&index) => earlier,
                _ => index,
            });
        }
        let merged = self.rename_operands(&|operand| target[operand]);
        let named = merged.named_operands();
        let kept: Vec<usize> = (0..ins.len())
            .filter(|operand| named.contains(operand) || listed.contains(operand))
            .collect();
        let position = |operand: usize| {
            kept.iter()
                .position(|&kept_operand| kept_operand == operand)
                .expect("a named operand stays")
        };
        let mut values = Vec::with_capacity(kept.len());
        for &operand in &kept {
            values.push(ins[operand]);
        }
        (merged.rename_operands(&position), values)
    }

    /// Whether the block writes one value per point of its domain: one plain fold of one new
    /// result, no constraints, and no read from a run of operands. A reader can compute such a
    /// block's elements itself.
    pub fn is_map(&self) -> bool {
        let [fold] = &self.folds[..] else {
            return false;
        };
        fold.combine == CombineKind::None
            && fold.writes.len() == 1
            && fold.writes[0].into.is_none()
            && self.constraints.is_empty()
            && self
                .reads
                .iter()
                .all(|read| matches!(read.source, Source::Operand(_)))
    }

    /// The parallel iterator, by position, that is the last coordinate of every write and no other
    /// coordinate of one: the iterator that the lanes of the block's vectors run along.
    pub fn lanes_iterator(&self) -> Option<usize> {
        let lane = self
            .folds
            .first()?
            .writes
            .first()?
            .coords
            .last()?
            .as_iter()?;
        if self.iters[lane].kind != IterKind::Par {
            return None;
        }
        for write in self.writes() {
            let (last, outer) = write.coords.split_last()?;
            if last.as_iter() != Some(lane) || outer.iter().any(|coord| coord.uses_iter(lane)) {
                return None;
            }
        }
        Some(lane)
    }
}

impl BlockOp<'_> {
    /// Whether the block only moves elements: a map with one read, whose body yields the value it
    /// reads.
    pub fn is_copy(&self) -> bool {
        let body = self.body();
        self.is_map()
            && self.reads.len() == 1
            && body.insts().count() == 1
            && body.yielded() == body.args()
    }
}

impl Fold {
    /// The fold of `combine` with the writes of its results, no start state, and its state where
    /// no holding puts it.
    pub fn new(combine: CombineKind, writes: Vec<Write>) -> Self {
        Self {
            combine,
            writes: writes.into(),
            init: SmallVec::new(),
            held: None,
        }
    }

    /// One yielded value, written at each point.
    pub fn plain(write: Write) -> Self {
        Self::new(CombineKind::None, vec![write])
    }

    /// Whether the fold updates an operand at elements that more than one point may write, so
    /// its points must write in their order: a write into an operand that combines with the
    /// element there, or whose coordinates are no translation of the points
    /// ([`Write::translates`]).
    pub fn updates_in_order(&self, iters: &Iters) -> bool {
        self.writes.iter().any(|write| {
            write.into.is_some() && (self.combine != CombineKind::None || !write.translates(iters))
        })
    }
}

impl Source {
    /// The operand that names the source: the operand itself, or the first of the list.
    pub fn first(&self) -> usize {
        match self {
            Source::Operand(operand) => *operand,
            Source::List { first, .. } => *first,
        }
    }

    /// The operands the read may take elements from.
    pub fn operands(&self) -> Range<usize> {
        match self {
            Source::Operand(operand) => *operand..*operand + 1,
            Source::List { first, count, .. } => *first..*first + *count,
        }
    }

    /// The source with operand `i` renamed to `rename(i)`; the pick keeps its expression.
    fn rename_operands(&self, rename: &dyn Fn(usize) -> usize) -> Source {
        match self {
            Source::Operand(operand) => Source::Operand(rename(*operand)),
            Source::List { first, count, pick } => Source::List {
                first: rename(*first),
                count: *count,
                pick: pick.clone(),
            },
        }
    }
}

/// The text form of a [`Source`]: one operand, `$2`, or a list of operands and the pick,
/// `$2..$4[$1[d0]]`.
#[derive(crate::Attr, Debug, Clone, PartialEq, Eq, Hash)]
#[attr(format = "$first (`..` $list^)?")]
pub(in crate::dialects::tile) struct SourceText {
    first: OperandRef,
    list: Option<OperandList>,
}

/// The last operand of a list and the pick, after `..`.
#[derive(crate::Attr, Debug, Clone, PartialEq, Eq, Hash)]
#[attr(format = "$last `[` $pick `]`")]
struct OperandList {
    last: OperandRef,
    pick: AccessExpr,
}

impl From<&Source> for SourceText {
    fn from(source: &Source) -> Self {
        match source {
            Source::Operand(operand) => SourceText {
                first: OperandRef(*operand),
                list: None,
            },
            Source::List { first, count, pick } => SourceText {
                first: OperandRef(*first),
                list: Some(OperandList {
                    last: OperandRef(first + count - 1),
                    pick: (**pick).clone(),
                }),
            },
        }
    }
}

impl TryFrom<SourceText> for Source {
    type Error = anyhow::Error;

    fn try_from(text: SourceText) -> Result<Self> {
        let first = text.first.0;
        let Some(list) = text.list else {
            return Ok(Source::Operand(first));
        };
        let last = list.last.0;
        ensure!(last >= first, "operand list ${first}..${last} is empty");
        Ok(Source::List {
            first,
            count: last - first + 1,
            pick: Box::new(list.pick),
        })
    }
}

/// The coordinates `coords`, in the form of an access.
fn coords_of(coords: impl IntoIterator<Item = impl Into<AccessExpr>>) -> Coords {
    coords.into_iter().map(Into::into).collect()
}

impl Read {
    /// A read of operand `operand` at `coords`, without padding.
    pub fn new(operand: usize, coords: impl IntoIterator<Item = impl Into<AccessExpr>>) -> Self {
        Self {
            source: Source::Operand(operand),
            coords: coords_of(coords),
            pad: None,
            at: None,
            held: None,
        }
    }

    /// Whether the read takes the iterator at `position` only in its last coordinate, with step
    /// 1, or not at all: consecutive points along it read consecutive elements, or one.
    pub fn contiguous_along(&self, position: usize) -> bool {
        let Some((last, outer)) = self.coords.split_last() else {
            return true;
        };
        !outer.iter().any(|coord| coord.uses_iter(position))
            && matches!(last.coefficient(position), Some(0 | 1))
    }

    /// The bytes of the elements that the read, of an operand of `ty`, takes over `points` of
    /// each iterator: along each axis, the span that its coordinate moves over, at most the axis,
    /// or the whole axis where the coordinate moves other than by a constant along an iterator.
    /// Infinite for an operand of dynamic extents.
    pub fn bytes_over(&self, ty: &TensorType, points: &[usize]) -> f64 {
        let Some(dims) = ty.static_dims() else {
            return f64::INFINITY;
        };
        let mut elements = 1.0;
        for (axis, coord) in self.coords.iter().enumerate() {
            let mut span = Some(1_i64);
            for position in coord.iters() {
                let coefficient = coord.coefficient(position);
                span = span
                    .zip(coefficient)
                    .map(|(spanned, step)| spanned + step.abs() * (points[position] as i64 - 1));
            }
            let axis_extent = dims[axis] as i64;
            elements *= span.map_or(axis_extent, |spanned| spanned.min(axis_extent)) as f64;
        }
        elements * ty.dtype.bits() as f64 / 8.0
    }

    /// Whether each coordinate of the read moves with the iterator at `position` by a constant
    /// multiple of it, or not at all, as a register tile's points need.
    pub fn moves_by_multiples(&self, position: usize) -> bool {
        self.coords
            .iter()
            .all(|coord| coord.coefficient(position).is_some())
    }

    /// The read with every expression rewritten by `rewrite`.
    pub fn map_exprs(&self, rewrite: &dyn Fn(&AccessExpr) -> AccessExpr) -> Read {
        Read {
            source: match &self.source {
                Source::Operand(operand) => Source::Operand(*operand),
                Source::List { first, count, pick } => Source::List {
                    first: *first,
                    count: *count,
                    pick: Box::new(rewrite(pick)),
                },
            },
            coords: self.coords.iter().map(rewrite).collect(),
            pad: self.pad,
            at: self.at,
            held: self.held.clone(),
        }
    }

    /// The expressions of the read: its coordinates and its pick.
    pub fn exprs(&self) -> Vec<&AccessExpr> {
        let mut out: Vec<&AccessExpr> = self.coords.iter().collect();
        if let Source::List { pick, .. } = &self.source {
            out.push(pick);
        }
        out
    }
}

impl Write {
    /// A write of a new result at `coords`.
    pub fn new(coords: impl IntoIterator<Item = impl Into<AccessExpr>>) -> Self {
        Self {
            into: None,
            coords: coords_of(coords),
        }
    }

    /// A write of an updated copy of operand `operand`, at `coords`.
    pub fn update(operand: usize, coords: impl IntoIterator<Item = impl Into<AccessExpr>>) -> Self {
        Self {
            into: Some(OperandRef(operand)),
            coords: coords_of(coords),
        }
    }

    /// Whether the write puts each point of a block with the iterators `iters` at an element of
    /// its own: each coordinate is one iterator plus an offset that no iterator moves, and every
    /// iterator of more than one point is in exactly one coordinate.
    pub fn translates(&self, iters: &Iters) -> bool {
        let mut used = Vec::new();
        for coord in &self.coords {
            let positions = coord.iters();
            if positions.len() > 1 {
                return false;
            }
            let Some(&position) = positions.first() else {
                continue;
            };
            if coord.coefficient(position) != Some(1) || used.contains(&position) {
                return false;
            }
            used.push(position);
        }
        iters
            .iter()
            .enumerate()
            .all(|(position, iter)| iter.extent.is_one() || used.contains(&position))
    }
}

/// What the vector code of a block computes: each lane-wise `isa` op with the dtype of its lanes,
/// and the dtypes its registers hold ([`Block::vector_work`]).
#[derive(Debug, Clone, Default)]
pub struct VectorWork {
    pub ops: Vec<(Op, DType)>,
    pub dtypes: Vec<DType>,
}

/// What a point contributes to the fold of its block where a padded read falls outside its
/// operand, when it contributes nothing the fold keeps.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PadSkip {
    /// Nothing: the body yields the pad, which is the fold's identity bit for bit, as `-inf` is
    /// `max`'s.
    Exact,
    /// Products of the pad 0 with the body's other values, into a float `sum`: nothing where those
    /// are finite, but for the sign of a sum that is zero.
    ZeroProducts,
}

impl IterList {
    /// The iterators at `positions`.
    #[track_caller]
    pub fn new(positions: impl IntoIterator<Item = usize>) -> Self {
        let mut list = ArrayVec::new();
        for position in positions {
            list.push(IterRef::new(position));
        }
        Self(list)
    }

    /// The positions, in order.
    pub fn positions(&self) -> impl Iterator<Item = usize> + '_ {
        self.0.iter().map(|iter| iter.position())
    }

    pub fn contains(&self, position: usize) -> bool {
        self.positions().any(|listed| listed == position)
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl IterKind {
    /// Whether the combine folds the iterator.
    pub fn is_reduced(self) -> bool {
        matches!(self, IterKind::Red | IterKind::RedOrdered)
    }
}

impl Iter {
    pub fn new(kind: IterKind, extent: Dim) -> Self {
        Self { kind, extent }
    }

    pub fn par(extent: Dim) -> Self {
        Self::new(IterKind::Par, extent)
    }

    pub fn red(extent: Dim) -> Self {
        Self::new(IterKind::Red, extent)
    }
}

/// Whether `op` is in a function, in no `tile.group` or `tile.block`: an op that `form-groups`
/// groups and `seal-dispatches` outlines.
pub(crate) fn in_function_outside_groups(module: &Module, op: Inst) -> bool {
    let mut region = module.parent_region(op);
    while let Some(current) = region {
        let Some(parent) = module.parent_op(current) else {
            return false;
        };
        if module.op_as::<Group>(parent).is_some() || module.op_as::<Block>(parent).is_some() {
            return false;
        }
        if module.op_as::<Func>(parent).is_some() {
            return true;
        }
        region = module.parent_region(parent);
    }
    false
}
