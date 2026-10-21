//! A block's body at one point: its reads, from the operand in place, a staged copy, or the
//! element the block before it wrote; its constraints; and its ops.

use std::collections::HashMap;

use anyhow::Result;
use smallvec::SmallVec;

use super::placed::Piece;
use super::scope::{Point, Scope};
use super::BlockLowering;
use crate::conversion::tile_to_loop::utils::{if_then, index, load, pick, predicate, prefetch};
use crate::dialects::arith;
use crate::dialects::tile::{Index, Source};
use crate::ir::{AccessExpr, Cmp, Constraint, IndexExpr, Level, OperandRef, Root, Value};
use crate::rewrite::Builder;
use crate::traits::Terminator;

/// The position of `value` in `operands`, which it joins when it is not there yet.
fn slot(operands: &mut Vec<Value>, value: Value) -> usize {
    match operands.iter().position(|&operand| operand == value) {
        Some(position) => position,
        None => {
            operands.push(value);
            operands.len() - 1
        }
    }
}

/// What the block before a block gives it at a point: nothing to the first block of a run; the
/// element of its result there to a placed block that runs at the point; and the piece that holds
/// its elements to one that reads across the piece.
#[derive(Clone, Copy)]
pub(super) enum Input<'v> {
    None,
    Element(Value),
    Piece(&'v Piece),
}

impl BlockLowering<'_> {
    /// Builds `body` where every constraint of the block holds at the point of `scope`: inside one
    /// `loop.if` per constraint, each checked once the ones before it hold.
    pub(super) fn guarded(
        &self,
        builder: &mut Builder,
        scope: &Scope,
        body: &mut dyn FnMut(&mut Builder) -> Result<()>,
    ) -> Result<()> {
        self.guarded_from(builder, 0, &scope.point, body)
    }

    fn guarded_from(
        &self,
        builder: &mut Builder,
        first: usize,
        point: &Point,
        body: &mut dyn FnMut(&mut Builder) -> Result<()>,
    ) -> Result<()> {
        let Some(constraint) = self.block.constraints.get(first) else {
            return body(builder);
        };
        let mut operands = Vec::new();
        let mut cmps = SmallVec::new();
        for cmp in &constraint.0 {
            let lhs = self.lower_expr(builder, &cmp.lhs, point, &mut operands);
            let rhs = self.lower_expr(builder, &cmp.rhs, point, &mut operands);
            cmps.push(Cmp::new(lhs, cmp.op, rhs));
        }
        let holds = predicate(builder, &[Constraint(cmps)], &operands);
        if_then(builder, holds, |then_builder| {
            self.guarded_from(then_builder, first + 1, point, body)
        })
    }

    /// Builds the reads and the body of the block at the point of `scope`, and gives what it
    /// yields. `input` is what the block before gives it there.
    pub(super) fn point(
        &self,
        builder: &mut Builder,
        scope: &Scope,
        input: Input,
    ) -> Result<Vec<Value>> {
        let module = self.module;
        let mut map: HashMap<Value, Value> = HashMap::new();
        for (read, &arg) in module.args(self.body).iter().enumerate() {
            map.insert(arg, self.read(builder, scope, read, input));
        }
        let terminator = module
            .last_inst(self.body)
            .filter(|&last| module.op(last).has_trait::<Terminator>())
            .expect("the verifier ends the body of a block with tile.yield");
        let block_location = builder.location().cloned();
        for inst in module.insts(self.body) {
            if inst == terminator {
                break;
            }
            let op = module.op(inst);
            // The ops built for a body op take its location.
            builder.set_origin(module, inst);
            if let Some(position) = op.get::<Index>() {
                let mut operands = Vec::new();
                let access = AccessExpr::from(position.expr.clone());
                let lowered = self.lower_expr(builder, &access, &scope.point, &mut operands);
                let at = index(builder, lowered, &operands);
                let value = builder.create(
                    arith::FromIndex {
                        dtype: position.dtype,
                    },
                    at,
                );
                map.insert(module.results(inst)[0], value);
                continue;
            }
            builder.create_mapped(
                op.clone(),
                module.operands(inst),
                module.operand_segments(inst).map(<[usize]>::to_vec),
                module.results(inst),
                &mut map,
            )?;
        }
        builder.set_location(block_location);
        let mut yielded = Vec::with_capacity(module.operands(terminator).len());
        for value in module.operands(terminator) {
            yielded.push(map[value]);
        }
        Ok(yielded)
    }

    /// The value that read `read` gives at the point of `scope`: what `input` gives of the
    /// operand that the block before writes, else from the operand's staged copy or in place.
    fn read(&self, builder: &mut Builder, scope: &Scope, read: usize, input: Input) -> Value {
        let block_read = &self.block.reads[read];
        let source = &block_read.source;
        let reads_input =
            matches!(source, Source::Operand(operand) if Some(*operand) == self.input);
        match (reads_input, input) {
            (true, Input::Element(element)) => return element,
            (true, Input::Piece(piece)) => {
                let coords = self.coords(builder, &block_read.coords, &scope.point);
                let at = piece.position(builder, &coords);
                return load(builder, piece.buffer, at, None);
            }
            _ => {}
        }
        if let Some(staged) = &scope.staged[source.first()] {
            let at = self.position_in_copy(builder, scope, staged);
            return load(builder, staged.buffer, at, None);
        }
        if let Some((at, level)) = self.prefetched(source.first()) {
            self.prefetch_next_tile(builder, scope, read, at, level);
        }
        self.read_in_place(builder, scope, read, &scope.point)
    }

    /// The loop of the nest and the cache that the reads of operand `operand` ask for the
    /// elements of the next tile, when a holding of its read places a prefetch. The walk's
    /// prefetches name operands of the run's first block, so the blocks placed after it ask for
    /// none.
    fn prefetched(&self, operand: usize) -> Option<(usize, Level)> {
        if self.input.is_some() {
            return None;
        }
        let prefetch = self
            .walk
            .prefetches
            .iter()
            .find(|entry| entry.operand == OperandRef(operand))?;
        Some((prefetch.at?, prefetch.level.clone()))
    }

    /// Asks the cache `level` for the element that read `read` takes at the same place of the
    /// next tile of loop `at` of the nest: at the point of `scope` one tile further along the
    /// iterator that loop walks. An element past the operand is asked for nothing.
    fn prefetch_next_tile(
        &self,
        builder: &mut Builder,
        scope: &Scope,
        read: usize,
        at: usize,
        level: Level,
    ) {
        let walked = &self.walk.nest[at];
        let position = walked.iter.position();
        let here = scope.point[position].expect("the body runs at a point of every iterator");
        let step = IndexExpr::iter(0) + IndexExpr::constant(walked.extent as i64);
        let mut next = scope.point;
        next[position] = Some(index(builder, step, &[here]));
        let block_read = &self.block.reads[read];
        let coords = self.coords(builder, &block_read.coords, &next);
        let buffer = self.source_buffer(builder, &block_read.source, &next);
        prefetch(builder, buffer, coords, level);
    }

    /// The value that read `read` gives at `point`, from the operand itself: without its pad
    /// where the loops keep it in range.
    pub(super) fn read_in_place(
        &self,
        builder: &mut Builder,
        scope: &Scope,
        read: usize,
        point: &Point,
    ) -> Value {
        let block_read = &self.block.reads[read];
        let coords = self.coords(builder, &block_read.coords, point);
        let pad = match scope.clips.kept(read) {
            true => None,
            false => block_read.pad,
        };
        let buffer = self.source_buffer(builder, &block_read.source, point);
        load(builder, buffer, coords, pad)
    }

    /// The buffer that `source` takes its elements from at `point`: the operand's, or that of the
    /// operand of the list that its choice names there.
    fn source_buffer(&self, builder: &mut Builder, source: &Source, point: &Point) -> Value {
        let (first, count, choice) = match source {
            Source::Operand(operand) => return self.operands[*operand].in_place(),
            Source::List { first, count, pick } => (*first, *count, pick),
        };
        let mut operands = Vec::new();
        let position = self.lower_expr(builder, choice, point, &mut operands);
        let picked = index(builder, position, &operands);
        let mut members = Vec::with_capacity(count);
        for offset in 0..count {
            members.push(self.operands[first + offset].in_place());
        }
        pick(builder, picked, members)
    }

    /// The index values of `exprs` at `point`.
    pub(super) fn coords(
        &self,
        builder: &mut Builder,
        exprs: &[AccessExpr],
        point: &Point,
    ) -> Vec<Value> {
        let mut operands = Vec::new();
        let mut lowered = Vec::with_capacity(exprs.len());
        for expr in exprs {
            lowered.push(self.lower_expr(builder, expr, point, &mut operands));
        }
        let mut values = Vec::with_capacity(lowered.len());
        for expr in lowered {
            values.push(index(builder, expr, &operands));
        }
        values
    }

    /// `expr` at `point` as an expression whose `di` is `operands[i]`: each iterator becomes its
    /// value, and each element it reads a load, which joins `operands` as an index.
    pub(super) fn lower_expr(
        &self,
        builder: &mut Builder,
        expr: &AccessExpr,
        point: &Point,
        operands: &mut Vec<Value>,
    ) -> IndexExpr {
        match expr.root() {
            Root::Const(value) => IndexExpr::constant(value),
            Root::Sym(sym) => IndexExpr::sym(sym),
            Root::Iter(position) => {
                let value = point[position].expect(
                    "the loops around an expression stand at a point of each iterator it takes",
                );
                IndexExpr::iter(slot(operands, value))
            }
            Root::Binary(op, lhs, rhs) => {
                let lowered_lhs = self.lower_expr(builder, &lhs, point, operands);
                let lowered_rhs = self.lower_expr(builder, &rhs, point, operands);
                IndexExpr::binary(op, lowered_lhs, lowered_rhs)
            }
            Root::Neg(operand) => -self.lower_expr(builder, &operand, point, operands),
            Root::Read(operand, coords) => {
                let mut values = Vec::with_capacity(coords.len());
                for coord in &coords {
                    let lowered = self.lower_expr(builder, coord, point, operands);
                    values.push(index(builder, lowered, operands));
                }
                let element = load(builder, self.operands[operand].in_place(), values, None);
                let value = builder.create(arith::ToIndex, element);
                IndexExpr::iter(slot(operands, value))
            }
        }
    }
}
