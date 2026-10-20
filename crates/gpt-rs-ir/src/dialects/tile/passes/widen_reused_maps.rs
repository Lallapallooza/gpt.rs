//! Widening a map's result once where its readers widen each element at several points.

use std::collections::HashMap;

use anyhow::Result;

use crate::dialects::arith;
use crate::dialects::tile::{copy_body_ops, Block, BlockOp, Read, Source, UnscheduledBlock};
use crate::ir::{DType, Value, ValueRef};
use crate::passes::{pass, run_patterns, Pass};
use crate::require;
use crate::rewrite::{pattern, NoMatch, Pattern, PatternSet, Rewriter, Rewrites, Rewriting};

/// Computes the result of a map wide where the blocks that read it only widen what they read, one
/// of them at several points for each element. The map then widens each element once, where its
/// readers widened it at each read, as the projections that read one normalized activation in
/// bf16 do. The values are the same, but the map's result takes more bytes, so it is a pass rather
/// than a canonical form. A pipeline runs it when its target steps the readers' folds in the wide
/// dtype. On a target whose step takes the narrow dtype as it is, such as a dot product of bf16
/// pairs, the readers read the result narrow.
///
/// - The map runs at each call: a value known before the run that grows past the parameters it
///   reads is no value `outline-static` stores, as a weight is not copied larger.
/// - The map writes each point of its iterators at an element of its own.
/// - Each reader reads it in one read, without a pad, holding or placement, and its body only
///   casts what the read gives, all readers to one float wider than the result's; the
///   coordinates of one reader's read leave out an iterator of more than one point.
#[pass(
    name = "widen-reused-maps",
    stats(widened = "map results widened once for readers that widen them")
)]
pub struct WidenReusedMaps;

fn populate_widen_reused_maps_patterns(set: &mut PatternSet) {
    set.add(WidenInMap);
}

impl Pass for WidenReusedMaps {
    fn rewrites(&self) -> Rewrites {
        Rewrites::patterns(populate_widen_reused_maps_patterns)
    }

    fn run(&self, module: &mut Rewriting) -> Result<bool> {
        run_patterns(module, self.rewrites(), &Self::WIDENED)
    }
}

/// A map whose readers each cast every value they read of it to one wider float, one of them at
/// several points for each element: the map yields its value cast to that float, and its readers
/// read it without the casts.
#[pattern]
struct WidenInMap;

impl Pattern for WidenInMap {
    type Root = UnscheduledBlock;

    fn rewrite(&self, map: BlockOp, rewriter: &mut Rewriter) -> Result<(), NoMatch> {
        let module = map.module();
        require!(
            map.is_map() && map.outs().len() == 1,
            "the block is no map of one result"
        );
        require!(
            map.folds[0].writes[0].translates(&map.iters),
            "the map writes no element of its own for each point"
        );
        let result = map.outs()[0];
        require!(
            !result.known_before_run(),
            "the map's result is known before the run"
        );
        let readers = widening_readers(&map)?;
        let wide = widened_to(&map, &readers)?;
        let map_body = map.body();
        let map_ins: Vec<Value> = map.ins().iter().map(|input| input.id()).collect();
        let wide_results = map.attrs().clone().build(rewriter, map_ins, |body, args| {
            let mut values = HashMap::new();
            for (arg, &value) in map_body.args().into_iter().zip(args) {
                values.insert(arg.id(), value);
            }
            let yielded = copy_body_ops(module, map_body.id(), body.builder, &mut values);
            vec![body.cast(yielded[0], wide)]
        });
        for (reader, index) in &readers {
            read_wide(reader, *index, (result.id(), wide_results[0]), rewriter);
        }
        rewriter.erase(map.inst());
        Ok(())
    }
}

/// The blocks that read the result of `map`, each with its read of it: each an unscheduled block
/// that reads it once and only widens it ([`widened_read`]).
fn widening_readers<'m>(map: &BlockOp<'m>) -> Result<Vec<(BlockOp<'m>, usize)>, NoMatch> {
    let module = map.module();
    let result = map.outs()[0];
    let mut readers: Vec<(BlockOp, usize)> = Vec::new();
    for use_site in result.uses() {
        require!(
            let Some(reader) = module.op_as::<Block>(use_site.inst).filter(|block| !block.scheduled()),
            "an op other than an unscheduled block reads the map's result"
        );
        require!(
            !readers
                .iter()
                .any(|(earlier, _)| earlier.inst() == reader.inst()),
            "a block reads the map's result twice"
        );
        require!(
            let Some(index) = widened_read(&reader, result.id()),
            "a reader takes the map's result other than in one read that it only widens"
        );
        readers.push((reader, index));
    }
    Ok(readers)
}

/// The float wider than the result of `map` that all of `readers` widen it to, when one of them
/// reads an element of it at several points.
fn widened_to(map: &BlockOp, readers: &[(BlockOp, usize)]) -> Result<DType, NoMatch> {
    let mut wides = Vec::new();
    let mut repeats = false;
    for (reader, index) in readers {
        let arg = reader.body().args()[*index];
        for cast in widening_casts(arg) {
            wides.push(cast.to);
        }
        repeats |= reads_repeatedly(reader, &reader.reads[*index]);
    }
    let narrow = map.outs()[0].tensor_type().dtype;
    require!(let Some(&wide) = wides.first(), "the map has no reader");
    require!(
        wides.iter().all(|&to| to == wide) && wide.is_float() && narrow.is_float(),
        "the readers widen the map's result to different dtypes or to no float"
    );
    require!(
        wide.bits() > narrow.bits(),
        "the readers cast the map's result to no wider float"
    );
    require!(
        repeats,
        "no reader reads an element of the map's result at several points"
    );
    Ok(wide)
}

/// Replaces `reader`, whose read `index` takes `narrow` and only widens it, by the block that reads
/// `wide` there, the widened values, without the casts.
fn read_wide(
    reader: &BlockOp,
    index: usize,
    (narrow, wide): (Value, Value),
    rewriter: &mut Rewriter,
) {
    let module = reader.module();
    let mut ins: Vec<Value> = reader.ins().iter().map(|input| input.id()).collect();
    for input in ins.iter_mut() {
        if *input == narrow {
            *input = wide;
        }
    }
    let reader_body = reader.body();
    let widened_arg = reader_body.args()[index];
    rewriter.set_insertion_point(reader.inst());
    let results = reader.attrs().clone().build(rewriter, ins, |body, args| {
        let mut values = HashMap::new();
        for (arg, &value) in reader_body.args().into_iter().zip(args) {
            values.insert(arg.id(), value);
        }
        for cast in widening_casts(widened_arg) {
            values.insert(cast.output().id(), args[index]);
        }
        copy_body_ops(module, reader_body.id(), body.builder, &mut values)
    });
    rewriter.replace_op(reader.inst(), results);
}

/// The read of `reader` that takes `value`, when it is the only one, takes it without a pad,
/// holding or placement, and the body only casts what it gives.
fn widened_read(reader: &BlockOp, value: Value) -> Option<usize> {
    let mut found = None;
    for (index, read) in reader.reads.iter().enumerate() {
        let Source::Operand(operand) = read.source else {
            continue;
        };
        if reader.ins()[operand].id() != value {
            continue;
        }
        if found.is_some() || read.pad.is_some() || read.at.is_some() || read.held.is_some() {
            return None;
        }
        found = Some(index);
    }
    let index = found?;
    let arg = reader.body().args()[index];
    let casts = widening_casts(arg);
    (!casts.is_empty() && arg.uses().len() == casts.len()).then_some(index)
}

/// Whether `read` of `reader` leaves out of its coordinates an iterator of more than one point, so
/// that the reader reads each element it takes at several points.
fn reads_repeatedly(reader: &BlockOp, read: &Read) -> bool {
    let mut repeats = false;
    for (position, iter) in reader.iters.iter().enumerate() {
        let several = iter.extent.as_static() != Some(1);
        repeats |= several && !read.uses_iter(position);
    }
    repeats
}

/// The `arith.cast` ops that read `value`.
fn widening_casts(value: ValueRef<'_>) -> Vec<arith::CastOp<'_>> {
    let module = value.module();
    let mut casts = Vec::new();
    for use_site in value.uses() {
        if let Some(cast) = module.op_as::<arith::Cast>(use_site.inst) {
            casts.push(cast);
        }
    }
    casts
}
