//! Computing a map of parameters in its readers, where its result is larger than what it reads.

use std::collections::HashMap;

use anyhow::Result;

use crate::dialects::tile::{
    copy_body_ops, Block, BlockOp, Index, IterKind, Read, Source, UnscheduledBlock,
};
use crate::ir::Value;
use crate::passes::{pass, run_patterns, OutlineStatic, Pass};
use crate::require;
use crate::rewrite::{pattern, NoMatch, Pattern, PatternSet, Rewriter, Rewrites, Rewriting};

/// Computes each map known before the run that `outline-static` would not store, because its result
/// takes more bytes than the parameters it reads ([`OutlineStatic::would_store_over`]). The blocks
/// that read the map compute it at each point they read. A weight cast to a wider dtype or
/// dequantized is then read as stored. No copy of it larger than the parameters is made at load
/// time or at each call. A block that reads an element at several points computes it at each, so it
/// is a pass rather than a canonical form.
///
/// - The map writes each point of its parallel iterators at an element of its own, at a
///   coordinate that is the iterator or 0, and its body reads no `tile.index`.
/// - The read of its result has no pad, holding or placement.
#[pass(
    name = "inline-static-maps",
    stats(inlined = "reads of maps computed in their readers")
)]
pub struct InlineStaticMaps;

fn populate_inline_static_maps_patterns(set: &mut PatternSet) {
    set.add(ComputeMapInReader);
}

impl Pass for InlineStaticMaps {
    fn rewrites(&self) -> Rewrites {
        Rewrites::patterns(populate_inline_static_maps_patterns)
    }

    fn run(&self, module: &mut Rewriting) -> Result<bool> {
        run_patterns(module, self.rewrites(), &Self::INLINED)
    }
}

/// A read of the result of a map known before the run that takes more bytes than its operands is
/// the map's body on the map's reads, at the point of the map that the read reaches.
#[pattern]
struct ComputeMapInReader;

impl Pattern for ComputeMapInReader {
    type Root = UnscheduledBlock;

    fn rewrite(&self, block: BlockOp, rewriter: &mut Rewriter) -> Result<(), NoMatch> {
        let module = block.module();
        let mut found = None;
        for (index, read) in block.reads.iter().enumerate() {
            if let Some(map) = larger_map(&block, read) {
                found = Some((index, map));
                break;
            }
        }
        require!(
            let Some((index, map)) = found,
            "no read takes a map known before the run that is larger than what it reads"
        );
        let mut ins: Vec<Value> = block.ins().iter().map(|input| input.id()).collect();
        let offset = ins.len();
        for input in map.ins() {
            ins.push(input.id());
        }
        let mut attrs = block.attrs().clone();
        let mut reads = attrs.reads.to_vec();
        reads.remove(index);
        for map_read in 0..map.reads.len() {
            reads.push(map.read_through(map_read, &block.reads[index], offset));
        }
        attrs.reads = reads.into();
        let reader_body = block.body();
        let map_body = map.body();
        let kept = block.reads.len() - 1;
        let results = attrs.build(rewriter, ins, |body, args| {
            let mut values = HashMap::new();
            for (arg, &value) in map_body.args().into_iter().zip(&args[kept..]) {
                values.insert(arg.id(), value);
            }
            let [computed] = copy_body_ops(module, map_body.id(), body.builder, &mut values)[..]
            else {
                unreachable!("a map yields one value")
            };
            let mut kept_args = args[..kept].iter();
            for (position, arg) in reader_body.args().into_iter().enumerate() {
                let value = match position == index {
                    true => computed,
                    false => *kept_args.next().expect("an argument for each kept read"),
                };
                values.insert(arg.id(), value);
            }
            copy_body_ops(module, reader_body.id(), body.builder, &mut values)
        });
        rewriter.replace_op(block.inst(), results);
        Ok(())
    }
}

/// The map that `read` of `block` takes the result of, when it is known before the run,
/// `outline-static` would not store it, and it computes its elements as the pass requires.
fn larger_map<'m>(block: &BlockOp<'m>, read: &Read) -> Option<BlockOp<'m>> {
    if read.pad.is_some() || read.at.is_some() || read.held.is_some() {
        return None;
    }
    let Source::Operand(operand) = read.source else {
        return None;
    };
    let map = block.ins()[operand].defining_op::<Block>()?;
    let module = map.module();
    let write = &map.folds.first()?.writes.first()?.coords;
    let each_point = map.is_map()
        && !map.scheduled()
        && map.iters.iter().all(|iter| iter.kind == IterKind::Par)
        && map.folds[0].writes[0].translates(&map.iters)
        && write
            .iter()
            .all(|coord| coord.as_iter().is_some() || coord.as_const() == Some(0));
    let indexes = map
        .body()
        .insts()
        .any(|inst| module.op(inst).get::<Index>().is_some());
    let result = map.outs()[0];
    let mut reads = Vec::with_capacity(map.ins().len());
    for input in map.ins() {
        reads.push(input.value_type());
    }
    let larger = !OutlineStatic::would_store_over(result.value_type(), &reads);
    (each_point && !indexes && result.known_before_run() && larger).then_some(map)
}
