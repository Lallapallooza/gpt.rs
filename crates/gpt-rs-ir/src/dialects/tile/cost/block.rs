//! The cost of `tile.block`.

use std::collections::BTreeSet;

use crate::analysis::{cost, tensor_bytes, tensor_elements, Cost, CostAt, OpCost};
use crate::dialects::tile::ir::*;
use crate::traits::Terminator;

/// A block runs its body once at each point of its domain: each op of the body but its constants
/// and `tile.index` values, which compute nothing, and one step of each fold other than `none`,
/// as a `tensor` reduction counts one operation for each element it folds.
///
/// It reads at most the elements it takes: of each operand, one element per point for each read
/// and each read of an index expression, as `tensor.gather` reads the rows it gathers, and the
/// whole operand that a fold starts from. It does not read an operand that a write only updates.
/// It writes each new result whole, and of a result that updates an operand the elements its
/// coordinates reach, one per point of the iterators they take, as `tensor.dynamic_update_slice`
/// writes its patch.
#[cost]
impl OpCost for Block {
    fn cost(&self, at: CostAt<Self>) -> Option<Cost> {
        let (module, inst) = (at.module, at.inst);
        let mut points: u64 = 1;
        for iter in &self.iters {
            points *= iter.extent.as_static()? as u64;
        }
        let mut per_point = 0;
        for op in module.walk(module.regions(inst)[0]) {
            let body_op = module.op(op);
            let computes = !body_op.has_trait::<Terminator>()
                && body_op.as_constant().is_none()
                && body_op.get::<Index>().is_none();
            if computes {
                per_point += module.cost(op)?.flops;
            }
        }
        for fold in &self.folds {
            if fold.combine != CombineKind::None {
                per_point += 1;
            }
        }
        let mut operand_types = Vec::new();
        for &operand in module.operands(inst) {
            operand_types.push(module.value_ref(operand).tensor_type());
        }
        let mut taken_bits = vec![0; operand_types.len()];
        let point_bits = |operand: usize| points * operand_types[operand].dtype.bits() as u64;
        for read in &self.reads {
            for operand in read.source.operands() {
                taken_bits[operand] += point_bits(operand);
            }
        }
        for expr in self.exprs() {
            for (operand, _) in expr.reads() {
                taken_bits[operand] += point_bits(operand);
            }
        }
        for fold in &self.folds {
            for start in &fold.init {
                let start_type = operand_types[start.0];
                taken_bits[start.0] +=
                    tensor_elements(start_type)? * start_type.dtype.bits() as u64;
            }
        }
        let mut bytes_read = 0;
        for (operand_type, bits) in operand_types.iter().zip(taken_bits) {
            bytes_read += tensor_bytes(operand_type)?.min(bits.div_ceil(8));
        }
        let mut bytes_written = 0;
        for (write, &result) in self.writes().zip(module.results(inst)) {
            let result_type = module.value_ref(result).tensor_type();
            let whole = tensor_bytes(result_type)?;
            if write.into.is_none() {
                bytes_written += whole;
                continue;
            }
            let mut taken = BTreeSet::new();
            for coord in &write.coords {
                taken.extend(coord.iters());
            }
            let mut written: u64 = 1;
            for position in taken {
                written *= self.iters[position].extent.as_static()? as u64;
            }
            let written_bits = written * result_type.dtype.bits() as u64;
            bytes_written += whole.min(written_bits.div_ceil(8));
        }
        Some(Cost {
            flops: points * per_point,
            bytes_read,
            bytes_written,
        })
    }
}
