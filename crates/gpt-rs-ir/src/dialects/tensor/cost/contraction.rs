//! The costs of the contraction ops.

use crate::analysis::{cost, tensor_bytes, tensor_elements, Cost, CostAt, OpCost};
use crate::dialects::tensor::ir::*;
use crate::dialects::tensor::utils::contraction::Side;

/// Two operations, a multiply and an add, for each product the dot sums: each result element sums
/// one product for each coordinate of the contracting axes. It reads both operands.
#[cost]
impl OpCost for DotGeneral {
    fn cost(&self, at: CostAt<Self>) -> Option<Cost> {
        let (ins, outs) = (at.ins, at.outs);
        let axes = self.axes(ins.lhs.rank(), ins.rhs.rank()).ok()?;
        let mut contracted = 1;
        for &axis in axes.contract(Side::Lhs) {
            contracted *= ins.lhs.static_dim(axis)? as u64;
        }
        Some(Cost {
            flops: 2 * tensor_elements(outs.output)? * contracted,
            bytes_read: tensor_bytes(ins.lhs)? + tensor_bytes(ins.rhs)?,
            bytes_written: tensor_bytes(outs.output)?,
        })
    }
}
