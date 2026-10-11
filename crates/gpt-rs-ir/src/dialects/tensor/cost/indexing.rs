//! The costs of the indexing ops.

use crate::cost::{cost, tensor_bytes, Cost, OpCost};
use crate::dialects::tensor::ir::*;
use crate::op::Types;

/// A gather reads the elements it gathers and its indices, not its whole input, such as the rows
/// of an embedding table that a few tokens look up.
#[cost]
impl OpCost for Gather {
    fn cost(&self, ins: GatherOperands<Types<'_>>, outs: GatherResults<Types<'_>>) -> Option<Cost> {
        let gathered = tensor_bytes(outs.output)?;
        Some(Cost {
            flops: 0,
            bytes_read: tensor_bytes(ins.indices)? + gathered,
            bytes_written: gathered,
        })
    }
}

/// A dynamic slice reads its block and its starts, not its whole input.
#[cost]
impl OpCost for DynamicSlice {
    fn cost(
        &self,
        ins: DynamicSliceOperands<Types<'_>>,
        outs: DynamicSliceResults<Types<'_>>,
    ) -> Option<Cost> {
        let block = tensor_bytes(outs.output)?;
        Some(Cost {
            flops: 0,
            bytes_read: tensor_bytes(ins.starts)? + block,
            bytes_written: block,
        })
    }
}

/// A dynamic update slice runs in place, as backends run a cache update: it reads the update and
/// its starts and writes the update into the base.
#[cost]
impl OpCost for DynamicUpdateSlice {
    fn cost(
        &self,
        ins: DynamicUpdateSliceOperands<Types<'_>>,
        _outs: DynamicUpdateSliceResults<Types<'_>>,
    ) -> Option<Cost> {
        let update = tensor_bytes(ins.update)?;
        Some(Cost {
            flops: 0,
            bytes_read: tensor_bytes(ins.starts)? + update,
            bytes_written: update,
        })
    }
}
