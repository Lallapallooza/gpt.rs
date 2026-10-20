//! The costs of the `host` ops, which `#[cost]` registers.

use crate::analysis::{cost, tensor_bytes, Cost, CostAt, OpCost};
use crate::dialects::host::ir::*;
use crate::dialects::tile;
use crate::ir::Module;

/// What a call of `callee` costs: the cost of the ops of a `tile.dispatch`'s body, with the ops
/// of each block counted once per point, or `None` for a `loop.func`: the count of a
/// `loop.func`'s ops leaves out its loops.
fn callee_cost(module: &Module, callee: &str) -> Option<Cost> {
    let symbol = module.symbol(callee)?;
    module.op(symbol).get::<tile::Dispatch>()?;
    let body = *module.regions(symbol).first()?;
    let summary = module.cost_summary(body, |_| String::new());
    (summary.unknown == 0).then_some(summary.total)
}

/// A call costs what its callee does: what a `tile.dispatch` costs, or nothing known for a
/// `loop.func`.
#[cost]
impl OpCost for Dispatch {
    fn cost(&self, at: CostAt<Self>) -> Option<Cost> {
        callee_cost(at.module, &self.callee.0)
    }
}

/// A launch has no known cost: it runs a `loop.func`, and the count of a `loop.func`'s ops leaves
/// out its loops.
#[cost]
impl OpCost for Launch {
    fn cost(&self, _at: CostAt<Self>) -> Option<Cost> {
        None
    }
}

/// A copy reads and writes the bytes of its buffer.
#[cost]
impl OpCost for Copy {
    fn cost(&self, at: CostAt<Self>) -> Option<Cost> {
        let bytes = tensor_bytes(&at.ins.from.tensor_type())?;
        Some(Cost {
            flops: 0,
            bytes_read: bytes,
            bytes_written: bytes,
        })
    }
}

/// An arena moves no data: it names memory.
#[cost]
impl OpCost for Arena {
    fn cost(&self, _at: CostAt<Self>) -> Option<Cost> {
        Some(Cost::default())
    }
}

/// A slice moves no data: it names bytes of its arena.
#[cost]
impl OpCost for Slice {
    fn cost(&self, _at: CostAt<Self>) -> Option<Cost> {
        Some(Cost::default())
    }
}

/// An allocation moves no data: it names memory.
#[cost]
impl OpCost for Alloc {
    fn cost(&self, _at: CostAt<Self>) -> Option<Cost> {
        Some(Cost::default())
    }
}

/// A buffer of a tensor's elements moves no data: it names the tensor's memory.
#[cost]
impl OpCost for ToBuffer {
    fn cost(&self, _at: CostAt<Self>) -> Option<Cost> {
        Some(Cost::default())
    }
}

/// A taken tensor moves no data: it names the tensor's memory.
#[cost]
impl OpCost for Take {
    fn cost(&self, _at: CostAt<Self>) -> Option<Cost> {
        Some(Cost::default())
    }
}
