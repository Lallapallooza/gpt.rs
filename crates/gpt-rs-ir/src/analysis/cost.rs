//! What an op costs to run.

use std::any::{Any, TypeId};
use std::collections::BTreeMap;
use std::ops::{Add, AddAssign};

pub use gpt_rs_ir_macros::cost;

use crate::ir::{
    Groups, Inst, Module, OpDyn, OpKind, Operands, Region, Results, TensorType, Type, Types,
};
use crate::traits::{Movement, Terminator};
use crate::Op;

/// The arithmetic and the memory traffic of running an op.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Cost {
    /// Arithmetic operations: a multiply-add counts two.
    pub flops: u64,
    pub bytes_read: u64,
    pub bytes_written: u64,
}

impl Cost {
    /// The bytes it moves, read and written.
    pub fn bytes(&self) -> u64 {
        self.bytes_read + self.bytes_written
    }
}

impl Add for Cost {
    type Output = Cost;

    fn add(self, other: Cost) -> Cost {
        Cost {
            flops: self.flops + other.flops,
            bytes_read: self.bytes_read + other.bytes_read,
            bytes_written: self.bytes_written + other.bytes_written,
        }
    }
}

impl AddAssign for Cost {
    fn add_assign(&mut self, other: Cost) {
        *self = *self + other;
    }
}

/// An op whose cost the default model gets wrong, such as one whose cost comes from its regions,
/// which it runs as often as its semantics say, or from the function it calls. `#[cost]` on the
/// impl registers it, and the ops of the regions of a registered op count in its cost alone.
pub trait OpCost: OpKind {
    /// The op's cost at `at`, or `None` when a dynamic extent leaves it unknown.
    fn cost(&self, at: CostAt<Self>) -> Option<Cost>;
}

/// Where an op's cost is asked: the module, the op's instruction, and the types of its operands
/// and results in its groups.
pub struct CostAt<'m, T: OpKind> {
    pub module: &'m Module,
    pub inst: Inst,
    pub ins: Operands<T, Types<'m>>,
    pub outs: Results<T, Types<'m>>,
}

/// An op's registered cost, erased to run on any op value.
pub type CostFn = fn(&dyn Any, &Module, Inst) -> Option<Cost>;

/// One op's registered cost.
pub struct CostEntry {
    /// The `TypeId` of the op struct.
    pub op: fn() -> TypeId,
    /// The name of the op struct, and the file of its cost, relative to the workspace.
    pub rust: &'static str,
    pub file: &'static str,
    pub cost: CostFn,
}

/// Every cost that `#[cost]` registers.
#[linkme::distributed_slice]
pub static COSTS: [CostEntry];

/// `T`'s cost as a [`CostFn`], which `#[cost]` registers.
pub fn cost_erased<T: OpCost + 'static>(
    erased_op: &dyn Any,
    module: &Module,
    inst: Inst,
) -> Option<Cost> {
    let op = erased_op.downcast_ref::<T>()?;
    let operand_types = module.operand_types(inst);
    let result_types = module.result_types(inst);
    let segments = module.operand_segments(inst);
    let ins =
        T::Groups::split_operands::<Types<'_>>(operand_types.iter().copied(), segments).ok()?;
    let outs = T::Groups::split_results::<Types<'_>>(result_types.iter().copied(), None).ok()?;
    op.cost(CostAt {
        module,
        inst,
        ins,
        outs,
    })
}

/// The registered cost of `op`, when it has one.
fn registered_cost(op: &Op) -> Option<CostFn> {
    let op_type = op.dyn_any().type_id();
    COSTS
        .iter()
        .find(|entry| (entry.op)() == op_type)
        .map(|entry| entry.cost)
}

/// The cost of an op that computes each result element once, from the operand elements:
///
/// - a `Movement` op does no arithmetic and reads as many bytes as the smaller of its operands
///   and results, as a slice reads its result and a broadcast its input;
/// - any other op does one operation for each element of its largest operand or result, and reads
///   every operand.
///
/// `None` when a dynamic extent leaves a size unknown.
pub fn default_cost(op: &Op, ins: &[&Type], outs: &[&Type]) -> Option<Cost> {
    let operand_bytes = total(ins, bytes)?;
    let result_bytes = total(outs, bytes)?;
    if op.has_trait::<Movement>() {
        return Some(Cost {
            flops: 0,
            bytes_read: operand_bytes.min(result_bytes),
            bytes_written: result_bytes,
        });
    }
    let mut largest = 0;
    for ty in [ins, outs].concat() {
        largest = largest.max(elements(ty)?);
    }
    Some(Cost {
        flops: largest,
        bytes_read: operand_bytes,
        bytes_written: result_bytes,
    })
}

/// The elements of a value of type `ty`: one for a scalar, an index or a buffer, whose value is
/// where its elements are, and every lane of a vector.
pub fn elements(ty: &Type) -> Option<u64> {
    match ty {
        Type::Tensor(tensor) => tensor_elements(tensor),
        Type::Vector(vector) => tensor_elements(&vector.tensor_type()),
        Type::Scalar(_) | Type::Index | Type::Buffer(_) => Some(1),
    }
}

/// The bytes a value of type `ty` takes: the address of its elements for a buffer.
pub fn bytes(ty: &Type) -> Option<u64> {
    match ty {
        Type::Tensor(tensor) => tensor_bytes(tensor),
        Type::Vector(vector) => tensor_bytes(&vector.tensor_type()),
        Type::Scalar(dtype) => Some((dtype.bits() as u64).div_ceil(8)),
        Type::Index | Type::Buffer(_) => Some(Type::INDEX_BYTES),
    }
}

/// The elements of a tensor of type `tensor`, when its extents are static.
pub fn tensor_elements(tensor: &TensorType) -> Option<u64> {
    tensor.element_count().map(|count| count as u64)
}

/// The bytes a tensor of type `tensor` takes, its elements packed as narrow dtypes store them.
pub fn tensor_bytes(tensor: &TensorType) -> Option<u64> {
    Some((tensor_elements(tensor)? * tensor.dtype.bits() as u64).div_ceil(8))
}

/// The sum of `measure` over `types`, or `None` when one is unknown.
fn total(types: &[&Type], measure: fn(&Type) -> Option<u64>) -> Option<u64> {
    let mut sum = 0;
    for ty in types {
        sum += measure(ty)?;
    }
    Some(sum)
}

/// The cost of the ops under a region, group by group and in all.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct CostSummary {
    /// For each group, how many ops it has and what they cost together.
    pub groups: BTreeMap<String, (usize, Cost)>,
    /// What every op costs together.
    pub total: Cost,
    /// The ops whose cost a dynamic extent leaves unknown, which the totals leave out.
    pub unknown: usize,
}

impl Module {
    /// The cost of the op of `inst`, or `None` when a dynamic extent leaves it unknown.
    pub fn cost(&self, inst: Inst) -> Option<Cost> {
        let op = self.op(inst);
        match registered_cost(op) {
            Some(cost) => cost(op.dyn_any(), self, inst),
            None => default_cost(op, &self.operand_types(inst), &self.result_types(inst)),
        }
    }

    /// The cost of the ops under `region`, those of nested regions once each, in the groups that
    /// `group` names, such as op names. Terminators move no data, so they are left out, and the
    /// ops in the regions of an op with a registered [`OpCost`] count in its cost alone.
    pub fn cost_summary(&self, region: Region, group: impl Fn(Inst) -> String) -> CostSummary {
        let mut summary = CostSummary::default();
        let mut pending: Vec<Inst> = self.insts(region).collect();
        pending.reverse();
        while let Some(inst) = pending.pop() {
            if registered_cost(self.op(inst)).is_none() {
                for &inner in self.regions(inst).iter().rev() {
                    let mut nested: Vec<Inst> = self.insts(inner).collect();
                    nested.reverse();
                    pending.extend(nested);
                }
            }
            let op = self.op(inst);
            if op.has_trait::<Terminator>() {
                continue;
            }
            let Some(cost) = self.cost(inst) else {
                summary.unknown += 1;
                continue;
            };
            let entry = summary.groups.entry(group(inst)).or_default();
            entry.0 += 1;
            entry.1 += cost;
            summary.total += cost;
        }
        summary
    }
}
