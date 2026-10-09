//! The dialects and their union, [`Op`].

pub mod func;
pub mod tensor;

use std::borrow::Cow;

use anyhow::Result;

use crate::dense::Dense;
use crate::host::HostValue;
use crate::module::{Module, Region};
use crate::op::{Effects, FoldIn, Folded, Group, OpDyn, RegionRunner, RegionSig};
use crate::text;
use crate::traits::{OpTrait, Traits};
use crate::types::Type;
use func::FuncOp;
use tensor::TensorOp;

/// Every op of every dialect.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Op {
    Func(FuncOp),
    Tensor(TensorOp),
}

/// The dialect an op belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Dialect {
    Func,
    Tensor,
}

macro_rules! each_dialect {
    ($self:expr, $op:ident => $body:expr) => {
        match $self {
            Op::Func($op) => $body,
            Op::Tensor($op) => $body,
        }
    };
}

impl Op {
    /// The dialect of the op.
    pub fn dialect(&self) -> Dialect {
        match self {
            Op::Func(_) => Dialect::Func,
            Op::Tensor(_) => Dialect::Tensor,
        }
    }

    pub fn name(&self) -> Cow<'static, str> {
        each_dialect!(self, op => op.name())
    }

    pub fn effects(&self) -> Effects {
        each_dialect!(self, op => op.effects())
    }

    pub fn traits(&self) -> Traits {
        each_dialect!(self, op => op.traits())
    }

    /// The trait `T`, with its parameters, when the op has it.
    pub fn get_trait<T: OpTrait>(&self) -> Option<&'static T> {
        self.traits().get::<T>()
    }

    pub fn has_trait<T: OpTrait>(&self) -> bool {
        self.traits().has::<T>()
    }

    /// The op's operand groups, in operand order.
    pub fn operand_groups(&self) -> &'static [Group] {
        each_dialect!(self, op => op.operand_groups())
    }

    /// The op's result groups, in result order.
    pub fn result_groups(&self) -> &'static [Group] {
        each_dialect!(self, op => op.result_groups())
    }

    pub fn region_names(&self) -> &'static [&'static str] {
        each_dialect!(self, op => op.region_names())
    }

    pub fn is_region_terminator(&self, index: usize, op: &Op) -> bool {
        each_dialect!(self, inner => inner.is_region_terminator(index, op))
    }

    pub fn print_attrs(&self, out: &mut Vec<(&'static str, String)>) {
        each_dialect!(self, op => op.print_attrs(out))
    }

    /// Parses the op named `name` and its attribute dictionary, or returns `None` when no dialect
    /// has an op of that name.
    pub fn parse(name: &str, c: &mut text::Cursor<'_>) -> Option<Result<Op>> {
        if let Some(result) = FuncOp::parse(name, c) {
            return Some(result.map(Op::Func));
        }
        if let Some(result) = TensorOp::parse(name, c) {
            return Some(result.map(Op::Tensor));
        }
        None
    }
}

impl OpDyn for Op {
    fn dyn_result_types(
        &self,
        ins: &[&Type],
        segments: Option<&[usize]>,
        regions: &[Region],
        module: &Module,
    ) -> Result<(crate::op::Flat<Type>, Option<Vec<usize>>)> {
        each_dialect!(self, op => op.dyn_result_types(ins, segments, regions, module))
    }

    fn dyn_region_signatures(
        &self,
        ins: &[&Type],
        segments: Option<&[usize]>,
    ) -> Result<Vec<RegionSig>> {
        each_dialect!(self, op => op.dyn_region_signatures(ins, segments))
    }

    fn dyn_fold(
        &self,
        ins: &[FoldIn<'_>],
        segments: Option<&[usize]>,
        results: &[Type],
    ) -> Option<Vec<Folded>> {
        each_dialect!(self, op => op.dyn_fold(ins, segments, results))
    }

    fn dyn_eval(
        &self,
        ins: &[&HostValue],
        segments: Option<&[usize]>,
        regions: &[Region],
        results: &[Type],
        runner: &mut dyn RegionRunner,
    ) -> Result<Vec<HostValue>> {
        each_dialect!(self, op => op.dyn_eval(ins, segments, regions, results, runner))
    }
}

/// The op that defines a constant with the value `dense`.
pub fn constant_op(dense: Dense) -> Op {
    tensor::Constant { value: dense }.into()
}
