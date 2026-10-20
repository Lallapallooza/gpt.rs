//! Op traits: facts that hold for every instance of an op, each with the generic behavior that
//! follows from it.
//!
//! An op lists its traits in `#[op(..., traits(...))]`, and an op family lists those its ops share
//! in `#[family(..., traits(...))]`. Folding calls every trait's [`OpTrait::fold`], and a pattern
//! rooted at [`Has<T>`](crate::rewrite::Has) runs on every op that has `T`. Each algebraic trait
//! states the equations it claims in [`OpTrait::laws`], which a test checks against the op's `Eval`
//! on random inputs.

mod algebra;
mod patterns;
mod structure;

use std::any::Any;
use std::fmt;

pub use self::algebra::{
    Absorbing, Associative, Commutative, DistributesOver, Idempotent, IntValue, Involution,
    PerClass, RightIdentity, NEG_ZERO,
};
pub use self::structure::{
    Elementwise, IsolatedFromAbove, Movement, NarrowInF32, OnlyUse, Terminator,
};
use crate::ir::{DType, Dense, Module, Type, Value};
use crate::rewrite::{Doc, FoldResult};
use crate::Op;

/// A fact about an op. Every hook has a default that does nothing, so a trait implements only the
/// behavior that follows from it. `Display` writes it with its parameters, as an op's entry in the
/// dialect reference shows it, and `#[derive(Doc)]` gives its name and doc comment.
pub trait OpTrait: Any + Sync + fmt::Debug + fmt::Display + Doc {
    /// What the op folds to by this trait alone.
    fn fold(&self, _op: &Operation) -> Option<FoldResult> {
        None
    }

    /// The equations the trait claims of the op applied to `arity` operands of dtype `dtype`.
    fn laws(&self, _arity: usize, _dtype: DType) -> Vec<Law> {
        Vec::new()
    }
}

/// The traits of one op: those of its family, then those the op lists.
#[derive(Clone, Copy)]
pub struct Traits {
    traits: &'static [&'static dyn OpTrait],
}

impl Traits {
    pub const fn new(traits: &'static [&'static dyn OpTrait]) -> Self {
        Self { traits }
    }

    pub fn iter(&self) -> impl Iterator<Item = &'static dyn OpTrait> {
        self.traits.iter().copied()
    }

    /// The trait `T`, with its parameters, when the op has it.
    pub fn get<T: OpTrait>(&self) -> Option<&'static T> {
        self.iter().find_map(|op_trait| {
            let any: &'static dyn Any = op_trait;
            any.downcast_ref()
        })
    }

    pub fn has<T: OpTrait>(&self) -> bool {
        self.get::<T>().is_some()
    }
}

/// An op with its operands and result types, untyped, as a trait's fold sees it: an op of the
/// module, or one a builder is about to create.
pub struct Operation<'a> {
    pub module: &'a Module,
    pub op: &'a Op,
    pub operands: &'a [Value],
    pub results: &'a [Type],
}

impl Operation<'_> {
    /// The constant that defines operand `index`, if one does.
    pub fn constant(&self, index: usize) -> Option<&Dense> {
        self.module.constant(self.operands[index])
    }

    /// Whether operand `index` is a constant whose every element has the bytes `element`.
    pub fn is_splat(&self, index: usize, element: &[u8]) -> bool {
        self.constant(index)
            .and_then(Dense::splat_bytes)
            .is_some_and(|bytes| bytes == element)
    }

    /// The operands of the op that defines operand `index`, when it is this same op.
    pub fn same_op_operands(&self, index: usize) -> Option<&[Value]> {
        let inst = self.module.def_inst(self.operands[index])?;
        (self.module.op(inst) == self.op).then(|| self.module.operands(inst))
    }

    /// The dtype of the first operand: its element dtype, or its own as a scalar.
    pub fn dtype(&self) -> Option<DType> {
        first_dtype(self.module, self.operands)
    }
}

fn first_dtype(module: &Module, operands: &[Value]) -> Option<DType> {
    module.ty(*operands.first()?).dtype()
}

/// A side of a law, whose values are tensors of the operand type, or scalars of it.
#[derive(Debug, Clone, PartialEq)]
pub enum Term {
    /// Variable `index`: a value of the operand type.
    Var(usize),
    /// The op itself applied to these terms.
    Apply(Vec<Term>),
    /// Another op applied to these terms.
    Other(Box<Op>, Vec<Term>),
    /// The term converted to the dtype: by `tensor.cast` on a tensor, by `arith.cast` on a scalar.
    Cast(DType, Box<Term>),
    /// A value of the operand type whose every element has these bytes.
    Splat(Vec<u8>),
}

/// An equation that holds bit for bit for every value of its variables, except for the sign and
/// payload of a NaN.
#[derive(Debug, Clone, PartialEq)]
pub struct Law {
    /// The equation in words, such as `f(a, b) = f(b, a)`.
    pub name: &'static str,
    pub lhs: Term,
    pub rhs: Term,
}
