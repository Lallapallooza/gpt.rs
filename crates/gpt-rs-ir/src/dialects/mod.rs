//! The dialects and their union, [`Op`].

pub mod arith;
pub mod func;
pub mod host;
pub mod isa;
pub mod loops;
pub mod tensor;
pub mod tile;
pub mod x86;

use anyhow::Result;

use self::arith::ArithOps;
use self::func::FuncOps;
use self::host::HostOps;
use self::isa::IsaOps;
use self::loops::LoopOps;
use self::tensor::TensorOps;
use self::tile::TileOps;
use crate::ir::{Dense, InOp, OpDyn, SymbolOp, Type};
use crate::traits::OpTrait;

/// Every op of every dialect.
#[derive(crate::DialectSet, Debug, Clone, PartialEq, Eq, Hash)]
#[dialect_set(kind(Dialect))]
pub enum Op {
    Func(FuncOps),
    Tensor(TensorOps),
    Arith(ArithOps),
    Tile(TileOps),
    Host(HostOps),
    Loop(LoopOps),
    Isa(IsaOps),
}

impl Op {
    /// The name of the top-level symbol this op defines, when it is a [`SymbolOp`].
    pub fn symbol_name(&self) -> Option<&str> {
        self.dyn_symbol().map(SymbolOp::symbol)
    }

    /// The trait `T`, with its parameters, when the op has it.
    pub fn get_trait<T: OpTrait>(&self) -> Option<&'static T> {
        self.traits().get::<T>()
    }

    pub fn has_trait<T: OpTrait>(&self) -> bool {
        self.traits().has::<T>()
    }

    /// The attributes of the op when it is a `T`.
    pub fn get<T: InOp>(&self) -> Option<&T> {
        T::cast(self)
    }

    /// The value of a [`ConstantValue`](crate::ir::ConstantValue), such as `tensor.constant`, or
    /// `arith.constant` as a rank-0 constant.
    pub fn as_constant(&self) -> Option<&Dense> {
        self.dyn_constant()
    }
}

/// The op that defines a constant of type `ty` with the value `dense`: `arith.constant` for a
/// scalar, `tensor.constant` for a tensor.
pub fn constant_op(dense: Dense, ty: &Type) -> Result<Op> {
    Ok(match ty {
        Type::Scalar(_) => arith::Constant {
            value: arith::Literal::new(crate::host::HostScalar::from_dense(&dense)?)?,
        }
        .into(),
        _ => tensor::Constant { value: dense }.into(),
    })
}
