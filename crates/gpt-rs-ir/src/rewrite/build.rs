//! [`Build`]: what [`Builder`](super::Builder) and [`Rewriter`](super::Rewriter) share for
//! building ops. It creates an op or what the op folds to, builds splat constants, and gives ops
//! as methods and operators on a value ([`Chain`]).

use std::ops;

use crate::dialects::constant_op;
use crate::dialects::tensor::{Add, BroadcastTo, Cast, Constant, Mul, Reshape, Transpose};
use crate::host::HostScalar;
use crate::ir::{
    Created, DType, Dim, Folded, Groups, InOp, Operands, TensorType, Type, Value, Values,
};
use crate::Op;

/// What builds IR: a [`Builder`](super::Builder) outside patterns and a
/// [`Rewriter`](super::Rewriter) inside them.
pub trait Build: Sized {
    /// Creates the op value `op`, which has no regions, on `operands` in the groups `segments`,
    /// unless the op folds. Then it gives what the op folds to instead, existing values or new
    /// constants. Gives the results and their groups.
    fn create_or_fold_op(
        &mut self,
        op: Op,
        operands: Vec<Value>,
        segments: Option<Vec<usize>>,
    ) -> (Vec<Value>, Option<Vec<usize>>);

    /// The type of `value`, a tensor.
    fn tensor_type(&self, value: Value) -> &TensorType;

    /// Creates `op`, which has no regions, on `operands`, or gives what it folds to, as `create`
    /// gives the results of an op.
    #[track_caller]
    fn create_or_fold<T: InOp>(
        &mut self,
        op: T,
        operands: impl Into<Operands<T, Values>>,
    ) -> Created<T> {
        let (operands, segments) = T::Groups::flatten_operands::<Values>(operands.into());
        let (values, result_segments) =
            self.create_or_fold_op(op.into(), operands.into_vec(), segments);
        let results =
            T::Groups::split_results::<Values>(values.into_iter(), result_segments.as_deref())
                .expect("the type rule gave the result groups");
        T::Groups::created(results)
    }

    /// `scalar` repeated to the extents `dims`: a constant of rank 0, broadcast to them.
    #[track_caller]
    fn splat(&mut self, scalar: HostScalar, dims: &[Dim]) -> Value {
        let value = scalar
            .to_dense()
            .expect("a scalar of a dtype is a dense constant");
        let constant = self.create_or_fold(Constant { value }, ());
        self.on(constant).broadcast_to(dims.iter().cloned()).value()
    }

    /// `value`, with the ops built on it as methods.
    fn on(&mut self, value: impl Into<Value>) -> Chain<'_, Self> {
        Chain {
            build: self,
            value: value.into(),
        }
    }
}

/// A value and the builder that builds ops on it, named as the ops of `DeviceTensor` are:
/// `(rewriter.on(x).transpose(perm) * y).cast(DType::F32).value()`. Each method and operator
/// creates its op or what the op folds to, so a transpose, reshape, broadcast or cast to the
/// value's own type leaves it as it is.
pub struct Chain<'b, B: Build> {
    build: &'b mut B,
    value: Value,
}

impl<B: Build> Chain<'_, B> {
    /// The value built so far.
    pub fn value(&self) -> Value {
        self.value
    }

    /// The type of the value built so far.
    pub fn tensor_type(&self) -> &TensorType {
        self.build.tensor_type(self.value)
    }

    /// The value after `op`, an op of one operand and one result.
    #[track_caller]
    pub fn apply<T: InOp>(self, op: T) -> Self
    where
        Value: Into<Operands<T, Values>>,
        Created<T>: Into<Value>,
    {
        let value = self.build.create_or_fold(op, self.value).into();
        Self {
            build: self.build,
            value,
        }
    }

    #[track_caller]
    pub fn transpose(self, perm: impl IntoIterator<Item = usize>) -> Self {
        self.apply(Transpose::new(perm))
    }

    #[track_caller]
    pub fn reshape(self, dims: impl IntoIterator<Item = impl Into<Dim>>) -> Self {
        self.apply(Reshape::new(dims))
    }

    #[track_caller]
    pub fn broadcast_to(self, dims: impl IntoIterator<Item = impl Into<Dim>>) -> Self {
        self.apply(BroadcastTo::new(dims))
    }

    #[track_caller]
    pub fn cast(self, dtype: DType) -> Self {
        self.apply(Cast { to: dtype })
    }
}

impl<B: Build, V: Into<Value>> ops::Mul<V> for Chain<'_, B> {
    type Output = Self;

    #[track_caller]
    fn mul(self, other: V) -> Self {
        let value = self.build.create_or_fold(Mul, (self.value, other.into()));
        Self {
            build: self.build,
            value,
        }
    }
}

impl<B: Build, V: Into<Value>> ops::Add<V> for Chain<'_, B> {
    type Output = Self;

    #[track_caller]
    fn add(self, other: V) -> Self {
        let value = self.build.create_or_fold(Add, (self.value, other.into()));
        Self {
            build: self.build,
            value,
        }
    }
}

/// The values of folded results of types `types`: an existing value, or the constant op that
/// `create` makes for a folded constant.
pub(crate) fn materialize(
    folded: Vec<Folded>,
    types: &[Type],
    mut create: impl FnMut(Op) -> Value,
) -> Vec<Value> {
    let mut values = Vec::with_capacity(folded.len());
    for (result, ty) in folded.into_iter().zip(types) {
        values.push(match result {
            Folded::Value(value) => value,
            Folded::Constant(dense) => create(
                constant_op(dense, ty).unwrap_or_else(|err| panic!("building a constant: {err:#}")),
            ),
        });
    }
    values
}
