use std::marker::PhantomData;

use super::utils::elementwise;
use crate::dialects::arith;
use crate::dialects::tensor::{
    Binary, Cast, CastOp, Compare, Select, SelectOp, StopGradient, StopGradientOp, Unary,
};
use crate::ir::OpRef;
use crate::rewrite::{pattern, NoMatch, Pattern, Rewriter};

/// `tensor.stop_gradient` is its input: a forward computation has no gradients.
#[pattern]
pub(super) struct StopGradientToTile;

impl Pattern for StopGradientToTile {
    type Root = StopGradient;

    fn rewrite(&self, stop: StopGradientOp, rewriter: &mut Rewriter) -> Result<(), NoMatch> {
        rewriter.replace_op(&stop, stop.input());
        Ok(())
    }
}

/// A `tensor` op of one operand is a parallel block with the `arith` op it applies.
#[pattern]
pub(super) struct UnaryToTile<T>(pub(super) PhantomData<T>);

impl<T: Unary> Pattern for UnaryToTile<T> {
    type Root = T;

    fn rewrite(&self, unary: OpRef<T>, rewriter: &mut Rewriter) -> Result<(), NoMatch> {
        let inputs = [unary.input().id()];
        let result = elementwise(
            rewriter,
            &inputs,
            unary.output().tensor_type(),
            |body, args| body.create(T::Scalar::default(), args[0]),
        );
        rewriter.replace_op(&unary, result);
        Ok(())
    }
}

/// A `tensor` op of two operands is a parallel block with the `arith` op it applies.
#[pattern]
pub(super) struct BinaryToTile<T>(pub(super) PhantomData<T>);

impl<T: Binary> Pattern for BinaryToTile<T> {
    type Root = T;

    fn rewrite(&self, binary: OpRef<T>, rewriter: &mut Rewriter) -> Result<(), NoMatch> {
        let inputs = [binary.lhs().id(), binary.rhs().id()];
        let result = elementwise(
            rewriter,
            &inputs,
            binary.output().tensor_type(),
            |body, args| body.create(T::Scalar::default(), (args[0], args[1])),
        );
        rewriter.replace_op(&binary, result);
        Ok(())
    }
}

/// A `tensor` comparison is a parallel block with the `arith` comparison it applies.
#[pattern]
pub(super) struct CompareToTile<T>(pub(super) PhantomData<T>);

impl<T: Compare> Pattern for CompareToTile<T> {
    type Root = T;

    fn rewrite(&self, compare: OpRef<T>, rewriter: &mut Rewriter) -> Result<(), NoMatch> {
        let inputs = [compare.lhs().id(), compare.rhs().id()];
        let result = elementwise(
            rewriter,
            &inputs,
            compare.output().tensor_type(),
            |body, args| body.create(T::Scalar::default(), (args[0], args[1])),
        );
        rewriter.replace_op(&compare, result);
        Ok(())
    }
}

/// `tensor.select` is a parallel block with `arith.select`.
#[pattern]
pub(super) struct SelectToTile;

impl Pattern for SelectToTile {
    type Root = Select;

    fn rewrite(&self, select: SelectOp, rewriter: &mut Rewriter) -> Result<(), NoMatch> {
        let inputs = [
            select.pred().id(),
            select.on_true().id(),
            select.on_false().id(),
        ];
        let result = elementwise(
            rewriter,
            &inputs,
            select.output().tensor_type(),
            |body, args| body.create(arith::Select, (args[0], args[1], args[2])),
        );
        rewriter.replace_op(&select, result);
        Ok(())
    }
}

/// `tensor.cast` is a parallel block with `arith.cast`.
#[pattern]
pub(super) struct CastToTile;

impl Pattern for CastToTile {
    type Root = Cast;

    fn rewrite(&self, cast: CastOp, rewriter: &mut Rewriter) -> Result<(), NoMatch> {
        let to = cast.to;
        let inputs = [cast.input().id()];
        let result = elementwise(
            rewriter,
            &inputs,
            cast.output().tensor_type(),
            |body, args| body.cast(args[0], to),
        );
        rewriter.replace_op(&cast, result);
        Ok(())
    }
}
