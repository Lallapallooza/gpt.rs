use std::marker::PhantomData;

use crate::dialects::arith;
use crate::dialects::arith::math::Draw;
use crate::dialects::tensor::Rng;
use crate::dialects::tile::{par_iters, Block, Fold, Read, Write};
use crate::ir::{DType, Dim, IndexExpr, Matched};
use crate::require;
use crate::rewrite::{pattern, NoMatch, Pattern, Rewriter};

/// A `tensor.rng_*` op is a block that draws element `i` of the state's stream with the `arith`
/// op of the same name, and a block that advances the state's counter by the blocks read.
#[pattern]
pub(super) struct RngToTile<T>(pub(super) PhantomData<T>);

impl<T: Rng> Pattern for RngToTile<T> {
    type Root = T;

    fn rewrite(&self, rng: Matched<'_, T>, rewriter: &mut Rewriter<'_>) -> Result<(), NoMatch> {
        let state = rng.state().id();
        let output = rng.output().tensor_type();
        let dtype = rng.dtype;
        require!(
            let Ok(words_per_element) = T::Scalar::words_per_element(dtype),
            "{} draws no {dtype} values",
            T::INFO.name
        );
        let (iters, coords) = par_iters(&output.dims);
        let position = IndexExpr::linearize(&coords, &output.dims);
        let key = Read::new(0, vec![IndexExpr::constant(0)]);
        let counter = Read::new(0, vec![IndexExpr::constant(1)]);
        let draws = Block::new(
            iters,
            vec![key, counter],
            vec![Fold::plain(Write::new(coords))],
        );
        let drawn = draws.build(rewriter, vec![state], |body, args| {
            let index = body.index(position, DType::Ui64);
            let draw = T::Scalar::from(arith::RngAttrs { dtype });
            vec![body.create(draw, (args[0], args[1], index))]
        })[0];
        // The blocks of the stream the draws read: `ceil(count * words_per_element / 4)`.
        let words = IndexExpr::volume(&output.dims) * IndexExpr::constant(words_per_element as i64);
        let blocks =
            IndexExpr::floordiv(words + IndexExpr::constant(3), IndexExpr::constant(4)).simplify();
        let (iters, coords) = par_iters(&[Dim::Static(2)]);
        let which = coords[0].clone();
        let read = Read::new(0, coords.clone());
        let advance = Block::new(iters, vec![read], vec![Fold::plain(Write::new(coords))]);
        let next_state = advance.build(rewriter, vec![state], |body, args| {
            let which = body.index(which, DType::Si64);
            let zero = body.int(DType::Si64, 0);
            let is_key = body.create(arith::Equal, (which, zero));
            let blocks = body.index(blocks, DType::Ui64);
            let advanced = body.create(arith::Add, (args[0], blocks));
            vec![body.create(arith::Select, (is_key, args[0], advanced))]
        })[0];
        rewriter.replace_op(&rng, [next_state, drawn]);
        Ok(())
    }
}
