//! `widen-narrow-floats`: the ops on bf16 and f16 computed in f32.

use anyhow::Result;

use crate::dialects::{arith, tensor};
use crate::ir::{DType, Value};
use crate::passes::{pass, run_patterns, Pass};
use crate::require;
use crate::rewrite::{
    pattern, Has, NoMatch, Pattern, PatternSet, Rewriter, Rewrites, Rewriting, WithTrait,
};
use crate::traits::NarrowInF32;

/// Computes each op with [`NarrowInF32`] whose operands are bf16 or f16 in f32: it casts the
/// operands to f32, applies the op, and casts a float result back, which rounds it once, as the
/// op's semantics do. A backend with no arithmetic on those floats runs it before it decides how
/// to compute the ops. The casts are work of their own, so it is a pass, not part of
/// `canonicalize`.
#[pass(
    name = "widen-narrow-floats",
    stats(widened = "ops on bf16 or f16 computed in f32")
)]
pub struct WidenNarrowFloats;

fn populate_widen_narrow_floats_patterns(set: &mut PatternSet) {
    set.add(ComputeInF32);
}

impl Pass for WidenNarrowFloats {
    fn rewrites(&self) -> Rewrites {
        Rewrites::patterns(populate_widen_narrow_floats_patterns)
    }

    fn run(&self, module: &mut Rewriting) -> Result<bool> {
        run_patterns(module, self.rewrites(), &Self::WIDENED)
    }
}

/// An op on bf16 or f16 operands is the op on their f32 values, its float result cast back to the
/// operands' dtype.
#[pattern]
struct ComputeInF32;

impl Pattern for ComputeInF32 {
    type Root = Has<NarrowInF32>;

    fn rewrite(
        &self,
        root: WithTrait<NarrowInF32>,
        rewriter: &mut Rewriter,
    ) -> Result<(), NoMatch> {
        require!(let Some(first) = root.operands.first(), "the op takes no operand");
        let scalar = first.value_type().as_scalar().is_some();
        require!(let Some(dtype) = first.value_type().dtype(), "the operand has no dtype");
        require!(
            dtype.computes_in_f32(),
            "the op computes on its operands' dtype"
        );
        let mut widened = Vec::with_capacity(root.operands.len());
        for operand in &root.operands {
            widened.push(cast(rewriter, operand.id(), scalar, DType::F32));
        }
        let wide = rewriter.create_op(root.op.clone(), widened);
        let mut results = Vec::with_capacity(wide.len());
        for value in wide {
            let result = match root.fact.rounds {
                true => cast(rewriter, value, scalar, dtype),
                false => value,
            };
            results.push(result);
        }
        rewriter.replace_op(&root, results);
        Ok(())
    }
}

/// `value` cast to `to`: with `arith.cast` when it is a scalar, else with `tensor.cast`.
fn cast(rewriter: &mut Rewriter, value: Value, scalar: bool, to: DType) -> Value {
    match scalar {
        true => rewriter.create(arith::Cast { to }, value),
        false => rewriter.create(tensor::Cast { to }, value),
    }
}
