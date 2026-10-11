//! Moving a static scale of a `tensor.dot_general` result into its weight.

use anyhow::Result;

use crate::dialects::tensor::{BroadcastTo, Cast, DotAxes, DotGeneral, Mul, Side};
use crate::module::{Module, ValueRef};
use crate::op::Matched;
use crate::passes::{pass, run_patterns, Pass, Rewrites};
use crate::require;
use crate::rewrite::{matched, pattern, Build, NoMatch, Pattern, PatternSet, Rewriter};
use crate::traits::Movement;
use crate::types::TensorType;

/// `dot_general(x, w) * broadcast_to(s)` becomes `dot_general(x, w * s')` when `w` and `s` are
/// static and `s` varies only along result axes that come from `w`. `s'` is `s` laid out along
/// those axes of `w` and broadcast to its shape. The scaled weight is static and reads a
/// parameter, so `outline-static` moves it to load time.
///
/// A weight read through a cast is left alone: scaling it would store a widened copy.
#[pattern]
struct ScaleIntoDot;

/// The extents of the scale laid out along the axes of the weight. Each axis of the scale that is
/// not 1 aligns with one result axis. It goes to the free axis of the weight that gives that result
/// axis. `None` when the scale varies along a result axis that does not come from the weight.
fn scale_dims(
    axes: &DotAxes,
    weight: &TensorType,
    scale_dims: &[usize],
    result_rank: usize,
) -> Option<Vec<usize>> {
    // The result axes are the batch axes, the free axes of the input, then the free axes of the
    // weight.
    let weight_free = axes.free(Side::Rhs);
    let first_weight_axis = result_rank - weight_free.len();
    // `broadcast_to` aligns the scale with the trailing result axes.
    let first_scale_axis = result_rank - scale_dims.len();
    let mut dims = vec![1; weight.rank()];
    for (axis, &extent) in scale_dims.iter().enumerate() {
        if extent == 1 {
            continue;
        }
        let free = (first_scale_axis + axis).checked_sub(first_weight_axis)?;
        dims[weight_free[free]] = extent;
    }
    Some(dims)
}

/// Whether the elements of `value` come from a `tensor.cast` through ops that only move them.
fn moves_cast(value: ValueRef<'_>) -> bool {
    if value.defining_op::<Cast>().is_some() {
        return true;
    }
    let Some(inst) = value.defining_inst() else {
        return false;
    };
    let module = value.module();
    let Some(movement) = module.op(inst).get_trait::<Movement>() else {
        return false;
    };
    let operands = module.operands(inst);
    let data = movement.data.map_or(operands, |count| &operands[..count]);
    data.iter()
        .any(|&operand| moves_cast(module.value_ref(operand)))
}

impl Pattern for ScaleIntoDot {
    type Root = Mul;

    fn rewrite(&self, mul: Matched<'_, Mul>, rewriter: &mut Rewriter<'_>) -> Result<(), NoMatch> {
        // A static operand is on the right of a commutative op.
        matched!(let dot @ DotGeneral(input, weight) = mul.lhs() else {
            return Err("lhs is not a dot_general".into());
        });
        matched!(let BroadcastTo(scale) = mul.rhs() else {
            return Err("rhs is not a broadcast_to".into());
        });
        require!(weight.known_before_run(), "the weight changes at run time");
        require!(scale.known_before_run(), "the scale changes at run time");
        require!(
            weight.depends_on_parameter(),
            "the weight depends on no parameter"
        );
        require!(
            mul.lhs().has_one_use(),
            "other users need the product unscaled"
        );
        require!(
            !moves_cast(weight),
            "scaling a cast weight would store a widened copy"
        );
        let (weight_type, product_type) = (weight.tensor_type(), mul.output().tensor_type());
        require!(
            weight_type.dtype == product_type.dtype,
            "the weight has another dtype"
        );
        require!(let Some(weight_dims) = weight_type.static_dims(), "the weight has a dynamic extent");
        require!(let Some(scale_extents) = scale.tensor_type().static_dims(), "the scale has a dynamic extent");
        let axes = dot.axes();
        require!(
            let Some(aligned_shape) = scale_dims(&axes, weight_type, &scale_extents, product_type.rank()),
            "the scale varies along an axis that does not come from the weight"
        );

        let weight_scale = rewriter
            .on(scale)
            .reshape(aligned_shape)
            .broadcast_to(weight_dims)
            .value();
        let scaled_weight = rewriter.create(Mul, (weight, weight_scale));
        let product = rewriter.create(dot.attrs().clone(), (input, scaled_weight));
        rewriter.replace_op(&mul, product);
        Ok(())
    }
}

/// Moves the static scales of dot results into the weights, so a convolution absorbs the
/// BatchNorm after it and a run saves the multiply.
///
/// The two sides are equal in real arithmetic, not bit for bit: the scale rounds into every
/// weight before the sum, instead of into the sum once. So it is a pass of its own, not part of
/// `canonicalize`.
#[pass(name = "dot-scale-fold", stats(moved = "scales moved into a weight"))]
pub struct DotScaleFold;

fn populate_dot_scale_fold_patterns(set: &mut PatternSet) {
    set.add(ScaleIntoDot);
}

impl Pass for DotScaleFold {
    fn rewrites(&self) -> Rewrites {
        Rewrites::patterns(populate_dot_scale_fold_patterns)
    }

    fn run(&self, module: &mut Module) -> Result<bool> {
        run_patterns(module, self.rewrites(), &Self::MOVED)
    }
}
