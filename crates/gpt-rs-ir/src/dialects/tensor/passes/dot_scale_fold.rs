//! Moving a static scale of a `tensor.dot_general` result into its weight.

use anyhow::Result;

use crate::dialects::tensor::{
    BroadcastTo, DotAxes, DotGeneral, DotGeneralOp, Mul, MulOp, Reshape, Side, Transpose,
};
use crate::ir::{Dim, Inst, Ranked, TensorType, Type, ValueRef};
use crate::passes::{pass, run_patterns, OutlineStatic, Pass};
use crate::require;
use crate::rewrite::{
    matched, pattern, Build, NoMatch, Pattern, PatternSet, Rewriter, Rewrites, Rewriting,
};

/// `dot_general(x, w) * broadcast_to(s)` becomes `dot_general(x, w * s')` when `w` and `s` are
/// static and `s` varies only along result axes that come from `w`: its batch axes and its free
/// axes. `s'` is `s` laid out along those axes of `w` and broadcast to its shape. The product may
/// reach the multiply through transposes and reshapes that only add or remove axes of extent 1,
/// each the one user of the value before it, which then move the scaled product. The scaled weight
/// is static and reads a parameter, so `outline-static` moves it to load time, and the rewrite
/// applies only where that pass would store it ([`OutlineStatic::would_store`]): a weight cast to a
/// wider dtype, which would take more bytes than its parameter, stays unscaled.
#[pattern]
struct ScaleIntoDot;

/// The extents of the scale laid out along the axes of the weight. `scaled_axes` gives, in order,
/// the result axis and the extent of each axis of the scale that is not 1. Each extent goes on the
/// weight axis that lands on that result axis ([`DotAxes::result_axes`]). `None` when the scale
/// varies along a result axis of the input, or its axes go to the weight's out of order.
fn scale_dims(
    axes: &DotAxes,
    weight: &TensorType,
    scaled_axes: &[(usize, usize)],
) -> Option<Ranked<usize>> {
    let landing = axes.result_axes(Side::Rhs);
    let mut dims = Ranked::new();
    dims.resize(weight.rank(), 1);
    let mut previous: Option<usize> = None;
    for &(result_axis, extent) in scaled_axes {
        let weight_axis = landing
            .iter()
            .position(|&lands| lands == Some(result_axis))?;
        if previous.is_some_and(|earlier| earlier >= weight_axis) {
            return None;
        }
        previous = Some(weight_axis);
        dims[weight_axis] = extent;
    }
    Some(dims)
}

/// A `tensor.dot_general` whose result transposes and reshapes move: the dot, the moving ops from
/// the dot on, and the result axis that each axis of the moved value holds, `None` for an axis of
/// extent 1 that a reshape added.
struct MovedDot<'m> {
    dot: DotGeneralOp<'m>,
    moves: Vec<Inst>,
    axis_of: Ranked<Option<usize>>,
}

/// The dot whose result `value` is, moved by transposes and by reshapes that only add or remove
/// axes of extent 1, each the one user of the value before it.
fn moved_dot(value: ValueRef<'_>) -> Option<MovedDot<'_>> {
    let mut current = value;
    let mut axis_of = Ranked::new();
    for axis in 0..value.tensor_type().rank() {
        axis_of.push(Some(axis));
    }
    let mut moves = Vec::new();
    loop {
        if let Some(dot) = current.defining_op::<DotGeneral>() {
            moves.reverse();
            return Some(MovedDot {
                dot,
                moves,
                axis_of,
            });
        }
        if let Some(transpose) = current.defining_op::<Transpose>() {
            let input = transpose.input();
            if !input.has_one_use() {
                return None;
            }
            for axis in axis_of.iter_mut().flatten() {
                *axis = transpose.perm[*axis];
            }
            moves.push(transpose.inst());
            current = input;
            continue;
        }
        let reshape = current.defining_op::<Reshape>()?;
        let input = reshape.input();
        if !input.has_one_use() {
            return None;
        }
        let kept_axes = |dims: &[Dim]| -> Ranked<usize> {
            let mut kept = Ranked::new();
            for (axis, dim) in dims.iter().enumerate() {
                if !dim.is_one() {
                    kept.push(axis);
                }
            }
            kept
        };
        let output_dims = &current.tensor_type().dims;
        let input_dims = &input.tensor_type().dims;
        let (output_kept, input_kept) = (kept_axes(output_dims), kept_axes(input_dims));
        if output_kept.len() != input_kept.len() {
            return None;
        }
        for (&output_axis, &input_axis) in output_kept.iter().zip(&input_kept) {
            if output_dims[output_axis] != input_dims[input_axis] {
                return None;
            }
        }
        for axis in axis_of.iter_mut() {
            let kept =
                axis.and_then(|position| output_kept.iter().position(|&kept| kept == position));
            *axis = kept.map(|index| input_kept[index]);
        }
        moves.push(reshape.inst());
        current = input;
    }
}

impl Pattern for ScaleIntoDot {
    type Root = Mul;

    fn rewrite(&self, mul: MulOp, rewriter: &mut Rewriter) -> Result<(), NoMatch> {
        // A static operand is on the right of a commutative op.
        require!(
            let Some(MovedDot { dot, moves, axis_of }) = moved_dot(mul.lhs()),
            "lhs is not a dot_general, or one that only transposes and reshapes move"
        );
        let (input, weight) = (dot.lhs(), dot.rhs());
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
        let (weight_type, product_type) = (weight.tensor_type(), mul.output().tensor_type());
        require!(
            weight_type.dtype == product_type.dtype,
            "the weight has another dtype"
        );
        require!(
            OutlineStatic::would_store(
                mul.module(),
                &Type::Tensor(weight_type.clone()),
                &[weight.id(), scale.id()]
            ),
            "outline-static would not store the scaled weight, which takes more bytes than the \
             parameters it reads"
        );
        require!(let Some(weight_dims) = weight_type.static_dims(), "the weight has a dynamic extent");
        require!(let Some(scale_extents) = scale.tensor_type().static_dims(), "the scale has a dynamic extent");
        // `broadcast_to` aligns the scale with the trailing axes of the product.
        let first_scale_axis = product_type.rank() - scale_extents.len();
        let mut scaled_axes: Ranked<(usize, usize)> = Ranked::new();
        for (axis, &extent) in scale_extents.iter().enumerate() {
            if extent == 1 {
                continue;
            }
            require!(
                let Some(result_axis) = axis_of[first_scale_axis + axis],
                "the scale varies along an axis that a reshape added"
            );
            scaled_axes.push((result_axis, extent));
        }
        let axes = dot.axes();
        require!(
            let Some(aligned_shape) = scale_dims(&axes, weight_type, &scaled_axes),
            "the scale varies along an axis that does not come from the weight"
        );

        let weight_scale = rewriter
            .on(scale)
            .reshape(aligned_shape)
            .broadcast_to(weight_dims)
            .value();
        let scaled_weight = rewriter.create(Mul, (weight, weight_scale));
        let mut moved = rewriter.create(dot.attrs().clone(), (input, scaled_weight));
        let module = mul.module();
        for &moving in &moves {
            moved = rewriter.create_op(module.op(moving).clone(), [moved])[0];
        }
        rewriter.replace_op(&mul, moved);
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

    fn run(&self, module: &mut Rewriting) -> Result<bool> {
        run_patterns(module, self.rewrites(), &Self::MOVED)
    }
}
