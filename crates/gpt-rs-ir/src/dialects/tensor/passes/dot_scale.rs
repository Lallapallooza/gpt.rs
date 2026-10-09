//! Moving a static scale of a `tensor.dot_general` result into its weight.

use anyhow::Result;

use crate::dialects::tensor::ir::shape::free_axes;
use crate::dialects::tensor::{
    Binary, BinaryKind, BroadcastTo, Cast, DotAxes, DotGeneral, Reshape,
};
use crate::module::{Module, Value};
use crate::op::Matched;
use crate::passes::{pass, Greedy, Pass, Rewrites};
use crate::rewrite::{pattern, Pattern, PatternSet, Rewriter};
use crate::traits::Movement;
use crate::types::{Dim, TensorType};

/// `dot_general(x, w) * broadcast_to(s)` becomes `dot_general(x, w * s')` when `w` and `s` are
/// static and `s` varies only along result axes that come from `w`. `s'` is `s` laid out along
/// those axes of `w` and broadcast to its shape. The scaled weight is static and reads a
/// parameter, so `outline-static` moves it to load time.
///
/// A weight read through a cast is left alone: scaling it would store a widened copy.
#[pattern]
struct ScaleIntoDot;

/// The extents of `s` laid out along the axes of `w`: each axis of `s` that is not 1 goes to the
/// free axis of `w` whose result axis the broadcast aligns it with. `None` when `s` varies along
/// a result axis that does not come from `w`.
fn scale_dims(dot: &DotAxes, w: &TensorType, s: &[usize], out_rank: usize) -> Option<Vec<usize>> {
    // The result axes are the batch axes, the free axes of `x`, then the free axes of `w`.
    let w_free = free_axes(w.rank(), &dot.batch_rhs, &dot.contract_rhs);
    let first_w_axis = out_rank - w_free.len();
    // `broadcast_to` aligns `s` with the trailing result axes.
    let offset = out_rank - s.len();
    let mut dims = vec![1; w.rank()];
    for (axis, &extent) in s.iter().enumerate().filter(|&(_, &e)| e != 1) {
        let free = (offset + axis).checked_sub(first_w_axis)?;
        dims[w_free[free]] = extent;
    }
    Some(dims)
}

/// Whether the elements of `value` come from a `tensor.cast` through ops that only move them.
fn moves_cast(module: &Module, value: Value) -> bool {
    if module.def_as::<Cast>(value).is_some() {
        return true;
    }
    let Some(inst) = module.def_inst(value) else {
        return false;
    };
    let Some(movement) = module.op(inst).get_trait::<Movement>() else {
        return false;
    };
    let operands = module.operands(inst);
    let data = movement.data.map_or(operands, |count| &operands[..count]);
    data.iter().any(|&operand| moves_cast(module, operand))
}

impl Pattern for ScaleIntoDot {
    type Root = Binary;
    fn rewrite(&self, root: Matched<Binary>, rw: &mut Rewriter<'_>) -> Result<bool> {
        if root.attrs.kind != BinaryKind::Mul {
            return Ok(false);
        }
        let module = &*rw.module;
        // A static operand is on the right of a commutative op.
        let (product, scale) = (root.operands.lhs, root.operands.rhs);
        let (Some(dot), Some(broadcast)) = (
            module.def_as::<DotGeneral>(product),
            module.def_as::<BroadcastTo>(scale),
        ) else {
            return Ok(false);
        };
        let (x, w, s) = (dot.operands.lhs, dot.operands.rhs, broadcast.operands.input);
        // Other users of the product need it unscaled, and scaling a cast weight would store a
        // widened copy of a narrower one.
        let applies = module.is_static(w)
            && module.is_static(s)
            && module.reads_param(w)
            && module.uses(product).len() == 1
            && !moves_cast(module, w);
        if !applies {
            return Ok(false);
        }
        let tensor = |v| module.ty(v).as_tensor().expect("a tensor");
        let (w_ty, s_ty, out_ty) = (tensor(w), tensor(s), tensor(root.results.output));
        if w_ty.dtype != out_ty.dtype {
            return Ok(false);
        }
        let (Some(w_dims), Some(s_dims)) = (w_ty.static_dims(), s_ty.static_dims()) else {
            return Ok(false);
        };
        let axes = dot.attrs.axes(tensor(x).rank(), w_ty.rank())?;
        let Some(dims) = scale_dims(&axes, w_ty, &s_dims, out_ty.rank()) else {
            return Ok(false);
        };
        let shape = dims.into_iter().map(Dim::Static).collect();
        let w_shape = w_dims.into_iter().map(Dim::Static).collect();
        let at = root.inst;
        let aligned = rw.insert_before(at, Reshape { shape }, s)?.output;
        let expanded = rw
            .insert_before(at, BroadcastTo { shape: w_shape }, aligned)?
            .output;
        let mul = Binary {
            kind: BinaryKind::Mul,
        };
        let scaled = rw.insert_before(at, mul, (w, expanded))?.output;
        let out = rw.insert_before(at, dot.attrs, (x, scaled))?.output;
        rw.replace(at, &[out])?;
        Ok(true)
    }
}

/// Adds the patterns that move static scales of dot results into the weights.
pub fn populate_dot_scale_patterns(set: &mut PatternSet) {
    set.add(ScaleIntoDot);
}

/// Moves the static scales of dot results into the weights, so a convolution absorbs the
/// BatchNorm after it and a run saves the multiply.
///
/// The two sides are equal in real arithmetic, not bit for bit: the scale rounds into every
/// weight before the sum, instead of into the sum once. So it is a pass of its own, not part of
/// `canonicalize`.
#[pass(name = "dot-scale-fold", stats(moved = "scales moved into a weight"))]
pub struct DotScaleFold;

impl Pass for DotScaleFold {
    fn rewrites(&self) -> Rewrites {
        let mut patterns = PatternSet::new();
        populate_dot_scale_patterns(&mut patterns);
        Rewrites {
            folds: false,
            patterns,
        }
    }

    fn run(&self, module: &mut Module) -> Result<bool> {
        let report = Greedy::new(self.rewrites()).erase_dead().run(module)?;
        DotScaleFold::MOVED.add(report.rewritten);
        Ok(report.changed())
    }
}
