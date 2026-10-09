//! Rewrites of `tensor.reduce_*` that follow from the traits of the op it combines with.

use anyhow::Result;
use smallvec::SmallVec;

use crate::dialects::tensor::ir::shape::{axis_index, free_axes};
use crate::dialects::tensor::{Binary, BroadcastTo, Concat, Reduce, ReduceKind, Reshape};
use crate::module::Value;
use crate::op::Matched;
use crate::rewrite::{pattern, Pattern, Rewriter};
use crate::traits::{Associative, Commutative, Idempotent};
use crate::types::{DType, Dim, TensorType};
use crate::Op;

/// The op `kind` combines two partial results with.
fn combiner(kind: ReduceKind) -> Op {
    Binary {
        kind: kind.combiner(),
    }
    .into()
}

/// Whether the elements of a `kind` reduction over `dtype` may combine in any order and grouping.
fn combines_in_any_order(kind: ReduceKind, dtype: DType) -> bool {
    let op = combiner(kind);
    op.get_trait::<Associative>()
        .is_some_and(|a| a.covers(dtype))
        && op.has_trait::<Commutative>()
}

/// The input type of a reduction that accumulates and returns in its input dtype.
fn plain_input<'a>(rw: &'a Rewriter<'_>, reduce: &Matched<Reduce>) -> Option<&'a TensorType> {
    let input = rw.module.ty(reduce.operands.input).as_tensor()?;
    let accum = reduce
        .attrs
        .accum
        .unwrap_or(input.dtype.default_accumulator());
    (accum == input.dtype && reduce.attrs.out.is_none()).then_some(input)
}

/// `value` reshaped to the type of `at`, inserted before the op that defines `at`. `value` itself
/// when it has that type already.
fn reshaped_to(rw: &mut Rewriter<'_>, at: Value, value: Value) -> Result<Value> {
    let want = rw.module.ty(at).as_tensor().expect("a tensor").clone();
    if rw.module.ty(value).as_tensor() == Some(&want) {
        return Ok(value);
    }
    let inst = rw.module.def_inst(at).expect("a result");
    let reshape = Reshape { shape: want.dims };
    Ok(rw.insert_before(inst, reshape, value)?.output)
}

/// `reduce(reduce(x, a), b)` is one reduction of `x` over the axes of both, reshaped to the
/// result, when the elements may combine in any order.
#[pattern(canonical)]
struct MergeReductions;

impl Pattern for MergeReductions {
    type Root = Reduce;
    fn rewrite(&self, root: Matched<Reduce>, rw: &mut Rewriter<'_>) -> Result<bool> {
        let Some(inner) = rw.def_as::<Reduce>(root.operands.input) else {
            return Ok(false);
        };
        let (Some(_), Some(x)) = (plain_input(rw, &root), plain_input(rw, &inner)) else {
            return Ok(false);
        };
        let applies = inner.attrs.kind == root.attrs.kind
            && combines_in_any_order(root.attrs.kind, x.dtype)
            && rw.module.uses(root.operands.input).len() == 1;
        if !applies {
            return Ok(false);
        }
        let rank = x.rank();
        let inner_axes = inner.attrs.reduced_axes(rank)?;
        // The root's axes, as axes of `x`.
        let outer: SmallVec<[usize; 4]> = if inner.attrs.keepdims {
            root.attrs.reduced_axes(rank)?
        } else {
            let kept = free_axes(rank, &inner_axes, &[]);
            let root_axes = root.attrs.reduced_axes(kept.len())?;
            root_axes.iter().map(|&axis| kept[axis]).collect()
        };
        let mut axes: Vec<usize> = inner_axes.iter().chain(&outer).copied().collect();
        axes.sort_unstable();
        axes.dedup();
        let merged = Reduce {
            kind: root.attrs.kind,
            axes: axes.into_iter().map(|axis| axis as i64).collect(),
            keepdims: inner.attrs.keepdims && root.attrs.keepdims,
            accum: None,
            out: None,
        };
        let merged = rw.insert_before(root.inst, merged, inner.operands.input)?;
        let output = reshaped_to(rw, root.results.output, merged.output)?;
        rw.replace(root.inst, &[output])?;
        Ok(true)
    }
}

/// A reduction of `broadcast_to(x)` is `x` reshaped to the result when the combining op is
/// idempotent, the reduction covers every axis the broadcast repeats `x` along, and its other
/// axes have extent 1. Each repeated axis must have a static extent above zero, since a
/// reduction over no elements gives its initial value.
#[pattern(canonical)]
struct ReduceOfBroadcast;

impl Pattern for ReduceOfBroadcast {
    type Root = Reduce;
    fn rewrite(&self, root: Matched<Reduce>, rw: &mut Rewriter<'_>) -> Result<bool> {
        let Some(broadcast) = rw.def_as::<BroadcastTo>(root.operands.input) else {
            return Ok(false);
        };
        if plain_input(rw, &root).is_none() || !combiner(root.attrs.kind).has_trait::<Idempotent>()
        {
            return Ok(false);
        }
        let x = rw
            .module
            .ty(broadcast.operands.input)
            .as_tensor()
            .expect("a tensor");
        let shape = &broadcast.attrs.shape;
        let added = shape.len() - x.rank();
        let one = Dim::Static(1);
        let source = |axis: usize| (axis >= added).then(|| &x.dims[axis - added]);
        let repeated = |axis: usize| source(axis).is_none_or(|d| *d == one) && shape[axis] != one;
        let axes = root.attrs.reduced_axes(shape.len())?;
        let reduced = |axis: usize| axes.contains(&axis);
        let exact = (0..shape.len()).all(|axis| {
            let ok_extent = matches!(shape[axis], Dim::Static(n) if n > 0);
            match (repeated(axis), reduced(axis)) {
                (true, true) => ok_extent,
                (true, false) => false,
                (false, true) => shape[axis] == one,
                (false, false) => true,
            }
        });
        if !exact {
            return Ok(false);
        }
        let output = reshaped_to(rw, root.results.output, broadcast.operands.input)?;
        rw.replace(root.inst, &[output])?;
        Ok(true)
    }
}

/// A reduction of `concat(x1, x2, ...)` that reduces the concatenated axis is the combining op
/// applied to the reductions of the parts, when the elements may combine in any order.
#[pattern(canonical)]
struct ReduceOfConcat;

impl Pattern for ReduceOfConcat {
    type Root = Reduce;
    fn rewrite(&self, root: Matched<Reduce>, rw: &mut Rewriter<'_>) -> Result<bool> {
        let Some(concat) = rw.def_as::<Concat>(root.operands.input) else {
            return Ok(false);
        };
        let Some(input) = plain_input(rw, &root) else {
            return Ok(false);
        };
        let rank = input.rank();
        let applies = concat.operands.inputs.len() > 1
            && root
                .attrs
                .reduced_axes(rank)?
                .contains(&axis_index(concat.attrs.axis, rank)?)
            && combines_in_any_order(root.attrs.kind, input.dtype)
            && rw.module.uses(root.operands.input).len() == 1;
        if !applies {
            return Ok(false);
        }
        let mut partials = Vec::with_capacity(concat.operands.inputs.len());
        for &part in &concat.operands.inputs {
            partials.push(
                rw.insert_before(root.inst, root.attrs.clone(), part)?
                    .output,
            );
        }
        let mut combined = partials[0];
        for &partial in &partials[1..] {
            let inst = rw.insert_op_before(
                root.inst,
                combiner(root.attrs.kind),
                vec![combined, partial],
            )?;
            combined = rw.module.results(inst)[0];
        }
        rw.replace(root.inst, &[combined])?;
        Ok(true)
    }
}
